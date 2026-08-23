"""Foreground supervision for two-node runtimes under systemd.

``lmswitch serve <name>`` is what a ``restart:``-managed unit runs. For
single-node runtimes the wrapper either execs the server (vLLM) or watches a
child process (llama). A dual model has neither shape: both ranks are detached
containers, the head is the one that owns the API port, and the worker lives on
another host over ssh. So the wrapper here starts the pair, then blocks on the
head container and exits the moment it is gone — systemd's ``Restart=`` is what
brings the pair back, and an exit is the only signal it understands.

Two failure modes this exists to avoid:

* A supervisor that sleeps forever keeps the unit looking healthy while the
  model is unreachable (the incident behind ``tests/test_cmd_serve.py``).
* A half-dead pair. If the head is gone the worker is still holding the peer's
  GPU, and the next start cannot bind the TP group. Every exit path here tears
  down both ranks first.

At boot the peer, the weights mount and dockerd are usually not ready yet. A
user unit cannot order itself after them (mounts and ``network-online.target``
live in the system manager, not the user one), so the wait is a gate in here
instead of ``After=``/``RequiresMountsFor=`` in the unit.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

from lmswitch.runtimes.base import runtime_registry
from lmswitch.system.checks import _docker_container
from lmswitch.system.memory import _memory_check


def _weights_ready(yaml: dict) -> bool:
    """Reports whether the head node's weights directory is populated.

    On this cluster ``model_path`` is usually an NFS mount of the peer's
    library, so an empty (or absent) directory means the mount has not come up
    yet — not that the model is missing.
    """
    raw = yaml.get("model_path")
    if not raw:
        return True
    path = os.path.expanduser(os.path.expandvars(str(raw)))
    try:
        return os.path.isdir(path) and bool(os.listdir(path))
    except OSError:
        return False


def _docker_ready() -> bool:
    """Reports whether the local docker daemon answers."""
    return subprocess.run(["docker", "info"], stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL, check=False).returncode == 0


def _worker_ready(yaml: dict) -> bool:
    """Reports whether the peer answers over ssh (BatchMode; never prompts)."""
    worker = yaml.get("worker_host")
    if not worker:
        return True
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", worker, "true"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
    ).returncode == 0


def _gate(name: str, yaml: dict, timeout: int, interval: int = 10) -> str | None:
    """Waits for the cluster preconditions, returning None once all pass.

    Returns:
        ``None`` when every precondition holds, or a string naming the one that
        was still failing when *timeout* expired.
    """
    checks = (
        ("docker daemon", lambda: _docker_ready()),
        ("weights mount", lambda: _weights_ready(yaml)),
        (f"worker {yaml.get('worker_host')}", lambda: _worker_ready(yaml)),
    )
    elapsed = 0
    while True:
        pending = [label for label, check in checks if not check()]
        if not pending:
            return None
        if elapsed >= timeout:
            return ", ".join(pending)
        print(f"  …waiting for {', '.join(pending)} ({elapsed}s)")
        time.sleep(interval)
        elapsed += interval


def _start_dual_foreground(name: str, yaml: dict, poll_interval: int = 5) -> None:
    """Foreground serve for dual runtimes — used by systemd.

    Blocks for the lifetime of the model. Never returns: every path out is a
    ``SystemExit``, so systemd sees the unit stop and applies its ``Restart=``
    policy.

    Args:
        name: Model id (the yaml stem).
        yaml: Parsed recipe.
        poll_interval: Seconds between head-container liveness checks.
    """
    runtime_name = yaml.get("runtime", "vllm-dual")
    runtime = runtime_registry.lookup(runtime_name)()

    def _teardown(signum, _frame):
        # `systemctl stop` sends SIGTERM here, not to the containers — without
        # this the pair keeps serving after the unit reports stopped.
        print(f"  signal {signum} received; stopping both ranks")
        runtime.stop(name, yaml)
        sys.exit(0)

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _teardown)

    try:
        gate_timeout = int(yaml.get("gate_timeout", 600))
    except (ValueError, TypeError):
        gate_timeout = 600
    blocked = _gate(name, yaml, gate_timeout)
    if blocked:
        sys.exit(f"{name} cannot start: {blocked} not ready after {gate_timeout}s")

    # systemd calls `lmswitch serve` directly, bypassing start_model — so the
    # RAM guard it applies has to be repeated here, or a restart-after-crash
    # would launch on top of whatever the user loaded meanwhile and OOM a
    # unified-memory box. _memory_check sizes dual runtimes per node from
    # gpu_memory_utilization, which is the figure that matters here.
    ok, why = _memory_check(name, yaml)
    if not ok and not yaml.get("force"):
        sys.exit(f"{name} refusing to start: {why}")

    state = runtime.start(name, yaml)
    if state.status != "ready":
        # Startup failed or timed out — drop the worker too, then exit so
        # systemd retries from a clean pair rather than a stale rank 1.
        runtime.stop(name, yaml)
        sys.exit(f"{name} failed to start ({state.status})")

    while True:
        time.sleep(poll_interval)
        if _docker_container(name) is None:
            runtime.stop(name, yaml)
            sys.exit(f"{name} head container exited; "
                     f"handing back to systemd for restart")
