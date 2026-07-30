"""llama.cpp dual-node (2x DGX Spark over CX7) runtime.

Serves one GGUF split across two Sparks with llama.cpp's RPC backend: the
head runs ``llama-server`` locally and offloads part of the layers to an
``rpc-server`` started over SSH on ``worker_host``. Unlike the vLLM dual
runtimes there is no container and no NCCL — the head reads the GGUF and
pushes the remote node's share of the tensors over the CX7 link (RDMA when
both binaries are built with ``GGML_RPC_RDMA=ON``), so the weights only need
to exist on this node.

Minimal YAML:

    runtime: llama-dual
    model: unsloth/Qwen3.5-397B-A17B-GGUF/UD-IQ4_NL/...-00001-of-00005.gguf
    port: 8105
    ctx: 8192
    worker_host: Gigabyte          # ssh alias
    worker_ip: 10.100.224.1        # worker's CX7 ip (the RPC endpoint)
    tensor_split: "0.45,0.55"      # CUDA0 (this node), RPC0 (worker)

Optional: ``rpc_port`` (50052), ``rpc_cache`` (true — the worker caches
received tensors under ``~/.cache/llama.cpp/rpc`` so restarts don't re-push
the whole share), ``rpc_threads`` (worker CPU threads), ``worker_rpc_bin``
(path to ``rpc-server`` on the worker), plus every first-class ``llama``
key (``ctx``, ``batch``, ``ubatch``, ``extra_args``, ``ready_timeout`` …).

``tensor_split`` order follows llama-server's device enumeration: local
CUDA devices first, then one ``RPC<n>`` entry per ``--rpc`` endpoint in the
order given. Verify with
``llama-server --list-devices --rpc <worker_ip>:<rpc_port>`` after a
llama.cpp upgrade — the order is not guaranteed across versions, and
getting it backwards silently loads the wrong share onto the wrong box.
"""

from __future__ import annotations

import socket
import subprocess
import time
from pathlib import Path

from lmswitch.system.io import RUN_DIR
from lmswitch.runtimes.base import RunningState
from lmswitch.runtimes.llama import LlamaRuntime

DEFAULT_RPC_PORT = 50052
DEFAULT_RPC_BIN = "~/utils/llama.cpp/build/bin/rpc-server"


def _ssh(host: str, cmd: str, **kw) -> subprocess.CompletedProcess:
    """Runs the shell string ``cmd`` on ``host`` over ssh (never prompts)."""
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, cmd],
        **kw,
    )


def _worker_path(raw: str) -> str:
    """Quotes a worker-side path for a remote shell, keeping ``~`` usable.

    ``shlex.quote`` would turn ``~/utils/...`` into a literal directory named
    ``~``; the worker's shell expands ``$HOME`` inside double quotes instead.
    """
    return '"' + str(raw).replace("~/", "$HOME/", 1).replace('"', r'\"') + '"'


def _port_open(host: str, port: int, timeout: float = 2.0) -> bool:
    """True when a TCP connect to ``host:port`` succeeds."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _pkill_cmd(rpc_port: int) -> str:
    """Kills only the ``rpc-server`` serving ``rpc_port`` on the worker.

    Scoped by port, not by binary name: a second llama-dual recipe on another
    RPC port must survive this one starting or stopping — the same collision
    the docker-name-scoped vLLM dual runtimes avoid.
    """
    return f'pkill -f "rpc-server.* -p {rpc_port}( |$)" || true'


def _wait_loaded(port: int, timeout: int, alive) -> str:
    """Polls ``/health`` until llama-server reports the weights are loaded.

    The shared ``_wait_ready`` helper only checks that the port answers, and
    llama-server binds it *before* loading — it replies 503 "Loading model"
    meanwhile. That gap is seconds for a local GGUF but many minutes here,
    where the worker's share is pushed over the wire, so a start would be
    announced ready while most of the model is still in flight.

    Args:
        port: llama-server's OpenAI-compatible port.
        timeout: Seconds to wait before giving up.
        alive: Zero-arg callable returning False once the server has exited.

    Returns:
        ``"ready"``, ``"dead"``, or ``"timeout"``.
    """
    import json
    import urllib.error
    import urllib.request

    elapsed = 0
    while elapsed < timeout:
        if not alive():
            return "dead"
        try:
            with urllib.request.urlopen(
                    f"http://localhost:{port}/health", timeout=5) as resp:
                if json.load(resp).get("status") == "ok":
                    return "ready"
        except (urllib.error.URLError, OSError, ValueError):
            pass
        time.sleep(5)
        elapsed += 5
        if elapsed % 60 == 0:
            print(f"  …still loading weights ({elapsed // 60}m)")
    return "timeout"


def local_share(yaml: dict) -> float:
    """Fraction of the model that stays on this node, from ``tensor_split``.

    Falls back to an even split when the key is absent or unparseable, which
    matches llama.cpp's own default when ``--tensor-split`` is not passed.
    """
    raw = yaml.get("tensor_split")
    if raw is None:
        return 0.5
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        return 0.5
    total = sum(vals)
    if not vals or total <= 0:
        return 0.5
    return vals[0] / total


class LlamaDualRuntime(LlamaRuntime):
    """Two-node llama.cpp: local llama-server + SSH-launched rpc-server.

    Inherits the whole single-node lifecycle (argv building, pid file,
    readiness polling, kill escalation) from ``LlamaRuntime`` and adds only
    the remote worker: started before the head, torn down after it.
    """

    def _rpc_endpoint(self, yaml: dict) -> str:
        """``host:port`` the head passes to ``--rpc``."""
        return f"{yaml['worker_ip']}:{yaml.get('rpc_port', DEFAULT_RPC_PORT)}"

    def _build_cmd(self, name: str, yaml: dict) -> tuple[list[str], Path]:
        """Adds ``--rpc`` (and ``--tensor-split``) to the llama-server argv."""
        cmd, model_path = super()._build_cmd(name, yaml)
        cmd += ["--rpc", self._rpc_endpoint(yaml)]
        split = yaml.get("tensor_split")
        if split and not any(a in ("-ts", "--tensor-split") for a in cmd):
            cmd += ["--tensor-split", str(split)]
        return cmd, model_path

    def _preflight(self, name: str, yaml: dict) -> str | None:
        """Returns an error string if the pair isn't ready to launch."""
        for key in ("worker_host", "worker_ip", "model"):
            if not yaml.get(key):
                return f"missing required yaml field: {key}"
        worker = yaml["worker_host"]
        if _ssh(worker, "true", capture_output=True).returncode != 0:
            return f"worker unreachable over ssh: {worker}"
        rpc_bin = yaml.get("worker_rpc_bin", DEFAULT_RPC_BIN)
        check = _ssh(worker, f"test -x {_worker_path(rpc_bin)}",
                     capture_output=True)
        if check.returncode != 0:
            return (f"rpc-server missing on {worker}: {rpc_bin} "
                    f"(rebuild llama.cpp there with -DGGML_RPC=ON)")
        return None

    def _start_worker(self, name: str, yaml: dict) -> str | None:
        """Starts ``rpc-server`` on the worker; returns an error or None.

        The remote process is detached with ``setsid`` so it outlives the ssh
        session, and its log stays on the worker — the head has no way to
        stream it back once the connection closes.
        """
        worker = yaml["worker_host"]
        port = int(yaml.get("rpc_port", DEFAULT_RPC_PORT))
        rpc_bin = yaml.get("worker_rpc_bin", DEFAULT_RPC_BIN)
        log = f"$HOME/.lmswitch-rpc-{name}.log"

        # Reason: a stranded rpc-server from a previous run still holds its
        # share of the weights, so the next start would find the port open
        # but the device short of memory.
        _ssh(worker, _pkill_cmd(port),
             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        time.sleep(1)

        args = [_worker_path(rpc_bin), "-H", "0.0.0.0", "-p", str(port)]
        if yaml.get("rpc_cache", True):
            args.append("-c")
        if yaml.get("rpc_threads"):
            args += ["-t", str(yaml["rpc_threads"])]
        launch = f"setsid nohup {' '.join(args)} > {log} 2>&1 &"
        result = _ssh(worker, launch, capture_output=True)
        if result.returncode != 0:
            return f"rpc-server launch failed on {worker} (exit {result.returncode})"

        for _ in range(30):
            if _port_open(yaml["worker_ip"], port):
                print(f"  Worker rpc-server up on {worker} "
                      f"({self._rpc_endpoint(yaml)}, "
                      f"log: {log.replace('$HOME', '~')})")
                return None
            time.sleep(1)
        return f"rpc-server on {worker} never opened port {port} (see {log})"

    def _stop_worker(self, yaml: dict) -> None:
        """Kills this recipe's ``rpc-server`` on the worker, unconditionally."""
        worker = yaml.get("worker_host")
        if not worker:
            return
        port = int(yaml.get("rpc_port", DEFAULT_RPC_PORT))
        print(f"Stopping rpc-server on {worker} (port {port})...")
        _ssh(worker, _pkill_cmd(port),
             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

    def start(self, name: str, yaml: dict) -> RunningState:
        err = self._preflight(name, yaml)
        if err:
            print(f"  ✗ {err}")
            return RunningState("dead", detail=err)

        print(f"Starting llama-dual {name} "
              f"(head=local, worker={yaml['worker_host']})...")
        err = self._start_worker(name, yaml)
        if err:
            print(f"  ✗ {err}")
            return RunningState("dead", detail=err)

        # Head last: llama-server queries the RPC device's free memory while
        # assigning layers, so the worker has to be listening already.
        state = super().start(name, yaml)
        if state.status == "ready":
            try:
                timeout = int(yaml.get("ready_timeout", 1800))
            except (ValueError, TypeError):
                timeout = 1800
            alive = (lambda: state.proc.poll() is None) if state.proc else (lambda: True)
            status = _wait_loaded(yaml.get("port", 8081), timeout, alive)
            state = RunningState(status, detail=state.detail, proc=state.proc)
            if status == "ready":
                share = local_share(yaml)
                print(f"  Weights loaded — split ~{share:.0%} local / "
                      f"{1 - share:.0%} {yaml['worker_host']}")
            elif status == "timeout":
                # Still alive, just slow — leave the pair up rather than
                # tearing down a load that may be minutes from finishing.
                print(f"  WARNING: {name} still loading after {timeout}s "
                      f"(`lmswitch off {name}` to abort)")
        if state.status == "dead":
            self._stop_worker(yaml)
        return state

    def stop(self, name: str, yaml: dict) -> None:
        """Stops the local server, then the worker even if the head was dead.

        A stranded rpc-server keeps the worker's share of the weights
        resident, which is most of a node's memory for the models this
        runtime exists to serve.
        """
        super().stop(name, yaml)
        self._stop_worker(yaml)
        (RUN_DIR / name).unlink(missing_ok=True)
