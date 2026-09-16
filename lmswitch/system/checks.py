"""Port detection, docker container checks, and process state."""

import os
import subprocess
import time
from datetime import datetime, timezone

from lmswitch.system.io import RUN_DIR, CONF_DIR, _load_yaml, _model_size_and_present
from lmswitch.system.memory import _ram_line


def _listening_ports() -> set[int]:
    ports: set[int] = set()
    try:
        out = subprocess.check_output(["ss", "-tlnH"], text=True,
                                      stderr=subprocess.DEVNULL)
    except Exception:
        return ports
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 4:
            port = parts[3].rsplit(":", 1)[-1]
            if port.isdigit():
                ports.add(int(port))
    return ports


def port_holder(port: int, own_container: str | None = None) -> str | None:
    """Describes what already holds *port*, or None if it is free.

    The container-name check in each runtime's ``start`` only notices a model
    restarting itself. It does not notice a *different* occupant of the same
    port -- another recipe on the same port, or a container this tool did not
    start, such as one launched by an external recipe's own script. Those
    launch anyway, bind-clash, and claim GPU memory twice, which on unified
    memory can take the box down.

    *own_container* is skipped so restarting a model that is already up is not
    reported as a conflict.
    """
    if port not in _listening_ports():
        return None

    # Attribution is best-effort. Every Docker-backed runtime here launches
    # with --network host, and a host-network container publishes no port map,
    # so `docker ps --filter publish=` cannot see it -- its silence says
    # nothing about whether a container holds the port. Name what can be named
    # and stay vague otherwise, rather than asserting "not a container" from
    # the absence of evidence.
    def _names(*filters: str) -> list[str]:
        try:
            out = subprocess.check_output(
                ["docker", "ps", "--format", "{{.Names}}", *filters],
                text=True, stderr=subprocess.DEVNULL)
        except Exception:
            return []
        return [n for n in out.split() if n and n != own_container]

    published = _names("--filter", f"publish={port}")
    if published:
        return f"port {port} is already held by container {', '.join(published)}"

    host_net = _names("--filter", "network=host")
    if host_net:
        # A hint, not an attribution: these containers share the host's
        # network namespace so any of them *could* hold the port, and so
        # could a plain host process.
        return (f"port {port} is already in use "
                f"(host-network containers that may hold it: "
                f"{', '.join(host_net)})")
    return f"port {port} is already in use"


# Container name prefix per Docker-backed runtime. Every vLLM recipe is
# named ``vllm-<model>``; sglang gets its own namespace so the same model
# wired for both backends never collides on a container name.
_CONTAINER_PREFIX = {"sglang": "sglang", "sglang-dual": "sglang"}


def _container_prefix(runtime: str) -> str:
    """Returns the docker container name prefix used by *runtime*.

    Args:
        runtime: Runtime type string (e.g. ``"vllm"``, ``"sglang"``).

    Returns:
        The prefix, defaulting to ``"vllm"`` for every runtime that predates
        this mapping.
    """
    return _CONTAINER_PREFIX.get(runtime, "vllm")


def _docker_container(name: str, prefix: str = "vllm") -> str | None:
    """Returns the running container ID for *name*, or None.

    Args:
        name: Model name (the yaml stem).
        prefix: Container name prefix — see ``_container_prefix``. Defaults
            to ``"vllm"`` so every pre-existing caller is unchanged.
    """
    try:
        out = subprocess.check_output(
            ["docker", "ps", "--filter", f"name=^/{prefix}-{name}$", "--format", "{{.ID}}"],
            text=True, stderr=subprocess.DEVNULL
        ).strip()
        return out or None
    except Exception:
        return None


# Runtimes whose liveness is Docker-container-backed. Their pidfile (when
# present) holds a container ID and is handled above; when the pidfile is
# missing/stale, liveness must still be checked by container NAME, never
# by port — every vllm-dual/vllm-dual-ray recipe conventionally shares
# port 8888, so a port-based fallback would mark every OTHER dual recipe
# as "running" the moment any single one of them actually is.
_DOCKER_BACKED_RUNTIMES = ("vllm", "vllm-dual", "vllm-dual-ray", "sglang",
                           "sglang-dual")


def _is_running(name: str, runtime: str) -> bool:
    prefix = _container_prefix(runtime)
    pid_file = RUN_DIR / name
    if pid_file.exists():
        content = pid_file.read_text().strip()
        # vLLM: container ID (hex string, 12+ chars) → check Docker
        if len(content) >= 12 and content.isalnum():
            return _docker_container(name, prefix) is not None
        # GGUF: PID (numeric) → check process
        try:
            pid = int(content)
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, ValueError, OSError):
            pass
    if runtime in _DOCKER_BACKED_RUNTIMES:
        return _docker_container(name, prefix) is not None
    yaml_path = CONF_DIR / f"{name}.yaml"
    if yaml_path.exists():
        try:
            yaml_cfg = _load_yaml(yaml_path)
            port = int(yaml_cfg.get("port", 0))
        except (ValueError, TypeError):
            port = 0
        if port and port in _listening_ports():
            return True
    return False


def _seconds_since(stamp: str) -> float:
    """Returns the seconds elapsed since an RFC-3339 timestamp.

    Args:
        stamp: Timestamp as Docker reports it, e.g.
            ``"2026-09-08T09:30:34.152106113Z"``.

    Returns:
        Elapsed seconds, or 0.0 when *stamp* cannot be parsed.
    """
    # Reason: Docker emits nanosecond precision and a literal `Z`. Neither is
    # accepted by `datetime.fromisoformat` on 3.10, so trim the fraction to
    # microseconds and spell the zone out before parsing.
    text = stamp.strip().replace("Z", "+00:00")
    head, dot, tail = text.partition(".")
    if dot:
        digits = ""
        idx = 0
        while idx < len(tail) and tail[idx].isdigit():
            digits += tail[idx]
            idx += 1
        text = f"{head}.{digits[:6]}{tail[idx:]}"
    try:
        started = datetime.fromisoformat(text)
    except ValueError:
        return 0.0
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - started).total_seconds())


def _container_uptime(name: str, prefix: str) -> float:
    """Returns the uptime of *name*'s Docker container in seconds."""
    cid = _docker_container(name, prefix)
    if cid is None:
        return 0.0
    try:
        started = subprocess.check_output(
            ["docker", "inspect", "-f", "{{.State.StartedAt}}", cid],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return 0.0
    return _seconds_since(started)


def _process_uptime(name: str) -> float:
    """Returns the uptime of *name*'s llama-server process in seconds."""
    pid_file = RUN_DIR / name
    if not pid_file.exists():
        return 0.0
    try:
        pid = int(pid_file.read_text().strip())
        # Reason: the /proc/<pid> directory inode is stamped at process
        # creation, so its ctime is the real process start — unlike the pid
        # file's mtime, which a re-exec under systemd would leave untouched.
        return max(0.0, time.time() - os.stat(f"/proc/{pid}").st_ctime)
    except (ValueError, OSError):
        return 0.0


def server_uptime(name: str, runtime: str) -> float:
    """Returns how long the server process behind *name* has been up.

    Reads the real process/container start time rather than lmswitch's own
    bookkeeping: a ``restart:`` recipe is respawned by systemd without lmswitch
    ever seeing it, so its last recorded start event can be weeks older than
    the process actually serving — and than the token counters scraped off it.

    Args:
        name: Model name (the yaml stem).
        runtime: Runtime type string (e.g. ``"vllm"``, ``"llama-dual"``).

    Returns:
        Uptime in seconds, or 0.0 when the server is down or unreadable.
    """
    if runtime in _DOCKER_BACKED_RUNTIMES:
        return _container_uptime(name, _container_prefix(runtime))
    return _process_uptime(name)
