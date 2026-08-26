"""Port detection, docker container checks, and process state."""

import os
import subprocess

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
