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


def _docker_container(name: str) -> str | None:
    try:
        out = subprocess.check_output(
            ["docker", "ps", "--filter", f"name=^/vllm-{name}$", "--format", "{{.ID}}"],
            text=True, stderr=subprocess.DEVNULL
        ).strip()
        return out or None
    except Exception:
        return None


def _is_running(name: str, runtime: str) -> bool:
    pid_file = RUN_DIR / name
    if pid_file.exists():
        content = pid_file.read_text().strip()
        # vLLM: container ID (hex string, 12+ chars) → check Docker
        if len(content) >= 12 and content.isalnum():
            return _docker_container(name) is not None
        # GGUF: PID (numeric) → check process
        try:
            pid = int(content)
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, ValueError, OSError):
            pass
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
