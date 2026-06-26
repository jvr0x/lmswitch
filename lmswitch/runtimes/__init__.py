"""Runtime abstraction for model servers."""

import os
import shlex
import subprocess
import time
from pathlib import Path

from lmswitch.system.io import (
    HOME,
    RUN_DIR,
    CONF_DIR,
    SCRIPT_DIR,
    _load_yaml,
)
from lmswitch.system.checks import _docker_container, _is_running, _listening_ports
from lmswitch.runtimes.llama import _extra_args, _start_llama_direct
from lmswitch.runtimes.vllm import (
    _vllm_args,
    _start_vllm_direct,
    _start_vllm_foreground,
)
from lmswitch.runtimes.systemd import _start_systemd
from lmswitch.runtimes.wait import _wait_ready
from lmswitch.system.memory import _memory_check

__all__ = [
    "start_model",
    "stop_model",
    "_wait_ready",
    "_extra_args",
    "_vllm_args",
    "_start_llama_direct",
    "_start_vllm_direct",
    "_start_vllm_foreground",
    "_start_systemd",
    "_memory_check",
]


def start_model(name: str, yaml: dict) -> None:
    # Reason: refuse loads that would exceed available RAM — on a unified-memory
    # box an OOM can lock the machine and force a reboot. `force: true` overrides.
    ok, why = _memory_check(name, yaml)
    if not ok and not yaml.get("force"):
        print(f"  ✗ refusing to start {name}: {why}.")
        print("    Free memory (`lmswitch off <model>`) or set `force: true` in its yaml to override.")
        return
    runtime = yaml.get("runtime", "llama")
    restart = yaml.get("restart")
    if restart:
        _start_systemd(name, yaml, restart)
        return
    if runtime == "vllm":
        _start_vllm_direct(name, yaml)
    else:
        _start_llama_direct(name, yaml)


def stop_model(name: str, runtime: str) -> None:
    if runtime == "vllm":
        cid = _docker_container(name)
        if cid:
            print(f"Stopping vLLM {name} (container {cid[:12]})...")
            subprocess.run(["docker", "stop", cid], check=False)
            subprocess.run(["docker", "rm", cid], check=False)
        else:
            print(f"vLLM {name} not running")
    else:
        pid_file = RUN_DIR / name
        if pid_file.exists():
            pid = int(pid_file.read_text().strip())
            try:
                os.kill(pid, 0)
                print(f"Stopping llama-server {name} (PID {pid})...")
                os.kill(pid, 15)
                pid_file.unlink()
            except (ProcessLookupError, ValueError, OSError):
                pid_file.unlink(missing_ok=True)
                print(f"{name} not running")
        else:
            print(f"{name} not running")
