"""Runtime abstraction for model servers.

Each runtime implements a common interface via ``BaseRuntime``:
  - ``start(name, yaml) -> RunningState``
  - ``stop(name, yaml) -> None``
  - ``is_running(name, runtime_name) -> bool``
  - ``is_ready(name, port, timeout) -> str``

Adding a new runtime means writing one file that subclasses
``BaseRuntime`` and registering it in ``runtime_registry``.
"""

from __future__ import annotations

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
from lmswitch.system.memory import _memory_check
from lmswitch.system import usage as usage_mod
from lmswitch.runtimes.base import BaseRuntime, RunningState, runtime_registry
from lmswitch.runtimes.llama import LlamaRuntime, _extra_args, _start_llama_direct
from lmswitch.runtimes.vllm import VLLMRuntime, _vllm_args, _start_vllm_direct, _start_vllm_foreground
from lmswitch.runtimes.vllm_dual import VLLMDualRuntime
from lmswitch.runtimes.systemd import _start_systemd
from lmswitch.runtimes.wait import _wait_ready

__all__ = [
    "start_model",
    "stop_model",
    "BaseRuntime",
    "RunningState",
    "runtime_registry",
    "_wait_ready",
    "_extra_args",
    "_vllm_args",
    "_start_llama_direct",
    "_start_vllm_direct",
    "_start_vllm_foreground",
    "_start_systemd",
    "_memory_check",
    "LlamaRuntime",
    "VLLMRuntime",
    "VLLMDualRuntime",
]

# Register runtimes — called at import time
runtime_registry.register("llama", LlamaRuntime)
runtime_registry.register("vllm", VLLMRuntime)
runtime_registry.register("vllm-dual", VLLMDualRuntime)


def start_model(name: str, yaml: dict) -> None:
    """Start a model server using RAM guard + runtime dispatch.

    This is the public entry point used by cli.py cmd_on / toggle().
    """
    # Reason: refuse loads that would exceed available RAM — on a unified-memory
    # box an OOM can lock the machine and force a reboot. `force: true` overrides.
    ok, why = _memory_check(name, yaml)
    if not ok and not yaml.get("force"):
        print(f"  ✗ refusing to start {name}: {why}.")
        print("    Free memory (`lmswitch off <model>`) or set `force: true` in its yaml to override.")
        return
    runtime_name = yaml.get("runtime", "llama")
    restart = yaml.get("restart")
    if restart:
        _start_systemd(name, yaml, restart)
        usage_mod.record_start(name, yaml)
        return
    runtime_cls = runtime_registry.lookup(runtime_name)
    runtime_cls().start(name, yaml)
    usage_mod.record_start(name, yaml)


def stop_model(name: str, runtime: str) -> None:
    """Stop a model server by runtime type."""
    # Load the model's YAML config so stop() has access to model-specific
    # settings (port, force, etc.) — callers should pass the runtime name.
    yaml = {}
    yaml_path = CONF_DIR / f"{name}.yaml"
    if yaml_path.exists():
        yaml = _load_yaml(yaml_path) or {}
    runtime_cls = runtime_registry.lookup(runtime)
    runtime_cls().stop(name, yaml)
    usage_mod.record_stop(name, 0.0)
