"""vLLM (Docker) runtime."""

from __future__ import annotations

import os
import shlex
import subprocess
import time
from pathlib import Path

from lmswitch.system.io import HOME, RUN_DIR, SCRIPT_DIR
from lmswitch.runtimes.base import BaseRuntime, RunningState, runtime_registry
from lmswitch.runtimes.wait import _wait_ready


def _extra_mounts(yaml: dict) -> list[str]:
    """Extra docker `-v` bind mounts for the vLLM container.

    Each entry is a `host:container[:ro]` spec (same syntax as `docker run -v`).
    The host path may use `~` and `$VARS`. These are `docker run` options, so
    they're emitted BEFORE the image — unlike `extra_args`, which are appended
    after the image and go to `vllm serve` inside the container. Accepts a YAML
    list or a single shell-split string:

        extra_mounts: ["~/models/foo/drafter:/drafter:ro"]
        extra_mounts: "~/models/foo/drafter:/drafter:ro"
    """
    raw = yaml.get("extra_mounts") or []
    if isinstance(raw, str):
        raw = shlex.split(raw)
    args: list[str] = []
    for spec in raw:
        host, sep, rest = str(spec).partition(":")
        host = os.path.expanduser(os.path.expandvars(host))
        args += ["-v", f"{host}{sep}{rest}" if sep else host]
    return args


def _env_args(yaml: dict) -> list[str]:
    """Extra docker `-e` environment variables for the vLLM container.

    `docker run` options, so emitted BEFORE the image. Accepts a YAML mapping
    (preferred) or a list of `KEY=VALUE` strings. Values are stringified so
    YAML scalars like `1` / `true` work without quoting:

        env:
          VLLM_TEST_FORCE_FP8_MARLIN: 1
          TORCH_CUDA_ARCH_LIST: 12.1a
        env: ["VLLM_TEST_FORCE_FP8_MARLIN=1", "TORCH_CUDA_ARCH_LIST=12.1a"]
    """
    raw = yaml.get("env") or {}
    if isinstance(raw, dict):
        items = list(raw.items())
    else:
        items = [str(x).partition("=")[::2] for x in raw]
    args: list[str] = []
    for key, val in items:
        args += ["-e", f"{key}={val}"]
    return args


def _entrypoint(yaml: dict) -> tuple[list[str], list[str]]:
    """Override the container ENTRYPOINT.

    Returns (docker_opts, leading_cmd): docker's `--entrypoint` takes a single
    executable, so a multi-token entrypoint (e.g. `vllm serve`) splits into the
    `--entrypoint <bin>` option (before the image) plus leading command tokens
    (after the image, before the model path). Needed for images whose ENTRYPOINT
    is not `vllm serve` — e.g. the AEON image's ENTRYPOINT is `/bin/bash`, so set
    `entrypoint: "vllm serve"`. Accepts a string (shell-split) or a YAML list.
    """
    ep = yaml.get("entrypoint")
    if not ep:
        return [], []
    if isinstance(ep, str):
        ep = shlex.split(ep)
    ep = [str(x) for x in ep]
    return ["--entrypoint", ep[0]], ep[1:]


def _vllm_args(yaml: dict) -> list[str]:
    from lmswitch.runtimes.llama import _extra_args
    args: list[str] = []
    if yaml.get("enforce_eager", True):
        args.append("--enforce-eager")
    if yaml.get("tool_call_parser"):
        args += ["--enable-auto-tool-choice", "--tool-call-parser", str(yaml["tool_call_parser"])]
    if yaml.get("reasoning_parser"):
        args.append(f"--reasoning-parser={yaml['reasoning_parser']}")
    if yaml.get("reasoning_parser_plugin"):
        args.append(f"--reasoning-parser-plugin={yaml['reasoning_parser_plugin']}")
    if yaml.get("trust_remote_code"):
        args.append("--trust-remote-code")
    if yaml.get("limit_mm_per_prompt"):
        args.append(f"--limit-mm-per-prompt={yaml['limit_mm_per_prompt']}")
    if yaml.get("chat_template") and not yaml.get("no_chat_template"):
        args.append(f"--chat-template={yaml['chat_template']}")
    if yaml.get("attention_backend"):
        args.append(f"--attention-backend={yaml['attention_backend']}")
    if yaml.get("gpu_memory_utilization"):
        args.append(f"--gpu-memory-utilization={yaml['gpu_memory_utilization']}")
    if yaml.get("max_num_seqs"):
        args.append(f"--max-num-seqs={yaml['max_num_seqs']}")
    if yaml.get("load_format"):
        args.append(f"--load-format={yaml['load_format']}")
    args += _extra_args(yaml)
    return args


class VLLMRuntime(BaseRuntime):
    """vLLM model runtime using Docker."""

    def _build_cmd(self, name: str, yaml: dict, detached: bool = True) -> list[str]:
        """Build the docker run command for this model."""
        models_dir = yaml.get("_models_dir")
        if models_dir is None:
            from lmswitch.system.io import _models_dir
            models_dir = _models_dir()
        model_path = models_dir / yaml["model"]
        port = yaml.get("port", 0)
        ctx = yaml.get("ctx", 65536)
        gpu_mem = yaml.get("gpu_memory_utilization", 0.15)
        image = yaml.get("image", "vllm/vllm-openai:cu130-nightly")
        wheels = SCRIPT_DIR / "spark-vllm-docker" / "wheels"

        vllm_args = _vllm_args(yaml)
        detach_flag = ["-d"] if detached else []
        ep_opts, ep_cmd = _entrypoint(yaml)

        cmd = [
            "docker", "run", *detach_flag,
            "--name", f"vllm-{name}",
            "--gpus", "all",
            "--network", "host",
            "--shm-size", "8g",
            "--log-driver", "json-file",
            "--log-opt", "max-size=10m",
            "--log-opt", "max-file=3",
            "-v", f"{model_path}:{model_path}:ro",
            "-v", f"{HOME}/.cache/huggingface/hub:/root/.cache/huggingface/hub",
            "-v", f"{HOME}/.cache/huggingface/token:/root/.cache/huggingface/token",
        ]
        if wheels.exists():
            cmd += ["-v", f"{wheels}:/wheels:ro"]
        cmd += _extra_mounts(yaml)
        cmd += _env_args(yaml)
        cmd += ep_opts

        cmd += [
            image,
            *ep_cmd,
            str(model_path),
            f"--served-model-name={name}",
            f"--port={port}",
            "--host=0.0.0.0",
            f"--max-model-len={ctx}",
            "--max-num-seqs=32",
            "--tensor-parallel-size=1",
            f"--gpu-memory-utilization={gpu_mem}",
            "--disable-log-stats",
        ] + vllm_args
        return cmd

    def start(self, name: str, yaml: dict) -> RunningState:
        from lmswitch.system.checks import _docker_container
        existing = _docker_container(name)
        if existing:
            print(f"vLLM {name} already running (container {existing[:12]})")
            return RunningState("ready")

        port = yaml.get("port", 0)
        print(f"Starting vLLM {name} on port {port}...")
        print(f"  Image: {yaml.get('image', 'vllm/vllm-openai:cu130-nightly')}")

        subprocess.run(["docker", "rm", "-f", f"vllm-{name}"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

        cmd = self._build_cmd(name, yaml, detached=True)
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"  ✗ docker run failed (exit {result.returncode}); not waiting for readiness.")
            return RunningState("dead", detail=f"docker exit {result.returncode}")

        try:
            timeout = int(yaml.get("ready_timeout", 600))
        except (ValueError, TypeError):
            timeout = 600
        status = _wait_ready(name, port, timeout, lambda: _docker_container(name) is not None)
        if status == "ready":
            print(f"  Ready on port {port}")
        elif status == "dead":
            print(f"  ✗ {name} container exited during startup — "
                  f"check: docker logs vllm-{name}")
        else:
            print(f"  WARNING: {name} did not become ready in {timeout}s "
                  f"(still loading? check docker logs vllm-{name})")
        return RunningState(status)

    def stop(self, name: str, yaml: dict) -> None:
        from lmswitch.system.checks import _docker_container
        cid = _docker_container(name)
        if cid:
            print(f"Stopping vLLM {name} (container {cid[:12]})...")
            subprocess.run(["docker", "stop", cid], check=False)
            subprocess.run(["docker", "rm", cid], check=False)
        else:
            print(f"vLLM {name} not running")

    def is_running(self, name: str, runtime_name: str) -> bool:
        from lmswitch.system.checks import _docker_container
        return _docker_container(name) is not None

    def is_ready(self, name: str, port: int, timeout: int = 300) -> str:
        from lmswitch.system.checks import _docker_container
        status = _wait_ready(name, port, timeout, lambda: _docker_container(name) is not None)
        return status


# Keep module-level functions for backward compat
def _start_vllm_direct(name: str, yaml: dict) -> None:
    VLLMRuntime().start(name, yaml)

def _start_vllm_foreground(name: str, yaml: dict) -> None:
    """Foreground serve — used by systemd for restart-managed models."""
    models_dir = yaml.get("_models_dir")
    if models_dir is None:
        from lmswitch.system.io import _models_dir
        models_dir = _models_dir()
    model_path = models_dir / yaml["model"]
    port = yaml.get("port", 0)
    ctx = yaml.get("ctx", 65536)
    gpu_mem = yaml.get("gpu_memory_utilization", 0.15)
    image = yaml.get("image", "vllm/vllm-openai:cu130-nightly")
    wheels = SCRIPT_DIR / "spark-vllm-docker" / "wheels"

    vllm_args = _vllm_args(yaml)
    ep_opts, ep_cmd = _entrypoint(yaml)

    cmd = [
        "docker", "run", "--rm",
        "--name", f"vllm-{name}",
        "--gpus", "all",
        "--network", "host",
        "--shm-size", "8g",
        "-v", f"{model_path}:{model_path}:ro",
        "-v", f"{HOME}/.cache/huggingface/hub:/root/.cache/huggingface/hub",
        "-v", f"{HOME}/.cache/huggingface/token:/root/.cache/huggingface/token",
    ]
    if wheels.exists():
        cmd += ["-v", f"{wheels}:/wheels:ro"]
    cmd += _extra_mounts(yaml)
    cmd += _env_args(yaml)
    cmd += ep_opts

    cmd += [
        image,
        *ep_cmd,
        str(model_path),
        f"--served-model-name={name}",
        f"--port={port}",
        "--host=0.0.0.0",
        f"--max-model-len={ctx}",
        "--max-num-seqs=32",
        "--tensor-parallel-size=1",
        f"--gpu-memory-utilization={gpu_mem}",
        "--disable-log-stats",
    ] + vllm_args

    print(f"Serving {name} on port {port} (systemd-managed)...")
    os.execvp("docker", cmd)
