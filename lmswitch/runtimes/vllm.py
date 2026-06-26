"""vLLM (Docker) runtime."""

import subprocess
import time
from pathlib import Path

from lmswitch.system.io import HOME, RUN_DIR, SCRIPT_DIR
from lmswitch.system.checks import _docker_container


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


def _start_vllm_direct(name: str, yaml: dict) -> None:
    models_dir = yaml.get("_models_dir")
    if models_dir is None:
        from lmswitch.system.io import _models_dir
        models_dir = _models_dir()
    existing = _docker_container(name)
    if existing:
        print(f"vLLM {name} already running (container {existing[:12]})")
        return

    model_path = models_dir / yaml["model"]
    port = yaml.get("port", 0)
    ctx = yaml.get("ctx", 65536)
    gpu_mem = yaml.get("gpu_memory_utilization", 0.15)
    image = yaml.get("image", "vllm/vllm-openai:cu130-nightly")
    wheels = SCRIPT_DIR / "spark-vllm-docker" / "wheels"

    vllm_args = _vllm_args(yaml)

    cmd = [
        "docker", "run", "-d",
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

    cmd += [
        image,
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

    print(f"Starting vLLM {name} on port {port}...")
    print(f"  Model: {model_path}")
    print(f"  Image: {image}")

    # Clear any stale container with this name before `docker run`. We already
    # returned above if it was running, so this only removes a stopped/created/
    # dead container, or no-ops. Without this, `docker run --name vllm-<name>`
    # fails with a name conflict on a leftover exited container.
    subprocess.run(["docker", "rm", "-f", f"vllm-{name}"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"  ✗ docker run failed (exit {result.returncode}); not waiting for readiness.")
        return

    # Reason: returncode-based readiness poll that also detects a crashed
    # container, so we don't spin the full timeout against a dead port.
    try:
        timeout = int(yaml.get("ready_timeout", 600))
    except (ValueError, TypeError):
        timeout = 600
    from lmswitch.runtimes.wait import _wait_ready
    status = _wait_ready(name, port, timeout, lambda: _docker_container(name) is not None)
    if status == "ready":
        print(f"  Ready on port {port}")
    elif status == "dead":
        print(f"  ✗ {name} container exited during startup — "
              f"check: docker logs vllm-{name}")
    else:
        print(f"  WARNING: {name} did not become ready in {timeout}s "
              f"(still loading? check docker logs vllm-{name})")


def _start_vllm_foreground(name: str, yaml: dict) -> None:
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

    cmd += [
        image,
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
    import os
    os.execvp("docker", cmd)
