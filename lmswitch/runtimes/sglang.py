"""SGLang (Docker) runtime.

Wraps ``python3 -m sglang.launch_server`` in a container, mirroring the
vLLM runtime's shape (detached ``docker run`` + ``docker logs -f`` follower
+ HTTP readiness poll) but speaking SGLang's flag surface: SGLang has no
``--gpu-memory-utilization`` / ``--max-model-len`` / ``--tensor-parallel-size``
and rejects them, using ``--mem-fraction-static`` / ``--context-length`` /
``--tp-size`` instead.

Recipe origin: MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark (the SGLang cookbook's
DGX Spark cell) — see ``ai-models/qwen3.8-27b-nvfp4-sglang.yaml``.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from lmswitch.system.io import HOME, RUN_DIR
from lmswitch.runtimes.base import BaseRuntime, RunningState
from lmswitch.runtimes.vllm import _extra_mounts, _env_args
from lmswitch.runtimes.wait import _wait_ready

# Reason: GB10 has 10 Cortex-X5 performance cores (5-9, 15-19) and 10 A725
# efficiency cores (0-4, 10-14). Pinning the container to the big cores keeps
# the SGLang scheduler/tokenizer Python loop off the 2.8GHz cores — worth
# +2-7% decode throughput in the upstream recipe's measurements.
DEFAULT_CPUSET = "5-9,15-19"

# Native context of the Qwen3.8 family; above this SGLang needs a YaRN rope
# override or it silently clamps back down to this value.
NATIVE_CTX = 262144


def _cpuset_args(yaml: dict) -> list[str]:
    """Docker ``--cpuset-cpus`` option, or nothing when pinning is disabled.

    Set ``cpuset: ""`` (or ``cpuset: false``) in the recipe to run unpinned.
    """
    cpuset = yaml.get("cpuset", DEFAULT_CPUSET)
    if not cpuset:
        return []
    return ["--cpuset-cpus", str(cpuset)]


def _sglang_args(yaml: dict) -> list[str]:
    """First-class SGLang server flags derived from the recipe's YAML keys.

    Only keys that are actually set emit a flag, so a recipe stays about as
    short as the cookbook cell it came from. Anything without a first-class
    key goes through ``extra_args``, appended last — argparse's last-wins
    rule lets it override anything here.
    """
    from lmswitch.runtimes.llama import _extra_args
    args: list[str] = []
    if yaml.get("trust_remote_code"):
        args.append("--trust-remote-code")
    args += ["--mem-fraction-static", str(yaml.get("mem_fraction_static", 0.95))]
    if yaml.get("attention_backend"):
        args += ["--attention-backend", str(yaml["attention_backend"])]
    if yaml.get("chunked_prefill_size"):
        args += ["--chunked-prefill-size", str(yaml["chunked_prefill_size"])]
    if yaml.get("disable_prefill_cuda_graph", True):
        args.append("--disable-prefill-cuda-graph")
    if yaml.get("kv_cache_dtype"):
        args += ["--kv-cache-dtype", str(yaml["kv_cache_dtype"])]
    if yaml.get("tp_size"):
        args += ["--tp-size", str(yaml["tp_size"])]
    if yaml.get("max_running_requests"):
        args += ["--max-running-requests", str(yaml["max_running_requests"])]
    # Reason: SGLang needs no vLLM-style --enable-auto-tool-choice; sending
    # `tools` in the request is enough once a parser is selected.
    if yaml.get("tool_call_parser"):
        args += ["--tool-call-parser", str(yaml["tool_call_parser"])]
    if yaml.get("reasoning_parser"):
        args += ["--reasoning-parser", str(yaml["reasoning_parser"])]
    if yaml.get("sampling_defaults"):
        args += ["--sampling-defaults", str(yaml["sampling_defaults"])]
    args += _extra_args(yaml)
    return args


class SGLangRuntime(BaseRuntime):
    """SGLang model runtime using Docker."""

    def _setup_logging(self, name: str) -> Path:
        """Mirrors the container's output into ``RUN_DIR/<name>.log``.

        Same detached ``docker logs -f`` follower the vLLM runtime uses, so
        SGLang models expose a plain log file homogeneous with every other
        runtime. Must run AFTER ``docker run`` so the named container exists;
        the follower exits by itself when the container stops.
        """
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        log_path = RUN_DIR / f"{name}.log"
        try:
            if log_path.is_symlink() or log_path.exists():
                log_path.unlink()
            log_fh = open(log_path, "wb")
            subprocess.Popen(
                ["docker", "logs", "-f", f"sglang-{name}"],
                stdout=log_fh, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            log_fh.close()
        except OSError as exc:
            print(f"  WARNING: could not write log file for {name} ({exc}); "
                  f"logs still available via: docker logs -f sglang-{name}")
        return log_path

    def _build_cmd(self, name: str, yaml: dict, detached: bool = True) -> list[str]:
        """Builds the ``docker run`` command for this model.

        Args:
            name: Model id (yaml stem) — names the container and the served
                model, so clients address it by the same string ``lmswitch``
                shows.
            yaml: Parsed model config.
            detached: Pass ``-d`` (the normal path); False builds the same
                command for a foreground run.

        Returns:
            The full argv for ``docker run``.
        """
        models_dir = yaml.get("_models_dir")
        if models_dir is None:
            from lmswitch.system.io import _models_dir
            models_dir = _models_dir()
        model_path = models_dir / yaml["model"]
        port = yaml.get("port", 0)
        ctx = int(yaml.get("ctx", NATIVE_CTX))
        image = yaml.get("image", "lmsysorg/sglang:qwen38-27b")

        detach_flag = ["-d"] if detached else []
        cmd = [
            "docker", "run", *detach_flag,
            "--name", f"sglang-{name}",
            "--gpus", "all",
            "--network", "host",
            "--ipc", "host",
            "--privileged",
            "--shm-size", str(yaml.get("shm_size", "32g")),
            *_cpuset_args(yaml),
            "--log-driver", "json-file",
            "--log-opt", "max-size=10m",
            "--log-opt", "max-file=3",
            "-v", f"{model_path}:{model_path}:ro",
            "-v", f"{HOME}/.cache/huggingface:/root/.cache/huggingface",
            "-v", f"{HOME}/.cache/triton:/root/.triton",
            "-e", "HF_HOME=/root/.cache/huggingface",
            "-e", "TRITON_CACHE_DIR=/root/.triton",
        ]
        if os.environ.get("HF_TOKEN"):
            cmd += ["-e", f"HF_TOKEN={os.environ['HF_TOKEN']}"]
        # Reason: SGLang ignores a --context-length above the checkpoint's
        # derived maximum unless this is set — it would silently serve 262K
        # when the recipe asked for 1M. The recipe still has to supply the
        # YaRN rope override itself via json_model_override_args.
        if ctx > NATIVE_CTX:
            cmd += ["-e", "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1"]
        cmd += _extra_mounts(yaml)
        cmd += _env_args(yaml)
        # Reason: the SGLang images carry no `sglang.launch_server`
        # ENTRYPOINT (the upstream recipe spells the module out), so pin the
        # entrypoint instead of inheriting whatever the image declares.
        cmd += ["--entrypoint", "python3", image,
                "-m", "sglang.launch_server",
                "--model-path", str(model_path),
                "--served-model-name", name,
                "--context-length", str(ctx),
                "--host", "0.0.0.0",
                "--port", str(port)]
        if yaml.get("json_model_override_args"):
            cmd += ["--json-model-override-args",
                    str(yaml["json_model_override_args"])]
        cmd += _sglang_args(yaml)
        return cmd

    def start(self, name: str, yaml: dict) -> RunningState:
        from lmswitch.system.checks import _docker_container
        existing = _docker_container(name, "sglang")
        if existing:
            print(f"SGLang {name} already running (container {existing[:12]})")
            return RunningState("ready")

        RUN_DIR.mkdir(parents=True, exist_ok=True)
        id_file = RUN_DIR / name

        port = yaml.get("port", 0)
        ctx = int(yaml.get("ctx", NATIVE_CTX))
        print(f"Starting SGLang {name} on port {port}...")
        print(f"  Image: {yaml.get('image', 'lmsysorg/sglang:qwen38-27b')}")
        if ctx > NATIVE_CTX and not yaml.get("json_model_override_args"):
            print(f"  WARNING: ctx={ctx} exceeds the native {NATIVE_CTX} but no "
                  f"json_model_override_args (YaRN) is set — SGLang will clamp "
                  f"back to {NATIVE_CTX}.")

        # Reason: a container left behind in Created/Exited state keeps the
        # name and makes every retry fail with a name conflict.
        subprocess.run(["docker", "container", "rm", "-f", f"sglang-{name}"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

        cmd = self._build_cmd(name, yaml, detached=True)
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"  ✗ docker run failed (exit {result.returncode}); not waiting for readiness.")
            return RunningState("dead", detail=f"docker exit {result.returncode}")

        log_path = self._setup_logging(name)

        try:
            timeout = int(yaml.get("ready_timeout", 900))
        except (ValueError, TypeError):
            timeout = 900
        status = _wait_ready(name, port, timeout,
                             lambda: _docker_container(name, "sglang") is not None)
        if status == "ready":
            container_id = _docker_container(name, "sglang")
            id_file.write_text(container_id or name)
            print(f"  Ready on port {port}")
            print(f"  PID file:  {id_file}")
            print(f"  Log file:  {log_path}")
        elif status == "dead":
            print(f"  ✗ {name} container exited during startup — check log: {log_path}")
        else:
            print(f"  WARNING: {name} did not become ready in {timeout}s "
                  f"(still loading? check {log_path})")
        return RunningState(status)

    def stop(self, name: str, yaml: dict) -> None:
        from lmswitch.system.checks import _docker_container
        id_file = RUN_DIR / name
        cid = _docker_container(name, "sglang")
        if cid:
            print(f"Stopping SGLang {name} (container {cid[:12]})...")
            subprocess.run(["docker", "stop", cid], check=False)
            subprocess.run(["docker", "container", "rm", cid], check=False)
            id_file.unlink(missing_ok=True)
        else:
            print(f"SGLang {name} not running")

    def is_running(self, name: str, runtime_name: str) -> bool:
        from lmswitch.system.checks import _docker_container
        cid = _docker_container(name, "sglang")
        if cid is None:
            (RUN_DIR / name).unlink(missing_ok=True)
        return cid is not None

    def is_ready(self, name: str, port: int, timeout: int = 300) -> str:
        from lmswitch.system.checks import _docker_container
        return _wait_ready(name, port, timeout,
                           lambda: _docker_container(name, "sglang") is not None)
