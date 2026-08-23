"""Tests for the SGLang runtime: command shape, guards, and registration.

All subprocess interaction is stubbed; nothing is launched and no GPU memory
is ever touched. Runs anywhere.
"""

import tempfile
from pathlib import Path
from unittest import mock

import lmswitch.runtimes.sglang as sglang_mod
from lmswitch.runtimes import runtime_registry
from lmswitch.runtimes.sglang import SGLangRuntime
from lmswitch.system import checks as checks_mod
from lmswitch.system import memory as memory_mod
from lmswitch.system.io import _load_yaml

CONF = Path(__file__).resolve().parents[1] / "ai-models" / "qwen3.8-27b-nvfp4-sglang.yaml"


def _Result(returncode=0):
    return type("Result", (), {"returncode": returncode})()


def _yaml(**over):
    """The shipped Qwen3.8 recipe, with a temp models dir and any overrides."""
    cfg = _load_yaml(CONF)
    cfg["_models_dir"] = Path(tempfile.mkdtemp())
    cfg.update(over)
    return cfg


# --------------------------------------------------------------------------
# Expected use
# --------------------------------------------------------------------------

def test_registered_under_sglang():
    """The registry must resolve "sglang" — a miss silently falls back to
    llama, which would try to boot llama-server on a safetensors dir."""
    assert runtime_registry.lookup("sglang") is SGLangRuntime


def test_build_cmd_shape():
    """The command carries SGLang's flag surface, not vLLM's, and launches
    the module explicitly rather than trusting the image ENTRYPOINT."""
    cfg = _yaml()
    cmd = SGLangRuntime()._build_cmd("qwen3.8-27b-nvfp4-sglang", cfg)

    assert cmd[:2] == ["docker", "run"]
    assert "-d" in cmd
    assert "sglang-qwen3.8-27b-nvfp4-sglang" in cmd
    # Entrypoint pinned, module spelled out after the image.
    img_i = cmd.index("lmsysorg/sglang:qwen38-27b")
    assert cmd[img_i - 2:img_i] == ["--entrypoint", "python3"]
    assert cmd[img_i + 1:img_i + 3] == ["-m", "sglang.launch_server"]

    pairs = list(zip(cmd, cmd[1:]))
    assert ("--mem-fraction-static", "0.95") in pairs
    assert ("--context-length", "262144") in pairs
    assert ("--served-model-name", "qwen3.8-27b-nvfp4-sglang") in pairs
    assert ("--port", "8145") in pairs
    assert ("--attention-backend", "flashinfer") in pairs
    assert ("--kv-cache-dtype", "fp8_e4m3") in pairs
    assert ("--tool-call-parser", "qwen3_coder") in pairs
    assert ("--reasoning-parser", "qwen3") in pairs
    assert "--disable-prefill-cuda-graph" in cmd
    assert "--trust-remote-code" in cmd
    # vLLM-only flags would make SGLang's argparse reject the launch.
    for banned in ("--gpu-memory-utilization", "--max-model-len",
                   "--tensor-parallel-size", "--enable-auto-tool-choice",
                   "--disable-log-stats"):
        assert banned not in cmd, f"{banned} is not an SGLang flag"


def test_build_cmd_docker_opts_and_extra_args():
    """DGX Spark container options from the upstream recipe survive the port,
    and extra_args land after the image so argparse's last-wins applies."""
    cfg = _yaml()
    cmd = SGLangRuntime()._build_cmd("m", cfg)
    pairs = list(zip(cmd, cmd[1:]))

    assert ("--ipc", "host") in pairs
    assert "--privileged" in cmd
    assert ("--shm-size", "32g") in pairs
    assert ("--cpuset-cpus", "5-9,15-19") in pairs
    assert ("-e", "HF_HOME=/root/.cache/huggingface") in pairs
    assert ("-e", "TRITON_CACHE_DIR=/root/.triton") in pairs
    assert ("-e", "SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=0") in pairs
    # Weights mounted read-only from the local models dir, not fetched by id.
    model_path = str(cfg["_models_dir"] / "RadixArk/Qwen3.8-27B-NVFP4")
    assert f"{model_path}:{model_path}:ro" in cmd
    assert ("--model-path", model_path) in pairs

    img_i = cmd.index("lmsysorg/sglang:qwen38-27b")
    assert cmd.index("--speculative-algorithm") > img_i
    assert ("--speculative-algorithm", "EAGLE") in pairs
    assert ("--speculative-num-draft-tokens", "4") in pairs
    assert ("--max-mamba-cache-size", "64") in pairs


# --------------------------------------------------------------------------
# Edge cases
# --------------------------------------------------------------------------

def test_ctx_above_native_needs_yarn_escape_hatch():
    """Above the native 262144 SGLang ignores --context-length unless the
    overwrite env var is set; at or below it the var must stay absent."""
    long_cmd = SGLangRuntime()._build_cmd("m", _yaml(ctx=1000000))
    assert ("-e", "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1") in \
        list(zip(long_cmd, long_cmd[1:]))
    assert ("--context-length", "1000000") in list(zip(long_cmd, long_cmd[1:]))

    native_cmd = SGLangRuntime()._build_cmd("m", _yaml())
    assert not any(a == "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1"
                   for a in native_cmd)


def test_cpuset_can_be_disabled():
    """An empty cpuset means "do not pin", not "pin to nothing"."""
    cmd = SGLangRuntime()._build_cmd("m", _yaml(cpuset=""))
    assert "--cpuset-cpus" not in cmd


def test_liveness_uses_the_sglang_container_prefix():
    """Containers are named sglang-<model>; probing the vllm- prefix would
    report a healthy server as dead the moment it comes up."""
    assert checks_mod._container_prefix("sglang") == "sglang"
    assert checks_mod._container_prefix("vllm") == "vllm"
    assert "sglang" in checks_mod._DOCKER_BACKED_RUNTIMES

    seen = []

    def fake_check_output(cmd, *a, **k):
        seen.append(cmd)
        return "abc123456789\n"

    with mock.patch.object(checks_mod.subprocess, "check_output", fake_check_output):
        assert checks_mod._is_running("m", "sglang") is True
    assert any("name=^/sglang-m$" in tok for cmd in seen for tok in cmd)


def test_memory_guard_sizes_the_static_reservation():
    """--mem-fraction-static is a hard up-front reservation, so the guard must
    size it like vLLM's utilization — sizing from the ~17GB of weights would
    wave through a start that grabs ~121Gi of a 128Gi box."""
    with mock.patch.object(memory_mod, "_ram_line", return_value=(128.0, 68.0, 60.0)):
        ok, why = memory_mod._memory_check("m", {"runtime": "sglang",
                                                 "model": "RadixArk/Qwen3.8-27B-NVFP4",
                                                 "mem_fraction_static": 0.95})
    assert ok is False
    assert "122" in why and "60Gi free" in why

    with mock.patch.object(memory_mod, "_ram_line", return_value=(128.0, 4.0, 124.0)):
        ok, _ = memory_mod._memory_check("m", {"runtime": "sglang",
                                               "model": "RadixArk/Qwen3.8-27B-NVFP4",
                                               "mem_fraction_static": 0.95})
    assert ok is True


# --------------------------------------------------------------------------
# Failure case
# --------------------------------------------------------------------------

def test_failed_docker_run_reports_dead_without_waiting():
    """A non-zero docker run must short-circuit — no readiness poll, and the
    stale container is cleared first so the next attempt is not a name clash."""
    calls = []

    def fake_run(cmd, *a, **k):
        cl = list(cmd)
        calls.append(cl)
        return _Result(1 if cl[:2] == ["docker", "run"] else 0)

    waited = []

    with mock.patch.object(sglang_mod.subprocess, "run", fake_run), \
         mock.patch.object(sglang_mod, "_wait_ready",
                           lambda *a, **k: waited.append(a) or "ready"), \
         mock.patch.object(checks_mod, "_docker_container", return_value=None):
        state = SGLangRuntime().start("m", _yaml())

    assert state.status == "dead"
    assert waited == [], "must not poll for readiness after a failed launch"
    run_i = next(i for i, c in enumerate(calls) if c[:2] == ["docker", "run"])
    rm_i = next(i for i, c in enumerate(calls) if c[:3] == ["docker", "container", "rm"])
    assert rm_i < run_i, "stale container must be cleared before docker run"
