"""Tests for vLLM start robustness, graceful abort, and TUI sync.

All subprocess interaction is stubbed; nothing is launched. Runs anywhere.
"""

import sys
import tempfile
from pathlib import Path
from unittest import mock

import lmswitch.runtimes.vllm as vllm_mod
import lmswitch.runtimes as runtime_mod
import lmswitch.cli as cli_mod
from lmswitch.system import checks as checks_mod


def _Result(returncode=0):
    return type("Result", (), {"returncode": returncode})()


def test_stale_container_cleared_before_run():
    """A docker rm for the model's container must precede docker run."""
    calls = []
    state = {"ran": False}

    def fake_run(cmd, *a, **k):
        cl = list(cmd)
        calls.append(cl)
        if cl[:2] == ["docker", "run"]:
            state["ran"] = True
        return _Result(0)

    yaml = {"runtime": "vllm", "model": "nvidia/qwen3.6-35b-a3b-nvfp4",
            "port": 8114, "ctx": 32768, "gpu_memory_utilization": 0.55,
            "_models_dir": Path(tempfile.mkdtemp())}

    with mock.patch.object(vllm_mod.subprocess, "run", fake_run), \
         mock.patch.object(vllm_mod.time, "sleep"), \
         mock.patch.object(checks_mod, "_docker_container", return_value=None):
        vllm_mod._start_vllm_direct("qwen3.6-35b-nvfp4-nvidia", yaml)

    run_idx = next((i for i, c in enumerate(calls) if c[:2] == ["docker", "run"]), None)
    rm_idx = next((i for i, c in enumerate(calls) if c[:2] == ["docker", "rm"]), None)
    assert run_idx is not None, f"expected a docker run; calls={calls}"
    assert rm_idx is not None, f"expected a docker rm; calls={calls}"
    assert rm_idx < run_idx, "docker rm must run before docker run"


def test_graceful_keyboardinterrupt():
    """Ctrl-C anywhere under main() must exit cleanly, not raise."""
    def boom():
        raise KeyboardInterrupt()

    cli_mod.cmd_list = boom
    old_argv = sys.argv
    sys.argv = ["lmswitch", "list"]
    try:
        cli_mod.main()
    except KeyboardInterrupt:
        raise AssertionError("Ctrl-C must be handled gracefully")
    except SystemExit:
        pass
    finally:
        sys.argv = old_argv


def test_toggle_syncs_opencode():
    """The interactive TUI toggle must refresh opencode.json."""
    called = {"regen": 0}
    cli_mod._resolve = lambda t: t
    cli_mod.time.sleep = lambda *a, **k: None

    # Create a temp model dir with the yaml file, mock CONF_DIR and _load_yaml in cli
    tmp = tempfile.mkdtemp()
    yaml_path = Path(tmp) / "qwen3-4b.yaml"
    yaml_path.write_text("runtime: llama\nport: 8085\n")

    with mock.patch("lmswitch.system.io.CONF_DIR", Path(tmp)), \
         mock.patch.object(cli_mod, "_load_yaml", return_value={"runtime": "llama", "port": 8085}), \
         mock.patch.object(cli_mod, "start_model"), \
         mock.patch("lmswitch.sync.regen_opencode", lambda: called.__setitem__("regen", called["regen"] + 1)):
        cli_mod.toggle("qwen3-4b", "on")
    assert called["regen"] >= 1, "TUI toggle must sync opencode"


def test_wait_ready_ready_on_success():
    """A 200 from the port (curl returncode 0) yields 'ready'."""
    with mock.patch.object(vllm_mod.subprocess, "run", return_value=_Result(0)), \
         mock.patch.object(vllm_mod.time, "sleep"):
        from lmswitch.runtimes.wait import _wait_ready
        assert _wait_ready("m", 8085, 10, lambda: True) == "ready"


def test_wait_ready_dead_when_backend_exits():
    """If the backend is not alive, readiness reports 'dead'."""
    with mock.patch.object(vllm_mod.time, "sleep"):
        from lmswitch.runtimes.wait import _wait_ready
        assert _wait_ready("m", 8085, 10, lambda: False) == "dead"


def _guarded_start(ram, yaml):
    """Runs start_model with _ram_line/launchers stubbed."""
    started = {"on": False}
    def _capture(*a, **k):
        started["on"] = True
    with mock.patch("lmswitch.system.memory._ram_line", return_value=ram), \
         mock.patch("lmswitch.runtimes.vllm.VLLMRuntime.start", _capture), \
         mock.patch("lmswitch.runtimes.llama.LlamaRuntime.start", _capture), \
         mock.patch("lmswitch.runtimes.systemd._start_systemd", _capture):
        runtime_mod.start_model("m", yaml)
    return started["on"]


def test_memory_guard_refuses_insufficient_vllm():
    """A vLLM reservation larger than free RAM must be refused."""
    started = _guarded_start((121.0, 116.0, 5.0),
                             {"runtime": "vllm", "gpu_memory_utilization": 0.55})
    assert started is False


def test_memory_guard_allows_when_enough():
    """A model that fits in free RAM must start."""
    started = _guarded_start((121.0, 20.0, 95.0),
                             {"runtime": "vllm", "gpu_memory_utilization": 0.55})
    assert started is True


def test_memory_guard_force_overrides():
    """`force: true` bypasses the guard."""
    started = _guarded_start((121.0, 116.0, 5.0),
                             {"runtime": "vllm", "gpu_memory_utilization": 0.55, "force": True})
    assert started is True


def test_memory_guard_llama_uses_model_size():
    """GGUF footprint is estimated from the on-disk weight size."""
    # Patch _model_size_and_present where it's used (system/memory)
    with mock.patch("lmswitch.system.memory._ram_line", return_value=(121.0, 111.0, 10.0)), \
         mock.patch("lmswitch.system.memory._model_size_and_present", return_value=(30 * 1024 ** 3, True)):
        from lmswitch.system.memory import _memory_check
        ok, why = _memory_check("big", {"runtime": "llama", "model": "big.gguf"})
        assert ok is False, "must refuse a GGUF whose weights+headroom exceed free RAM"


if __name__ == "__main__":
    failures = 0
    for fn in (test_stale_container_cleared_before_run,
               test_graceful_keyboardinterrupt,
               test_toggle_syncs_opencode,
               test_wait_ready_ready_on_success,
               test_wait_ready_dead_when_backend_exits,
               test_memory_guard_refuses_insufficient_vllm,
               test_memory_guard_allows_when_enough,
               test_memory_guard_force_overrides,
               test_memory_guard_llama_uses_model_size):
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL {fn.__name__}: {e}")
    raise SystemExit(failures)
