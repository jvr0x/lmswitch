"""Reproducing tests for vLLM start robustness, graceful abort, and TUI sync.

Bugs reproduced (all currently FAIL):
  1. A stale/exited `vllm-<name>` container blocks `docker run` with a name
     conflict, because detection only checks running containers. The start path
     must clear a stale container before `docker run`.
  2. Ctrl-C during a toggle dumps a traceback instead of aborting cleanly.
  3. The interactive TUI toggle never refreshes opencode.json, so a model can
     show as running while opencode doesn't list it.

All subprocess interaction is stubbed; nothing is launched. Runs anywhere.
"""

import importlib.util
import tempfile
from importlib.machinery import SourceFileLoader
from pathlib import Path

_LMS = Path(__file__).resolve().parent.parent / "lmswitch"


def _load():
    loader = SourceFileLoader("lmswitch_mod", str(_LMS))
    spec = importlib.util.spec_from_loader("lmswitch_mod", loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


class _Result:
    def __init__(self, returncode=0):
        self.returncode = returncode


def test_stale_container_cleared_before_run():
    """A docker rm for the model's container must precede docker run."""
    mod = _load()
    calls = []
    state = {"ran": False}

    def fake_run(cmd, *a, **k):
        cl = list(cmd)
        calls.append(cl)
        if cl[:2] == ["docker", "run"]:
            state["ran"] = True
        return _Result(0)

    def fake_check_output(cmd, *a, **k):
        cl = list(cmd)
        calls.append(cl)
        if "ps" in cl:
            if "-a" in cl:                       # stale-container probe
                return "stale123\n" if not state["ran"] else ""
            return "run123\n" if state["ran"] else ""   # running only after run
        return ""

    mod.subprocess.run = fake_run
    mod.subprocess.check_output = fake_check_output
    mod.time.sleep = lambda *a, **k: None

    yaml = {"runtime": "vllm", "model": "nvidia/qwen3.6-35b-a3b-nvfp4",
            "port": 8114, "ctx": 32768, "gpu_memory_utilization": 0.55}
    mod._start_vllm_direct("qwen3.6-35b-nvfp4-nvidia", yaml)

    run_idx = next((i for i, c in enumerate(calls) if c[:2] == ["docker", "run"]), None)
    rm_idx = next((i for i, c in enumerate(calls) if c[:2] == ["docker", "rm"]), None)
    assert run_idx is not None, f"expected a docker run; calls={calls}"
    assert rm_idx is not None, f"expected a docker rm to clear a stale container; calls={calls}"
    assert rm_idx < run_idx, "docker rm must run before docker run to avoid a name conflict"


def test_graceful_keyboardinterrupt():
    """Ctrl-C anywhere under main() must exit cleanly, not raise."""
    mod = _load()

    def boom():
        raise KeyboardInterrupt()

    mod.cmd_list = boom
    mod.sys.argv = ["lmswitch", "list"]
    try:
        mod.main()
    except KeyboardInterrupt:
        raise AssertionError("Ctrl-C must be handled gracefully, not propagate KeyboardInterrupt")
    except SystemExit:
        pass  # graceful exit is acceptable


def test_toggle_syncs_opencode():
    """The interactive TUI toggle must refresh opencode.json."""
    mod = _load()
    called = {"regen": 0}
    mod.regen_opencode = lambda: called.__setitem__("regen", called["regen"] + 1)
    mod.start_model = lambda name, y: None
    mod._resolve = lambda t: t
    mod._load_yaml = lambda p: {"runtime": "llama", "port": 8085}
    mod.time.sleep = lambda *a, **k: None

    mod.toggle("qwen3-4b", "on")
    assert called["regen"] >= 1, "TUI toggle must sync opencode (call regen_opencode)"


def test_wait_ready_ready_on_success():
    """A 200 from the port (curl returncode 0) yields 'ready'."""
    mod = _load()
    mod.subprocess.run = lambda *a, **k: _Result(0)
    mod.time.sleep = lambda *a, **k: None
    assert mod._wait_ready("m", 8085, 10, lambda: True) == "ready"


def test_wait_ready_dead_when_backend_exits():
    """If the backend is not alive, readiness reports 'dead' without polling."""
    mod = _load()
    mod.time.sleep = lambda *a, **k: None
    assert mod._wait_ready("m", 8085, 10, lambda: False) == "dead"


if __name__ == "__main__":
    failures = 0
    for fn in (test_stale_container_cleared_before_run,
               test_graceful_keyboardinterrupt,
               test_toggle_syncs_opencode,
               test_wait_ready_ready_on_success,
               test_wait_ready_dead_when_backend_exits):
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL {fn.__name__}: {e}")
    raise SystemExit(failures)
