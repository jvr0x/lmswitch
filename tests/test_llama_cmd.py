"""Reproduces the llama-server launch bug.

lmswitch built every llama-server flag in `--flag=value` form. This llama.cpp
build rejects that syntax ("error: invalid argument: --model=..."), so the
process exits 1 before binding its port. The failure was invisible because the
child's stdout/stderr went to DEVNULL *and* `--log-disable` muted llama-server's
own logs.

These tests stub out subprocess.Popen to capture the argv that lmswitch would
exec, without launching anything, so they run anywhere (no GPU / no spark).

The fix must:
  * emit space-separated args (`--model PATH`, not `--model=PATH`)
  * stop passing `--log-disable`
"""

import importlib.util
import re
import tempfile
from importlib.machinery import SourceFileLoader
from pathlib import Path

_LMS = Path(__file__).resolve().parent.parent / "lmswitch"


def _load():
    """Loads the extension-less `lmswitch` script as a module."""
    loader = SourceFileLoader("lmswitch_mod", str(_LMS))
    spec = importlib.util.spec_from_loader("lmswitch_mod", loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def _capture_llama_cmd() -> list:
    """Invokes the llama start path with Popen stubbed; returns the argv list."""
    mod = _load()
    captured: dict = {}

    class _FakeProc:
        pid = 999999

        def poll(self):
            # Report "still alive" so any instant-exit detection in the fix
            # treats the launch as successful.
            return None

        def wait(self, *a, **k):
            return 0

    def _fake_popen(cmd, *a, **k):
        captured["cmd"] = cmd
        return _FakeProc()

    mod.subprocess.Popen = _fake_popen
    # The readiness probe shells out to curl via subprocess.run; stub it to
    # report "ready" immediately so the test neither hits the network nor waits.
    mod.subprocess.run = lambda *a, **k: type("R", (), {"returncode": 0})()
    mod.RUN_DIR = Path(tempfile.mkdtemp())
    # Keep tests fast: the fix adds a real `time.sleep(2)` instant-exit probe.
    mod.time.sleep = lambda *a, **k: None

    yaml = {
        "runtime": "llama",
        "model": "unsloth/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf",
        "port": 8085,
        "ctx": 65536,
        "display_name": "Qwen3-4B",
    }
    mod._start_llama_direct("qwen3-4b", yaml)
    return captured["cmd"]


def test_no_equals_form_args():
    """Each flag must be its own argv element, not `--flag=value`."""
    cmd = _capture_llama_cmd()
    bad = [a for a in cmd if isinstance(a, str) and re.match(r"^--[\w-]+=", a)]
    assert not bad, f"llama args must be space-separated, found equals-form: {bad}"


def test_model_path_is_separate_arg():
    """`--model` must be followed by the gguf path as a distinct element."""
    cmd = _capture_llama_cmd()
    assert "--model" in cmd, f"missing space-separated --model flag: {cmd}"
    val = cmd[cmd.index("--model") + 1]
    assert val.endswith(".gguf"), f"--model must be followed by the gguf path, got: {val!r}"


def test_diagnostics_not_suppressed():
    """`--log-disable` hides startup errors and must not be passed."""
    cmd = _capture_llama_cmd()
    assert "--log-disable" not in cmd, "--log-disable hides startup errors; remove it"


if __name__ == "__main__":
    failures = 0
    for fn in (test_no_equals_form_args,
               test_model_path_is_separate_arg,
               test_diagnostics_not_suppressed):
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL {fn.__name__}: {e}")
    raise SystemExit(failures)
