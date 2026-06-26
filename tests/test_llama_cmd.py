"""Tests for llama-server command construction.

These tests import from the lmswitch package and stub subprocess.Popen
to capture the argv without launching anything.
"""

import re
import tempfile
from pathlib import Path
from unittest import mock

import lmswitch.runtimes.llama as llama_mod


def _capture_llama_cmd() -> list:
    """Invokes the llama start path with Popen stubbed; returns the argv list."""
    captured: dict = {}

    class _FakeProc:
        pid = 999999

        def poll(self):
            return None

        def wait(self, *a, **k):
            return 0

    def _fake_popen(cmd, *a, **k):
        captured["cmd"] = list(cmd)
        return _FakeProc()

    yaml = {
        "runtime": "llama",
        "model": "unsloth/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf",
        "port": 8085,
        "ctx": 65536,
        "display_name": "Qwen3-4B",
        "_models_dir": Path(tempfile.mkdtemp()),
    }

    with mock.patch.object(llama_mod.subprocess, "Popen", _fake_popen), \
         mock.patch.object(llama_mod.subprocess, "run", return_value=type("R", (), {"returncode": 0})()), \
         mock.patch.object(llama_mod.time, "sleep"):
        llama_mod._start_llama_direct("qwen3-4b", yaml)
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


def test_fit_disabled_by_default():
    """`-fit off` is passed by default to avoid the auto-fit cudaMemGetInfo abort."""
    cmd = _capture_llama_cmd()
    assert "-fit" in cmd, f"expected -fit flag: {cmd}"
    assert cmd[cmd.index("-fit") + 1] == "off"


if __name__ == "__main__":
    failures = 0
    for fn in (test_no_equals_form_args,
               test_model_path_is_separate_arg,
               test_diagnostics_not_suppressed,
               test_fit_disabled_by_default):
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL {fn.__name__}: {e}")
    raise SystemExit(failures)
