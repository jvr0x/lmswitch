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


# ---------------------------------------------------------------------------
# --metrics — llama.cpp serves /metrics only when asked, and `lmswitch stats`
# reads its token counters from there.
#
# Expected use: a plain recipe gets the flag. Edge: a recipe that already sets
# it, in either the list or the string form of extra_args, is not given a
# duplicate. Failure guard: extra_args still lands last, so a recipe keeps the
# final say under llama.cpp's last-wins parser.
# ---------------------------------------------------------------------------

def _build(**over) -> list:
    """Builds the llama argv for a minimal recipe plus *over*."""
    from pathlib import Path as _Path
    from lmswitch.runtimes.llama import LlamaRuntime
    yaml = {"model": "a/b.gguf", "port": 8085, "_models_dir": _Path("/tmp")}
    yaml.update(over)
    cmd, _ = LlamaRuntime()._build_cmd("m", yaml)
    return cmd


def test_metrics_endpoint_is_enabled():
    """Without --metrics the server 404s /metrics and no tokens are logged."""
    assert "--metrics" in _build()


def test_metrics_flag_is_not_duplicated_from_a_list():
    cmd = _build(extra_args=["--metrics", "--jinja"])
    assert cmd.count("--metrics") == 1


def test_metrics_flag_is_not_duplicated_from_a_string():
    """extra_args in string form is shell-split before the check."""
    cmd = _build(extra_args="--metrics --jinja")
    assert cmd.count("--metrics") == 1


def test_extra_args_still_come_last():
    cmd = _build(extra_args=["-fa", "on"])
    assert cmd[-2:] == ["-fa", "on"]
    assert cmd.index("--metrics") < cmd.index("-fa")


if __name__ == "__main__":
    failures = 0
    for fn in (test_no_equals_form_args,
               test_model_path_is_separate_arg,
               test_diagnostics_not_suppressed,
               test_fit_disabled_by_default,
               test_metrics_endpoint_is_enabled,
               test_metrics_flag_is_not_duplicated_from_a_list,
               test_metrics_flag_is_not_duplicated_from_a_string,
               test_extra_args_still_come_last):
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL {fn.__name__}: {e}")
    raise SystemExit(failures)
