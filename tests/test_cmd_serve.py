"""Tests for cmd_serve's foreground supervision (the systemd Restart=always path).

Production incident this covers: llama-server crashed under a systemd
``restart: on-failure`` unit. The wrapper process (`lmswitch serve <name>`)
stayed alive forever sleeping in a loop that never checked its child, so
systemd saw a healthy unit and never restarted anything — the model was
silently unreachable for hours, with the crashed llama-server left as an
unreaped zombie. The fix: cmd_serve must actually exit when the backing
process dies, using ``Popen.poll()`` (which detects AND reaps) rather than a
PID-liveness check (``os.kill(pid, 0)``), since a zombie keeps its PID valid
until reaped and would otherwise look "alive" forever too.
"""

import pytest
from unittest import mock

import lmswitch.cli as cli_mod
from lmswitch.runtimes.base import RunningState


def _write_model_yaml(name: str) -> None:
    (cli_mod.CONF_DIR / f"{name}.yaml").write_text(
        "runtime: llama\nmodel: dummy.gguf\nport: 8085\n"
    )


class _FakeProc:
    """A Popen stand-in whose poll() sequence is scripted call-by-call."""

    def __init__(self, poll_sequence):
        self._seq = list(poll_sequence)
        self.returncode = None

    def poll(self):
        if self._seq:
            val = self._seq.pop(0)
        else:
            val = self.returncode or 1
        self.returncode = val
        return val


def test_cmd_serve_exits_when_child_crashes(lmswitch_data_dir):
    """Expected use: llama-server runs fine for a bit, then dies — cmd_serve
    must exit (not sleep forever) so systemd's Restart=always can recover it."""
    _write_model_yaml("m")
    proc = _FakeProc(poll_sequence=[None, None, 1])  # alive, alive, then dead
    state = RunningState("ready", proc=proc)

    with mock.patch.object(cli_mod, "_start_llama_direct", return_value=state), \
         mock.patch.object(cli_mod.time, "sleep"):
        with pytest.raises(SystemExit) as exc:
            cli_mod.cmd_serve("m")

    msg = str(exc.value)
    assert "exited" in msg
    assert "restart" in msg.lower()
    assert proc.poll() is not None, "poll() must have actually reaped the child"


def test_cmd_serve_exits_immediately_on_failed_startup(lmswitch_data_dir):
    """Edge case: startup itself failed (status != "ready") — must exit right
    away rather than entering the supervision loop pretending it's serving."""
    _write_model_yaml("m")
    state = RunningState("dead", detail="startup crash")

    with mock.patch.object(cli_mod, "_start_llama_direct", return_value=state):
        with pytest.raises(SystemExit) as exc:
            cli_mod.cmd_serve("m")

    assert "failed to start" in str(exc.value)


def test_cmd_serve_missing_config_exits(lmswitch_data_dir):
    """Failure case: no YAML for the requested model — must not proceed to
    launch anything."""
    with pytest.raises(SystemExit) as exc:
        cli_mod.cmd_serve("does-not-exist")
    assert "Config not found" in str(exc.value)
