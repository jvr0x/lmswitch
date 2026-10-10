"""systemone runtime: command, registry, and chat-sync exclusion."""

import tempfile
from pathlib import Path
from unittest import mock

from lmswitch.runtimes import runtime_registry
from lmswitch.runtimes.systemone import SystemOneRuntime
from lmswitch.sync import _is_chat


def test_registry_has_systemone():
    assert runtime_registry.lookup("systemone") is SystemOneRuntime


def test_start_argv_points_at_serve_script():
    captured = {}

    class _Fake:
        pid = 4242

        def poll(self):
            return None

    def _popen(cmd, *a, **k):
        captured["cmd"] = list(cmd)
        captured["env"] = k.get("env") or {}
        return _Fake()

    yaml = {
        "runtime": "systemone",
        "kind": "d1",
        "model": "LiquidAI/d1-3B",
        "port": 8170,
        "python": "/usr/bin/python3",
        "_models_dir": Path("/tmp/models"),
    }
    with mock.patch("lmswitch.runtimes.systemone.subprocess.Popen", _popen), \
         mock.patch("lmswitch.runtimes.systemone._wait_ready", return_value="ready"), \
         tempfile.TemporaryDirectory() as td, \
         mock.patch("lmswitch.runtimes.systemone.RUN_DIR", Path(td)):
        state = SystemOneRuntime().start("d1-3b", yaml)
    assert state.status == "ready"
    assert captured["cmd"][0] == "/usr/bin/python3"
    assert captured["cmd"][1].endswith("systemone_serve.py")
    assert captured["env"]["SYSTEMONE_KIND"] == "d1"
    assert captured["env"]["SYSTEMONE_MODEL"].endswith("LiquidAI/d1-3B")
    assert captured["env"]["SYSTEMONE_PORT"] == "8170"


def test_decision_models_are_not_chat():
    assert _is_chat({"chat": False}) is False
    assert _is_chat({}) is True
