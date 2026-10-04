from pathlib import Path
from unittest import mock

import pytest

import lmswitch.runtimes.llama as llama_mod
from lmswitch.system.io import _load_yaml

CONF_DIR = Path(__file__).resolve().parents[1] / "ai-models"
RECIPE = CONF_DIR / "qwen3-4b-pirate.yaml"


def _cmd(yaml: dict, models_dir: Path) -> list[str]:
    """Builds the llama-server argv for the recipe with process launch stubbed out."""
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

    with mock.patch.object(llama_mod.subprocess, "Popen", _fake_popen), \
         mock.patch.object(llama_mod.subprocess, "run", return_value=type("R", (), {"returncode": 0})()), \
         mock.patch.object(llama_mod.time, "sleep"):
        llama_mod._start_llama_direct("qwen3-4b-pirate", {**yaml, "_models_dir": models_dir})
    return captured["cmd"]


def test_serves_the_finetuned_gguf_with_its_own_template(tmp_path: Path):
    cmd = _cmd(_load_yaml(RECIPE), tmp_path)
    assert cmd[cmd.index("--model") + 1] == str(tmp_path / "jvr0x/Qwen3-4B-Pirate-GGUF/Qwen3-4B-Pirate.Q8_0.gguf")
    assert cmd[cmd.index("--port") + 1] == "8149"
    assert "--jinja" in cmd


def test_base_model_sampling_is_the_last_word(tmp_path: Path):
    cmd = _cmd(_load_yaml(RECIPE), tmp_path)
    for flag, value in (("--temp", "0.7"), ("--top-p", "0.8"), ("--top-k", "20"), ("--min-p", "0")):
        last = len(cmd) - 1 - cmd[::-1].index(flag)
        assert cmd[last + 1] == value


def test_port_collides_with_no_other_recipe():
    ours = _load_yaml(RECIPE)["port"]
    clashes = [p.name for p in CONF_DIR.glob("*.yaml") if p != RECIPE and _load_yaml(p).get("port") == ours]
    assert not clashes, f"port {ours} already used by {clashes}"


def test_weights_exist_on_this_node():
    gguf = Path("~/models").expanduser() / _load_yaml(RECIPE)["model"]
    if not gguf.parent.exists():
        pytest.skip("pirate GGUF not downloaded on this node")
    assert gguf.is_file()
