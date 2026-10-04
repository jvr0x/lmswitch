from pathlib import Path

import pytest

from lmswitch.runtimes.vllm_dual import VLLMDualRuntime
from lmswitch.system.io import _load_yaml


REPO = Path(__file__).resolve().parents[1]
CONF_DIR = REPO / "ai-models"
TEMPLATE = CONF_DIR / "glm53-exl3-overlay" / "chat_template.jinja"
LANES = ("glm5.3-flash-nvfp4-dual", "glm5.3-flash-uncensored-nvfp4-dual")
MOUNT = f"{TEMPLATE}:/opt/glm53-chat-template.jinja:ro"


def _cmds(lane: str) -> tuple[list[str], list[str]]:
    yaml = _load_yaml(CONF_DIR / f"{lane}.yaml")
    rt = VLLMDualRuntime()
    return rt._node_cmd(lane, yaml, 0), rt._node_cmd(lane, yaml, 1)


@pytest.mark.parametrize("lane", LANES)
def test_thinking_defaults_off_on_both_ranks(lane: str):
    for cmd in _cmds(lane):
        assert cmd[cmd.index("--default-chat-template-kwargs") + 1] == '{"enable_thinking": false}'
        assert "--chat-template=/opt/glm53-chat-template.jinja" in cmd
        assert MOUNT in cmd


def test_gated_template_honours_enable_thinking():
    text = TEMPLATE.read_text()
    assert "enable_thinking is defined" in text
    assert "'<think>' if thinking_enabled else '<think></think>'" in text


@pytest.mark.parametrize("lane", LANES)
def test_checkpoint_template_alone_cannot_disable_thinking(lane: str):
    yaml = _load_yaml(CONF_DIR / f"{lane}.yaml")
    shipped = Path(yaml["model_path"]).expanduser() / "chat_template.jinja"
    if not shipped.exists():
        pytest.skip("checkpoint not mounted on this node")
    assert "enable_thinking" not in shipped.read_text()
    assert yaml["chat_template"] == "/opt/glm53-chat-template.jinja"
