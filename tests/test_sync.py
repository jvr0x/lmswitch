"""Tests for lmswitch sync to opencode, hermes, and grok.

These tests import from the lmswitch package and use temp directories.
"""

import json
import tempfile
from pathlib import Path
from unittest import mock

from lmswitch.sync import regen_opencode, regen_hermes, regen_grok, regen_all
from lmswitch.system import _get_sync_targets
from lmswitch.cli import cmd_on, cmd_off


def _make_model_cfg(tmp: Path, name: str, port: int, ctx: int = 65536,
                    display: str = "", runtime: str = "llama",
                    model_path: str = "dummy/model.gguf"):
    if not display:
        display = name.replace("-", " ").title()
    cfg = (
        f"runtime: {runtime}\n"
        f"model: {model_path}\n"
        f"port: {port}\n"
        f"ctx: {ctx}\n"
        f'display_name: "{display}"\n'
    )
    p = tmp / f"{name}.yaml"
    p.write_text(cfg)
    return p


def _write_lmswitch_config(models_dir: Path, extra: dict = None):
    cfg_lines = [f'MODELS_DIR="{models_dir}"\n']
    if extra:
        for k, v in extra.items():
            cfg_lines.append(f'{k}={v}\n')
    (models_dir.parent / ".lmswitch").write_text("".join(cfg_lines))


# ---------------------------------------------------------------------------
# regen_opencode
# ---------------------------------------------------------------------------

def test_regen_opencode_writes_providers():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        models_dir = tmp / "models"
        models_dir.mkdir()
        _make_model_cfg(models_dir, "qwen3.6-35b", 8089, 262144, "Qwen3.6-35B")
        _make_model_cfg(models_dir, "qwen3-vl-8b", 8109, 65536, "Qwen3-VL-8B")
        _make_model_cfg(models_dir, "ornith-35b-q8", 8115, 262144, "Ornith-35B")
        _write_lmswitch_config(models_dir, {"SYNC_OPENCODE": "true"})

        opencode_cfg = tmp / "opencode.json"
        opencode_cfg.write_text("{}")
        export_cfg = tmp / "export.json"

        with mock.patch("lmswitch.sync._listening_ports", return_value={8089, 8109, 8115}), \
             mock.patch("lmswitch.sync.OPENCODE", opencode_cfg), \
             mock.patch("lmswitch.sync.OPENCODE_EXPORT", export_cfg), \
             mock.patch("lmswitch.sync.SPARK_HOST", "spark-8912.local"), \
             mock.patch("lmswitch.models.loader.CONF_DIR", models_dir), \
             mock.patch("lmswitch.system.io.CONFIG_FILE", models_dir.parent / ".lmswitch"):
            changed = regen_opencode()

        assert changed is True
        cfg = json.loads(opencode_cfg.read_text())
        assert "provider" in cfg
        for model_id in ("qwen3.6-35b", "qwen3-vl-8b", "ornith-35b-q8"):
            assert model_id in cfg["provider"]
            prov = cfg["provider"][model_id]
            assert prov["npm"] == "@ai-sdk/openai-compatible"
            assert "spark-8912.local" in prov["options"]["baseURL"]
        assert len(cfg["provider"]) == 3


def test_regen_opencode_skips_non_running_models():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        models_dir = tmp / "models"
        models_dir.mkdir()
        _make_model_cfg(models_dir, "running-model", 8089, 65536)
        _make_model_cfg(models_dir, "stopped-model", 9999, 65536)
        _write_lmswitch_config(models_dir, {"SYNC_OPENCODE": "true"})

        opencode_cfg = tmp / "opencode.json"
        opencode_cfg.write_text("{}")

        with mock.patch("lmswitch.sync._listening_ports", return_value={8089}), \
             mock.patch("lmswitch.sync.OPENCODE", opencode_cfg), \
             mock.patch("lmswitch.models.loader.CONF_DIR", models_dir), \
             mock.patch("lmswitch.system.io.CONFIG_FILE", models_dir.parent / ".lmswitch"):
            changed = regen_opencode()

        assert changed is True
        cfg = json.loads(opencode_cfg.read_text())
        assert "running-model" in cfg["provider"]
        assert "stopped-model" not in cfg["provider"]


def test_regen_opencode_idempotent():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        models_dir = tmp / "models"
        models_dir.mkdir()
        _make_model_cfg(models_dir, "qwen3.6-35b", 8089, 65536)
        _write_lmswitch_config(models_dir, {"SYNC_OPENCODE": "true"})
        opencode_cfg = tmp / "opencode.json"
        opencode_cfg.write_text("{}")

        with mock.patch("lmswitch.sync._listening_ports", return_value={8089}), \
             mock.patch("lmswitch.sync.OPENCODE", opencode_cfg), \
             mock.patch("lmswitch.models.loader.CONF_DIR", models_dir), \
             mock.patch("lmswitch.system.io.CONFIG_FILE", models_dir.parent / ".lmswitch"):
            regen_opencode()
            changed = regen_opencode()
        assert changed is False


# ---------------------------------------------------------------------------
# regen_hermes
# ---------------------------------------------------------------------------

def test_regen_hermes_updates_primary_model():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        models_dir = tmp / "models"
        models_dir.mkdir()
        _make_model_cfg(models_dir, "qwen3.6-35b", 8089, 262144, "Qwen3.6-35B")
        _make_model_cfg(models_dir, "qwen3-vl-8b", 8109, 65536, "Qwen3-VL-8B")
        _make_model_cfg(models_dir, "ornith-35b-q8", 8115, 262144, "Ornith-35B")
        _write_lmswitch_config(models_dir, {"SYNC_HERMES": "true"})

        hermes_cfg = tmp / "hermes.yaml"
        hermes_cfg.write_text(
            "model:\n  default: old-model\n  provider: custom\n"
            "  base_url: http://localhost:9999/v1\n  api_key: none\n"
            "  context_length: 4096\n"
        )

        with mock.patch("lmswitch.sync._listening_ports", return_value={8089, 8109, 8115}), \
             mock.patch("lmswitch.sync.HERMES_CONFIG", hermes_cfg), \
             mock.patch("lmswitch.sync.SPARK_HOST", "spark-8912.local"), \
             mock.patch("lmswitch.models.loader.CONF_DIR", models_dir), \
             mock.patch("lmswitch.system.io.CONFIG_FILE", models_dir.parent / ".lmswitch"):
            changed = regen_hermes()

        assert changed is True
        import yaml
        cfg = yaml.safe_load(hermes_cfg.read_text())
        assert cfg["model"]["default"] == "qwen3.6-35b"
        assert cfg["model"]["base_url"] == "http://spark-8912.local:8089/v1"


def test_regen_hermes_registers_all_models_as_custom_providers():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        models_dir = tmp / "models"
        models_dir.mkdir()
        _make_model_cfg(models_dir, "qwen3.6-35b", 8089, 262144, "Qwen3.6-35B")
        _make_model_cfg(models_dir, "qwen3-vl-8b", 8109, 65536, "Qwen3-VL-8B")
        _make_model_cfg(models_dir, "gemma-4-12b-it", 8088, 262144, "Gemma-4-12B")
        _write_lmswitch_config(models_dir, {"SYNC_HERMES": "true"})

        hermes_cfg = tmp / "hermes.yaml"
        hermes_cfg.write_text("model:\n  default: qwen3.6-35b\n")

        with mock.patch("lmswitch.sync._listening_ports", return_value={8088, 8089, 8109}), \
             mock.patch("lmswitch.sync.HERMES_CONFIG", hermes_cfg), \
             mock.patch("lmswitch.models.loader.CONF_DIR", models_dir), \
             mock.patch("lmswitch.system.io.CONFIG_FILE", models_dir.parent / ".lmswitch"):
            assert regen_hermes() is True

        import yaml
        cfg = yaml.safe_load(hermes_cfg.read_text())
        cps = {e["name"]: e for e in cfg["custom_providers"]}
        assert set(cps) == {"qwen3.6-35b", "qwen3-vl-8b", "gemma-4-12b-it"}
        assert all(e["discover_models"] is False for e in cfg["custom_providers"])


def test_regen_hermes_custom_providers_drop_stopped_keep_foreign():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        models_dir = tmp / "models"
        models_dir.mkdir()
        _make_model_cfg(models_dir, "qwen3.6-35b", 8089, 262144)
        _make_model_cfg(models_dir, "gemma-4-12b-it", 8088, 262144)
        _write_lmswitch_config(models_dir, {"SYNC_HERMES": "true"})

        hermes_cfg = tmp / "hermes.yaml"
        hermes_cfg.write_text(
            "model:\n  default: qwen3.6-35b\n"
            "custom_providers:\n"
            "- name: my-litellm\n  base_url: http://localhost:4000/v1\n"
            "- name: gemma-4-12b-it\n  base_url: http://spark-8912.local:9999/v1\n"
        )

        with mock.patch("lmswitch.sync._listening_ports", return_value={8089}), \
             mock.patch("lmswitch.sync.HERMES_CONFIG", hermes_cfg), \
             mock.patch("lmswitch.sync.SPARK_HOST", "spark-8912.local"), \
             mock.patch("lmswitch.models.loader.CONF_DIR", models_dir), \
             mock.patch("lmswitch.system.io.CONFIG_FILE", models_dir.parent / ".lmswitch"):
            assert regen_hermes() is True

        import yaml
        cfg = yaml.safe_load(hermes_cfg.read_text())
        assert "my-litellm" in [e["name"] for e in cfg["custom_providers"]]
        assert "gemma-4-12b-it" not in [e["name"] for e in cfg["custom_providers"]]


def test_regen_hermes_keeps_running_default_sticky():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        models_dir = tmp / "models"
        models_dir.mkdir()
        _make_model_cfg(models_dir, "qwen3.6-35b", 8089, 262144, "Qwen3.6-35B")
        _make_model_cfg(models_dir, "gemma-4-12b-it", 8088, 262144, "Gemma-4-12B")
        _write_lmswitch_config(models_dir, {"SYNC_HERMES": "true"})

        hermes_cfg = tmp / "hermes.yaml"
        hermes_cfg.write_text(
            "model:\n  default: gemma-4-12b-it\n  provider: custom\n"
            "  base_url: http://localhost:8088/v1\n  api_key: none\n"
            "  context_length: 262144\n"
        )

        with mock.patch("lmswitch.sync._listening_ports", return_value={8088, 8089}), \
             mock.patch("lmswitch.sync.HERMES_CONFIG", hermes_cfg), \
             mock.patch("lmswitch.models.loader.CONF_DIR", models_dir), \
             mock.patch("lmswitch.system.io.CONFIG_FILE", models_dir.parent / ".lmswitch"):
            regen_hermes()

        import yaml
        cfg = yaml.safe_load(hermes_cfg.read_text())
        assert cfg["model"]["default"] == "gemma-4-12b-it"


def test_regen_hermes_skipped_when_disabled():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        models_dir = tmp / "models"
        models_dir.mkdir()
        _make_model_cfg(models_dir, "qwen3.6-35b", 8089, 65536)
        _write_lmswitch_config(models_dir, {"SYNC_HERMES": "false"})

        hermes_cfg = tmp / "hermes.yaml"
        hermes_cfg.write_text("model:\n  default: old\n")

        with mock.patch("lmswitch.sync._listening_ports", return_value={8089}), \
             mock.patch("lmswitch.sync.HERMES_CONFIG", hermes_cfg), \
             mock.patch("lmswitch.models.loader.CONF_DIR", models_dir), \
             mock.patch("lmswitch.system.io.CONFIG_FILE", models_dir.parent / ".lmswitch"):
            assert regen_hermes() is False


def test_regen_hermes_skipped_when_config_missing():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        models_dir = tmp / "models"
        models_dir.mkdir()
        _make_model_cfg(models_dir, "qwen3.6-35b", 8089, 65536)
        _write_lmswitch_config(models_dir, {"SYNC_HERMES": "true"})

        hermes_cfg = tmp / "nonexistent.yaml"

        with mock.patch("lmswitch.sync._listening_ports", return_value={8089}), \
             mock.patch("lmswitch.sync.HERMES_CONFIG", hermes_cfg), \
             mock.patch("lmswitch.models.loader.CONF_DIR", models_dir), \
             mock.patch("lmswitch.system.io.CONFIG_FILE", models_dir.parent / ".lmswitch"):
            assert regen_hermes() is False


# ---------------------------------------------------------------------------
# regen_grok
# ---------------------------------------------------------------------------

def test_regen_grok_adds_model_sections():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        models_dir = tmp / "models"
        models_dir.mkdir()
        _make_model_cfg(models_dir, "qwen3.6-35b", 8089, 262144, "Qwen3.6-35B")
        _make_model_cfg(models_dir, "qwen3-vl-8b", 8109, 65536, "Qwen3-VL-8B")
        _make_model_cfg(models_dir, "ornith-35b-q8", 8115, 262144, "Ornith-35B")
        _write_lmswitch_config(models_dir, {"SYNC_GROK": "true"})

        grok_cfg = tmp / "grok.toml"
        grok_cfg.write_text("[cli]\ninstaller = \"internal\"\n")

        with mock.patch("lmswitch.sync._listening_ports", return_value={8089, 8109, 8115}), \
             mock.patch("lmswitch.sync.GROK_CONFIG", grok_cfg), \
             mock.patch("lmswitch.models.loader.CONF_DIR", models_dir), \
             mock.patch("lmswitch.system.io.CONFIG_FILE", models_dir.parent / ".lmswitch"):
            assert regen_grok() is True

        content = grok_cfg.read_text()
        assert "[model.qwen3-6-35b]" in content
        assert "[cli]" in content


def test_regen_grok_preserves_existing_sections():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        models_dir = tmp / "models"
        models_dir.mkdir()
        _make_model_cfg(models_dir, "qwen3.6-35b", 8089, 65536)
        _write_lmswitch_config(models_dir, {"SYNC_GROK": "true"})

        grok_cfg = tmp / "grok.toml"
        grok_cfg.write_text("[cli]\ninstaller = \"internal\"\n\n[analytics]\nenabled = true\n")

        with mock.patch("lmswitch.sync._listening_ports", return_value={8089}), \
             mock.patch("lmswitch.sync.GROK_CONFIG", grok_cfg), \
             mock.patch("lmswitch.models.loader.CONF_DIR", models_dir), \
             mock.patch("lmswitch.system.io.CONFIG_FILE", models_dir.parent / ".lmswitch"):
            assert regen_grok() is True

        content = grok_cfg.read_text()
        assert "[analytics]" in content
        assert "enabled = true" in content


def test_regen_grok_removes_old_model_sections():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        models_dir = tmp / "models"
        models_dir.mkdir()
        _make_model_cfg(models_dir, "qwen3.6-35b", 8089, 262144, "Qwen3.6-35B")
        _write_lmswitch_config(models_dir, {"SYNC_GROK": "true"})

        grok_cfg = tmp / "grok.toml"
        grok_cfg.write_text("[model.old-model]\nmodel = \"old-model\"\n")

        with mock.patch("lmswitch.sync._listening_ports", return_value={8089}), \
             mock.patch("lmswitch.sync.GROK_CONFIG", grok_cfg), \
             mock.patch("lmswitch.models.loader.CONF_DIR", models_dir), \
             mock.patch("lmswitch.system.io.CONFIG_FILE", models_dir.parent / ".lmswitch"):
            assert regen_grok() is True

        content = grok_cfg.read_text()
        assert "[model.old-model]" not in content
        assert "[model.qwen3-6-35b]" in content


def test_regen_grok_idempotent():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        models_dir = tmp / "models"
        models_dir.mkdir()
        _make_model_cfg(models_dir, "qwen3.6-35b", 8089, 65536)
        _write_lmswitch_config(models_dir, {"SYNC_GROK": "true"})
        grok_cfg = tmp / "grok.toml"
        grok_cfg.write_text("")

        with mock.patch("lmswitch.sync._listening_ports", return_value={8089}), \
             mock.patch("lmswitch.sync.GROK_CONFIG", grok_cfg), \
             mock.patch("lmswitch.models.loader.CONF_DIR", models_dir), \
             mock.patch("lmswitch.system.io.CONFIG_FILE", models_dir.parent / ".lmswitch"):
            regen_grok()
            assert regen_grok() is False


def test_regen_grok_skipped_when_disabled():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        models_dir = tmp / "models"
        models_dir.mkdir()
        _make_model_cfg(models_dir, "qwen3.6-35b", 8089, 65536)
        _write_lmswitch_config(models_dir, {"SYNC_GROK": "false"})
        grok_cfg = tmp / "grok.toml"
        grok_cfg.write_text("")

        with mock.patch("lmswitch.sync._listening_ports", return_value={8089}), \
             mock.patch("lmswitch.sync.GROK_CONFIG", grok_cfg), \
             mock.patch("lmswitch.models.loader.CONF_DIR", models_dir), \
             mock.patch("lmswitch.system.io.CONFIG_FILE", models_dir.parent / ".lmswitch"):
            assert regen_grok() is False


def test_regen_grok_skipped_when_parent_missing():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        models_dir = tmp / "models"
        models_dir.mkdir()
        _make_model_cfg(models_dir, "qwen3.6-35b", 8089, 65536)
        _write_lmswitch_config(models_dir, {"SYNC_GROK": "true"})
        grok_cfg = tmp / "nonexistent" / "grok.toml"

        with mock.patch("lmswitch.sync._listening_ports", return_value={8089}), \
             mock.patch("lmswitch.sync.GROK_CONFIG", grok_cfg), \
             mock.patch("lmswitch.models.loader.CONF_DIR", models_dir), \
             mock.patch("lmswitch.system.io.CONFIG_FILE", models_dir.parent / ".lmswitch"):
            assert regen_grok() is False


# ---------------------------------------------------------------------------
# _get_sync_targets
# ---------------------------------------------------------------------------

def test_get_sync_targets_defaults():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        cfg_file = tmp / ".lmswitch"
        cfg_file.write_text("")
        with mock.patch("lmswitch.system.io.CONFIG_FILE", cfg_file):
            targets = _get_sync_targets()
        assert "opencode" in targets
        assert "hermes" in targets
        assert "grok" in targets


def test_get_sync_targets_respects_config():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        cfg_file = tmp / ".lmswitch"
        cfg_file.write_text("SYNC_OPENCODE=true\nSYNC_HERMES=false\nSYNC_GROK=true\n")
        with mock.patch("lmswitch.system.io.CONFIG_FILE", cfg_file):
            targets = _get_sync_targets()
        assert "opencode" in targets
        assert "hermes" not in targets
        assert "grok" in targets


def test_get_sync_targets_fallback_to_opencode():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        cfg_file = tmp / ".lmswitch"
        cfg_file.write_text("SYNC_OPENCODE=false\nSYNC_HERMES=false\nSYNC_GROK=false\n")
        with mock.patch("lmswitch.system.io.CONFIG_FILE", cfg_file):
            targets = _get_sync_targets()
        assert targets == ["opencode"]


# ---------------------------------------------------------------------------
# regen_all
# ---------------------------------------------------------------------------

def test_regen_all_calls_all_sync_functions():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        models_dir = tmp / "models"
        models_dir.mkdir()
        _make_model_cfg(models_dir, "qwen3.6-35b", 8089, 65536)
        _write_lmswitch_config(models_dir, {
            "SYNC_OPENCODE": "true",
            "SYNC_HERMES": "true",
            "SYNC_GROK": "true",
        })

        opencode_cfg = tmp / "opencode.json"
        opencode_cfg.write_text("{}")
        hermes_cfg = tmp / "hermes.yaml"
        hermes_cfg.write_text("model:\n  default: old\n  provider: custom\n  base_url: http://localhost:9999/v1\n  api_key: none\n  context_length: 4096\n")
        grok_cfg = tmp / "grok.toml"
        grok_cfg.write_text("[cli]\ninstaller = \"internal\"\n")

        with mock.patch("lmswitch.sync._listening_ports", return_value={8089}), \
             mock.patch("lmswitch.sync.OPENCODE", opencode_cfg), \
             mock.patch("lmswitch.sync.OPENCODE_EXPORT", tmp / "export.json"), \
             mock.patch("lmswitch.sync.HERMES_CONFIG", hermes_cfg), \
             mock.patch("lmswitch.sync.GROK_CONFIG", grok_cfg), \
             mock.patch("lmswitch.models.loader.CONF_DIR", models_dir), \
             mock.patch("lmswitch.system.io.CONFIG_FILE", models_dir.parent / ".lmswitch"):
            changed = regen_all()

        assert changed is True
        assert "qwen3.6-35b" in json.loads(opencode_cfg.read_text())["provider"]
        import yaml
        hermes = yaml.safe_load(hermes_cfg.read_text())
        assert hermes["model"]["default"] == "qwen3.6-35b"


def test_regen_all_skips_disabled_targets():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        models_dir = tmp / "models"
        models_dir.mkdir()
        _make_model_cfg(models_dir, "qwen3.6-35b", 8089, 65536)
        _write_lmswitch_config(models_dir, {
            "SYNC_OPENCODE": "true",
            "SYNC_HERMES": "false",
            "SYNC_GROK": "false",
        })

        opencode_cfg = tmp / "opencode.json"
        opencode_cfg.write_text("{}")
        hermes_cfg = tmp / "hermes.yaml"
        hermes_cfg.write_text("model:\n  default: old\n  provider: custom\n  base_url: http://localhost:9999/v1\n  api_key: none\n  context_length: 4096\n")
        grok_cfg = tmp / "grok.toml"
        grok_cfg.write_text("")

        with mock.patch("lmswitch.sync._listening_ports", return_value={8089}), \
             mock.patch("lmswitch.sync.OPENCODE", opencode_cfg), \
             mock.patch("lmswitch.sync.OPENCODE_EXPORT", tmp / "export.json"), \
             mock.patch("lmswitch.sync.HERMES_CONFIG", hermes_cfg), \
             mock.patch("lmswitch.sync.GROK_CONFIG", grok_cfg), \
             mock.patch("lmswitch.models.loader.CONF_DIR", models_dir), \
             mock.patch("lmswitch.system.io.CONFIG_FILE", models_dir.parent / ".lmswitch"):
            changed = regen_all()

        assert changed is True
        import yaml
        hermes = yaml.safe_load(hermes_cfg.read_text())
        assert hermes["model"]["default"] == "old"
        assert grok_cfg.read_text() == ""


# ---------------------------------------------------------------------------
# Integration: cmd_on/cmd_off
# ---------------------------------------------------------------------------

def test_cmd_on_calls_regen_all():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        models_dir = tmp / "models"
        models_dir.mkdir()
        _make_model_cfg(models_dir, "qwen3.6-35b", 8089, 65536)
        _write_lmswitch_config(models_dir, {
            "SYNC_OPENCODE": "true",
            "SYNC_HERMES": "true",
            "SYNC_GROK": "true",
        })

        opencode_cfg = tmp / "opencode.json"
        opencode_cfg.write_text("{}")
        hermes_cfg = tmp / "hermes.yaml"
        hermes_cfg.write_text("model:\n  default: old\n  provider: custom\n  base_url: http://localhost:9999/v1\n  api_key: none\n  context_length: 4096\n")
        grok_cfg = tmp / "grok.toml"
        grok_cfg.write_text("")

        with mock.patch("lmswitch.sync._listening_ports", return_value={8089}), \
             mock.patch("lmswitch.sync.OPENCODE", opencode_cfg), \
             mock.patch("lmswitch.sync.OPENCODE_EXPORT", tmp / "export.json"), \
             mock.patch("lmswitch.sync.HERMES_CONFIG", hermes_cfg), \
             mock.patch("lmswitch.sync.GROK_CONFIG", grok_cfg), \
             mock.patch("lmswitch.models.loader.CONF_DIR", models_dir), \
             mock.patch("lmswitch.system.io.CONF_DIR", models_dir), \
             mock.patch("lmswitch.cli.CONF_DIR", models_dir), \
             mock.patch("lmswitch.system.io.CONFIG_FILE", models_dir.parent / ".lmswitch"), \
             mock.patch("lmswitch.cli.start_model"):
            cmd_on("qwen3.6-35b")

        assert "qwen3.6-35b" in json.loads(opencode_cfg.read_text())["provider"]
        import yaml
        hermes = yaml.safe_load(hermes_cfg.read_text())
        assert hermes["model"]["default"] == "qwen3.6-35b"


def test_cmd_off_calls_regen_all():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        models_dir = tmp / "models"
        models_dir.mkdir()
        _make_model_cfg(models_dir, "qwen3.6-35b", 8089, 65536)
        _write_lmswitch_config(models_dir, {
            "SYNC_OPENCODE": "true",
            "SYNC_HERMES": "true",
            "SYNC_GROK": "true",
        })

        opencode_cfg = tmp / "opencode.json"
        opencode_cfg.write_text("{}")
        hermes_cfg = tmp / "hermes.yaml"
        hermes_cfg.write_text("model:\n  default: qwen3.6-35b\n  provider: custom\n  base_url: http://localhost:8089/v1\n  api_key: none\n  context_length: 65536\n")
        grok_cfg = tmp / "grok.toml"
        grok_cfg.write_text("[model.qwen3_6_35b]\nmodel = \"qwen3.6-35b\"\nbase_url = \"http://localhost:8089/v1\"\nname = \"Qwen3.6-35B\"\ncontext_window = 65536\n")

        with mock.patch("lmswitch.sync._listening_ports", return_value=set()), \
             mock.patch("lmswitch.sync.OPENCODE", opencode_cfg), \
             mock.patch("lmswitch.sync.OPENCODE_EXPORT", tmp / "export.json"), \
             mock.patch("lmswitch.sync.HERMES_CONFIG", hermes_cfg), \
             mock.patch("lmswitch.sync.GROK_CONFIG", grok_cfg), \
             mock.patch("lmswitch.models.loader.CONF_DIR", models_dir), \
             mock.patch("lmswitch.system.io.CONF_DIR", models_dir), \
             mock.patch("lmswitch.cli.CONF_DIR", models_dir), \
             mock.patch("lmswitch.system.io.CONFIG_FILE", models_dir.parent / ".lmswitch"), \
             mock.patch("lmswitch.cli.start_model"):
            cmd_off("qwen3.6-35b")

        opencode = json.loads(opencode_cfg.read_text())
        assert "qwen3.6-35b" not in opencode.get("provider", {})


def test_regen_grok_repairs_dangling_default():
    """A [models] default pointing at a stopped model falls back to a live one."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        models_dir = tmp / "models"
        models_dir.mkdir()
        _make_model_cfg(models_dir, "qwen3.6-35b", 8089, 262144, "Qwen3.6-35B")
        _write_lmswitch_config(models_dir, {"SYNC_GROK": "true"})

        grok_cfg = tmp / "grok.toml"
        grok_cfg.write_text(
            '[cli]\ninstaller = "internal"\n\n'
            '[models]\ndefault = "deepseek-v4-flash-dspark"\n\n'
            '[model.deepseek-v4-flash-dspark]\nmodel = "deepseek-v4-flash-dspark"\n'
        )

        with mock.patch("lmswitch.sync._listening_ports", return_value={8089}), \
             mock.patch("lmswitch.sync.GROK_CONFIG", grok_cfg), \
             mock.patch("lmswitch.models.loader.CONF_DIR", models_dir), \
             mock.patch("lmswitch.system.io.CONFIG_FILE", models_dir.parent / ".lmswitch"):
            assert regen_grok() is True

        content = grok_cfg.read_text()
        assert 'default = "qwen3.6-35b"' in content
        assert "deepseek-v4-flash-dspark" not in content


def test_regen_grok_keeps_default_while_serving():
    """A [models] default that IS being served stays untouched (sticky)."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        models_dir = tmp / "models"
        models_dir.mkdir()
        _make_model_cfg(models_dir, "qwen3.6-35b", 8089, 262144, "Qwen3.6-35B")
        _make_model_cfg(models_dir, "ornith-35b-q8", 8115, 262144, "Ornith-35B")
        _write_lmswitch_config(models_dir, {"SYNC_GROK": "true"})

        grok_cfg = tmp / "grok.toml"
        grok_cfg.write_text(
            '[models]\ndefault = "ornith-35b-q8"\n'
        )

        with mock.patch("lmswitch.sync._listening_ports", return_value={8089, 8115}), \
             mock.patch("lmswitch.sync.GROK_CONFIG", grok_cfg), \
             mock.patch("lmswitch.models.loader.CONF_DIR", models_dir), \
             mock.patch("lmswitch.system.io.CONFIG_FILE", models_dir.parent / ".lmswitch"):
            regen_grok()

        assert 'default = "ornith-35b-q8"' in grok_cfg.read_text()
