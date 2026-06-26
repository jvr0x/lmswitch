"""Tests for lmswitch sync to opencode, hermes, and grok.

These tests stub out subprocess calls and use temp directories so they run
without a GPU, without spark-8912.local, and without any models actually
running.
"""

import importlib.util
import json
import tempfile
from importlib.machinery import SourceFileLoader
from pathlib import Path
from unittest import mock

_LMS = Path(__file__).resolve().parent.parent / "lmswitch"


def _load():
    """Load the extension-less `lmswitch` script as a module."""
    loader = SourceFileLoader("lmswitch_mod", str(_LMS))
    spec = importlib.util.spec_from_loader("lmswitch_mod", loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def _make_model_cfg(tmp: Path, name: str, port: int, ctx: int = 65536,
                    display: str = "", runtime: str = "llama",
                    model_path: str = "dummy/model.gguf"):
    """Write a minimal YAML model config and return its path."""
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


def _stub_ports(mod, ports: set[int]):
    """Make _listening_ports() return the given set."""
    mod._listening_ports = lambda: ports


def _write_lmswitch(mod, tmp: Path, extra: dict = None):
    """Write an .lmswitch config file and reload the module's config."""
    cfg_lines = [f'MODELS_DIR="{tmp}"\n']
    if extra:
        for k, v in extra.items():
            cfg_lines.append(f'{k}={v}\n')
    lmswitch_cfg = tmp.parent / ".lmswitch"
    lmswitch_cfg.write_text("".join(cfg_lines))
    # Re-read config in module
    mod.CONFIG_FILE = lmswitch_cfg
    mod.CONF_DIR = tmp


# ---------------------------------------------------------------------------
# regen_opencode
# ---------------------------------------------------------------------------

def test_regen_opencode_writes_providers():
    mod = _load()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        models_dir = tmp / "models"
        models_dir.mkdir()
        _make_model_cfg(models_dir, "qwen3.6-35b", 8089, 262144, "Qwen3.6-35B")
        _make_model_cfg(models_dir, "qwen3-vl-8b", 8109, 65536, "Qwen3-VL-8B")
        _make_model_cfg(models_dir, "ornith-35b-q8", 8115, 262144, "Ornith-35B")

        opencode_cfg = tmp / "opencode.json"
        opencode_cfg.write_text("{}")
        mod.OPENCODE = opencode_cfg
        mod.OPENCODE_EXPORT = tmp / "export.json"
        mod.OPENCODE_EXPORT.parent.mkdir(parents=True, exist_ok=True)

        _stub_ports(mod, {8089, 8109, 8115})
        _write_lmswitch(mod, models_dir)

        changed = mod.regen_opencode()
        assert changed is True

        cfg = json.loads(opencode_cfg.read_text())
        assert "provider" in cfg
        assert "qwen3.6-35b" in cfg["provider"]
        assert "qwen3-vl-8b" in cfg["provider"]
        assert "ornith-35b-q8" in cfg["provider"]

        # Check provider structure
        for model_id in ("qwen3.6-35b", "qwen3-vl-8b", "ornith-35b-q8"):
            prov = cfg["provider"][model_id]
            assert prov["npm"] == "@ai-sdk/openai-compatible"
            assert "baseURL" in prov["options"]
            assert mod.SPARK_HOST in prov["options"]["baseURL"]
            assert model_id in prov["models"]

        # Check all 3 models are present
        assert len(cfg["provider"]) == 3


def test_regen_opencode_skips_non_running_models():
    mod = _load()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        models_dir = tmp / "models"
        models_dir.mkdir()
        _make_model_cfg(models_dir, "running-model", 8089, 65536)
        _make_model_cfg(models_dir, "stopped-model", 9999, 65536)  # port not listening

        opencode_cfg = tmp / "opencode.json"
        opencode_cfg.write_text("{}")
        mod.OPENCODE = opencode_cfg
        mod.OPENCODE_EXPORT = tmp / "export.json"
        mod.OPENCODE_EXPORT.parent.mkdir(parents=True, exist_ok=True)

        _stub_ports(mod, {8089})  # only 8089 is listening
        _write_lmswitch(mod, models_dir)

        changed = mod.regen_opencode()
        assert changed is True

        cfg = json.loads(opencode_cfg.read_text())
        assert "running-model" in cfg["provider"]
        assert "stopped-model" not in cfg["provider"]


def test_regen_opencode_idempotent():
    mod = _load()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        models_dir = tmp / "models"
        models_dir.mkdir()
        _make_model_cfg(models_dir, "qwen3.6-35b", 8089, 65536)
        opencode_cfg = tmp / "opencode.json"
        opencode_cfg.write_text("{}")
        mod.OPENCODE = opencode_cfg
        mod.OPENCODE_EXPORT = tmp / "export.json"
        mod.OPENCODE_EXPORT.parent.mkdir(parents=True, exist_ok=True)
        _stub_ports(mod, {8089})
        _write_lmswitch(mod, models_dir)

        mod.regen_opencode()
        changed = mod.regen_opencode()
        assert changed is False


# ---------------------------------------------------------------------------
# regen_hermes
# ---------------------------------------------------------------------------

def test_regen_hermes_updates_primary_model():
    mod = _load()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        models_dir = tmp / "models"
        models_dir.mkdir()
        _make_model_cfg(models_dir, "qwen3.6-35b", 8089, 262144, "Qwen3.6-35B")
        _make_model_cfg(models_dir, "qwen3-vl-8b", 8109, 65536, "Qwen3-VL-8B")
        _make_model_cfg(models_dir, "ornith-35b-q8", 8115, 262144, "Ornith-35B")

        hermes_cfg = tmp / "hermes.yaml"
        hermes_cfg.write_text(
            "model:\n"
            "  default: old-model\n"
            "  provider: custom\n"
            "  base_url: http://localhost:9999/v1\n"
            "  api_key: none\n"
            "  context_length: 4096\n"
        )
        mod.HERMES_CONFIG = hermes_cfg

        _stub_ports(mod, {8089, 8109, 8115})
        _write_lmswitch(mod, models_dir, extra={"SYNC_HERMES": "true"})

        changed = mod.regen_hermes()
        assert changed is True

        import yaml
        cfg = yaml.safe_load(hermes_cfg.read_text())
        assert cfg["model"]["default"] == "qwen3.6-35b"
        assert cfg["model"]["base_url"] == f"http://{mod.SPARK_HOST}:8089/v1"
        assert cfg["model"]["context_length"] == 262144
        assert cfg["model"]["provider"] == "custom"


def test_regen_hermes_registers_all_models_as_custom_providers():
    """Every serving model must be registered under custom_providers so it
    shows up in hermes' /model picker — not just the single active model."""
    mod = _load()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        models_dir = tmp / "models"
        models_dir.mkdir()
        _make_model_cfg(models_dir, "qwen3.6-35b", 8089, 262144, "Qwen3.6-35B")
        _make_model_cfg(models_dir, "qwen3-vl-8b", 8109, 65536, "Qwen3-VL-8B")
        _make_model_cfg(models_dir, "gemma-4-12b-it", 8088, 262144, "Gemma-4-12B")

        hermes_cfg = tmp / "hermes.yaml"
        hermes_cfg.write_text("model:\n  default: qwen3.6-35b\n")
        mod.HERMES_CONFIG = hermes_cfg

        _stub_ports(mod, {8088, 8089, 8109})
        _write_lmswitch(mod, models_dir, extra={"SYNC_HERMES": "true"})

        assert mod.regen_hermes() is True

        import yaml
        cfg = yaml.safe_load(hermes_cfg.read_text())
        cps = {e["name"]: e for e in cfg["custom_providers"]}
        assert set(cps) == {"qwen3.6-35b", "qwen3-vl-8b", "gemma-4-12b-it"}
        assert cps["qwen3-vl-8b"]["base_url"] == f"http://{mod.SPARK_HOST}:8109/v1"
        assert cps["qwen3-vl-8b"]["context_length"] == 65536
        # Each provider lists its own model so the picker shows it.
        assert cps["gemma-4-12b-it"]["models"] == {"gemma-4-12b-it": {}}
        # discover_models must be false on every entry, or hermes' /model picker
        # live-probes each endpoint on open and hangs on a slow/stopped one.
        assert all(e["discover_models"] is False for e in cfg["custom_providers"])


def test_regen_hermes_custom_providers_drop_stopped_keep_foreign():
    """Stopped models are removed; user's own (non-spark) custom providers stay."""
    mod = _load()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        models_dir = tmp / "models"
        models_dir.mkdir()
        _make_model_cfg(models_dir, "qwen3.6-35b", 8089, 262144)
        _make_model_cfg(models_dir, "gemma-4-12b-it", 8088, 262144)

        hermes_cfg = tmp / "hermes.yaml"
        hermes_cfg.write_text(
            "model:\n  default: qwen3.6-35b\n"
            "custom_providers:\n"
            "- name: my-litellm\n"
            "  base_url: http://localhost:4000/v1\n"
            "  model: gpt-4o\n"
            "- name: gemma-4-12b-it\n"  # stale spark entry — should be replaced
            "  base_url: http://spark-8912.local:9999/v1\n"
            "  model: gemma-4-12b-it\n"
        )
        mod.HERMES_CONFIG = hermes_cfg

        # Only qwen is serving now; gemma stopped.
        _stub_ports(mod, {8089})
        _write_lmswitch(mod, models_dir, extra={"SYNC_HERMES": "true"})

        assert mod.regen_hermes() is True

        import yaml
        cfg = yaml.safe_load(hermes_cfg.read_text())
        names = [e["name"] for e in cfg["custom_providers"]]
        base_by_name = {e["name"]: e["base_url"] for e in cfg["custom_providers"]}
        # Foreign provider preserved.
        assert "my-litellm" in names
        assert base_by_name["my-litellm"] == "http://localhost:4000/v1"
        # Only the running spark model is registered; stale gemma entry gone.
        assert base_by_name["qwen3.6-35b"] == f"http://{mod.SPARK_HOST}:8089/v1"
        assert names.count("gemma-4-12b-it") == 0


def test_regen_hermes_updates_vision_model():
    mod = _load()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        models_dir = tmp / "models"
        models_dir.mkdir()
        _make_model_cfg(models_dir, "qwen3.6-35b", 8089, 262144, "Qwen3.6-35B")
        _make_model_cfg(models_dir, "qwen3-vl-8b", 8109, 65536, "Qwen3-VL-8B")

        hermes_cfg = tmp / "hermes.yaml"
        hermes_cfg.write_text(
            "model:\n"
            "  default: qwen3.6-35b\n"
            "  provider: custom\n"
            "  base_url: http://localhost:8089/v1\n"
            "  api_key: none\n"
            "  context_length: 262144\n"
            "auxiliary:\n"
            "  vision:\n"
            "    provider: custom\n"
            "    model: old-vision\n"
            "    base_url: http://localhost:9999/v1\n"
            "    api_key: none\n"
        )
        mod.HERMES_CONFIG = hermes_cfg

        _stub_ports(mod, {8089, 8109})
        _write_lmswitch(mod, models_dir, extra={"SYNC_HERMES": "true"})

        changed = mod.regen_hermes()
        assert changed is True

        import yaml
        cfg = yaml.safe_load(hermes_cfg.read_text())
        assert cfg["auxiliary"]["vision"]["model"] == "qwen3-vl-8b"
        assert cfg["auxiliary"]["vision"]["base_url"] == f"http://{mod.SPARK_HOST}:8109/v1"


def test_regen_hermes_keeps_running_default_sticky():
    """If the configured default is still serving, sync must NOT switch it to
    another running model — it only refreshes the endpoint."""
    mod = _load()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        models_dir = tmp / "models"
        models_dir.mkdir()
        _make_model_cfg(models_dir, "qwen3.6-35b", 8089, 262144, "Qwen3.6-35B")
        _make_model_cfg(models_dir, "gemma-4-12b-it", 8088, 262144, "Gemma-4-12B")

        hermes_cfg = tmp / "hermes.yaml"
        hermes_cfg.write_text(
            "model:\n"
            "  default: gemma-4-12b-it\n"
            "  provider: custom\n"
            "  base_url: http://localhost:8088/v1\n"
            "  api_key: none\n"
            "  context_length: 262144\n"
        )
        mod.HERMES_CONFIG = hermes_cfg

        _stub_ports(mod, {8088, 8089})
        _write_lmswitch(mod, models_dir, extra={"SYNC_HERMES": "true"})

        mod.regen_hermes()

        import yaml
        cfg = yaml.safe_load(hermes_cfg.read_text())
        # gemma is still running, so it must remain the default.
        assert cfg["model"]["default"] == "gemma-4-12b-it"
        assert cfg["model"]["base_url"] == f"http://{mod.SPARK_HOST}:8088/v1"


def test_regen_hermes_switches_when_default_stops():
    """When the configured default is no longer serving, fall back to a
    running non-vision model."""
    mod = _load()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        models_dir = tmp / "models"
        models_dir.mkdir()
        _make_model_cfg(models_dir, "qwen3.6-35b", 8089, 262144, "Qwen3.6-35B")
        _make_model_cfg(models_dir, "gemma-4-12b-it", 8088, 262144, "Gemma-4-12B")

        hermes_cfg = tmp / "hermes.yaml"
        hermes_cfg.write_text(
            "model:\n"
            "  default: gemma-4-12b-it\n"
            "  provider: custom\n"
            "  base_url: http://localhost:8088/v1\n"
            "  api_key: none\n"
            "  context_length: 262144\n"
        )
        mod.HERMES_CONFIG = hermes_cfg

        # gemma (8088) stopped; only qwen (8089) is serving.
        _stub_ports(mod, {8089})
        _write_lmswitch(mod, models_dir, extra={"SYNC_HERMES": "true"})

        changed = mod.regen_hermes()
        assert changed is True

        import yaml
        cfg = yaml.safe_load(hermes_cfg.read_text())
        assert cfg["model"]["default"] == "qwen3.6-35b"
        assert cfg["model"]["base_url"] == f"http://{mod.SPARK_HOST}:8089/v1"


def test_regen_hermes_skipped_when_disabled():
    mod = _load()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        models_dir = tmp / "models"
        models_dir.mkdir()
        _make_model_cfg(models_dir, "qwen3.6-35b", 8089, 65536)

        hermes_cfg = tmp / "hermes.yaml"
        hermes_cfg.write_text("model:\n  default: old\n")
        mod.HERMES_CONFIG = hermes_cfg

        _stub_ports(mod, {8089})
        _write_lmswitch(mod, models_dir, extra={"SYNC_HERMES": "false"})

        changed = mod.regen_hermes()
        assert changed is False

        # Config should be unchanged
        import yaml
        cfg = yaml.safe_load(hermes_cfg.read_text())
        assert cfg["model"]["default"] == "old"


def test_regen_hermes_skipped_when_config_missing():
    mod = _load()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        models_dir = tmp / "models"
        models_dir.mkdir()
        _make_model_cfg(models_dir, "qwen3.6-35b", 8089, 65536)

        hermes_cfg = tmp / "nonexistent.yaml"
        mod.HERMES_CONFIG = hermes_cfg

        _stub_ports(mod, {8089})
        _write_lmswitch(mod, models_dir, extra={"SYNC_HERMES": "true"})

        changed = mod.regen_hermes()
        assert changed is False


# ---------------------------------------------------------------------------
# regen_grok
# ---------------------------------------------------------------------------

def test_regen_grok_adds_model_sections():
    mod = _load()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        models_dir = tmp / "models"
        models_dir.mkdir()
        _make_model_cfg(models_dir, "qwen3.6-35b", 8089, 262144, "Qwen3.6-35B")
        _make_model_cfg(models_dir, "qwen3-vl-8b", 8109, 65536, "Qwen3-VL-8B")
        _make_model_cfg(models_dir, "ornith-35b-q8", 8115, 262144, "Ornith-35B")

        grok_cfg = tmp / "grok.toml"
        grok_cfg.write_text(
            "[cli]\n"
            "installer = \"internal\"\n"
            "\n"
            "[ui]\n"
            "max_thoughts_width = 120\n"
            "fork_secondary_model = \"grok-build\"\n"
            "yolo = false\n"
            "compact_mode = false\n"
        )
        mod.GROK_CONFIG = grok_cfg

        _stub_ports(mod, {8089, 8109, 8115})
        _write_lmswitch(mod, models_dir, extra={"SYNC_GROK": "true"})

        changed = mod.regen_grok()
        assert changed is True

        content = grok_cfg.read_text()
        # Check all 3 models are present (dots replaced with dashes in section names)
        assert "[model.qwen3-6-35b]" in content
        assert "[model.qwen3-vl-8b]" in content
        assert "[model.ornith-35b-q8]" in content

        # Check model properties
        assert 'model = "qwen3.6-35b"' in content
        assert f'base_url = "http://{mod.SPARK_HOST}:8089/v1"' in content
        assert 'context_window = 262144' in content

        # Check original sections preserved
        assert "[cli]" in content
        assert "installer = \"internal\"" in content
        assert "[ui]" in content
        assert "max_thoughts_width = 120" in content
        assert "fork_secondary_model = \"grok-build\"" in content


def test_regen_grok_preserves_existing_sections():
    mod = _load()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        models_dir = tmp / "models"
        models_dir.mkdir()
        _make_model_cfg(models_dir, "qwen3.6-35b", 8089, 65536)

        grok_cfg = tmp / "grok.toml"
        grok_cfg.write_text(
            "[cli]\n"
            "installer = \"internal\"\n"
            "\n"
            "[ui]\n"
            "max_thoughts_width = 120\n"
            "fork_secondary_model = \"grok-build\"\n"
            "yolo = false\n"
            "compact_mode = false\n"
            "\n"
            "[analytics]\n"
            "enabled = true\n"
        )
        mod.GROK_CONFIG = grok_cfg

        _stub_ports(mod, {8089})
        _write_lmswitch(mod, models_dir, extra={"SYNC_GROK": "true"})

        changed = mod.regen_grok()
        assert changed is True

        content = grok_cfg.read_text()
        # All original sections should be preserved
        assert "[cli]" in content
        assert "[ui]" in content
        assert "[analytics]" in content
        assert "enabled = true" in content


def test_regen_grok_removes_old_model_sections():
    mod = _load()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        models_dir = tmp / "models"
        models_dir.mkdir()
        _make_model_cfg(models_dir, "qwen3.6-35b", 8089, 262144, "Qwen3.6-35B")

        grok_cfg = tmp / "grok.toml"
        grok_cfg.write_text(
            "[cli]\n"
            "installer = \"internal\"\n"
            "\n"
            "[model.old-model]\n"
            "model = \"old-model\"\n"
            "base_url = \"http://localhost:9999/v1\"\n"
            "name = \"Old Model\"\n"
            "context_window = 4096\n"
        )
        mod.GROK_CONFIG = grok_cfg

        _stub_ports(mod, {8089})
        _write_lmswitch(mod, models_dir, extra={"SYNC_GROK": "true"})

        changed = mod.regen_grok()
        assert changed is True

        content = grok_cfg.read_text()
        # Old model section should be removed
        assert "[model.old-model]" not in content
        assert 'model = "old-model"' not in content
        # New model section should be present
        assert "[model.qwen3-6-35b]" in content


def test_regen_grok_idempotent():
    mod = _load()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        models_dir = tmp / "models"
        models_dir.mkdir()
        _make_model_cfg(models_dir, "qwen3.6-35b", 8089, 65536)

        grok_cfg = tmp / "grok.toml"
        grok_cfg.write_text("")
        mod.GROK_CONFIG = grok_cfg

        _stub_ports(mod, {8089})
        _write_lmswitch(mod, models_dir, extra={"SYNC_GROK": "true"})

        mod.regen_grok()
        changed = mod.regen_grok()
        assert changed is False


def test_regen_grok_skipped_when_disabled():
    mod = _load()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        models_dir = tmp / "models"
        models_dir.mkdir()
        _make_model_cfg(models_dir, "qwen3.6-35b", 8089, 65536)

        grok_cfg = tmp / "grok.toml"
        grok_cfg.write_text("")
        mod.GROK_CONFIG = grok_cfg

        _stub_ports(mod, {8089})
        _write_lmswitch(mod, models_dir, extra={"SYNC_GROK": "false"})

        changed = mod.regen_grok()
        assert changed is False


def test_regen_grok_skipped_when_parent_missing():
    mod = _load()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        models_dir = tmp / "models"
        models_dir.mkdir()
        _make_model_cfg(models_dir, "qwen3.6-35b", 8089, 65536)

        grok_cfg = tmp / "nonexistent" / "grok.toml"
        mod.GROK_CONFIG = grok_cfg

        _stub_ports(mod, {8089})
        _write_lmswitch(mod, models_dir, extra={"SYNC_GROK": "true"})

        changed = mod.regen_grok()
        assert changed is False


# ---------------------------------------------------------------------------
# _get_sync_targets
# ---------------------------------------------------------------------------

def test_get_sync_targets_defaults():
    mod = _load()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _write_lmswitch(mod, tmp)
        targets = mod._get_sync_targets()
        # When no config, defaults to all enabled
        assert "opencode" in targets
        assert "hermes" in targets
        assert "grok" in targets


def test_get_sync_targets_respects_config():
    mod = _load()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _write_lmswitch(mod, tmp, extra={
            "SYNC_OPENCODE": "true",
            "SYNC_HERMES": "false",
            "SYNC_GROK": "true",
        })
        targets = mod._get_sync_targets()
        assert "opencode" in targets
        assert "hermes" not in targets
        assert "grok" in targets


def test_get_sync_targets_fallback_to_opencode():
    mod = _load()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _write_lmswitch(mod, tmp, extra={
            "SYNC_OPENCODE": "false",
            "SYNC_HERMES": "false",
            "SYNC_GROK": "false",
        })
        targets = mod._get_sync_targets()
        # Should fallback to opencode when all disabled
        assert targets == ["opencode"]


# ---------------------------------------------------------------------------
# regen_all
# ---------------------------------------------------------------------------

def test_regen_all_calls_all_sync_functions():
    mod = _load()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        models_dir = tmp / "models"
        models_dir.mkdir()
        _make_model_cfg(models_dir, "qwen3.6-35b", 8089, 65536)

        opencode_cfg = tmp / "opencode.json"
        opencode_cfg.write_text("{}")
        mod.OPENCODE = opencode_cfg
        mod.OPENCODE_EXPORT = tmp / "export.json"
        mod.OPENCODE_EXPORT.parent.mkdir(parents=True, exist_ok=True)

        hermes_cfg = tmp / "hermes.yaml"
        hermes_cfg.write_text("model:\n  default: old\n  provider: custom\n  base_url: http://localhost:9999/v1\n  api_key: none\n  context_length: 4096\n")
        mod.HERMES_CONFIG = hermes_cfg

        grok_cfg = tmp / "grok.toml"
        grok_cfg.write_text("[cli]\ninstaller = \"internal\"\n")
        mod.GROK_CONFIG = grok_cfg

        _stub_ports(mod, {8089})
        _write_lmswitch(mod, models_dir, extra={
            "SYNC_OPENCODE": "true",
            "SYNC_HERMES": "true",
            "SYNC_GROK": "true",
        })

        changed = mod.regen_all()
        assert changed is True

        # Verify all configs were updated
        assert "qwen3.6-35b" in json.loads(opencode_cfg.read_text())["provider"]
        import yaml
        hermes = yaml.safe_load(hermes_cfg.read_text())
        assert hermes["model"]["default"] == "qwen3.6-35b"
        grok_content = grok_cfg.read_text()
        assert "[model.qwen3-6-35b]" in grok_content


def test_regen_all_skips_disabled_targets():
    mod = _load()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        models_dir = tmp / "models"
        models_dir.mkdir()
        _make_model_cfg(models_dir, "qwen3.6-35b", 8089, 65536)

        opencode_cfg = tmp / "opencode.json"
        opencode_cfg.write_text("{}")
        mod.OPENCODE = opencode_cfg
        mod.OPENCODE_EXPORT = tmp / "export.json"
        mod.OPENCODE_EXPORT.parent.mkdir(parents=True, exist_ok=True)

        hermes_cfg = tmp / "hermes.yaml"
        hermes_cfg.write_text("model:\n  default: old\n  provider: custom\n  base_url: http://localhost:9999/v1\n  api_key: none\n  context_length: 4096\n")
        mod.HERMES_CONFIG = hermes_cfg

        grok_cfg = tmp / "grok.toml"
        grok_cfg.write_text("")
        mod.GROK_CONFIG = grok_cfg

        _stub_ports(mod, {8089})
        _write_lmswitch(mod, models_dir, extra={
            "SYNC_OPENCODE": "true",
            "SYNC_HERMES": "false",
            "SYNC_GROK": "false",
        })

        changed = mod.regen_all()
        assert changed is True

        # Hermes should be unchanged
        import yaml
        hermes = yaml.safe_load(hermes_cfg.read_text())
        assert hermes["model"]["default"] == "old"

        # Grok should be unchanged
        assert grok_cfg.read_text() == ""


# ---------------------------------------------------------------------------
# Integration: cmd_on/cmd_off/toggle call regen_all
# ---------------------------------------------------------------------------

def test_cmd_on_calls_regen_all():
    mod = _load()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        models_dir = tmp / "models"
        models_dir.mkdir()
        _make_model_cfg(models_dir, "qwen3.6-35b", 8089, 65536)

        # Stub start_model to avoid actually launching llama-server
        mod.start_model = lambda name, yaml: None

        opencode_cfg = tmp / "opencode.json"
        opencode_cfg.write_text("{}")
        mod.OPENCODE = opencode_cfg
        mod.OPENCODE_EXPORT = tmp / "export.json"
        mod.OPENCODE_EXPORT.parent.mkdir(parents=True, exist_ok=True)

        hermes_cfg = tmp / "hermes.yaml"
        hermes_cfg.write_text("model:\n  default: old\n  provider: custom\n  base_url: http://localhost:9999/v1\n  api_key: none\n  context_length: 4096\n")
        mod.HERMES_CONFIG = hermes_cfg

        grok_cfg = tmp / "grok.toml"
        grok_cfg.write_text("")
        mod.GROK_CONFIG = grok_cfg

        _stub_ports(mod, {8089})
        _write_lmswitch(mod, models_dir, extra={
            "SYNC_OPENCODE": "true",
            "SYNC_HERMES": "true",
            "SYNC_GROK": "true",
        })

        # Call cmd_on
        mod.cmd_on("qwen3.6-35b")

        # Verify configs were updated
        assert "qwen3.6-35b" in json.loads(opencode_cfg.read_text())["provider"]
        import yaml
        hermes = yaml.safe_load(hermes_cfg.read_text())
        assert hermes["model"]["default"] == "qwen3.6-35b"
        assert "[model.qwen3-6-35b]" in grok_cfg.read_text()


def test_cmd_off_calls_regen_all():
    mod = _load()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        models_dir = tmp / "models"
        models_dir.mkdir()
        _make_model_cfg(models_dir, "qwen3.6-35b", 8089, 65536)

        # Stub stop_model to avoid actually killing processes
        mod.stop_model = lambda name, runtime: None

        opencode_cfg = tmp / "opencode.json"
        opencode_cfg.write_text("{}")
        mod.OPENCODE = opencode_cfg
        mod.OPENCODE_EXPORT = tmp / "export.json"
        mod.OPENCODE_EXPORT.parent.mkdir(parents=True, exist_ok=True)

        hermes_cfg = tmp / "hermes.yaml"
        hermes_cfg.write_text("model:\n  default: qwen3.6-35b\n  provider: custom\n  base_url: http://localhost:8089/v1\n  api_key: none\n  context_length: 65536\n")
        mod.HERMES_CONFIG = hermes_cfg

        grok_cfg = tmp / "grok.toml"
        grok_cfg.write_text("[model.qwen3_6_35b]\nmodel = \"qwen3.6-35b\"\nbase_url = \"http://localhost:8089/v1\"\nname = \"Qwen3.6-35B\"\ncontext_window = 65536\n")
        mod.GROK_CONFIG = grok_cfg

        _stub_ports(mod, set())  # No ports listening
        _write_lmswitch(mod, models_dir, extra={
            "SYNC_OPENCODE": "true",
            "SYNC_HERMES": "true",
            "SYNC_GROK": "true",
        })

        # Call cmd_off
        mod.cmd_off("qwen3.6-35b")

        # Verify configs were updated (model should be removed since port not listening)
        opencode = json.loads(opencode_cfg.read_text())
        assert "qwen3.6-35b" not in opencode.get("provider", {})
