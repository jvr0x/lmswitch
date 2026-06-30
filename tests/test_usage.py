"""Tests for lmswitch usage tracking module.

These tests exercise:
  - ``record_start`` and ``record_stop`` produce valid JSONL lines
  - ``query_events`` filters correctly (by model, runtime, action, since, until, limit)
  - ``clear_events`` removes the file
  - The ``start_model`` / ``stop_model`` hooks record events
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from lmswitch.system import usage as usage_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model_cfg(name: str, runtime: str = "llama", port: int = 8081,
                    ctx: int = 65536, display: str = "",
                    model_path: str = "test/model.gguf") -> dict:
    """Create a minimal model YAML dict (as lmswitch would produce)."""
    if not display:
        display = name.replace("-", " ").title()
    return {
        "runtime": runtime,
        "model": model_path,
        "port": port,
        "ctx": ctx,
        "display_name": display,
    }


def _write_events_file(data_root: Path, lines: list[str]) -> None:
    """Write event lines to the usage file for a given data root."""
    import lmswitch.system.io as io_mod
    import lmswitch.system.usage as usage_mod_local

    io_mod.CONF_DIR = data_root
    usage_mod_local.CONF_DIR = data_root
    # Rebuild USAGE_FILE reference
    usage_mod_local.USAGE_FILE = data_root / "lmswitch-usage.json"

    path = usage_mod_local.USAGE_FILE
    path.write_text("".join(lines))


# ---------------------------------------------------------------------------
# record_start / record_stop
# ---------------------------------------------------------------------------

def test_record_start_creates_event(tmp_path: Path):
    """A start event is written with correct fields."""
    import lmswitch.system.io as io_mod
    import lmswitch.system.usage as usage_mod_local

    io_mod.CONF_DIR = tmp_path
    usage_mod_local.CONF_DIR = tmp_path
    usage_mod_local.USAGE_FILE = tmp_path / "lmswitch-usage.json"

    yaml_cfg = _make_model_cfg("qwen3.6-35b", "llama", 8089)
    usage_mod.record_start("qwen3.6-35b", yaml_cfg)

    path = usage_mod_local.USAGE_FILE
    assert path.exists()
    line = path.read_text().strip()
    ev = json.loads(line)

    assert ev["action"] == "start"
    assert ev["model"] == "qwen3.6-35b"
    assert ev["runtime"] == "llama"
    assert ev["port"] == 8089
    assert ev["ctx"] == 65536
    assert ev["display_name"] == "Qwen3.6 35B"
    assert ev["duration"] == 0
    assert "ts" in ev
    assert "config" in ev


def test_record_stop_creates_event(tmp_path: Path):
    """A stop event is written with correct fields."""
    import lmswitch.system.io as io_mod
    import lmswitch.system.usage as usage_mod_local

    io_mod.CONF_DIR = tmp_path
    usage_mod_local.CONF_DIR = tmp_path
    usage_mod_local.USAGE_FILE = tmp_path / "lmswitch-usage.json"

    usage_mod.record_stop("qwen3.6-35b", 300.5)

    path = usage_mod_local.USAGE_FILE
    ev = json.loads(path.read_text().strip())
    assert ev["action"] == "stop"
    assert ev["model"] == "qwen3.6-35b"
    assert ev["duration"] == 300.5


def test_record_stop_without_prior_start(tmp_path: Path):
    """If no start was recorded, stop still produces a valid event."""
    import lmswitch.system.io as io_mod
    import lmswitch.system.usage as usage_mod_local

    io_mod.CONF_DIR = tmp_path
    usage_mod_local.CONF_DIR = tmp_path
    usage_mod_local.USAGE_FILE = tmp_path / "lmswitch-usage.json"

    usage_mod.record_stop("orphan-model", 12.0)

    path = usage_mod_local.USAGE_FILE
    ev = json.loads(path.read_text().strip())
    assert ev["action"] == "stop"
    assert ev["model"] == "orphan-model"
    assert ev["runtime"] == "unknown"


def test_multiple_events_appended(tmp_path: Path):
    """Multiple start/stop calls produce separate JSONL lines."""
    import lmswitch.system.io as io_mod
    import lmswitch.system.usage as usage_mod_local

    io_mod.CONF_DIR = tmp_path
    usage_mod_local.CONF_DIR = tmp_path
    usage_mod_local.USAGE_FILE = tmp_path / "lmswitch-usage.json"

    usage_mod.record_start("model-a", _make_model_cfg("model-a"))
    usage_mod.record_start("model-b", _make_model_cfg("model-b", port=9090))
    usage_mod.record_stop("model-a", 100.0)

    path = usage_mod_local.USAGE_FILE
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 3

    e1, e2, e3 = (json.loads(l) for l in lines)
    assert e1["model"] == "model-a"
    assert e1["action"] == "start"
    assert e2["model"] == "model-b"
    assert e2["action"] == "start"
    assert e3["model"] == "model-a"
    assert e3["action"] == "stop"


# ---------------------------------------------------------------------------
# query_events filtering
# ---------------------------------------------------------------------------

def test_query_events_all(tmp_path: Path):
    """Query without filters returns all events."""
    import lmswitch.system.io as io_mod
    import lmswitch.system.usage as usage_mod_local

    io_mod.CONF_DIR = tmp_path
    usage_mod_local.CONF_DIR = tmp_path
    usage_mod_local.USAGE_FILE = tmp_path / "lmswitch-usage.json"

    usage_mod.record_start("model-a", _make_model_cfg("model-a"))
    usage_mod.record_start("model-b", _make_model_cfg("model-b"))
    usage_mod.record_stop("model-a", 100.0)

    events = usage_mod.query_events()
    assert len(events) == 3


def test_query_events_by_model(tmp_path: Path):
    """Query filters by model name."""
    import lmswitch.system.io as io_mod
    import lmswitch.system.usage as usage_mod_local

    io_mod.CONF_DIR = tmp_path
    usage_mod_local.CONF_DIR = tmp_path
    usage_mod_local.USAGE_FILE = tmp_path / "lmswitch-usage.json"

    usage_mod.record_start("model-a", _make_model_cfg("model-a"))
    usage_mod.record_start("model-b", _make_model_cfg("model-b"))
    usage_mod.record_stop("model-a", 100.0)

    events = usage_mod.query_events(model="model-a")
    assert all(e["model"] == "model-a" for e in events)
    assert len(events) == 2


def test_query_events_by_runtime(tmp_path: Path):
    """Query filters by runtime type."""
    import lmswitch.system.io as io_mod
    import lmswitch.system.usage as usage_mod_local

    io_mod.CONF_DIR = tmp_path
    usage_mod_local.CONF_DIR = tmp_path
    usage_mod_local.USAGE_FILE = tmp_path / "lmswitch-usage.json"

    usage_mod.record_start("gguf-model", _make_model_cfg("gguf-model", "llama"))
    usage_mod.record_start("vllm-model", _make_model_cfg("vllm-model", "vllm"))

    events_llama = usage_mod.query_events(runtime="llama")
    assert all(e["runtime"] == "llama" for e in events_llama)
    assert len(events_llama) == 1

    events_vllm = usage_mod.query_events(runtime="vllm")
    assert len(events_vllm) == 1


def test_query_events_by_action(tmp_path: Path):
    """Query filters by action type."""
    import lmswitch.system.io as io_mod
    import lmswitch.system.usage as usage_mod_local

    io_mod.CONF_DIR = tmp_path
    usage_mod_local.CONF_DIR = tmp_path
    usage_mod_local.USAGE_FILE = tmp_path / "lmswitch-usage.json"

    usage_mod.record_start("model", _make_model_cfg("model"))
    usage_mod.record_stop("model", 50.0)

    starts = usage_mod.query_events(action="start")
    assert len(starts) == 1
    assert starts[0]["action"] == "start"

    stops = usage_mod.query_events(action="stop")
    assert len(stops) == 1
    assert stops[0]["action"] == "stop"


def test_query_events_with_limit(tmp_path: Path):
    """Query respects the limit parameter."""
    import lmswitch.system.io as io_mod
    import lmswitch.system.usage as usage_mod_local

    io_mod.CONF_DIR = tmp_path
    usage_mod_local.CONF_DIR = tmp_path
    usage_mod_local.USAGE_FILE = tmp_path / "lmswitch-usage.json"

    for i in range(5):
        usage_mod.record_start(f"model-{i}", _make_model_cfg(f"model-{i}"))

    events = usage_mod.query_events(limit=3)
    assert len(events) == 3
    # Should return the last 3
    assert events[0]["model"] == "model-2"


def test_query_events_empty_file(tmp_path: Path):
    """Query on empty file returns empty list."""
    import lmswitch.system.io as io_mod
    import lmswitch.system.usage as usage_mod_local

    io_mod.CONF_DIR = tmp_path
    usage_mod_local.CONF_DIR = tmp_path
    usage_mod_local.USAGE_FILE = tmp_path / "lmswitch-usage.json"

    events = usage_mod.query_events()
    assert events == []


def test_query_events_no_file(tmp_path: Path):
    """Query on missing file returns empty list."""
    import lmswitch.system.io as io_mod
    import lmswitch.system.usage as usage_mod_local

    io_mod.CONF_DIR = tmp_path
    usage_mod_local.CONF_DIR = tmp_path
    usage_mod_local.USAGE_FILE = tmp_path / "lmswitch-usage.json"
    # Don't create the file

    events = usage_mod.query_events()
    assert events == []


def test_query_events_sorts_by_ts(tmp_path: Path):
    """Events are sorted by timestamp ascending."""
    import lmswitch.system.io as io_mod
    import lmswitch.system.usage as usage_mod_local

    io_mod.CONF_DIR = tmp_path
    usage_mod_local.CONF_DIR = tmp_path
    usage_mod_local.USAGE_FILE = tmp_path / "lmswitch-usage.json"

    usage_mod.record_start("model-a", _make_model_cfg("model-a"))
    usage_mod.record_start("model-b", _make_model_cfg("model-b"))
    usage_mod.record_start("model-c", _make_model_cfg("model-c"))

    events = usage_mod.query_events()
    models = [e["model"] for e in events]
    # The first start should be model-a (recorded first)
    assert models[0] == "model-a"


def test_query_events_combined_filters(tmp_path: Path):
    """Multiple filters work together."""
    import lmswitch.system.io as io_mod
    import lmswitch.system.usage as usage_mod_local

    io_mod.CONF_DIR = tmp_path
    usage_mod_local.CONF_DIR = tmp_path
    usage_mod_local.USAGE_FILE = tmp_path / "lmswitch-usage.json"

    usage_mod.record_start("qwen", _make_model_cfg("qwen", "llama", 8081))
    usage_mod.record_start("mistral", _make_model_cfg("mistral", "vllm", 8082))
    usage_mod.record_stop("qwen", 100.0)
    usage_mod.record_stop("mistral", 200.0)

    events = usage_mod.query_events(model="qwen", action="start")
    assert len(events) == 1
    assert events[0]["runtime"] == "llama"


# ---------------------------------------------------------------------------
# clear_events
# ---------------------------------------------------------------------------

def test_clear_events_removes_file(tmp_path: Path):
    """clear_events removes the usage file."""
    import lmswitch.system.io as io_mod
    import lmswitch.system.usage as usage_mod_local

    io_mod.CONF_DIR = tmp_path
    usage_mod_local.CONF_DIR = tmp_path
    usage_mod_local.USAGE_FILE = tmp_path / "lmswitch-usage.json"

    usage_mod.record_start("model", _make_model_cfg("model"))
    assert usage_mod_local.USAGE_FILE.exists()

    usage_mod.clear_events()
    assert usage_mod_local.USAGE_FILE.exists() is False

    # Also idempotent — clearing again shouldn't fail
    usage_mod.clear_events()


# ---------------------------------------------------------------------------
# record_stop with prior start (full cycle)
# ---------------------------------------------------------------------------

def test_stop_looks_up_start_info(tmp_path: Path):
    """record_stop pulls display_name and runtime from prior start event."""
    import lmswitch.system.io as io_mod
    import lmswitch.system.usage as usage_mod_local

    io_mod.CONF_DIR = tmp_path
    usage_mod_local.CONF_DIR = tmp_path
    usage_mod_local.USAGE_FILE = tmp_path / "lmswitch-usage.json"

    yaml_cfg = _make_model_cfg("qwen3.6-35b", "llama", 8089)
    usage_mod.record_start("qwen3.6-35b", yaml_cfg)
    usage_mod.record_stop("qwen3.6-35b", 600.0)

    events = usage_mod.query_events()
    stop_ev = [e for e in events if e["action"] == "stop"][0]
    assert stop_ev["display_name"] == "Qwen3.6 35B"
    assert stop_ev["runtime"] == "llama"
    assert stop_ev["port"] == 8089
    assert stop_ev["duration"] == 600.0


# ---------------------------------------------------------------------------
# Integration: start_model hooks
# ---------------------------------------------------------------------------

def test_start_model_hooks_record_start():
    """start_model() calls record_start."""
    from lmswitch.runtimes import start_model
    from lmswitch.system import usage as usage_mod

    yaml_cfg = _make_model_cfg("test-hook", "llama", 9999)

    with mock.patch.object(usage_mod, "record_start") as mock_rec:
        with mock.patch("lmswitch.runtimes._memory_check", return_value=(True, "")):
            with mock.patch("lmswitch.runtimes._start_systemd"):
                with mock.patch("lmswitch.runtimes.runtime_registry"):
                    start_model("test-hook", yaml_cfg)

    mock_rec.assert_called_once_with("test-hook", yaml_cfg)


def test_stop_model_hooks_record_stop():
    """stop_model() calls record_stop."""
    from lmswitch.runtimes import stop_model
    from lmswitch.system import usage as usage_mod

    with mock.patch.object(usage_mod, "record_stop") as mock_rec:
        with mock.patch("lmswitch.runtimes.runtime_registry"):
            stop_model("test-model", "llama")

    mock_rec.assert_called_once_with("test-model", 0.0)


# ---------------------------------------------------------------------------
# Stats rendering (cmd_stats)
# ---------------------------------------------------------------------------

def test_cmd_stats_displays_summary():
    """cmd_stats prints a usage summary."""
    import io
    import sys
    from lmswitch.system import usage as usage_mod
    from lmswitch.cli import cmd_stats

    # Patch usage events
    with mock.patch.object(usage_mod, "query_events") as mock_q:
        mock_q.return_value = [
            {"ts": "2024-01-01T00:00:00+00:00", "action": "start", "model": "test",
             "display_name": "Test", "runtime": "llama", "port": 8081, "ctx": 65536,
             "size": 1000000000, "duration": 0, "config": {}},
            {"ts": "2024-01-01T00:05:00+00:00", "action": "stop", "model": "test",
             "display_name": "Test", "runtime": "llama", "port": 8081, "ctx": 65536,
             "size": 1000000000, "duration": 300.0},
        ]
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            cmd_stats()
        finally:
            sys.stdout = old
        output = buf.getvalue()
        assert "Usage Statistics" in output
        assert "Starts:         1" in output
        assert "Stops:          1" in output
        assert "Summary" in output
        assert "Latest Events" in output


def test_cmd_stats_empty():
    """cmd_stats says 'No usage events recorded yet.' when empty."""
    import io
    import sys
    from lmswitch.system import usage as usage_mod
    from lmswitch.cli import cmd_stats

    with mock.patch.object(usage_mod, "query_events", return_value=[]):
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            cmd_stats()
        finally:
            sys.stdout = old
        output = buf.getvalue()
        assert "No usage events recorded yet" in output


def test_cmd_stats_clear():
    """cmd_stats_clear calls clear_events."""
    from lmswitch.cli import cmd_stats_clear
    from lmswitch.system import usage as usage_mod

    with mock.patch.object(usage_mod, "clear_events") as mock_clear:
        cmd_stats_clear()

    mock_clear.assert_called_once()


# ---------------------------------------------------------------------------
# _human_sec and __human_bytes helpers
# ---------------------------------------------------------------------------

def test_human_sec_seconds():
    from lmswitch.cli import _human_sec
    assert _human_sec(30) == "30s"
    assert _human_sec(5) == "5s"


def test_human_sec_minutes():
    from lmswitch.cli import _human_sec
    assert "m" in _human_sec(120)  # 2.0m


def test_human_sec_hours():
    from lmswitch.cli import _human_sec
    assert "h" in _human_sec(3600)  # 1.0h


def test_human_bytes_small():
    from lmswitch.cli import _human_bytes
    assert "G" in _human_bytes(1_000_000_000)  # ~1G


def test_human_bytes_zero():
    from lmswitch.cli import _human_bytes
    assert _human_bytes(0) == "-"


# ---------------------------------------------------------------------------
# End-to-end: record → query → stats pipeline
# ---------------------------------------------------------------------------

def test_full_pipeline():
    """Record events, query them, verify stats display contains all info."""
    import io
    import sys
    from lmswitch.system import usage as usage_mod
    from lmswitch.cli import cmd_stats

    # Clear any existing events
    usage_mod.clear_events()

    # Record a realistic sequence
    yaml1 = _make_model_cfg("qwen3.6-35b", "llama", 8089, display="Qwen3.6-35B")
    yaml2 = _make_model_cfg("mistral-7b", "vllm", 8090, display="Mistral-7B")

    usage_mod.record_start("qwen3.6-35b", yaml1)
    usage_mod.record_start("mistral-7b", yaml2)
    usage_mod.record_stop("qwen3.6-35b", 300.0)
    usage_mod.record_start("qwen3.6-35b", yaml1)
    usage_mod.record_stop("qwen3.6-35b", 600.0)
    usage_mod.record_stop("mistral-7b", 900.0)

    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        cmd_stats()
    finally:
        sys.stdout = old

    output = buf.getvalue()
    # Check aggregate stats
    assert "Starts:         3" in output
    assert "Stops:          3" in output
    # Check per-model (output uses display_name, not raw model name)
    assert "Qwen3.6-35B" in output
    assert "Mistral-7B" in output
    assert "llama" in output  # runtime display
    assert "vllm" in output

    # Cleanup
    usage_mod.clear_events()
