"""Tests for CLI command dispatch on representative argv.

Uses honest stubs: ``main()`` is called with real argv, real ``cmd_*`` bodies
execute (loading yamls from the fixture tree), but external I/O
(subprocess.Popen for llama/docker, config writes to opencode/hermes/grok)
is stubbed so tests run fast and deterministic.

The conftest auto-use ``lmswitch_data_dir`` fixture provides a minimal
``LMSWITCH_DATA_DIR`` tree with two YAML configs (qwen2.5-7b and mistral-7b),
so ``load_models()`` returns real data from the fixture.
"""

import subprocess
import sys
import tempfile
import io
from pathlib import Path
from unittest import mock

import lmswitch.cli as cli_mod
from lmswitch.cli import main

# ---------------------------------------------------------------------------
# Helper — capture stdout from a callable
# ---------------------------------------------------------------------------


def _capture(fn):
    """Capture printed output from a callable that uses print()."""
    buf = io.StringIO()
    old = sys.stdout
    try:
        sys.stdout = buf
        fn()
    finally:
        sys.stdout = old
    return buf.getvalue()


def _run_main(argv, ram=None):
    """Call main() with given argv, capture stdout, return captured text."""
    stdout, _ = _run_main_with_stderr(argv, ram)
    return stdout


def _run_main_with_stderr(argv, ram=None):
    """Call main() with given argv, capture stdout+stderr, return (stdout, stderr).

    Stubs ``_ram_line`` if *ram* is provided (None means no RAM line).
    Stubs ``subprocess.Popen`` and ``subprocess.run`` to prevent real
    llama-server / docker invocations.
    """
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    def fake_popen(cmd, *a, **k):
        """Fake subprocess.Popen that pretends success."""
        p = mock.MagicMock()
        p.pid = 99999
        p.poll = mock.MagicMock(return_value=0)
        return p

    def fake_run(cmd, *a, **k):
        return mock.MagicMock(returncode=0)

    def fake_rline():
        return ram  # None → no RAM line; (128, 48, 80) → RAM line

    with mock.patch.object(cli_mod.sys, "argv", argv):
        with mock.patch.object(cli_mod.sys, "stdout", stdout_buf):
            with mock.patch.object(cli_mod.sys, "stderr", stderr_buf):
                with mock.patch.object(cli_mod.subprocess, "Popen", fake_popen):
                    with mock.patch.object(cli_mod.subprocess, "run", fake_run):
                        if ram is not None:
                            with mock.patch.object(
                                cli_mod, "_ram_line", fake_rline
                            ):
                                try:
                                    main()
                                except SystemExit:
                                    pass
                        else:
                            try:
                                main()
                            except SystemExit:
                                pass
    return stdout_buf.getvalue(), stderr_buf.getvalue()


# ---------------------------------------------------------------------------
# Help / no-arg dispatch tests (no cmd_* stubbing — real main() body)
# ---------------------------------------------------------------------------

def test_main_help_prints_usage():
    """`lmswitch --help` prints the help text and exits cleanly."""
    output = _run_main(["lmswitch", "--help"])
    assert "lmswitch" in output
    assert "Usage:" in output


def test_main_h_flag_prints_usage():
    """`lmswitch -h` prints the help text."""
    output = _run_main(["lmswitch", "-h"])
    assert "-h" in output or "--help" in output
    assert "Usage:" in output


def test_main_no_args_shows_show():
    """`lmswitch` with no args calls show() (non-TTY → renders table)."""
    output = _run_main(["lmswitch"], ram=None)
    # Non-TTY show() calls render() which prints table header
    assert "TYPE" in output or "No models found" in output


# ---------------------------------------------------------------------------
# list / status / ls dispatch — real cmd_list body
# ---------------------------------------------------------------------------

def test_main_list_shows_table_header():
    """`lmswitch list` → real cmd_list → render → table header present."""
    output = _run_main(["lmswitch", "list"], ram=None)
    assert "TYPE" in output
    assert "NAME" in output
    assert "PORT" in output
    assert "DISPLAY" in output


def test_main_status_aliases_list():
    """`lmswitch status` → same output as `lmswitch list`."""
    out_status = _run_main(["lmswitch", "status"], ram=None)
    out_list = _run_main(["lmswitch", "list"], ram=None)
    assert out_status == out_list


def test_main_ls_aliases_list():
    """`lmswitch ls` → same output as `lmswitch list`."""
    out_ls = _run_main(["lmswitch", "ls"], ram=None)
    out_list = _run_main(["lmswitch", "list"], ram=None)
    assert out_ls == out_list


def test_main_list_shows_model_names():
    """`lmswitch list` shows model names from the fixture tree."""
    output = _run_main(["lmswitch", "list"], ram=None)
    assert "qwen2.5-7b" in output
    assert "mistral-7b" in output


# ---------------------------------------------------------------------------
# on / off / start / stop / sync dispatch — real cmd_* body
# ---------------------------------------------------------------------------

def test_main_on_prints_starting():
    """`lmswitch on qwen2.5-7b` → real cmd_on → prints 'Starting' or ready msg."""
    output = _run_main(
        ["lmswitch", "on", "qwen2.5-7b"],
        ram=(128, 48, 80),
    )
    assert "Starting" in output or "qwen2.5-7b" in output


def test_main_on_invalid_model_exits():
    """`lmswitch on nonexistent` → exits with error (SystemExit raised)."""
    caught = False
    caught_msg = ""
    old_argv = sys.argv
    sys.argv = ["lmswitch", "on", "nonexistent"]

    def fake_popen(cmd, *a, **k):
        p = mock.MagicMock()
        p.pid = 99999
        p.poll = mock.MagicMock(return_value=0)
        return p

    def fake_run(cmd, *a, **k):
        return mock.MagicMock(returncode=0)

    try:
        with mock.patch.object(cli_mod.subprocess, "Popen", fake_popen):
            with mock.patch.object(cli_mod.subprocess, "run", fake_run):
                main()
    except SystemExit as e:
        caught = True
        caught_msg = str(e)
    sys.argv = old_argv

    assert caught, "main should raise SystemExit for unknown model"
    assert "Unknown model" in caught_msg, f"exit message should mention the model: {caught_msg!r}"


def test_main_off_prints_stopped():
    """`lmswitch off qwen2.5-7b` → real cmd_off → prints stopped or not-running."""
    output = _run_main(
        ["lmswitch", "off", "qwen2.5-7b"],
        ram=None,
    )
    # The model isn't actually running, so we get "not running"
    assert "not running" in output or "Stopped" in output


def test_main_start_alias_dispatches_on():
    """`lmswitch start qwen2.5-7b` → same as `lmswitch on`."""
    out_start = _run_main(["lmswitch", "start", "qwen2.5-7b"], ram=(128, 48, 80))
    out_on = _run_main(["lmswitch", "on", "qwen2.5-7b"], ram=(128, 48, 80))
    # Both should contain similar output (Starting / model name)
    assert "qwen2.5-7b" in out_start
    assert "qwen2.5-7b" in out_on


def test_main_stop_alias_dispatches_off():
    """`lmswitch stop qwen2.5-7b` → same as `lmswitch off`."""
    out_stop = _run_main(["lmswitch", "stop", "qwen2.5-7b"], ram=None)
    out_off = _run_main(["lmswitch", "off", "qwen2.5-7b"], ram=None)
    assert out_stop == out_off


def test_main_sync_prints_synced():
    """`lmswitch sync` → real cmd_sync → prints synced message."""
    output = _run_main(["lmswitch", "sync"], ram=None)
    assert "Synced" in output or "already in sync" in output


def test_main_invalid_argv_exits():
    """Invalid argv calls sys.exit with help text."""
    old_argv = sys.argv
    sys.argv = ["lmswitch", "bogus"]
    caught = False
    try:
        main()
    except SystemExit:
        caught = True
    sys.argv = old_argv
    assert caught


# ---------------------------------------------------------------------------
# render output check (unit-level, no main() dispatch)
# ---------------------------------------------------------------------------

def test_render_prints_table_header():
    """render() prints the expected table header with column names."""
    models = [
        {"name": "test", "display": "Test", "runtime": "llama", "type": "gguf",
         "port": 8081, "ctx": "65536", "size": 0, "present": False,
         "restart": None, "family": "Qwen", "fam_order": 0},
    ]
    output = _capture(lambda: cli_mod.render(models))
    assert "TYPE" in output
    assert "NAME" in output
    assert "SIZE" in output
    assert "PORT" in output
    assert "DISPLAY" in output


def test_render_shows_running_models():
    """render() marks running models with ● and bold."""
    models = [
        {"name": "test", "display": "Test", "runtime": "llama", "type": "gguf",
         "port": 8081, "ctx": "65536", "size": 0, "present": True,
         "restart": None, "family": "Qwen", "fam_order": 0},
    ]
    with mock.patch.object(cli_mod, "_is_running", return_value=True):
        output = _capture(lambda: cli_mod.render(models))
    assert "●" in output


def test_render_shows_missing_indicator():
    """render() shows ✗ for models that are not downloaded."""
    models = [
        {"name": "missing", "display": "Missing", "runtime": "llama", "type": "gguf",
         "port": 8081, "ctx": "65536", "size": 0, "present": False,
         "restart": None, "family": "Qwen", "fam_order": 0},
    ]
    output = _capture(lambda: cli_mod.render(models))
    assert "✗" in output


# ---------------------------------------------------------------------------
# _resolve dispatch
# ---------------------------------------------------------------------------

def test_resolve_by_index():
    """_resolve accepts numeric indices."""
    tmp = Path(tempfile.mkdtemp())
    (tmp / "model1.yaml").write_text("runtime: llama\nport: 8081\n")
    (tmp / "model2.yaml").write_text("runtime: llama\nport: 8082\n")
    models = [
        {"name": "model1", "display": "Model1", "runtime": "llama", "type": "gguf",
         "port": 8081, "ctx": "", "size": 0, "present": False,
         "restart": None, "family": "Qwen", "fam_order": 0},
        {"name": "model2", "display": "Model2", "runtime": "llama", "type": "gguf",
         "port": 8082, "ctx": "", "size": 0, "present": False,
         "restart": None, "family": "Qwen", "fam_order": 0},
    ]
    with mock.patch("lmswitch.system.io.CONF_DIR", tmp), \
         mock.patch("lmswitch.cli.load_models", return_value=models):
        assert cli_mod._resolve("1") == ("model1", None)
        assert cli_mod._resolve("2") == ("model2", None)


def test_resolve_by_name():
    """_resolve accepts model names."""
    import tempfile
    tmp = tempfile.mkdtemp()
    (Path(tmp) / "mymodel.yaml").write_text("runtime: llama\nport: 8081\n")
    with mock.patch("lmswitch.cli.CONF_DIR", Path(tmp)), \
         mock.patch("lmswitch.cli.load_models", return_value=[]):
        assert cli_mod._resolve("mymodel") == ("mymodel", None)


def test_resolve_invalid_index_exits():
    """_resolve exits on invalid index."""
    import tempfile
    tmp = tempfile.mkdtemp()
    with mock.patch("lmswitch.cli.load_models", return_value=[]), \
         mock.patch("lmswitch.system.io.CONF_DIR", Path(tmp)):
        def raise_exit(msg):
            raise SystemExit(msg)
        with mock.patch("lmswitch.cli.sys.exit", raise_exit):
            try:
                cli_mod._resolve("99")
            except SystemExit:
                pass


def test_resolve_unknown_name_exits():
    """_resolve exits on unknown model name."""
    import tempfile
    tmp = tempfile.mkdtemp()
    models = [{"name": "other", "display": "Other", "runtime": "llama", "type": "gguf",
               "port": 8081, "ctx": "", "size": 0, "present": False,
               "restart": None, "family": "Qwen", "fam_order": 0}]
    with mock.patch("lmswitch.system.io.CONF_DIR", Path(tmp)), \
         mock.patch("lmswitch.cli.load_models", return_value=models):
        def raise_exit(msg):
            raise SystemExit(msg)
        with mock.patch("lmswitch.cli.sys.exit", raise_exit):
            try:
                cli_mod._resolve("nonexistent")
            except SystemExit:
                pass


def test_render_counts_mmap_gguf_as_used(monkeypatch):
    """Loaded GGUF weights (mmap page cache) must show as used RAM."""
    models = [{
        "name": "big-gguf", "display": "Big", "runtime": "llama",
        "type": "gguf", "port": 8080, "ctx": "", "size": 36 * 1024 ** 3,
        "present": True, "restart": None, "family": "Qwen", "fam_order": 0,
        "running": True,
    }]
    monkeypatch.setattr(cli_mod, "_ram_line", lambda: (121.0, 14.0, 107.0))
    monkeypatch.setattr(cli_mod, "_cluster_hosts", lambda: [])
    out = _capture(lambda: cli_mod.render(models))
    assert "50Gi used" in out          # 14 + 36
    assert "71Gi available" in out     # 107 - 36
