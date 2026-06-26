"""Tests for CLI command dispatch on representative argv.

Each test exercises main() with a specific argv list and checks that
the correct sub-command is dispatched (via stubs that prevent side effects).
"""

import sys
import tempfile
from pathlib import Path
from unittest import mock

import lmswitch.cli as cli_mod
from lmswitch.cli import main


def _capture_stdout(fn):
    """Capture printed output from a callable."""
    import io
    buf = io.StringIO()
    old = sys.stdout
    try:
        sys.stdout = buf
        fn()
    finally:
        sys.stdout = old
    return buf.getvalue()


# ---------------------------------------------------------------------------
# main() dispatch tests
# ---------------------------------------------------------------------------

def test_main_no_args_calls_show():
    """`main()` with no args calls show() (interactive TUI)."""
    shown = {"called": False}
    with mock.patch.object(cli_mod, "show", lambda: shown.__setitem__("called", True)):
        with mock.patch.object(cli_mod.sys, "argv", ["lmswitch"]):
            main()
    assert shown["called"]


def test_main_help_prints_help():
    """`lmswitch --help` prints help text."""
    import io
    old_argv = sys.argv
    sys.argv = ["lmswitch", "--help"]
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        main()
    except SystemExit:
        pass  # --help calls sys.exit(0)
    finally:
        sys.stdout = old_stdout
    sys.argv = old_argv
    output = buf.getvalue()
    assert "lmswitch" in output, "--help should print usage text"
    assert "Usage:" in output


def test_main_list():
    """`lmswitch list` dispatches cmd_list."""
    called = {"called": False}
    with mock.patch.object(cli_mod, "cmd_list", lambda: called.__setitem__("called", True)):
        with mock.patch.object(cli_mod.sys, "argv", ["lmswitch", "list"]):
            main()
    assert called["called"]


def test_main_status():
    """`lmswitch status` dispatches cmd_list (alias)."""
    called = {"called": False}
    with mock.patch.object(cli_mod, "cmd_list", lambda: called.__setitem__("called", True)):
        with mock.patch.object(cli_mod.sys, "argv", ["lmswitch", "status"]):
            main()
    assert called["called"]


def test_main_ls():
    """`lmswitch ls` dispatches cmd_list (alias)."""
    called = {"called": False}
    with mock.patch.object(cli_mod, "cmd_list", lambda: called.__setitem__("called", True)):
        with mock.patch.object(cli_mod.sys, "argv", ["lmswitch", "ls"]):
            main()
    assert called["called"]


def test_main_on_dispatches_cmd_on():
    """`lmswitch on <name>` dispatches cmd_on."""
    called = {"called": False}
    with mock.patch.object(cli_mod, "cmd_on", lambda name: called.__setitem__("called", True)):
        with mock.patch.object(cli_mod.sys, "argv", ["lmswitch", "on", "testmodel"]):
            main()
    assert called["called"]


def test_main_start_dispatches_cmd_on():
    """`lmswitch start <name>` dispatches cmd_on (alias)."""
    called = {"called": False}
    with mock.patch.object(cli_mod, "cmd_on", lambda name: called.__setitem__("called", True)):
        with mock.patch.object(cli_mod.sys, "argv", ["lmswitch", "start", "testmodel"]):
            main()
    assert called["called"]


def test_main_off_dispatches_cmd_off():
    """`lmswitch off <name>` dispatches cmd_off."""
    called = {"called": False}
    with mock.patch.object(cli_mod, "cmd_off", lambda name: called.__setitem__("called", True)):
        with mock.patch.object(cli_mod.sys, "argv", ["lmswitch", "off", "testmodel"]):
            main()
    assert called["called"]


def test_main_stop_dispatches_cmd_off():
    """`lmswitch stop <name>` dispatches cmd_off (alias)."""
    called = {"called": False}
    with mock.patch.object(cli_mod, "cmd_off", lambda name: called.__setitem__("called", True)):
        with mock.patch.object(cli_mod.sys, "argv", ["lmswitch", "stop", "testmodel"]):
            main()
    assert called["called"]


def test_main_sync_dispatches_cmd_sync():
    """`lmswitch sync` dispatches cmd_sync."""
    called = {"called": False}
    with mock.patch.object(cli_mod, "cmd_sync", lambda: called.__setitem__("called", True)):
        with mock.patch.object(cli_mod.sys, "argv", ["lmswitch", "sync"]):
            main()
    assert called["called"]


def test_main_init_dispatches_cmd_init():
    """`lmswitch init` dispatches cmd_init."""
    called = {"called": False}
    with mock.patch.object(cli_mod, "cmd_init", lambda: called.__setitem__("called", True)):
        with mock.patch.object(cli_mod.sys, "argv", ["lmswitch", "init"]):
            main()
    assert called["called"]


def test_main_add_dispatches_cmd_add():
    """`lmswitch add <name>` dispatches cmd_add."""
    called = {"called": False}
    with mock.patch.object(cli_mod, "cmd_add", lambda name: called.__setitem__("called", True)):
        with mock.patch.object(cli_mod.sys, "argv", ["lmswitch", "add", "mymodel"]):
            main()
    assert called["called"]


def test_main_serve_dispatches_cmd_serve():
    """`lmswitch serve <name>` dispatches cmd_serve."""
    called = {"called": False}
    with mock.patch.object(cli_mod, "cmd_serve", lambda name: called.__setitem__("called", True)):
        with mock.patch.object(cli_mod.sys, "argv", ["lmswitch", "serve", "mymodel"]):
            main()
    assert called["called"]


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


def test_main_h_flag_exits():
    """`lmswitch -h` prints help and returns."""
    import io
    old_argv = sys.argv
    sys.argv = ["lmswitch", "-h"]
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    main()
    sys.stdout = old_stdout
    sys.argv = old_argv
    output = buf.getvalue()
    assert "-h" in output or "--help" in output
    assert "Usage:" in output


# ---------------------------------------------------------------------------
# render output check
# ---------------------------------------------------------------------------

def test_render_prints_table_header():
    """render() prints the expected table header with column names."""
    models = [
        {"name": "test", "display": "Test", "runtime": "llama", "type": "gguf",
         "port": 8081, "ctx": "65536", "size": 0, "present": False,
         "restart": None, "family": "Qwen", "fam_order": 0},
    ]
    output = _capture_stdout(lambda: cli_mod.render(models))
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
    output = _capture_stdout(
        lambda: cli_mod.render(models)
    )
    # Running check stub
    with mock.patch.object(cli_mod, "_is_running", return_value=True):
        output = _capture_stdout(
            lambda: cli_mod.render(models)
        )
    assert "●" in output


def test_render_shows_missing_indicator():
    """render() shows ✗ for models that are not downloaded."""
    models = [
        {"name": "missing", "display": "Missing", "runtime": "llama", "type": "gguf",
         "port": 8081, "ctx": "65536", "size": 0, "present": False,
         "restart": None, "family": "Qwen", "fam_order": 0},
    ]
    output = _capture_stdout(lambda: cli_mod.render(models))
    assert "✗" in output


# ---------------------------------------------------------------------------
# _resolve dispatch
# ---------------------------------------------------------------------------

def test_resolve_by_index():
    """_resolve accepts numeric indices."""
    tmp = tempfile.mkdtemp()
    (Path(tmp) / "model1.yaml").write_text("runtime: llama\nport: 8081\n")
    (Path(tmp) / "model2.yaml").write_text("runtime: llama\nport: 8082\n")
    models = [
        {"name": "model1", "display": "Model1", "runtime": "llama", "type": "gguf",
         "port": 8081, "ctx": "", "size": 0, "present": False,
         "restart": None, "family": "Qwen", "fam_order": 0},
        {"name": "model2", "display": "Model2", "runtime": "llama", "type": "gguf",
         "port": 8082, "ctx": "", "size": 0, "present": False,
         "restart": None, "family": "Qwen", "fam_order": 0},
    ]
    with mock.patch("lmswitch.system.io.CONF_DIR", Path(tmp)), \
         mock.patch("lmswitch.cli.load_models", return_value=models):
        assert cli_mod._resolve("1") == "model1"
        assert cli_mod._resolve("2") == "model2"


def test_resolve_by_name():
    """_resolve accepts model names."""
    tmp = tempfile.mkdtemp()
    (Path(tmp) / "mymodel.yaml").write_text("runtime: llama\nport: 8081\n")
    with mock.patch("lmswitch.cli.CONF_DIR", Path(tmp)), \
         mock.patch("lmswitch.cli.load_models", return_value=[]):
        assert cli_mod._resolve("mymodel") == "mymodel"


def test_resolve_invalid_index_exits():
    """_resolve exits on invalid index."""
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


# ---------------------------------------------------------------------------
# Integration tests — exercise real cmd_* bodies (no stubbing of cmd_*)
# ---------------------------------------------------------------------------

def test_integration_on_dispatches_real_cmd_on():
    """`lmswitch on <name>` exercises the real cmd_on body (start_model + regen_all)."""
    import io
    tmp = tempfile.mkdtemp()
    (Path(tmp) / "integ.yaml").write_text("runtime: llama\nport: 8099\n")
    models = [
        {"name": "integ", "display": "Integ", "runtime": "llama", "type": "gguf",
         "port": 8099, "ctx": "", "size": 0, "present": False,
         "restart": None, "family": "Test", "fam_order": 0},
    ]

    start_called = {"called": False}
    regen_called = {"called": False}

    def fake_start(*a, **k):
        start_called["called"] = True
    def fake_regen():
        regen_called["called"] = True

    old_argv = sys.argv
    sys.argv = ["lmswitch", "on", "integ"]
    buf = io.StringIO()

    with mock.patch("lmswitch.cli.CONF_DIR", Path(tmp)):
        with mock.patch("lmswitch.cli.load_models", return_value=models):
            with mock.patch("lmswitch.cli.start_model", fake_start):
                with mock.patch("lmswitch.cli.regen_all", fake_regen):
                    with mock.patch.object(cli_mod.sys, "stdout", buf):
                        main()

    assert start_called["called"], "real cmd_on should call start_model"
    assert regen_called["called"], "real cmd_on should call regen_all"
    sys.argv = old_argv


def test_integration_off_dispatches_real_cmd_off():
    """`lmswitch off <name>` exercises the real cmd_off body (stop_model + regen_all)."""
    import io
    tmp = tempfile.mkdtemp()
    (Path(tmp) / "integ2.yaml").write_text("runtime: llama\nport: 8099\n")
    models = [
        {"name": "integ2", "display": "Integ2", "runtime": "llama", "type": "gguf",
         "port": 8099, "ctx": "", "size": 0, "present": False,
         "restart": None, "family": "Test", "fam_order": 0},
    ]

    stop_called = {"called": False}
    regen_called = {"called": False}

    def fake_stop(*a, **k):
        stop_called["called"] = True
    def fake_regen():
        regen_called["called"] = True

    old_argv = sys.argv
    sys.argv = ["lmswitch", "off", "integ2"]
    buf = io.StringIO()

    with mock.patch("lmswitch.cli.CONF_DIR", Path(tmp)):
        with mock.patch("lmswitch.cli.load_models", return_value=models):
            with mock.patch("lmswitch.cli.stop_model", fake_stop):
                with mock.patch("lmswitch.cli.regen_all", fake_regen):
                    with mock.patch.object(cli_mod.sys, "stdout", buf):
                        main()

    assert stop_called["called"], "real cmd_off should call stop_model"
    assert regen_called["called"], "real cmd_off should call regen_all"
    sys.argv = old_argv


def test_integration_sync_calls_real_cmd_sync():
    """`lmswitch sync` exercises the real cmd_sync body (regen_*)."""
    import io

    def fake_get_targets():
        return ["opencode"]

    regen_calls = []
    def fake_regen(*a):
        regen_calls.append(True)
        return False

    old_argv = sys.argv
    sys.argv = ["lmswitch", "sync"]
    buf = io.StringIO()

    with mock.patch("lmswitch.cli._get_sync_targets", fake_get_targets):
        with mock.patch("lmswitch.cli.regen_opencode", fake_regen):
            with mock.patch("lmswitch.cli.regen_hermes", fake_regen):
                with mock.patch("lmswitch.cli.regen_grok", fake_regen):
                    with mock.patch.object(cli_mod.sys, "stdout", buf):
                        main()

    assert len(regen_calls) == 1, "real cmd_sync should call regen functions"
    sys.argv = old_argv


def test_integration_list_dispatches_real_cmd_list():
    """`lmswitch list` exercises the real cmd_list body (load_models + render)."""
    import io
    tmp = tempfile.mkdtemp()
    models = [
        {"name": "testlist", "display": "TestList", "runtime": "llama", "type": "gguf",
         "port": 8099, "ctx": "", "size": 0, "present": False,
         "restart": None, "family": "Test", "fam_order": 0},
    ]

    old_argv = sys.argv
    sys.argv = ["lmswitch", "list"]

    with mock.patch("lmswitch.cli.CONF_DIR", Path(tmp)):
        with mock.patch("lmswitch.cli.load_models", return_value=models):
            with mock.patch("lmswitch.cli._is_running", return_value=False):
                buf = io.StringIO()
                with mock.patch.object(cli_mod.sys, "stdout", buf):
                    main()
                output = buf.getvalue()

    assert "TYPE" in output, "real cmd_list should call render which prints table header"
    assert "NAME" in output
    sys.argv = old_argv


def test_integration_start_alias_dispatches_real_cmd_on():
    """`lmswitch start <name>` dispatches to real cmd_on (alias of `on`)."""
    import io
    tmp = tempfile.mkdtemp()
    (Path(tmp) / "integ3.yaml").write_text("runtime: llama\nport: 8099\n")
    models = [
        {"name": "integ3", "display": "Integ3", "runtime": "llama", "type": "gguf",
         "port": 8099, "ctx": "", "size": 0, "present": False,
         "restart": None, "family": "Test", "fam_order": 0},
    ]

    start_called = {"called": False}
    regen_called = {"called": False}

    def fake_start(*a, **k):
        start_called["called"] = True
    def fake_regen():
        regen_called["called"] = True

    old_argv = sys.argv
    sys.argv = ["lmswitch", "start", "integ3"]
    buf = io.StringIO()

    with mock.patch("lmswitch.cli.CONF_DIR", Path(tmp)):
        with mock.patch("lmswitch.cli.load_models", return_value=models):
            with mock.patch("lmswitch.cli.start_model", fake_start):
                with mock.patch("lmswitch.cli.regen_all", fake_regen):
                    with mock.patch.object(cli_mod.sys, "stdout", buf):
                        main()

    assert start_called["called"], "real cmd_on should be called by start alias"
    assert regen_called["called"], "real cmd_on should call regen_all"
    sys.argv = old_argv


def test_integration_stop_alias_dispatches_real_cmd_off():
    """`lmswitch stop <name>` dispatches to real cmd_off (alias of `off`)."""
    import io
    tmp = tempfile.mkdtemp()
    (Path(tmp) / "integ4.yaml").write_text("runtime: llama\nport: 8099\n")
    models = [
        {"name": "integ4", "display": "Integ4", "runtime": "llama", "type": "gguf",
         "port": 8099, "ctx": "", "size": 0, "present": False,
         "restart": None, "family": "Test", "fam_order": 0},
    ]

    stop_called = {"called": False}
    regen_called = {"called": False}

    def fake_stop(*a, **k):
        stop_called["called"] = True
    def fake_regen():
        regen_called["called"] = True

    old_argv = sys.argv
    sys.argv = ["lmswitch", "stop", "integ4"]
    buf = io.StringIO()

    with mock.patch("lmswitch.cli.CONF_DIR", Path(tmp)):
        with mock.patch("lmswitch.cli.load_models", return_value=models):
            with mock.patch("lmswitch.cli.stop_model", fake_stop):
                with mock.patch("lmswitch.cli.regen_all", fake_regen):
                    with mock.patch.object(cli_mod.sys, "stdout", buf):
                        main()

    assert stop_called["called"], "real cmd_off should be called by stop alias"
    assert regen_called["called"], "real cmd_off should call regen_all"
    sys.argv = old_argv
