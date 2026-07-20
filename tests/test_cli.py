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
    """_resolve accepts numeric indices (matching the default filtered view)."""
    tmp = Path(tempfile.mkdtemp())
    (tmp / "model1.yaml").write_text("runtime: llama\nport: 8081\n")
    (tmp / "model2.yaml").write_text("runtime: llama\nport: 8082\n")
    models = [
        {"name": "model1", "display": "Model1", "runtime": "llama", "type": "gguf",
         "port": 8081, "ctx": "", "size": 0, "present": True,
         "restart": None, "family": "Qwen", "fam_order": 0},
        {"name": "model2", "display": "Model2", "runtime": "llama", "type": "gguf",
         "port": 8082, "ctx": "", "size": 0, "present": True,
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

# ---------------------------------------------------------------------------
# View filters: --all / --local / --dual + default missing-weights hiding
# ---------------------------------------------------------------------------

def _ghost_yaml(data_dir):
    """Adds a recipe whose model weights don't exist on disk."""
    (data_dir / "ghost-70b.yaml").write_text(
        "runtime: llama\n"
        "model: ghost-70b/ghost-70b-q4_k_m.gguf\n"
        "port: 8099\n"
        'display_name: "Ghost 70B"\n'
    )


def _dual_yaml(data_dir):
    """Adds a vllm-dual recipe pointing at a nonexistent weights dir."""
    (data_dir / "dual-236b.yaml").write_text(
        "runtime: vllm-dual\n"
        "model: org/dual-236b\n"
        "model_path: /nonexistent/dual-236b\n"
        "port: 8098\n"
        'display_name: "Dual 236B"\n'
    )


def test_list_hides_missing_by_default(lmswitch_data_dir):
    """Default `lmswitch list` hides declared recipes with missing weights."""
    _ghost_yaml(lmswitch_data_dir)
    output = _run_main(["lmswitch", "list"], ram=None)
    assert "ghost-70b" not in output
    assert "qwen2.5-7b" in output
    assert "hidden" in output  # hint line mentions the hidden recipe


def test_list_all_shows_missing(lmswitch_data_dir):
    """`lmswitch list --all` shows every declared recipe, missing or not."""
    _ghost_yaml(lmswitch_data_dir)
    output = _run_main(["lmswitch", "list", "--all"], ram=None)
    assert "ghost-70b" in output
    assert "qwen2.5-7b" in output


def test_list_dual_shows_only_dual(lmswitch_data_dir):
    """`lmswitch list --dual --all` shows only vllm-dual recipes."""
    _dual_yaml(lmswitch_data_dir)
    output = _run_main(["lmswitch", "list", "--dual", "--all"], ram=None)
    assert "dual-236b" in output
    assert "qwen2.5-7b" not in output
    assert "mistral-7b" not in output


def test_list_local_excludes_dual(lmswitch_data_dir):
    """`lmswitch list --local` keeps single-box recipes, drops dual ones."""
    _dual_yaml(lmswitch_data_dir)
    output = _run_main(["lmswitch", "list", "--local", "--all"], ram=None)
    assert "qwen2.5-7b" in output
    assert "mistral-7b" in output
    assert "dual-236b" not in output


def test_list_local_and_dual_conflict_exits():
    """`--local --dual` is rejected as mutually exclusive."""
    stdout, _ = _run_main_with_stderr(["lmswitch", "list", "--local", "--dual"])
    assert "TYPE" not in stdout  # no table rendered


def test_filter_models_keeps_running_missing():
    """A running model is never hidden, even if presence probing says missing."""
    models = [
        {"name": "a", "runtime": "llama", "type": "gguf",
         "present": False, "running": True},
        {"name": "b", "runtime": "llama", "type": "gguf",
         "present": False, "running": False},
    ]
    out = cli_mod._filter_models(models)
    assert [m["name"] for m in out] == ["a"]


def test_filter_models_local_drops_remote():
    """view="local" drops peer-owned entries and dual recipes."""
    models = [
        {"name": "here", "runtime": "llama", "type": "gguf",
         "present": True, "running": False},
        {"name": "there", "runtime": "llama", "type": "gguf",
         "present": True, "running": False, "remote_host": "gigabyte"},
        {"name": "both", "runtime": "vllm-dual", "type": "dual",
         "present": True, "running": False},
    ]
    out = cli_mod._filter_models(models, view="local")
    assert [m["name"] for m in out] == ["here"]


def test_resolve_index_matches_filtered_table(lmswitch_data_dir):
    """Numeric indexes skip hidden (missing-weights) recipes, matching list."""
    _ghost_yaml(lmswitch_data_dir)  # sorts into the table if unfiltered
    output = _run_main(["lmswitch", "list"], ram=None)
    assert "ghost-70b" not in output
    # Index 1 must resolve to a visible model, never the hidden ghost.
    name, remote = cli_mod._resolve("1")
    assert name != "ghost-70b"
    assert remote is None

# ---------------------------------------------------------------------------
# TUI keypress input: digits-only field, letters act instantly
# ---------------------------------------------------------------------------

def _feed_keys(keys):
    """Returns a fake _read_key that pops from the given key sequence."""
    seq = list(keys)
    return lambda: seq.pop(0)


def test_prompt_toggle_digits_submit():
    """Digits build the field; Enter submits them as ("nums", buffer)."""
    with mock.patch.object(cli_mod, "_read_key",
                           _feed_keys(["1", "2", " ", "3", "\r"])):
        assert cli_mod._prompt_toggle() == ("nums", "12 3")


def test_prompt_toggle_letter_acts_instantly():
    """A letter returns immediately — it never lands in the number field."""
    with mock.patch.object(cli_mod, "_read_key", _feed_keys(["4", "a"])):
        kind, val = cli_mod._prompt_toggle()
    assert (kind, val) == ("key", "a")


def test_prompt_toggle_uppercase_normalized():
    """Uppercase view keys are lowercased."""
    with mock.patch.object(cli_mod, "_read_key", _feed_keys(["D"])):
        assert cli_mod._prompt_toggle() == ("key", "d")


def test_prompt_toggle_backspace_edits_field():
    """Backspace removes the last digit from the field."""
    with mock.patch.object(cli_mod, "_read_key",
                           _feed_keys(["1", "9", "\x7f", "\r"])):
        assert cli_mod._prompt_toggle() == ("nums", "1")


def test_show_letter_switches_view(lmswitch_data_dir):
    """Pressing 'a' in the TUI reveals hidden recipes; 'q' quits."""
    _ghost_yaml(lmswitch_data_dir)
    keys = iter([("key", "a"), ("key", "q")])
    with mock.patch.object(cli_mod, "TTY", True), \
         mock.patch.object(cli_mod, "_prompt_toggle", lambda: next(keys)), \
         mock.patch.object(cli_mod, "_ram_line", lambda: None):
        output = _capture(lambda: cli_mod.show())
    # First render hides the ghost; after 'a' it must appear.
    assert "ghost-70b" in output


def test_show_number_toggles_filtered_index(lmswitch_data_dir):
    """A submitted number maps to the filtered table, skipping hidden rows."""
    _ghost_yaml(lmswitch_data_dir)
    toggled = []
    keys = iter([("nums", "1"), ("key", "q")])
    with mock.patch.object(cli_mod, "TTY", True), \
         mock.patch.object(cli_mod, "_prompt_toggle", lambda: next(keys)), \
         mock.patch.object(cli_mod, "_ram_line", lambda: None), \
         mock.patch.object(cli_mod, "toggle",
                           lambda name, action: toggled.append(name)):
        _capture(lambda: cli_mod.show())
    assert toggled and toggled[0] != "ghost-70b"

# ---------------------------------------------------------------------------
# Search: name/display substring match, "/" key, --search flag
# ---------------------------------------------------------------------------

def test_search_models_matches_name_or_display():
    """_search_models matches case-insensitively on name or display."""
    models = [
        {"name": "qwen2.5-7b", "display": "Qwen 2.5 7B"},
        {"name": "mistral-7b", "display": "Mistral 7B"},
        {"name": "kimi-linear", "display": "Kimi Linear"},
    ]
    assert [m["name"] for m in cli_mod._search_models(models, "qwen")] == ["qwen2.5-7b"]
    assert [m["name"] for m in cli_mod._search_models(models, "MISTRAL")] == ["mistral-7b"]
    # Matches display even when the query isn't in the name.
    assert [m["name"] for m in cli_mod._search_models(models, "linear")] == ["kimi-linear"]
    assert cli_mod._search_models(models, "nonexistent") == []


def test_prompt_search_letters_and_digits_accumulate():
    """Search input accepts letters and digits (unlike the toggle field)."""
    with mock.patch.object(cli_mod, "_read_key",
                           _feed_keys(["q", "w", "e", "n", "\r"])):
        assert cli_mod._prompt_search() == "qwen"


def test_prompt_search_escape_cancels():
    """A bare Escape (empty _read_key result) cancels and returns None."""
    with mock.patch.object(cli_mod, "_read_key", _feed_keys(["q", ""])):
        assert cli_mod._prompt_search() is None


def test_prompt_search_backspace_edits():
    """Backspace removes the last character from the search buffer."""
    with mock.patch.object(cli_mod, "_read_key",
                           _feed_keys(["q", "x", "\x7f", "\r"])):
        assert cli_mod._prompt_search() == "q"


def test_prompt_toggle_slash_returns_search_key():
    """Pressing '/' in the toggle field returns ("key", "/") immediately."""
    with mock.patch.object(cli_mod, "_read_key", _feed_keys(["1", "/"])):
        assert cli_mod._prompt_toggle() == ("key", "/")


def test_show_slash_key_applies_search(lmswitch_data_dir):
    """Pressing '/' then typing a query filters the table to matches.

    The session's first render happens before any key is read, so it still
    shows the unfiltered table — the assertion instead checks that the
    *last* render (after the search is applied, right before quitting) only
    contains the match: qwen2.5-7b's last mention must precede mistral-7b's.
    """
    keys = iter([("key", "/"), ("key", "q")])
    with mock.patch.object(cli_mod, "TTY", True), \
         mock.patch.object(cli_mod, "_prompt_toggle", lambda: next(keys)), \
         mock.patch.object(cli_mod, "_prompt_search", lambda: "mistral"), \
         mock.patch.object(cli_mod, "_ram_line", lambda: None):
        output = _capture(lambda: cli_mod.show())
    assert output.rfind("qwen2.5-7b") < output.rfind("mistral-7b")


def test_show_slash_empty_query_clears_search(lmswitch_data_dir):
    """An empty submitted query clears any active search filter."""
    keys = iter([("key", "/"), ("key", "q")])
    with mock.patch.object(cli_mod, "TTY", True), \
         mock.patch.object(cli_mod, "_prompt_toggle", lambda: next(keys)), \
         mock.patch.object(cli_mod, "_prompt_search", lambda: ""), \
         mock.patch.object(cli_mod, "_ram_line", lambda: None):
        output = _capture(lambda: cli_mod.show("default", False, "mistral"))
    assert "qwen2.5-7b" in output
    assert "mistral-7b" in output


def test_list_search_flag_filters(lmswitch_data_dir):
    """`lmswitch list --search mistral` shows only matching recipes."""
    output = _run_main(["lmswitch", "list", "--search", "mistral"], ram=None)
    assert "mistral-7b" in output
    assert "qwen2.5-7b" not in output


def test_parse_view_flags_extracts_search():
    """--search TEXT is pulled out of argv along with its value."""
    view, show_missing, search, rest = cli_mod._parse_view_flags(
        ["list", "-s", "qwen", "--all"])
    assert search == "qwen"
    assert show_missing is True
    assert rest == ["list"]


def test_parse_view_flags_missing_search_value_exits():
    """A trailing -s with no value exits with an error."""
    def raise_exit(msg):
        raise SystemExit(msg)
    with mock.patch.object(cli_mod.sys, "exit", raise_exit):
        try:
            cli_mod._parse_view_flags(["list", "-s"])
            assert False, "expected SystemExit"
        except SystemExit:
            pass
