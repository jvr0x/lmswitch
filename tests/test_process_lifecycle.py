"""Process lifecycle test: real on → off cycle with live _is_running checks.

Builds a minimal temp tree, runs the real ``lmswitch on`` then ``lmswitch off``,
and asserts:
  - After ``on``: ``_is_running(..., "llama")`` returns True
  - After ``off``: ``_is_running(..., "llama")`` returns False

This test exercises the SHIPPED ``on→off`` path end-to-end and will fail
if ``stop_llama_by_pid`` doesn't actually kill the process (port fallback included).
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import lmswitch.system.checks as checks_mod
import lmswitch.system.io as io_mod
from lmswitch.system.io import RUN_DIR


def test_real_on_then_off_clears_is_running():
    """Real lmswitch on → off: _is_running must go True → False.

    Uses the no-op llama-server wrapper from the fixture builder so the
    process actually starts, binds a port, and reports Ready.
    """
    sys.path.insert(0, "tests")
    from tests.support.minimal_tree import build_minimal_tree

    data_root = build_minimal_tree()

    env = {
        **os.environ,
        "LMSWITCH_DATA_DIR": str(data_root),
        "PYTHONDONTWRITEBYTECODE": "1",
    }

    # Patch RUN_DIR in BOTH modules so _is_running reads/writes the temp tree.
    # The checks module imports RUN_DIR at load time, so patching io_mod alone
    # isn't enough — we must patch checks_mod.RUN_DIR too.
    temp_run_dir = data_root / "running"
    old_io_run_dir = io_mod.RUN_DIR
    old_checks_run_dir = checks_mod.RUN_DIR
    io_mod.RUN_DIR = temp_run_dir
    checks_mod.RUN_DIR = temp_run_dir

    try:
        # --- ON ---
        r = subprocess.run(
            [sys.executable, "-m", "lmswitch", "on", "qwen2.5-7b"],
            capture_output=True, text=True, timeout=30, env=env,
        )
        assert r.returncode == 0, f"on failed: {r.stderr[:500]}\n{r.stdout[:500]}"
        assert "Ready" in r.stdout or "Stopping" in r.stdout, f"Unexpected: {r.stdout[:200]}"

        # Give the process a moment to bind
        time.sleep(1)

        # Verify _is_running is True
        running_before = checks_mod._is_running("qwen2.5-7b", "llama")
        assert running_before, f"_is_running should be True after on, got {running_before}"

        # --- OFF ---
        r = subprocess.run(
            [sys.executable, "-m", "lmswitch", "off", "qwen2.5-7b"],
            capture_output=True, text=True, timeout=30, env=env,
        )
        assert r.returncode == 0, f"off failed: {r.stderr[:200]}\n{r.stdout[:200]}"

        # Give process a moment to exit
        time.sleep(1)

        # Verify _is_running is False
        running_after = checks_mod._is_running("qwen2.5-7b", "llama")
        assert not running_after, (
            f"_is_running should be False after off, got {running_after}\n"
            f"off output: {r.stdout}"
        )
    finally:
        io_mod.RUN_DIR = old_io_run_dir
        checks_mod.RUN_DIR = old_checks_run_dir
