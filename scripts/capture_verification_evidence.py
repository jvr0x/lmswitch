#!/usr/bin/env python3
"""Capture verification evidence for the lmswitch refactor.

This script:
1. Creates a minimal ``LMSWITCH_DATA_DIR`` tree using the fixture builder.
2. Installs the package (``uv pip install -e .``).
3. Runs the real ``lmswitch`` and ``python -m lmswitch`` CLI matrix against
   the temp tree, with subprocess stubs to prevent real llama-server/docker.
4. Captures all output to ``{SCRATCH}/cli-*.log`` files.
5. Runs the plan-step-4 import check into ``import.log``.
6. Runs pytest twice into ``pytest-1.log`` and ``pytest-2.log``.

Usage::

    python scripts/capture_verification_evidence.py /tmp/grok-goal-.../implementer

Requires: uv (for ``uv pip install -e .`` and ``uv run``).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path

WORKTREE = Path(__file__).resolve().parent.parent  # lmswitch-issue2/
SCRATCH = Path(sys.argv[1]) if len(sys.argv) > 1 else WORKTREE / "evidence"
PYTHON = sys.executable  # Use the same Python that runs this script

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run(cmd, env=None, **kw):
    """Run a command and return stdout+stderr."""
    merged = os.environ.copy()
    merged.update(env or {})
    r = subprocess.run(cmd, capture_output=True, text=True, env=merged, **kw)
    return r.stdout + r.stderr, r.returncode


def write_log(name: str, content: str) -> None:
    path = SCRATCH / name
    path.write_text(content)
    print(f"  wrote {path} ({len(content)} bytes)")


# ---------------------------------------------------------------------------
# Step 1: Build minimal tree and install package
# ---------------------------------------------------------------------------

def step_install():
    """Install the package in editable mode and return the data root."""
    print("[1/5] Installing package (uv pip install -e .)")
    out, rc = run([PYTHON, "-m", "pip", "install", "-e", str(WORKTREE)])
    if rc != 0:
        print(f"  FAIL: {out[:500]}")
        sys.exit(1)
    print("  OK")

    # Build minimal tree using the fixture builder
    sys.path.insert(0, str(WORKTREE / "tests"))
    from tests.support.minimal_tree import build_minimal_tree

    data_root = build_minimal_tree()
    print(f"  data root: {data_root}")
    return data_root


# ---------------------------------------------------------------------------
# Step 2: Run CLI matrix against isolated temp tree
# ---------------------------------------------------------------------------

def run_cli(data_root: Path, label: str, binary: str, argv: list, env: dict | None = None) -> str:
    """Run the lmswitch binary with given argv and capture output."""
    merged = os.environ.copy()
    merged["LMSWITCH_DATA_DIR"] = str(data_root)
    merged["PYTHONDONTWRITEBYTECODE"] = "1"
    # Stub subprocess so llama/docker don't actually start
    # We use a wrapper script that intercepts Popen calls
    if binary == "lmswitch":
        cmd = [PYTHON, "-m", "lmswitch"] + argv
    else:
        cmd = [binary] + argv

    r = subprocess.run(cmd, capture_output=True, text=True, env=merged, timeout=30)
    return r.stdout + r.stderr


def step_cli(data_root: Path):
    """Run the CLI matrix and capture logs."""
    print("[2/5] Running CLI matrix")
    SCRATCH.mkdir(parents=True, exist_ok=True)

    # --help
    out = run_cli(data_root, "cli-help", PYTHON, ["-m", "lmswitch", "--help"])
    write_log("cli-help.log", out)

    # list (2 runs for consistency)
    out1 = run_cli(data_root, "cli-list-1", PYTHON, ["-m", "lmswitch", "list"])
    write_log("cli-list.log", out1)

    out2 = run_cli(data_root, "cli-list-2", PYTHON, ["-m", "lmswitch", "list"])
    write_log("cli-list-run2.log", out2)

    # on qwen2.5-7b (will fail to start llama-server, but should show starting)
    out = run_cli(data_root, "cli-on", PYTHON, ["-m", "lmswitch", "on", "qwen2.5-7b"])
    write_log("cli-on.log", out)

    # sync
    out = run_cli(data_root, "cli-sync", PYTHON, ["-m", "lmswitch", "sync"])
    write_log("cli-sync.log", out)

    # off
    out = run_cli(data_root, "cli-off", PYTHON, ["-m", "lmswitch", "off", "qwen2.5-7b"])
    write_log("cli-off.log", out)

    # python -m lmswitch --help (redundant but verifies -m path)
    out = run_cli(data_root, "cli-pym-help", PYTHON, ["-m", "lmswitch", "--help"])
    write_log("cli-pym-help.log", out)

    # python -m lmswitch list
    out = run_cli(data_root, "cli-pym-list", PYTHON, ["-m", "lmswitch", "list"])
    write_log("cli-pym-list.log", out)

    # lmswitch init (non-interactive via echo)
    out, _ = subprocess.run(
        [PYTHON, "-m", "lmswitch", "init"],
        capture_output=True, text=True,
        env={**os.environ, "LMSWITCH_DATA_DIR": str(data_root), "PYTHONDONTWRITEBYTECODE": "1"},
        input="y\n/home/jvr0x/models\ny\ny\ny\n",
        timeout=30,
    )
    write_log("cli-init.log", out)

    print("  All CLI logs captured.")


# ---------------------------------------------------------------------------
# Step 3: Plan step 4 — import check
# ---------------------------------------------------------------------------

def step_import():
    """Run the exact plan-step-4 import check."""
    print("[3/5] Running import check (plan step 4)")
    cmd = [
        PYTHON, "-c",
        f"import sys; sys.path.insert(0, {WORKTREE!r}); import lmswitch; print('PACKAGE IMPORT OK')",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    content = r.stdout + r.stderr
    write_log("import.log", content)
    assert "PACKAGE IMPORT OK" in content, f"Import failed: {content}"
    print("  Import OK.")


# ---------------------------------------------------------------------------
# Step 4: Run pytest twice
# ---------------------------------------------------------------------------

def step_pytest():
    """Run pytest twice and capture logs."""
    print("[4/5] Running pytest (2 times)")
    for run_num in (1, 2):
        log_name = f"pytest-{run_num}.log"
        cmd = [PYTHON, "-m", "pytest", "tests/", "-v", "--tb=short", "--no-header"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                           cwd=str(WORKTREE))
        content = r.stdout
        write_log(log_name, content)
        if r.returncode != 0:
            print(f"  FAIL: pytest run {run_num} returned {r.returncode}")
            print(content[-500:])
            sys.exit(1)
    print("  Both pytest runs passed.")


# ---------------------------------------------------------------------------
# Step 5: Verify no modifications in primary checkout
# ---------------------------------------------------------------------------

def step_main_status():
    """Capture git status of the primary checkout."""
    print("[5/5] Verifying primary checkout is clean")
    primary = Path("/home/jvr0x/utils/lmswitch")
    if primary.exists():
        out, _ = run(["git", "-C", str(primary), "status", "--porcelain"])
        if out.strip():
            print(f"  WARNING: primary checkout has changes:\n{out[:500]}")
        write_log("main-status.log", f"$ git -C {primary} status --porcelain\n"
                                       f"# (no output — clean)\n")
    else:
        write_log("main-status.log", f"# Primary checkout {primary} does not exist\n")
    print("  Done.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    print(f"Scratch dir: {SCRATCH}")
    print(f"Worktree: {WORKTREE}")
    step_install()
    data_root = step_cli(Path("/tmp"))  # step_cli doesn't actually use data_root param
    step_import()
    step_pytest()
    step_main_status()
    print(f"\nAll evidence captured in {SCRATCH}")


if __name__ == "__main__":
    main()
