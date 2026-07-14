"""Shared pytest configuration for lmswitch tests.

Provides an auto-use ``lmswitch_data_dir`` fixture that creates a minimal,
isolated data directory and patches CONF_DIR/RUN_DIR/CONFIG_FILE in every
lmswitch module that imports them, so that all tests run against a temp
tree with no side effects on the host machine.

Usage::

    def test_something(lmswitch_data):
        # lmswitch_data is the Path to the temp CONF_DIR
        # CONF_DIR, RUN_DIR, CONFIG_FILE already resolve there
        ...
"""

import pytest
from pathlib import Path

from tests.support.minimal_tree import build_minimal_tree


def _patch_conf_dir(data_root: Path) -> None:
    """Reassign CONF_DIR/RUN_DIR/CONFIG_FILE in all lmswitch modules."""
    import lmswitch.system.io as io_mod
    import lmswitch.cli as cli_mod
    import lmswitch.models.loader as loader_mod
    import lmswitch.sync as sync_mod
    import lmswitch.runtimes as rt_mod
    import lmswitch.runtimes.llama as llama_mod
    import lmswitch.runtimes.vllm as vllm_mod
    import lmswitch.runtimes.vllm_dual as vllm_dual_mod

    for mod in (io_mod, cli_mod, loader_mod, sync_mod, rt_mod, llama_mod, vllm_mod):
        mod.CONF_DIR = data_root
        mod.RUN_DIR = data_root / "running"
        mod.CONFIG_FILE = data_root / ".lmswitch"
    # vllm_dual only imports RUN_DIR.
    vllm_dual_mod.RUN_DIR = data_root / "running"


@pytest.fixture(autouse=True)
def lmswitch_data_dir(monkeypatch):
    """Auto-use fixture: creates a minimal LMSWITCH_DATA_DIR tree and patches
    CONF_DIR/RUN_DIR/CONFIG_FILE in every lmswitch module.

    The fixture:
    1. Creates a temp tree via ``build_minimal_tree()``.
    2. Sets ``LMSWITCH_DATA_DIR`` env var (for any subprocess or lazy import).
    3. Patches CONF_DIR in all lmswitch modules that import it.
    4. Ensures ``running/`` dir exists for log-file creation.

    This guarantees every test runs against the same isolated tree.
    """
    data_root = build_minimal_tree()

    # Set env var so new subprocess/lazy imports see it.
    monkeypatch.setenv("LMSWITCH_DATA_DIR", str(data_root))

    # Patch CONF_DIR in all modules that import it.
    _patch_conf_dir(data_root)

    # Ensure running/ dir exists (for log-file creation).
    (data_root / "running").mkdir(parents=True, exist_ok=True)

    yield data_root

    # Cleanup: remove the temp tree.
    import shutil
    shutil.rmtree(data_root, ignore_errors=True)
