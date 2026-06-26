"""Build a minimal, reproducible lmswitch data directory for tests.

Creates a temporary directory tree with:
  - Two model YAML configs (one llama, one vllm) that produce ``present=True``
    because the underlying model files are empty placeholder files.
  - A ``.lmswitch`` config file with ``MODELS_DIR`` pointing to a ``models/``
    subdir inside the temp directory.
  - A ``running/`` subdirectory (so log-file creation doesn't fail).
  - Placeholder model files under ``models/`` so ``_model_size_and_present``
    reports ``present=True`` without needing real weights.

Usage::

    from tests.support.minimal_tree import build_minimal_tree

    tmp = Path("/tmp/my_test")
    data_dir = build_minimal_tree(tmp)
    # Now set LMSWITCH_DATA_DIR=data_dir and import the package.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def _write_yaml(path: Path, **kwargs: str) -> None:
    """Write a simple YAML file (no quotes, no complex types)."""
    lines = [f"{k}: {v}" for k, v in kwargs.items()]
    path.write_text("\n".join(lines) + "\n")


def build_minimal_tree(parent: Path | None = None) -> Path:
    """Create and return a minimal ``LMSWITCH_DATA_DIR`` tree.

    Parameters
    ----------
    parent:
        Directory under which to create the temp tree.  Defaults to
        ``tempfile.mkdtemp()``.

    Returns
    -------
    Path
        The absolute path to the created data root (suitable for
        ``LMSWITCH_DATA_DIR``).
    """
    if parent is None:
        parent = Path(tempfile.mkdtemp())

    # ---- Data root ----
    data_root = parent / "lmswitch_data"
    data_root.mkdir(parents=True, exist_ok=True)

    # ---- models/ subdir with placeholder weights ----
    models_dir = data_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # llama model placeholder (empty .gguf file)
    llama_model_dir = models_dir / "qwen2.5-7b"
    llama_model_dir.mkdir(parents=True, exist_ok=True)
    (llama_model_dir / "qwen2.5-7b-instruct-q4_k_m.gguf").touch()

    # ---- ai-models/ (CONF_DIR) subdir ----
    conf_dir = data_root  # data root IS CONF_DIR

    # .lmswitch config
    config_file = conf_dir / ".lmswitch"
    config_file.write_text(
        f'MODELS_DIR="{str(models_dir)}"\n'
        "SYNC_OPENCODE=false\n"
        "SYNC_HERMES=false\n"
        "SYNC_GROK=false\n"
    )

    # ---- YAML configs ----
    # llama model
    llama_yaml = conf_dir / "qwen2.5-7b.yaml"
    llama_yaml.write_text(
        "runtime: llama\n"
        "model: qwen2.5-7b/qwen2.5-7b-instruct-q4_k_m.gguf\n"
        "port: 8081\n"
        "ctx: 65536\n"
        'display_name: "Qwen 2.5 7B"\n'
        "type: gguf\n"
    )

    # vllm model placeholder (dir-based for safetensors)
    vllm_model_dir = models_dir / "mistral-7b"
    vllm_model_dir.mkdir(parents=True, exist_ok=True)
    (vllm_model_dir / "config.json").touch()  # minimal safetensors marker
    vllm_yaml = conf_dir / "mistral-7b.yaml"
    vllm_yaml.write_text(
        "runtime: vllm\n"
        "model: mistral-7b\n"
        "port: 8082\n"
        "ctx: 32768\n"
        'display_name: "Mistral 7B"\n'
        "type: vllm\n"
        "gpu_memory_utilization: 0.15\n"
    )

    # ---- running/ dir ----
    running_dir = conf_dir / "running"
    running_dir.mkdir(parents=True, exist_ok=True)

    return data_root
