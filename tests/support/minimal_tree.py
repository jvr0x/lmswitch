"""Build a minimal, reproducible lmswitch data directory for tests.

Creates a temporary directory tree with:
  - Two model YAML configs (one llama, one vllm) that produce ``present=True``
    because the underlying model files are valid GGUF/safetensors placeholders.
  - A ``.lmswitch`` config file with ``MODELS_DIR`` pointing to a ``models/``
    subdir inside the temp directory.
  - A ``running/`` subdirectory (so log-file creation doesn't fail).
  - Placeholder model files under ``models/`` so ``_model_size_and_present``
    reports ``present=True`` without needing real weights.
  - A no-op ``llama-server`` wrapper for evidence capture so ``lmswitch on``
    succeeds without launching a real backend.

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

    # llama model placeholder (valid GGUF header so llama-server starts cleanly)
    llama_model_dir = models_dir / "qwen2.5-7b"
    llama_model_dir.mkdir(parents=True, exist_ok=True)
    _write_valid_gguf(llama_model_dir / "qwen2.5-7b-instruct-q4_k_m.gguf")

    # vllm model placeholder (dir-based for safetensors)
    vllm_model_dir = models_dir / "mistral-7b"
    vllm_model_dir.mkdir(parents=True, exist_ok=True)
    (vllm_model_dir / "config.json").touch()  # minimal safetensors marker

    # ---- no-op llama-server wrapper (for evidence capture / test_mode) ----
    llama_bin = data_root / "bin"
    llama_bin.mkdir(exist_ok=True)
    _write_llama_wrapper(llama_bin / "llama-server")

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
    # llama model (uses no-op wrapper via llama_bin)
    llama_yaml = conf_dir / "qwen2.5-7b.yaml"
    llama_yaml.write_text(
        f"runtime: llama\n"
        f"model: qwen2.5-7b/qwen2.5-7b-instruct-q4_k_m.gguf\n"
        f"port: 8081\n"
        f"ctx: 65536\n"
        f'display_name: "Qwen 2.5 7B"\n'
        f"type: gguf\n"
        f'llama_bin: "{llama_bin / "llama-server"}"\n'
    )

    # vllm model
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


def _write_valid_gguf(path: Path) -> None:
    """Write a minimal GGUF header (magic + version + 0 tensors + 0 KV pairs).

    This is enough for llama-server to pass the header check and start
    serving (it will fail on actual model inference, but the server
    process will be alive and the port will be bound).
    """
    import struct
    # GGUF magic + version 1 + 0 tensors + 0 KV pairs
    header = b"GGUF" + struct.pack("<I", 1) + struct.pack("<Q", 0) + struct.pack("<Q", 0)
    # Pad to ~64KB so the file is "big enough" to not be rejected
    padding = b"\x00" * (65536 - len(header))
    path.write_bytes(header + padding)


def _write_llama_wrapper(path: Path) -> None:
    """Write a no-op llama-server wrapper that binds a port and responds.

    The wrapper:
    - Parses ``--port N`` from args (default 8081)
    - Starts an HTTP server on ``0.0.0.0:N``
    - Responds to ``/v1/models`` with a valid OpenAI-style response
    - Stays alive until killed

    This allows ``_wait_ready`` to detect the port is ready and return
    ``"ready"`` — giving a clean ``lmswitch on`` success path for evidence.

    Usage: llama-server [any args]
    """
    wrapper = path.parent / "llama-server"
    wrapper.write_text(
        '#!/usr/bin/env python3\n'
        '# No-op llama-server for evidence capture.\n'
        '# Parses --port from args, binds HTTP server on /v1/models.\n'
        'import os, sys, signal, json, http.server, socketserver\n'
        '\n'
        '# Parse --port\n'
        'port = 8081\n'
        'for i, a in enumerate(sys.argv):\n'
        '    if a == "--port" and i + 1 < len(sys.argv):\n'
        '        port = int(sys.argv[i + 1])\n'
        '\n'
        'print(f"llama-server (no-op) running on port {port}", flush=True)\n'
        '\n'
        'class Handler(http.server.BaseHTTPRequestHandler):\n'
        '    def do_GET(self):\n'
        '        if self.path == "/v1/models":\n'
        '            body = json.dumps({"data": [{"id": "qwen2.5-7b", "object": "model", "owned_by": "test"}]}).encode()\n'
        '            self.send_response(200)\n'
        '            self.send_header("Content-Type", "application/json")\n'
        '            self.send_header("Content-Length", str(len(body)))\n'
        '            self.end_headers()\n'
        '            self.wfile.write(body)\n'
        '        else:\n'
        '            self.send_response(404)\n'
        '            self.end_headers()\n'
        '    def log_message(self, format, *args):\n'
        '        pass  # silence logs\n'
        '\n'
        'with socketserver.TCPServer(("0.0.0.0", port), Handler) as httpd:\n'
        '    signal.signal(signal.SIGTERM, lambda s, f: os._exit(0))\n'
        '    httpd.serve_forever()\n'
    )
    wrapper.chmod(0o755)
