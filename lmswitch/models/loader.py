"""Model loading from YAML configs."""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import os

from lmswitch.system.io import (
    CONF_DIR,
    _load_yaml,
    _family,
    _model_size_and_present,
    _hf_snapshot_size_and_present,
    _dir_size_and_present,
)

# TYPE column reflects the backend, not topology — vllm-dual/vllm-dual-ray
# are both vLLM running across two nodes (mp vs Ray executor), so both show
# "vllm" here. The HOST column's "dual" label (see below) is what tells the
# two-node story.
_TYPE_BY_RUNTIME = {"vllm": "vllm", "vllm-dual": "vllm", "vllm-dual-ray": "vllm"}
_DUAL_RUNTIMES = ("vllm-dual", "vllm-dual-ray")


def load_models() -> list[dict]:
    models: list[dict] = []
    if not CONF_DIR.is_dir():
        return models
    local_host = socket.gethostname()
    for path in sorted(CONF_DIR.glob("*.yaml")):
        name = path.stem
        try:
            env = _load_yaml(path)
        except Exception as e:
            print(f"Warning: skipping {path}: {e}", file=sys.stderr)
            continue
        runtime = env.get("runtime", "llama")
        rel = env.get("model", "")
        if runtime in _DUAL_RUNTIMES:
            # Dual models point at a plain weights directory (model_path,
            # possibly an NFS mount of the peer's ~/models) or, failing that,
            # an HF repo id inside the shared cluster cache.
            model_path = env.get("model_path")
            if model_path:
                size, present = _dir_size_and_present(
                    Path(os.path.expanduser(os.path.expandvars(str(model_path)))))
            else:
                size, present = _hf_snapshot_size_and_present(
                    rel, env.get("hf_cache", "~/hf-cluster"))
        else:
            size, present = _model_size_and_present(rel, runtime)
        try:
            port = int(env.get("port", 0))
        except (ValueError, TypeError):
            port = 0
        fam_order, fam_label = _family(name)
        models.append({
            "name": name,
            "display": env.get("display_name", name),
            "runtime": runtime,
            "type": _TYPE_BY_RUNTIME.get(runtime, "gguf"),
            "port": port,
            "ctx": env.get("ctx", ""),
            "size": size,
            "present": present,
            "restart": env.get("restart"),
            "family": fam_label,
            "fam_order": fam_order,
            "host": "dual" if runtime in _DUAL_RUNTIMES else local_host,
        })
    models.sort(key=lambda m: (m["fam_order"], m["name"]))
    return models
