"""Model loading from YAML configs."""

from __future__ import annotations

import sys
from pathlib import Path

from lmswitch.system.io import (
    CONF_DIR,
    _load_yaml,
    _family,
    _model_size_and_present,
)


def load_models() -> list[dict]:
    models: list[dict] = []
    if not CONF_DIR.is_dir():
        return models
    for path in sorted(CONF_DIR.glob("*.yaml")):
        name = path.stem
        try:
            env = _load_yaml(path)
        except Exception as e:
            print(f"Warning: skipping {path}: {e}", file=sys.stderr)
            continue
        runtime = env.get("runtime", "llama")
        rel = env.get("model", "")
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
            "type": "vllm" if runtime == "vllm" else "gguf",
            "port": port,
            "ctx": env.get("ctx", ""),
            "size": size,
            "present": present,
            "restart": env.get("restart"),
            "family": fam_label,
            "fam_order": fam_order,
        })
    models.sort(key=lambda m: (m["fam_order"], m["name"]))
    return models
