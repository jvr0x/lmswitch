"""Cluster model gathering: merges peer nodes' model tables over SSH."""

from __future__ import annotations

import json
import os
import subprocess

from lmswitch.system.io import _cluster_hosts, CONF_DIR, RUN_DIR, _load_yaml
from lmswitch.system.checks import _listening_ports


def _is_running(name: str) -> bool:
    """Check if a model is actually running on this machine."""
    pid_file = RUN_DIR / name
    if pid_file.exists():
        content = pid_file.read_text().strip()
        if len(content) >= 12 and content.isalnum():
            return False  # vLLM container check skipped here; YAML port check below
        try:
            pid = int(content)
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, ValueError, OSError):
            pass
    yaml_path = CONF_DIR / f"{name}.yaml"
    if yaml_path.exists():
        try:
            yaml_cfg = _load_yaml(yaml_path)
            port = int(yaml_cfg.get("port", 0))
        except (ValueError, TypeError):
            port = 0
        if port and port in _listening_ports():
            return True
    return False


def gather_cluster_models(local_names: set[str]) -> list[dict]:
    """Merges model tables from the other cluster nodes over SSH.

    Each peer runs ``lmswitch list --json`` and reports its own YAMLs with
    running state and its own ``serve_host`` (the mDNS name it uses for its
    own config sync, e.g. ``gigabyte.local``) already resolved on that node.
    Entries whose name is in ``local_names`` AND are actually running on this
    machine are skipped (dual models have a YAML only on the head node — the
    local row wins there, and the head's export is what peers see). Models
    whose YAML exists locally but aren't running here are still included from
    the cluster so config sync picks up remote endpoints. Unreachable peers
    are skipped silently: the cluster view is best-effort and must never break
    the local table or a local sync.

    A model's own ``host`` field (set by the loader — a hostname, or
    ``"dual"`` for a two-node model) is left untouched; only ``remote_host``
    (the SSH alias used to reach it) and ``serve_host`` (where its API
    actually listens) are added.
    """
    remote: list[dict] = []
    for host in _cluster_hosts():
        try:
            # Full path: non-interactive ssh doesn't source the profile, so
            # ~/.local/bin (the uv-tool install target) isn't on PATH.
            out = subprocess.check_output(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=3",
                 host, "$HOME/.local/bin/lmswitch", "list", "--json"],
                text=True, stderr=subprocess.DEVNULL, timeout=10,
            )
            payload = json.loads(out)
        except Exception:
            continue
        serve_host = payload.get("serve_host") or payload.get("host", host)
        for m in payload.get("models", []):
            # Only skip if the model is both known locally AND actually
            # running here — prevents filtering out remote models whose
            # YAML happens to exist locally (e.g. shared model configs).
            if m.get("name") in local_names and _is_running(m.get("name")):
                continue
            m["remote_host"] = host
            m["serve_host"] = serve_host
            remote.append(m)
    return remote
