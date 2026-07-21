"""Cluster model gathering: merges peer nodes' model tables over SSH."""

from __future__ import annotations

import json
import subprocess

from lmswitch.system.io import _cluster_hosts
from lmswitch.system.checks import _is_running as _checks_is_running


def _resolved_host(alias: str) -> str | None:
    """Resolves an SSH alias to its configured ``Hostname`` via ``ssh -G``.

    A peer's own ``lmswitch list --json`` reports its self-computed
    ``serve_host`` — by default ``<hostname>.local``, resolved via mDNS. That
    hostname has proven intermittently flaky for plain HTTP clients (grok
    failed outright; raw curl reproduced ~30% stalls to the full 5s timeout,
    even though the underlying network path is fine). The SSH alias used to
    reach the peer is already a known-good address — management access
    depends on it working — so prefer that resolved address for HTTP
    base_urls too, rather than the peer's own mDNS self-report.
    """
    try:
        out = subprocess.check_output(
            ["ssh", "-G", alias], text=True, stderr=subprocess.DEVNULL, timeout=3,
        )
    except Exception:
        return None
    for line in out.splitlines():
        if line.startswith("hostname "):
            return line.split(None, 1)[1].strip() or None
    return None


def _is_running(name: str, runtime: str = "llama") -> bool:
    """Check if a model is actually running on this machine.

    Delegates to ``system.checks._is_running`` — this module used to carry
    its own diverged copy that unconditionally skipped the Docker container
    check for vLLM-style pidfiles and fell straight to a port-liveness
    check, which is wrong for any dual runtime (they conventionally share
    port 8888, so it would mark every other dual recipe as "running" too
    whenever any single one of them actually was).
    """
    return _checks_is_running(name, runtime)


def gather_cluster_models(local_names: set[str]) -> list[dict]:
    """Merges model tables from the other cluster nodes over SSH.

    Each peer runs ``lmswitch list --json`` and reports its own YAMLs with
    running state and its own ``serve_host`` (the mDNS name it uses for its
    own config sync, e.g. ``gigabyte.local``) already resolved on that node.
    That self-reported mDNS hostname is used only as a fallback — the SSH
    alias's own resolved ``Hostname`` (see ``_resolved_host``) is preferred
    since it's already a proven-reliable address, avoiding mDNS flakiness
    that has caused real, intermittent client-facing connection failures.
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
        serve_host = _resolved_host(host) or payload.get("serve_host") or payload.get("host", host)
        for m in payload.get("models", []):
            # Only skip if the model is both known locally AND actually
            # running here — prevents filtering out remote models whose
            # YAML happens to exist locally (e.g. shared model configs).
            if m.get("name") in local_names and _is_running(m.get("name"), m.get("runtime", "llama")):
                continue
            m["remote_host"] = host
            m["serve_host"] = serve_host
            remote.append(m)
    return remote
