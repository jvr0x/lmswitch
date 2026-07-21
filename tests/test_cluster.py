"""Tests for models.cluster — serve_host resolution and cluster gathering.

Covers the grok/gigabyte connectivity bug: a peer's self-reported mDNS
``serve_host`` (e.g. ``spark.local``) is intermittently flaky for plain HTTP
clients (grok failed outright reaching it; raw curl reproduced ~30% stalls
to the full 5s timeout from gigabyte, even with the underlying network path
otherwise healthy). The SSH alias used to reach the peer already resolves to
a proven-reliable address (management access depends on it) — ``ssh -G``
exposes that resolved ``Hostname``, which should be preferred over the
peer's own mDNS self-report.
"""

import subprocess
from unittest import mock

import lmswitch.models.cluster as cluster_mod


# ---------------------------------------------------------------------------
# _resolved_host
# ---------------------------------------------------------------------------

def test_resolved_host_expected_use():
    """`ssh -G <alias>` succeeds and reports a concrete Hostname -> used."""
    ssh_g_output = (
        "user jvr0x\n"
        "hostname 10.100.224.2\n"
        "port 22\n"
    )
    with mock.patch.object(subprocess, "check_output", return_value=ssh_g_output):
        assert cluster_mod._resolved_host("Spark") == "10.100.224.2"


def test_resolved_host_edge_no_hostname_line():
    """Well-formed ssh -G output missing a hostname line -> None, not a crash."""
    with mock.patch.object(subprocess, "check_output", return_value="user jvr0x\nport 22\n"):
        assert cluster_mod._resolved_host("Spark") is None


def test_resolved_host_failure_ssh_errors():
    """ssh -G failing (bad alias, timeout, ssh not on PATH) -> None, no raise."""
    with mock.patch.object(subprocess, "check_output",
                           side_effect=subprocess.TimeoutExpired("ssh", 3)):
        assert cluster_mod._resolved_host("Spark") is None


# ---------------------------------------------------------------------------
# gather_cluster_models — serve_host precedence
# ---------------------------------------------------------------------------

def _payload(serve_host="spark.local"):
    return {
        "serve_host": serve_host,
        "host": "spark.local",
        "models": [{"name": "deepseek-v4-flash-dspark-dual", "port": 8888}],
    }


def test_gather_prefers_resolved_ssh_host_over_peer_self_report(monkeypatch):
    """THE bug: the peer reports its own flaky spark.local, but the SSH
    alias used to reach it resolves to a known-good CX7 address — that
    resolved address must win."""
    monkeypatch.setattr(cluster_mod, "_cluster_hosts", lambda: ["Spark"])
    with mock.patch.object(cluster_mod.subprocess, "check_output") as co:
        def side_effect(cmd, *a, **k):
            if cmd[0] == "ssh" and "-G" in cmd:
                return "hostname 10.100.224.2\n"
            return __import__("json").dumps(_payload())
        co.side_effect = side_effect
        result = cluster_mod.gather_cluster_models(local_names=set())
    assert len(result) == 1
    assert result[0]["serve_host"] == "10.100.224.2"


def test_gather_falls_back_to_peer_self_report_when_ssh_g_fails(monkeypatch):
    """Edge: ssh -G resolution unavailable -> falls back to the peer's own
    reported serve_host (previous behavior), not a hard failure."""
    monkeypatch.setattr(cluster_mod, "_cluster_hosts", lambda: ["Spark"])
    with mock.patch.object(cluster_mod.subprocess, "check_output") as co:
        def side_effect(cmd, *a, **k):
            if cmd[0] == "ssh" and "-G" in cmd:
                raise subprocess.CalledProcessError(1, cmd)
            return __import__("json").dumps(_payload(serve_host="spark.local"))
        co.side_effect = side_effect
        result = cluster_mod.gather_cluster_models(local_names=set())
    assert result[0]["serve_host"] == "spark.local"


def test_gather_falls_back_to_alias_when_nothing_else_available(monkeypatch):
    """Failure case: neither ssh -G nor the peer's payload provide a usable
    serve_host -> falls back to the bare alias, matching prior behavior."""
    monkeypatch.setattr(cluster_mod, "_cluster_hosts", lambda: ["Spark"])
    with mock.patch.object(cluster_mod.subprocess, "check_output") as co:
        def side_effect(cmd, *a, **k):
            if cmd[0] == "ssh" and "-G" in cmd:
                raise subprocess.CalledProcessError(1, cmd)
            payload = _payload()
            del payload["serve_host"]
            del payload["host"]
            return __import__("json").dumps(payload)
        co.side_effect = side_effect
        result = cluster_mod.gather_cluster_models(local_names=set())
    assert result[0]["serve_host"] == "Spark"
