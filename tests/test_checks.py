"""Tests for system.checks._is_running — port-liveness fallback correctness.

Covers the three required cases per project convention:
- expected use: a running docker-backed model (valid pidfile) is detected
  via its container; a running llama model is detected via its port.
- edge: a docker-backed model with NO pidfile is checked by container name,
  never by port — this is the regression case. Every vllm-dual/vllm-dual-ray
  recipe conventionally shares port 8888, so a stale/missing pidfile used to
  fall through to "is port 8888 listening", which is true for EVERY dual
  recipe the instant ANY single one of them is actually running. Observed
  live: 13 dual recipes simultaneously showing "running" with only one
  container actually up.
- failure: a genuinely-stopped docker-backed model with no pidfile and no
  container reports not-running even while its port is (coincidentally or
  not) listening.
"""

from pathlib import Path
from unittest import mock

import lmswitch.system.checks as checks_mod


def _yaml(tmp_path: Path, name: str, port: int) -> None:
    (tmp_path / f"{name}.yaml").write_text(f"runtime: vllm-dual\nport: {port}\n")


# ---------------------------------------------------------------------------
# Expected use
# ---------------------------------------------------------------------------

def test_docker_backed_running_via_valid_pidfile(tmp_path, monkeypatch):
    """A container-ID pidfile + a live matching container -> running."""
    monkeypatch.setattr(checks_mod, "RUN_DIR", tmp_path)
    (tmp_path / "model-a").write_text("abcdef123456")
    with mock.patch.object(checks_mod, "_docker_container", return_value="abcdef123456"):
        assert checks_mod._is_running("model-a", "vllm-dual") is True


def test_llama_running_via_port(tmp_path, monkeypatch):
    """A llama model with no pidfile but its own port listening -> running
    (unchanged behavior — llama recipes have unique ports, no shared-port
    ambiguity, so the port fallback is safe there)."""
    monkeypatch.setattr(checks_mod, "RUN_DIR", tmp_path)
    monkeypatch.setattr(checks_mod, "CONF_DIR", tmp_path)
    (tmp_path / "model-b.yaml").write_text("runtime: llama\nport: 8081\n")
    with mock.patch.object(checks_mod, "_listening_ports", return_value={8081}):
        assert checks_mod._is_running("model-b", "llama") is True


# ---------------------------------------------------------------------------
# Edge: the actual regression — shared port 8888 must not cross-contaminate
# ---------------------------------------------------------------------------

def test_dual_model_not_fooled_by_shared_port_from_a_different_recipe(tmp_path, monkeypatch):
    """THE bug: model-a is genuinely running (container up), model-b is a
    completely separate, stopped vllm-dual recipe that also declares
    port: 8888 (the standard dual convention). model-b must NOT be reported
    as running just because port 8888 happens to be listening — the port
    is model-a's, not model-b's, and only container-name lookup can tell
    them apart."""
    monkeypatch.setattr(checks_mod, "RUN_DIR", tmp_path)
    monkeypatch.setattr(checks_mod, "CONF_DIR", tmp_path)
    _yaml(tmp_path, "model-b", 8888)
    # No pidfile for model-b at all (the realistic case: it was never
    # started this boot, or its pidfile was cleaned up on a prior stop).
    with mock.patch.object(checks_mod, "_listening_ports", return_value={8888}), \
         mock.patch.object(checks_mod, "_docker_container", return_value=None):
        assert checks_mod._is_running("model-b", "vllm-dual") is False


def test_dual_ray_model_also_not_fooled_by_shared_port(tmp_path, monkeypatch):
    """Same regression, for the Ray-based dual runtime."""
    monkeypatch.setattr(checks_mod, "RUN_DIR", tmp_path)
    monkeypatch.setattr(checks_mod, "CONF_DIR", tmp_path)
    _yaml(tmp_path, "model-c", 8888)
    with mock.patch.object(checks_mod, "_listening_ports", return_value={8888}), \
         mock.patch.object(checks_mod, "_docker_container", return_value=None):
        assert checks_mod._is_running("model-c", "vllm-dual-ray") is False


def test_dual_model_correctly_running_via_container_name(tmp_path, monkeypatch):
    """The positive case: model-b has no pidfile, but its OWN container
    (vllm-model-b) is genuinely up — must report running, via container
    lookup, not the port."""
    monkeypatch.setattr(checks_mod, "RUN_DIR", tmp_path)
    monkeypatch.setattr(checks_mod, "CONF_DIR", tmp_path)
    _yaml(tmp_path, "model-b", 8888)
    with mock.patch.object(checks_mod, "_listening_ports", return_value={8888}), \
         mock.patch.object(checks_mod, "_docker_container", return_value="deadbeef0001"):
        assert checks_mod._is_running("model-b", "vllm-dual") is True


# ---------------------------------------------------------------------------
# Failure: genuinely stopped, no false positive
# ---------------------------------------------------------------------------

def test_docker_backed_stopped_is_stopped(tmp_path, monkeypatch):
    monkeypatch.setattr(checks_mod, "RUN_DIR", tmp_path)
    monkeypatch.setattr(checks_mod, "CONF_DIR", tmp_path)
    _yaml(tmp_path, "model-d", 8081)
    with mock.patch.object(checks_mod, "_listening_ports", return_value=set()), \
         mock.patch.object(checks_mod, "_docker_container", return_value=None):
        assert checks_mod._is_running("model-d", "vllm-dual") is False


# ---------------------------------------------------------------------------
# models.cluster._is_running must delegate to the fixed implementation
# above, not maintain its own diverged copy (it used to: an older version
# unconditionally skipped the Docker check for any vLLM-style pidfile and
# fell straight to the same shared-port bug, and didn't even accept a
# runtime argument to route around it).
# ---------------------------------------------------------------------------

def test_cluster_is_running_delegates_and_passes_runtime():
    import lmswitch.models.cluster as cluster_mod
    with mock.patch.object(cluster_mod, "_checks_is_running",
                           return_value="sentinel") as delegate:
        result = cluster_mod._is_running("some-dual-model", "vllm-dual-ray")
    delegate.assert_called_once_with("some-dual-model", "vllm-dual-ray")
    assert result == "sentinel"
