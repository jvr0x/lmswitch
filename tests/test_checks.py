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


# ---------------------------------------------------------------------------
# server_uptime — real process/container start time
#
# Expected use: a live llama PID and a live container both report a plausible
# uptime, and server_uptime routes to the right one per runtime.
# Edge: Docker's nanosecond + `Z` stamp, which datetime.fromisoformat rejects
# verbatim on 3.10, still parses.
# Failure: a missing pid file, a stale PID, a vanished container and an
# unparsable stamp all report 0.0 rather than raising into the stop path.
# ---------------------------------------------------------------------------

import os
from datetime import datetime, timedelta, timezone


def _docker_stamp(seconds_ago: float) -> str:
    """Builds a Docker-shaped RFC-3339 stamp *seconds_ago* in the past."""
    moment = datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)
    return moment.strftime("%Y-%m-%dT%H:%M:%S.%f") + "000Z"


def test_process_uptime_reads_the_real_process_start(tmp_path, monkeypatch):
    """A live PID reports its own age, not the pid file's."""
    monkeypatch.setattr(checks_mod, "RUN_DIR", tmp_path)
    (tmp_path / "mymodel").write_text(str(os.getpid()))

    uptime = checks_mod.server_uptime("mymodel", "llama")

    # The test process is older than 0s and younger than a day.
    assert 0.0 < uptime < 86400.0


def test_container_uptime_from_docker_started_at(tmp_path, monkeypatch):
    """A Docker-backed runtime reads State.StartedAt off the container."""
    monkeypatch.setattr(checks_mod, "_docker_container", lambda *a, **k: "deadbeef1234")
    monkeypatch.setattr(checks_mod.subprocess, "check_output",
                        lambda *a, **k: _docker_stamp(600.0))

    uptime = checks_mod.server_uptime("mymodel", "vllm-dual")

    assert 595.0 < uptime < 610.0


def test_server_uptime_routes_by_runtime(tmp_path, monkeypatch):
    """Docker-backed runtimes never take the pid-file path, and vice versa."""
    monkeypatch.setattr(checks_mod, "RUN_DIR", tmp_path)
    (tmp_path / "mymodel").write_text(str(os.getpid()))
    monkeypatch.setattr(checks_mod, "_docker_container", lambda *a, **k: None)

    # llama has no container, but does have the pid file.
    assert checks_mod.server_uptime("mymodel", "llama") > 0.0
    # sglang is container-backed, and its container is gone.
    assert checks_mod.server_uptime("mymodel", "sglang") == 0.0


def test_seconds_since_parses_docker_nanoseconds_and_zulu():
    """Docker's 9-digit fraction and literal Z are both normalised."""
    assert 55.0 < checks_mod._seconds_since(_docker_stamp(60.0)) < 65.0


def test_seconds_since_parses_an_explicit_offset():
    """A stamp already carrying a numeric offset needs no rewriting."""
    moment = datetime.now(timezone.utc) - timedelta(seconds=30)
    assert 25.0 < checks_mod._seconds_since(moment.isoformat()) < 35.0


def test_process_uptime_without_pid_file_is_zero(tmp_path, monkeypatch):
    """A model that was never started here reports no uptime."""
    monkeypatch.setattr(checks_mod, "RUN_DIR", tmp_path)
    assert checks_mod.server_uptime("never-started", "llama") == 0.0


def test_process_uptime_with_a_stale_pid_is_zero(tmp_path, monkeypatch):
    """A pid file left behind by a dead process reports no uptime."""
    monkeypatch.setattr(checks_mod, "RUN_DIR", tmp_path)
    # Above the kernel's default pid_max — cannot name a live process.
    (tmp_path / "mymodel").write_text("999999999")
    assert checks_mod.server_uptime("mymodel", "llama") == 0.0


def test_container_uptime_when_docker_fails_is_zero(monkeypatch):
    """A failing `docker inspect` must not raise into the stop path."""
    monkeypatch.setattr(checks_mod, "_docker_container", lambda *a, **k: "deadbeef1234")

    def _boom(*args, **kwargs):
        raise OSError("docker daemon is not running")

    monkeypatch.setattr(checks_mod.subprocess, "check_output", _boom)
    assert checks_mod.server_uptime("mymodel", "vllm") == 0.0


def test_seconds_since_garbage_is_zero():
    """An unparsable stamp accounts as no uptime."""
    assert checks_mod._seconds_since("not-a-timestamp") == 0.0
    assert checks_mod._seconds_since("") == 0.0


def test_port_holder_free_port_is_none(monkeypatch):
    """A port nothing is listening on is not a conflict."""
    monkeypatch.setattr(checks_mod, "_listening_ports", lambda: {8000})
    assert checks_mod.port_holder(8888) is None


def test_port_holder_names_a_published_container(monkeypatch):
    """A bridge-mode container publishing the port is named outright."""
    monkeypatch.setattr(checks_mod, "_listening_ports", lambda: {8888})
    monkeypatch.setattr(checks_mod.subprocess, "check_output",
                        lambda *a, **k: "vllm-other\n")
    held = checks_mod.port_holder(8888)
    assert "vllm-other" in held and "8888" in held


def test_port_holder_skips_our_own_container(monkeypatch):
    """Restarting a model that already holds the port is not a conflict.

    The published-name branch must ignore own_container; the port is still in
    use, so a message is still returned, but it must not accuse us of clashing
    with ourselves.
    """
    monkeypatch.setattr(checks_mod, "_listening_ports", lambda: {8888})
    monkeypatch.setattr(checks_mod.subprocess, "check_output",
                        lambda *a, **k: "vllm-mine\n")
    assert "vllm-mine" not in checks_mod.port_holder(
        8888, own_container="vllm-mine")


def test_port_holder_host_network_is_a_hint_not_an_accusation(monkeypatch):
    """Host-network containers publish no port map, so they can only be hinted.

    `docker ps --filter publish=` cannot see them, and its silence says nothing
    about whether a container holds the port. The message must not claim the
    holder is not a container.
    """
    monkeypatch.setattr(checks_mod, "_listening_ports", lambda: {8888})
    calls = []

    def _fake(cmd, *a, **k):
        calls.append(cmd)
        return "" if "publish=8888" in cmd else "vllm-fn\n"

    monkeypatch.setattr(checks_mod.subprocess, "check_output", _fake)
    held = checks_mod.port_holder(8888)
    assert "vllm-fn" in held
    assert "may hold it" in held
    assert "non-container" not in held


def test_port_holder_survives_docker_being_down(monkeypatch):
    """Docker unreachable still reports the conflict, just unattributed."""
    monkeypatch.setattr(checks_mod, "_listening_ports", lambda: {8888})

    def _boom(*a, **k):
        raise OSError("docker daemon is not running")

    monkeypatch.setattr(checks_mod.subprocess, "check_output", _boom)
    assert checks_mod.port_holder(8888) == "port 8888 is already in use"
