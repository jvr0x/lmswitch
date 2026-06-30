"""Tests for vLLM log-file homogeneity with the GGUF runtime.

vLLM models run in Docker, so unlike llama-server they don't write a
``running/<name>.log`` directly. ``VLLMRuntime._setup_logging`` closes that gap
by spawning a detached ``docker logs -f`` follower that mirrors the container's
output into the same ``RUN_DIR/<name>.log`` path the GGUF runtime writes, making
``lmswitch`` log inspection identical across runtimes.
"""

import subprocess
from pathlib import Path
from unittest import mock

from lmswitch.runtimes.vllm import VLLMRuntime


def test_setup_logging_spawns_docker_logs_follower(lmswitch_data_dir):
    """Expected use: spawns ``docker logs -f vllm-<name>`` into RUN_DIR/<name>.log."""
    run_dir = lmswitch_data_dir / "running"
    captured = {}

    def fake_popen(cmd, *a, **k):
        captured["cmd"] = cmd
        captured["stdout"] = k.get("stdout")
        captured["stderr"] = k.get("stderr")
        captured["start_new_session"] = k.get("start_new_session")
        return mock.MagicMock()

    with mock.patch.object(subprocess, "Popen", fake_popen):
        log_path = VLLMRuntime()._setup_logging("qwen3-test")

    # Returns the homogeneous <name>.log path and the file is created.
    assert log_path == run_dir / "qwen3-test.log"
    assert log_path.exists()

    # Follows the named container's docker logs.
    assert captured["cmd"][:3] == ["docker", "logs", "-f"]
    assert captured["cmd"][-1] == "vllm-qwen3-test"

    # Container stderr merged into stdout, detached session, stdout → log file.
    assert captured["stderr"] == subprocess.STDOUT
    assert captured["start_new_session"] is True
    assert Path(captured["stdout"].name) == log_path


def test_setup_logging_survives_popen_failure(lmswitch_data_dir):
    """Failure case: a follower that can't spawn must not abort the model start."""
    def boom(*a, **k):
        raise OSError("docker not found")

    with mock.patch.object(subprocess, "Popen", boom):
        # Must not raise — the follower is auxiliary to the running container.
        log_path = VLLMRuntime()._setup_logging("qwen3-test")

    assert log_path == lmswitch_data_dir / "running" / "qwen3-test.log"


def test_setup_logging_replaces_stale_symlink(lmswitch_data_dir, tmp_path):
    """Regression: a leftover symlink at <name>.log (an older version's link to a
    root-owned docker json log) must be replaced with a fresh regular file, not
    followed/written-through (which on the Spark caused EACCES)."""
    run_dir = lmswitch_data_dir / "running"
    log_path = run_dir / "qwen3-test.log"
    target = tmp_path / "fake-docker-json.log"  # stands in for the root-owned target
    log_path.symlink_to(target)
    assert log_path.is_symlink()

    with mock.patch.object(subprocess, "Popen", lambda *a, **k: mock.MagicMock()):
        result = VLLMRuntime()._setup_logging("qwen3-test")

    assert result == log_path
    assert not log_path.is_symlink(), "stale symlink should be removed"
    assert log_path.is_file(), "a fresh regular log file should be created"
    assert not target.exists(), "must not write through the symlink into the target"


def test_start_attaches_follower_after_docker_run(lmswitch_data_dir, tmp_path, monkeypatch):
    """Wiring: start() attaches the follower (and creates <name>.log) only once
    the container is up — the follower is the sole spawned Popen."""
    import lmswitch.runtimes.vllm as vllm_mod
    import lmswitch.system.checks as checks_mod

    run_dir = lmswitch_data_dir / "running"
    popen_cmds: list = []

    # Container is absent on the pre-flight check, present once "ready".
    cids = iter([None, "cid123abc456"])
    monkeypatch.setattr(checks_mod, "_docker_container",
                        lambda name: next(cids, "cid123abc456"))
    monkeypatch.setattr(vllm_mod, "_wait_ready", lambda *a, **k: "ready")
    monkeypatch.setattr(vllm_mod.subprocess, "run",
                        lambda *a, **k: mock.MagicMock(returncode=0))

    def fake_popen(cmd, *a, **k):
        popen_cmds.append(cmd)
        return mock.MagicMock()
    monkeypatch.setattr(vllm_mod.subprocess, "Popen", fake_popen)

    yaml = {"model": "m", "_models_dir": tmp_path, "port": 8000}
    state = VLLMRuntime().start("m", yaml)

    assert state.status == "ready"
    # Homogeneous log file was created during start().
    assert (run_dir / "m.log").exists()
    # The follower is the only process spawned, and only after docker run.
    assert popen_cmds == [["docker", "logs", "-f", "vllm-m"]]
