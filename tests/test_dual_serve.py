"""Tests for foreground supervision of two-node runtimes (`lmswitch serve`).

Before this path existed, ``cmd_serve`` routed every non-``vllm`` runtime to
``_start_llama_direct`` — so a ``restart: always`` dual recipe produced a
systemd unit that tried to launch llama-server against a vLLM TP=2 recipe.

Covers the three required cases:
- expected use: the pair starts, the head container later disappears, and the
  supervisor exits (so systemd's Restart= can recover it) after stopping the
  worker.
- edge: the boot gate waits for a peer that is still booting, then proceeds
  once it answers.
- failure: a start that never reaches ready tears both ranks down and exits;
  a gate that never passes exits without starting anything.
"""

import pytest
from unittest import mock

import lmswitch.cli as cli_mod
import lmswitch.runtimes.dual_serve as ds_mod
from lmswitch.runtimes.base import RunningState
from lmswitch.runtimes.systemd import _SYSTEMD_UNIT, _DUAL_UNIT_EXTRA


DUAL_YAML = {
    "runtime": "vllm-dual",
    "image": "img:test",
    "model": "org/Model",
    "model_path": "/nonexistent-mount",
    "port": 8888,
    "worker_host": "Gigabyte",
    "master_addr": "10.100.224.2",
}


class _FakeRuntime:
    """Stands in for VLLMDualRuntime: records start/stop, scripts the status."""

    def __init__(self, status="ready"):
        self.status = status
        self.started = 0
        self.stopped = 0

    def start(self, name, yaml):
        self.started += 1
        return RunningState(self.status)

    def stop(self, name, yaml):
        self.stopped += 1


def _install(monkeypatch, runtime, gate=None, containers=None):
    """Patches the registry, gate and container probe used by dual_serve."""
    monkeypatch.setattr(ds_mod.runtime_registry, "lookup",
                        lambda _n: (lambda: runtime))
    if gate is not None:
        monkeypatch.setattr(ds_mod, "_gate", gate)
    if containers is not None:
        seq = list(containers)
        # Takes the prefix too: sglang-dual's containers are named
        # ``sglang-<id>``, so probing under the default ``vllm-`` prefix would
        # report a live head as gone and tear the pair down on the first poll.
        monkeypatch.setattr(ds_mod, "_docker_container",
                            lambda _n, _p="vllm": seq.pop(0) if seq else None)
    monkeypatch.setattr(ds_mod.time, "sleep", lambda _s: None)
    monkeypatch.setattr(ds_mod, "_memory_check", lambda _n, _y: (True, ""))


# ---------------------------------------------------------------------------
# Expected use
# ---------------------------------------------------------------------------

def test_exits_when_head_container_disappears(monkeypatch):
    """The pair serves, then the head container dies — the supervisor must
    exit so systemd restarts it, and must not leave the worker running."""
    rt = _FakeRuntime("ready")
    _install(monkeypatch, rt, gate=lambda *a, **k: None,
             containers=["abc123", "abc123", None])

    with pytest.raises(SystemExit) as exc:
        ds_mod._start_dual_foreground("m", DUAL_YAML)

    assert "exited" in str(exc.value)
    assert "restart" in str(exc.value).lower()
    assert rt.started == 1
    assert rt.stopped == 1, "worker must be torn down before handing back"


def test_cmd_serve_routes_dual_runtimes(monkeypatch, lmswitch_data_dir):
    """cmd_serve must dispatch vllm-dual to the dual supervisor, never to the
    llama child-process path (which would launch llama-server on a TP=2 recipe)."""
    (cli_mod.CONF_DIR / "d.yaml").write_text(
        "runtime: vllm-dual\nimage: img:test\nmodel: org/Model\nport: 8888\n"
        "worker_host: Gigabyte\nmaster_addr: 10.0.0.1\n")
    calls = []
    monkeypatch.setattr(cli_mod, "_start_dual_foreground",
                        lambda name, yaml: calls.append(name))
    monkeypatch.setattr(cli_mod, "_start_llama_direct",
                        lambda *a, **k: pytest.fail("llama path must not run"))

    cli_mod.cmd_serve("d")
    assert calls == ["d"]


def test_cmd_serve_routes_sglang_dual(monkeypatch, lmswitch_data_dir):
    """sglang-dual is a two-node runtime too — the llama path would try to boot
    llama-server against a TP=2 safetensors recipe."""
    (cli_mod.CONF_DIR / "s.yaml").write_text(
        "runtime: sglang-dual\nimage: img:test\nmodel_path: /nonexistent\n"
        "port: 8888\nworker_host: Gigabyte\nmaster_addr: 10.0.0.1\n")
    calls = []
    monkeypatch.setattr(cli_mod, "_start_dual_foreground",
                        lambda name, yaml: calls.append(name))
    monkeypatch.setattr(cli_mod, "_start_llama_direct",
                        lambda *a, **k: pytest.fail("llama path must not run"))

    cli_mod.cmd_serve("s")
    assert calls == ["s"]


def test_head_probe_uses_the_runtimes_container_prefix(monkeypatch):
    """The head-liveness poll must probe ``sglang-<id>`` for sglang-dual. Under
    the default ``vllm-`` prefix a healthy head reads as gone, and the
    supervisor tears the pair down on its first poll."""
    seen = []
    rt = _FakeRuntime("ready")
    monkeypatch.setattr(ds_mod.runtime_registry, "lookup", lambda _n: (lambda: rt))
    monkeypatch.setattr(ds_mod, "_gate", lambda *a, **k: None)
    monkeypatch.setattr(ds_mod.time, "sleep", lambda _s: None)
    monkeypatch.setattr(ds_mod, "_memory_check", lambda _n, _y: (True, ""))
    monkeypatch.setattr(ds_mod, "_docker_container",
                        lambda n, p="vllm": seen.append(p) or None)

    with pytest.raises(SystemExit):
        ds_mod._start_dual_foreground("m", {**DUAL_YAML, "runtime": "sglang-dual"})
    assert seen == ["sglang"]


def test_dual_unit_disables_start_rate_limit_and_logs():
    """A dual unit must retry indefinitely (a booting peer outlasts the default
    5-in-10s budget) and keep its output where a failed boot can be read."""
    unit = _SYSTEMD_UNIT.format(name="d", restart="always",
                                unit_extra=_DUAL_UNIT_EXTRA, restart_sec=30,
                                stdout="journal", stderr="journal")
    assert "StartLimitIntervalSec=0" in unit
    assert "RestartSec=30" in unit
    assert "StandardOutput=journal" in unit
    assert "ExecStart=%h/.local/bin/lmswitch serve d" in unit


# ---------------------------------------------------------------------------
# Edge
# ---------------------------------------------------------------------------

def test_gate_waits_for_a_peer_that_is_still_booting(monkeypatch):
    """At boot the peer answers late; the gate must keep waiting and then pass
    rather than failing the unit on the first refused ssh."""
    monkeypatch.setattr(ds_mod.time, "sleep", lambda _s: None)
    monkeypatch.setattr(ds_mod, "_docker_ready", lambda: True)
    monkeypatch.setattr(ds_mod, "_weights_ready", lambda _y: True)
    answers = [False, False, True]
    monkeypatch.setattr(ds_mod, "_worker_ready", lambda _y: answers.pop(0))

    assert ds_mod._gate("m", DUAL_YAML, timeout=600, interval=10) is None
    assert answers == []


def test_gate_reports_the_unmet_precondition(monkeypatch):
    """A weights mount that never appears must be named in the exit message —
    an NFS mount lagging behind boot is the expected cause."""
    monkeypatch.setattr(ds_mod.time, "sleep", lambda _s: None)
    monkeypatch.setattr(ds_mod, "_docker_ready", lambda: True)
    monkeypatch.setattr(ds_mod, "_weights_ready", lambda _y: False)
    monkeypatch.setattr(ds_mod, "_worker_ready", lambda _y: True)

    blocked = ds_mod._gate("m", DUAL_YAML, timeout=20, interval=10)
    assert blocked == "weights mount"


def test_weights_ready_rejects_an_unmounted_path(tmp_path):
    """An empty directory is an NFS mount that has not come up, not a model."""
    empty = tmp_path / "models-gigabyte"
    empty.mkdir()
    assert ds_mod._weights_ready({"model_path": str(empty)}) is False
    (empty / "config.json").write_text("{}")
    assert ds_mod._weights_ready({"model_path": str(empty)}) is True
    # No model_path (hf_cache recipes) is not a failure.
    assert ds_mod._weights_ready({}) is True


# ---------------------------------------------------------------------------
# Failure
# ---------------------------------------------------------------------------

def test_failed_start_stops_both_ranks(monkeypatch):
    """A head that never reaches ready must not leave rank 1 holding the peer's
    GPU — the next start could not bind the TP group."""
    rt = _FakeRuntime("timeout")
    _install(monkeypatch, rt, gate=lambda *a, **k: None, containers=["x"])

    with pytest.raises(SystemExit) as exc:
        ds_mod._start_dual_foreground("m", DUAL_YAML)

    assert "failed to start" in str(exc.value)
    assert rt.stopped == 1


def test_blocked_gate_exits_without_starting(monkeypatch):
    """If the cluster never becomes ready, nothing is launched at all."""
    rt = _FakeRuntime("ready")
    _install(monkeypatch, rt, gate=lambda *a, **k: "worker Gigabyte")

    with pytest.raises(SystemExit) as exc:
        ds_mod._start_dual_foreground("m", DUAL_YAML)

    assert "worker Gigabyte" in str(exc.value)
    assert rt.started == 0


def test_sigterm_stops_the_pair_and_exits_clean(monkeypatch):
    """`systemctl stop` signals this wrapper, not the containers — without a
    handler the pair keeps serving after the unit reports stopped."""
    rt = _FakeRuntime("ready")
    handlers = {}
    _install(monkeypatch, rt, gate=lambda *a, **k: None, containers=["abc"])
    monkeypatch.setattr(ds_mod.signal, "signal",
                        lambda sig, fn: handlers.setdefault(sig, fn))
    # Head stays alive; interrupt the poll loop by firing the handler from sleep.
    def _sleep(_s):
        handlers[ds_mod.signal.SIGTERM](ds_mod.signal.SIGTERM, None)
    monkeypatch.setattr(ds_mod.time, "sleep", _sleep)

    with pytest.raises(SystemExit) as exc:
        ds_mod._start_dual_foreground("m", DUAL_YAML)

    assert exc.value.code == 0
    assert rt.stopped == 1


def test_ram_guard_refuses_a_start_that_would_not_fit(monkeypatch):
    """systemd calls `lmswitch serve` directly, bypassing start_model — the RAM
    guard must be repeated here or a restart-after-crash launches on top of a
    model the user loaded meanwhile and OOMs the box."""
    rt = _FakeRuntime("ready")
    _install(monkeypatch, rt, gate=lambda *a, **k: None, containers=["abc"])
    monkeypatch.setattr(ds_mod, "_memory_check",
                        lambda _n, _y: (False, "vLLM reserves ~98Gi, but only 30Gi free"))

    with pytest.raises(SystemExit) as exc:
        ds_mod._start_dual_foreground("m", DUAL_YAML)

    assert "refusing to start" in str(exc.value)
    assert "30Gi free" in str(exc.value)
    assert rt.started == 0


def test_force_overrides_the_ram_guard(monkeypatch):
    """`force: true` is the documented override on start_model; serve keeps the
    same contract so a recipe does not behave differently under systemd."""
    rt = _FakeRuntime("ready")
    _install(monkeypatch, rt, gate=lambda *a, **k: None, containers=["abc", None])
    monkeypatch.setattr(ds_mod, "_memory_check", lambda _n, _y: (False, "tight"))

    with pytest.raises(SystemExit) as exc:
        ds_mod._start_dual_foreground("m", dict(DUAL_YAML, force=True))

    assert "exited" in str(exc.value)
    assert rt.started == 1
