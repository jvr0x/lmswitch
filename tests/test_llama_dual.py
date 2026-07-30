"""Tests for the llama-dual (llama.cpp RPC) runtime.

Cover command construction, the tensor_split share math the memory guard
depends on, and the failure path where the worker is unreachable — all with
ssh/Popen stubbed, so nothing is launched and no node is touched.
"""

import tempfile
from pathlib import Path
from unittest import mock

import lmswitch.runtimes.llama as llama_mod
import lmswitch.runtimes.llama_dual as dual_mod
from lmswitch.runtimes.llama_dual import LlamaDualRuntime, local_share


def _yaml(**over) -> dict:
    cfg = {
        "runtime": "llama-dual",
        "model": "unsloth/Qwen3.5-397B-A17B-GGUF/UD-IQ4_NL/model-00001-of-00005.gguf",
        "port": 8105,
        "ctx": 8192,
        "worker_host": "Gigabyte",
        "worker_ip": "10.100.224.1",
        "tensor_split": "0.45,0.55",
        "_models_dir": Path(tempfile.mkdtemp()),
    }
    cfg.update(over)
    return cfg


def test_rpc_endpoint_and_split_in_cmd():
    """Expected use: the head gets --rpc <worker>:<port> and --tensor-split."""
    cmd, _path = LlamaDualRuntime()._build_cmd("qwen397b", _yaml())
    assert "--rpc" in cmd, f"missing --rpc: {cmd}"
    assert cmd[cmd.index("--rpc") + 1] == "10.100.224.1:50052"
    assert cmd[cmd.index("--tensor-split") + 1] == "0.45,0.55"
    # Still a normal llama-server invocation underneath.
    assert cmd[cmd.index("--model") + 1].endswith(".gguf")
    assert cmd[cmd.index("--port") + 1] == "8105"


def test_explicit_rpc_port_overrides_default():
    """Edge case: a recipe on a non-default RPC port is honoured."""
    cmd, _path = LlamaDualRuntime()._build_cmd("m", _yaml(rpc_port=50999))
    assert cmd[cmd.index("--rpc") + 1] == "10.100.224.1:50999"


def test_tensor_split_from_extra_args_is_not_duplicated():
    """Edge case: -ts passed by hand must not gain a second --tensor-split."""
    cfg = _yaml(extra_args=["-ts", "0.3,0.7"])
    cmd, _path = LlamaDualRuntime()._build_cmd("m", cfg)
    assert "--tensor-split" not in cmd, f"duplicate split flag: {cmd}"
    assert cmd[cmd.index("-ts") + 1] == "0.3,0.7"


def test_local_share_math():
    """The memory guard's share: first entry is this node, normalised."""
    assert local_share({"tensor_split": "0.45,0.55"}) == 0.45
    assert local_share({"tensor_split": "1,1"}) == 0.5
    # Missing or malformed → even split, matching llama.cpp's own default.
    assert local_share({}) == 0.5
    assert local_share({"tensor_split": "abc"}) == 0.5
    assert local_share({"tensor_split": "0,0"}) == 0.5


def test_unreachable_worker_refuses_to_start():
    """Failure case: no ssh to the worker → dead, and no llama-server spawned."""
    spawned = []

    def _fake_ssh(host, cmd, **kw):
        return type("R", (), {"returncode": 255})()

    with mock.patch.object(dual_mod, "_ssh", _fake_ssh), \
         mock.patch.object(llama_mod.subprocess, "Popen",
                           lambda *a, **k: spawned.append(a) or None):
        state = LlamaDualRuntime().start("qwen397b", _yaml())

    assert state.status == "dead"
    assert "unreachable" in state.detail
    assert not spawned, "llama-server must not start without its worker"


def test_missing_rpc_binary_refuses_to_start():
    """Failure case: worker reachable but rpc-server absent → clear error."""
    def _fake_ssh(host, cmd, **kw):
        # `true` succeeds (reachable), the `test -x` probe fails.
        return type("R", (), {"returncode": 0 if cmd == "true" else 1})()

    with mock.patch.object(dual_mod, "_ssh", _fake_ssh):
        state = LlamaDualRuntime().start("qwen397b", _yaml())

    assert state.status == "dead"
    assert "rpc-server missing" in state.detail


def test_worker_kill_is_scoped_to_its_rpc_port():
    """Edge case: stopping one recipe must not kill another's rpc-server."""
    cmd = dual_mod._pkill_cmd(50052)
    assert "50052" in cmd, cmd
    assert cmd.count("rpc-server") == 1 and "-p 50052" in cmd, cmd


def test_stop_kills_worker_even_when_head_is_gone():
    """A stranded rpc-server would hold ~99 GiB on the worker."""
    calls = []

    def _fake_ssh(host, cmd, **kw):
        calls.append((host, cmd))
        return type("R", (), {"returncode": 0})()

    with mock.patch.object(dual_mod, "_ssh", _fake_ssh), \
         mock.patch.object(dual_mod.LlamaDualRuntime.__mro__[1], "stop",
                           lambda self, name, yaml: None):
        LlamaDualRuntime().stop("qwen397b", _yaml())

    assert any(host == "Gigabyte" and "pkill" in cmd for host, cmd in calls), calls
