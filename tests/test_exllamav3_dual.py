"""Tests for the exllamav3-dual (pipeline-parallel EXL3 job) runtime.

Cover per-rank command construction, the splits share math the memory guard
depends on, and the failure path where the worker is unreachable — all with
ssh/Popen stubbed, so nothing is launched and no node is touched.
"""

from unittest import mock

import lmswitch.runtimes.exllamav3_dual as dual_mod
from lmswitch.runtimes.exllamav3_dual import Exllamav3DualRuntime, pipeline_share
from lmswitch.runtimes.base import runtime_registry


def _yaml(**over) -> dict:
    cfg = {
        "runtime": "exllamav3-dual",
        "model": "vcruz305/RED-SNOW-5.3-FLASH-EXL3-SAGE-4.91bpw",
        "model_path": "~/models-gigabyte/vcruz305/RED-SNOW-5.3-FLASH-EXL3-SAGE-4.91bpw",
        "worker_model_path": "~/models/vcruz305/RED-SNOW-5.3-FLASH-EXL3-SAGE-4.91bpw",
        "port": 0,
        "worker_host": "Gigabyte",
        "worker_ip": "10.100.224.1",
        "master_addr": "10.100.224.2",
        "transport": "tcp",
        "transport_port": 29650,
        "splits": "0:23,23:45",
        "ctx": 8192,
        "cache_quant": "4",
        "prompt": "Say hi.",
        "max_new": 200,
    }
    cfg.update(over)
    return cfg


def test_runtime_is_registered():
    """The new runtime resolves through the registry (no cli changes needed)."""
    assert runtime_registry.lookup("exllamav3-dual") is Exllamav3DualRuntime


def test_rank_cmds():
    """Expected use: rank 0 drives generation, rank 1 waits headless."""
    rt = Exllamav3DualRuntime()
    cmd0, env0 = rt._rank_cmd(_yaml(), 0)
    assert cmd0[cmd0.index("--rank") + 1] == "0"
    assert cmd0[cmd0.index("--addrs") + 1] == "10.100.224.2,10.100.224.1"
    assert cmd0[cmd0.index("--splits") + 1] == "0:23,23:45"
    assert cmd0[cmd0.index("--prompt") + 1] == "Say hi."
    assert cmd0[cmd0.index("--max_new") + 1] == "200"
    assert cmd0[cmd0.index("--cache_quant") + 1] == "4"
    assert env0 == {}, f"tcp must not set NCCL env: {env0}"

    cmd1, _env1 = rt._rank_cmd(_yaml(), 1)
    assert cmd1[cmd1.index("--rank") + 1] == "1"
    assert "--prompt" not in cmd1, f"rank 1 takes no prompt: {cmd1}"


def test_nccl_transport_sets_roce_env():
    """Edge case: nccl transport pins the CX7 interface on both ranks."""
    cfg = _yaml(transport="nccl",
                nccl={"ifname": "enp1s0f1np1", "hca": "rocep1s0f1"})
    _cmd, env = Exllamav3DualRuntime()._rank_cmd(cfg, 0)
    assert env["NCCL_SOCKET_IFNAME"] == "enp1s0f1np1"
    assert env["NCCL_IB_HCA"] == "rocep1s0f1"


def test_pipeline_share_math():
    """The memory guard's slice: rank-0 layers over total decoder layers."""
    assert pipeline_share({"splits": "0:23,23:45"}) == 23 / 45
    assert pipeline_share({"splits": "0:26,26:46,46:62,62:78"}) == 26 / 78
    # Absent → the default splits apply (same default _rank_cmd uses).
    assert pipeline_share({}) == 23 / 45
    # Malformed → even split.
    assert pipeline_share({"splits": "abc"}) == 0.5
    assert pipeline_share({"splits": "0:0,0:0"}) == 0.5


def test_unreachable_worker_refuses_to_start():
    """Failure case: no ssh to the worker → dead, and nothing is spawned."""
    spawned = []

    def _fake_ssh(host, cmd, **kw):
        return type("R", (), {"returncode": 255, "stdout": "", "stderr": ""})()

    with mock.patch.object(dual_mod, "_ssh", _fake_ssh), \
         mock.patch("subprocess.Popen",
                    lambda *a, **k: spawned.append(a) or None):
        state = Exllamav3DualRuntime().start("red-snow", _yaml())

    assert state.status == "dead"
    assert "unreachable" in state.detail
    assert spawned == []


def test_missing_field_refuses_to_start():
    """Failure case: a recipe without worker_ip is rejected by preflight."""
    cfg = _yaml()
    del cfg["worker_ip"]
    with mock.patch("subprocess.Popen",
                    side_effect=AssertionError("must not spawn")):
        state = Exllamav3DualRuntime().start("red-snow", cfg)
    assert state.status == "dead"
    assert "worker_ip" in state.detail
