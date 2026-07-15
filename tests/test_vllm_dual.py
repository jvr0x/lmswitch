"""Tests for the vllm-dual (two-node TP=2) runtime and cluster visibility.

Covers the three required cases per project convention:
- expected use: dual start launches worker (ssh) then head (local docker run)
  with the right --nnodes/--node-rank/--headless split and NCCL env.
- edge: cluster view merges a peer's models, dedupes shared dual YAMLs, and
  survives an unreachable peer.
- failure: a failed head launch tears the worker back down; missing yaml
  fields refuse to start.
"""

import json
from pathlib import Path
from unittest import mock

import lmswitch.cli as cli_mod
import lmswitch.models.cluster as cluster_mod
import lmswitch.runtimes.vllm_dual as dual_mod
from lmswitch.models.loader import load_models
from lmswitch.runtimes.base import runtime_registry
from lmswitch.runtimes.vllm_dual import VLLMDualRuntime


DUAL_YAML = {
    "runtime": "vllm-dual",
    "image": "vllm-dspark-runtime:test",
    "model": "org/Model-DSpark",
    "hf_cache": "~/hf-cluster",
    "port": 8888,
    "worker_host": "Gigabyte",
    "master_addr": "10.100.224.2",
    "worker_ip": "10.100.224.1",
    "master_port": 25000,
    "nccl": {"ifname": "enp1s0f1np1", "hca": "rocep1s0f1", "gid_index": 3},
}


class _Result:
    def __init__(self, returncode=0):
        self.returncode = returncode


# ---------------------------------------------------------------------------
# Expected use
# ---------------------------------------------------------------------------

def test_dual_runtime_registered():
    assert runtime_registry.lookup("vllm-dual") is VLLMDualRuntime


def test_node_cmd_head_vs_worker():
    rt = VLLMDualRuntime()
    head = rt._node_cmd("m", DUAL_YAML, node_rank=0)
    worker = rt._node_cmd("m", DUAL_YAML, node_rank=1)

    assert "--node-rank=0" in head and "--headless" not in head
    assert "--node-rank=1" in worker and worker[-1] == "--headless"
    for cmd in (head, worker):
        assert "--nnodes=2" in cmd
        assert "--tensor-parallel-size=2" in cmd
        assert "--master-addr=10.100.224.2" in cmd
        assert "--device" in cmd and "/dev/infiniband:/dev/infiniband" in cmd
        assert "NCCL_IB_HCA=rocep1s0f1" in cmd
        assert "NCCL_SOCKET_IFNAME=enp1s0f1np1" in cmd
    # VLLM_HOST_IP differs per node: head=master, worker=worker_ip.
    assert "VLLM_HOST_IP=10.100.224.2" in head
    assert "VLLM_HOST_IP=10.100.224.1" in worker


def test_dual_start_worker_before_head():
    order = []

    def fake_ssh(host, cmd, **kw):
        order.append(("ssh", cmd[:2]))
        return _Result(0)

    def fake_run(cmd, *a, **k):
        order.append(("local", list(cmd)[:2]))
        return _Result(0)

    with mock.patch.object(dual_mod, "_ssh", fake_ssh), \
         mock.patch.object(dual_mod.subprocess, "run", fake_run), \
         mock.patch.object(VLLMDualRuntime, "_preflight", return_value=None), \
         mock.patch.object(VLLMDualRuntime, "_setup_logging",
                           return_value=Path("/tmp/x.log")), \
         mock.patch("lmswitch.system.checks._docker_container",
                    side_effect=[None, "abc123def456"]), \
         mock.patch.object(dual_mod, "_wait_ready", return_value="ready"):
        state = VLLMDualRuntime().start("m", DUAL_YAML)

    assert state.status == "ready"
    docker_runs = [(w, c) for w, c in order if c == ["docker", "run"]]
    assert [w for w, _ in docker_runs] == ["ssh", "local"], \
        "worker container must start before the head"


def test_node_cmd_model_path_per_node():
    """model_path mounts each node's own host path at canonical /model."""
    yaml = dict(DUAL_YAML,
                model_path="/home/u/models-gigabyte/org/M",
                worker_model_path="/home/u/models/org/M")
    rt = VLLMDualRuntime()
    head = rt._node_cmd("m", yaml, node_rank=0)
    worker = rt._node_cmd("m", yaml, node_rank=1)
    assert "/home/u/models-gigabyte/org/M:/model:ro" in head
    assert "/home/u/models/org/M:/model:ro" in worker
    for cmd in (head, worker):
        assert "/model" in cmd            # serve target is the mount
        assert "HF_HOME=/cache/huggingface" not in cmd


# ---------------------------------------------------------------------------
# Edge: cluster view merge
# ---------------------------------------------------------------------------

def _remote_payload():
    return json.dumps({"host": "gigabyte", "serve_host": "gigabyte.local", "models": [
        {"name": "remote-only", "display": "R", "runtime": "vllm",
         "type": "vllm", "port": 8001, "ctx": "", "size": 1, "present": True,
         "restart": None, "family": "Other", "fam_order": 10, "running": True,
         "host": "gigabyte"},
        {"name": "qwen2.5-7b", "display": "dupe-of-local", "runtime": "llama",
         "type": "gguf", "port": 8080, "ctx": "", "size": 1, "present": True,
         "restart": None, "family": "Qwen", "fam_order": 0, "running": False,
         "host": "gigabyte"},
        # A dual model exported from its head node keeps the loader's own
        # "dual" label — the merge must not stomp it with the payload's
        # machine-level "host" (a bug in an earlier version of this code).
        {"name": "some-dual-model", "display": "Dual", "runtime": "vllm-dual",
         "type": "dual", "port": 8888, "ctx": "", "size": 1, "present": True,
         "restart": None, "family": "Other", "fam_order": 10, "running": True,
         "host": "dual"},
    ]})


def test_cluster_merge_dedupes_local_names(lmswitch_data_dir):
    with mock.patch.object(cluster_mod, "_cluster_hosts", return_value=["Gigabyte"]), \
         mock.patch.object(cluster_mod.subprocess, "check_output",
                           return_value=_remote_payload()):
        remote = cli_mod._gather_cluster_models()
    by_name = {m["name"]: m for m in remote}
    assert "remote-only" in by_name
    # qwen2.5-7b exists in the local fixture tree — the local row wins.
    assert "qwen2.5-7b" not in by_name
    assert by_name["remote-only"]["host"] == "gigabyte"
    assert by_name["remote-only"]["remote_host"] == "Gigabyte"
    assert by_name["remote-only"]["serve_host"] == "gigabyte.local"
    assert by_name["remote-only"]["running"] is True
    # The dual model's own "dual" host label survives the merge.
    assert by_name["some-dual-model"]["host"] == "dual"
    assert by_name["some-dual-model"]["serve_host"] == "gigabyte.local"


def test_cluster_merge_survives_unreachable_peer():
    def boom(*a, **k):
        raise cluster_mod.subprocess.CalledProcessError(255, "ssh")

    with mock.patch.object(cluster_mod, "_cluster_hosts", return_value=["Gigabyte"]), \
         mock.patch.object(cluster_mod.subprocess, "check_output", boom):
        assert cli_mod._gather_cluster_models() == []


def test_render_shows_host_column_for_cluster(capsys):
    models = [{
        "name": "dualmodel", "display": "Dual", "runtime": "vllm-dual",
        "type": "dual", "port": 8888, "ctx": "", "size": 0, "present": True,
        "restart": None, "family": "Other", "fam_order": 10,
        "host": "dual", "running": False,
    }]
    cli_mod.render(models)
    out = capsys.readouterr().out
    assert "HOST" in out
    assert "dual" in out


def test_render_no_host_column_single_box(capsys):
    models = [{
        "name": "solo", "display": "Solo", "runtime": "llama", "type": "gguf",
        "port": 8080, "ctx": "", "size": 0, "present": True, "restart": None,
        "family": "Other", "fam_order": 10, "running": False,
    }]
    with mock.patch.object(cli_mod, "_cluster_hosts", return_value=[]):
        cli_mod.render(models)
    assert "HOST" not in capsys.readouterr().out


def test_loader_dual_model_host_and_type(lmswitch_data_dir, tmp_path):
    cache = tmp_path / "hf"
    snap = cache / "hub" / "models--org--Model-DSpark" / "snapshots" / "rev1"
    snap.mkdir(parents=True)
    (snap / "model-00001-of-00002.safetensors").write_bytes(b"x" * 64)
    yaml_text = (
        "runtime: vllm-dual\n"
        "image: img:t\n"
        "model: org/Model-DSpark\n"
        f"hf_cache: {cache}\n"
        "port: 8888\n"
        "worker_host: Gigabyte\n"
        "master_addr: 10.100.224.2\n"
    )
    (lmswitch_data_dir / "zz-dual.yaml").write_text(yaml_text)
    dual = [m for m in load_models() if m["name"] == "zz-dual"]
    assert dual and dual[0]["type"] == "dual"
    assert dual[0]["host"] == "dual"
    assert dual[0]["present"] is True
    assert dual[0]["size"] == 64


# ---------------------------------------------------------------------------
# Failure cases
# ---------------------------------------------------------------------------

def test_dual_start_refuses_on_missing_fields():
    state = VLLMDualRuntime().start("m", {"runtime": "vllm-dual"})
    assert state.status == "dead"
    assert "missing required yaml field" in state.detail


def test_dual_head_failure_tears_down_worker():
    ssh_calls = []

    def fake_ssh(host, cmd, **kw):
        ssh_calls.append(list(cmd))
        return _Result(0)

    def fake_run(cmd, *a, **k):
        # Local head docker run fails.
        if list(cmd)[:2] == ["docker", "run"]:
            return _Result(1)
        return _Result(0)

    with mock.patch.object(dual_mod, "_ssh", fake_ssh), \
         mock.patch.object(dual_mod.subprocess, "run", fake_run), \
         mock.patch.object(VLLMDualRuntime, "_preflight", return_value=None), \
         mock.patch("lmswitch.system.checks._docker_container",
                    return_value=None):
        state = VLLMDualRuntime().start("m", DUAL_YAML)

    assert state.status == "dead"
    assert ["docker", "rm", "-f", "vllm-m"] in ssh_calls, \
        "worker must be cleaned up when the head fails to launch"


def test_dual_stop_stops_both_nodes():
    ssh_calls = []

    def fake_ssh(host, cmd, **kw):
        ssh_calls.append((host, list(cmd)))
        return _Result(0)

    with mock.patch.object(dual_mod, "_ssh", fake_ssh), \
         mock.patch.object(VLLMDualRuntime.__bases__[0], "stop") as head_stop:
        VLLMDualRuntime().stop("m", DUAL_YAML)

    head_stop.assert_called_once()
    assert ("Gigabyte", ["docker", "rm", "-f", "vllm-m"]) in ssh_calls
