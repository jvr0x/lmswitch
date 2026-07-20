"""Tests for the vllm-dual-ray (two-node Ray-backed TP=2) runtime.

Covers the three required cases per project convention:
- expected use: _container_cmd builds idle (`sleep infinity`) containers with
  the right model_path per node; _serve_cmd builds a distributed-executor-
  backend=ray vllm serve line; NCCL exports differ per side.
- edge: head/worker_extra_mounts are exclusive to their own side, same
  asymmetry as vllm_dual.py; runtime registers and round-trips via the
  registry like every other runtime.
- failure: missing required yaml fields refuse to start via _preflight.
"""

from unittest import mock

from lmswitch.runtimes.base import runtime_registry
from lmswitch.runtimes.vllm_dual_ray import VLLMDualRayRuntime


RAY_YAML = {
    "runtime": "vllm-dual-ray",
    "image": "vllm-mimo-v25-lmswitch:test",
    "model_path": "/home/u/models/lukealonso/MiMo-V2.5-NVFP4",
    "worker_model_path": "/home/u/models-spark/lukealonso/MiMo-V2.5-NVFP4",
    "port": 8888,
    "worker_host": "Gigabyte",
    "master_addr": "10.100.224.2",
    "worker_ip": "10.100.224.1",
    "nccl": {"ifname": "enp1s0f1np1", "hca": "rocep1s0f1", "gid_index": 3},
}


# ---------------------------------------------------------------------------
# Expected use
# ---------------------------------------------------------------------------

def test_registered_in_runtime_registry():
    """vllm-dual-ray resolves to VLLMDualRayRuntime via the shared registry."""
    assert runtime_registry.lookup("vllm-dual-ray") is VLLMDualRayRuntime


def test_container_cmd_idle_with_model_path_per_node(tmp_path):
    """Each node's container starts idle (`sleep infinity`) with its own
    model_path mounted at /model_cache — same asymmetry as vllm_dual.py."""
    rt = VLLMDualRayRuntime()
    with mock.patch.object(rt, "_workspace", return_value=str(tmp_path)):
        head = rt._container_cmd("mimo", RAY_YAML, head=True)
        worker = rt._container_cmd("mimo", RAY_YAML, head=False)
    assert "/home/u/models/lukealonso/MiMo-V2.5-NVFP4:/model_cache:ro" in head
    assert "/home/u/models-spark/lukealonso/MiMo-V2.5-NVFP4:/model_cache:ro" in worker
    for cmd in (head, worker):
        assert cmd[-2:] == ["sleep", "infinity"]     # idle, not vllm serve
        assert "--gpus" in cmd and "all" in cmd
        assert f"{tmp_path}:/workspace" in cmd


def test_serve_cmd_uses_ray_executor_backend():
    """The vllm serve line built for the head uses --distributed-executor-
    backend=ray (not vllm_dual.py's --nnodes/--node-rank multiproc split)."""
    rt = VLLMDualRayRuntime()
    cmd = rt._serve_cmd("mimo", RAY_YAML, master_addr="10.100.224.2")
    assert "--distributed-executor-backend=ray" in cmd
    assert "--tensor-parallel-size=2" in cmd
    assert "--served-model-name=mimo" in cmd
    assert "vllm serve /model_cache" in cmd
    assert "> /workspace/vllm.log 2>&1" in cmd
    # No --nnodes/--node-rank — those are vllm_dual.py's multiproc-only flags.
    assert "--nnodes" not in cmd
    assert "--node-rank" not in cmd


def test_serve_cmd_includes_serve_args():
    """Recipe-provided serve_args are appended to the built command."""
    rt = VLLMDualRayRuntime()
    yaml = dict(RAY_YAML, serve_args=["--enable-auto-tool-choice", "--tool-call-parser=mimo"])
    cmd = rt._serve_cmd("mimo", yaml, master_addr="10.100.224.2")
    assert "--enable-auto-tool-choice" in cmd
    assert "--tool-call-parser=mimo" in cmd


def test_serve_cmd_resets_stale_baked_env():
    """The exec that runs vllm serve unsets the image's stale
    RAY_OVERRIDE_NODE_IP_ADDRESS/RAY_NODE_IP_ADDRESS (baked in from a
    different cluster) and re-asserts the full NCCL/VLLM_HOST_IP wiring
    for the head — vLLM spawns its OWN fresh Ray worker actors from this
    exec's process tree, so env set inside the earlier ray-start exec
    (which doesn't carry over between separate `docker exec` calls) isn't
    enough; this command must set it all again itself."""
    rt = VLLMDualRayRuntime()
    cmd = rt._serve_cmd("mimo", RAY_YAML, master_addr="10.100.224.2")
    assert "unset RAY_OVERRIDE_NODE_IP_ADDRESS RAY_NODE_IP_ADDRESS" in cmd
    assert "VLLM_HOST_IP=10.100.224.2" in cmd
    assert "NCCL_SOCKET_IFNAME=enp1s0f1np1" in cmd
    assert "NCCL_IB_HCA=rocep1s0f1" in cmd


def test_nccl_env_exports_differ_per_side():
    """VLLM_HOST_IP (and GID index override) differ between head and worker,
    matching vllm_dual.py's per-node NCCL wiring for the same CX7 link."""
    rt = VLLMDualRayRuntime()
    head_env = rt._nccl_env_exports(RAY_YAML, head=True, master_addr="10.100.224.2")
    worker_env = rt._nccl_env_exports(RAY_YAML, head=False, master_addr="10.100.224.2")
    assert "VLLM_HOST_IP=10.100.224.2" in head_env
    assert "VLLM_HOST_IP=10.100.224.1" in worker_env
    assert "NCCL_IB_HCA=rocep1s0f1" in head_env
    assert "NCCL_IB_HCA=rocep1s0f1" in worker_env


# ---------------------------------------------------------------------------
# Edge: head/worker_extra_mounts asymmetry
# ---------------------------------------------------------------------------

def test_container_cmd_head_worker_extra_mounts_are_per_side(tmp_path):
    """head_extra_mounts / worker_extra_mounts apply to their own side only —
    same contract as vllm_dual.py's identically-named fields."""
    yaml = dict(RAY_YAML,
                head_extra_mounts=["/home/u/a:/extra:ro"],
                worker_extra_mounts=["/home/u/b:/extra:ro"])
    rt = VLLMDualRayRuntime()
    with mock.patch.object(rt, "_workspace", return_value=str(tmp_path)):
        head = rt._container_cmd("mimo", yaml, head=True)
        worker = rt._container_cmd("mimo", yaml, head=False)
    assert "/home/u/a:/extra:ro" in head
    assert "/home/u/a:/extra:ro" not in worker
    assert "/home/u/b:/extra:ro" not in head
    assert "/home/u/b:/extra:ro" in worker


# ---------------------------------------------------------------------------
# Failure: preflight refuses on missing config / unreachable worker
# ---------------------------------------------------------------------------

def test_preflight_missing_model_path():
    rt = VLLMDualRayRuntime()
    yaml = {k: v for k, v in RAY_YAML.items() if k != "model_path"}
    err = rt._preflight("mimo", yaml)
    assert err is not None
    assert "model_path" in err


def test_preflight_missing_worker_host():
    rt = VLLMDualRayRuntime()
    yaml = {k: v for k, v in RAY_YAML.items() if k != "worker_host"}
    err = rt._preflight("mimo", yaml)
    assert err is not None
    assert "worker_host" in err


def test_start_refuses_on_preflight_failure():
    """start() surfaces the preflight error as a dead RunningState without
    touching docker at all."""
    rt = VLLMDualRayRuntime()
    yaml = {k: v for k, v in RAY_YAML.items() if k != "master_addr"}
    with mock.patch("lmswitch.system.checks._docker_container", return_value=None), \
         mock.patch("subprocess.run") as run_mock:
        state = rt.start("mimo", yaml)
    assert state.status == "dead"
    assert "master_addr" in state.detail
    run_mock.assert_not_called()
