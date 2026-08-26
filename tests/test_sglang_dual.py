"""Tests for the sglang-dual (two-node TP=2) runtime.

Covers the three required cases per project convention:
- expected use: the head/worker split carries SGLang's multi-node flag surface,
  the per-node weights mount, and the CX7 NCCL env; start launches the worker
  over ssh before the local head.
- edge: a recipe-set tp_size is not duplicated, an over-native ctx opts into
  the SGLang override env, worker_env wins over env, and the container name is
  namespaced `sglang-` rather than colliding with the vLLM runtimes.
- failure: a failed head launch tears the worker back down, and missing yaml
  fields refuse to start.

All subprocess interaction is stubbed; nothing is launched and no GPU memory
is ever touched. Runs anywhere.
"""

from pathlib import Path
from unittest import mock

import lmswitch.cli as cli_mod
import lmswitch.runtimes.sglang_dual as dual_mod
from lmswitch.models.loader import load_models
from lmswitch.runtimes.base import runtime_registry
from lmswitch.runtimes.sglang_dual import SGLangDualRuntime
from lmswitch.system import checks as checks_mod
from lmswitch.system import memory as memory_mod
from lmswitch.system.io import _load_yaml

CONF = (Path(__file__).resolve().parents[1] / "ai-models"
        / "qwen3.8-flash-next-nvfp4-dual.yaml")

DUAL_YAML = {
    "runtime": "sglang-dual",
    "image": "lmsysorg/sglang:test",
    "model": "Org/Model-NVFP4",
    "model_path": "~/models-gigabyte/Org/Model-NVFP4",
    "worker_model_path": "~/models/Org/Model-NVFP4",
    "port": 8888,
    "ctx": 262144,
    "worker_host": "Gigabyte",
    "master_addr": "10.100.224.2",
    "worker_ip": "10.100.224.1",
    "master_port": 25000,
    "nccl": {"ifname": "enp1s0f1np1", "hca": "rocep1s0f1", "gid_index": 3},
    "mem_fraction_static": 0.80,
}


class _Result:
    def __init__(self, returncode=0):
        self.returncode = returncode


# ---------------------------------------------------------------------------
# Expected use
# ---------------------------------------------------------------------------

def test_dual_runtime_registered():
    """A registry miss silently falls back to llama, which would try to boot
    llama-server on a safetensors dir."""
    assert runtime_registry.lookup("sglang-dual") is SGLangDualRuntime


def test_node_cmd_head_vs_worker():
    rt = SGLangDualRuntime()
    head = rt._node_cmd("m", DUAL_YAML, node_rank=0)
    worker = rt._node_cmd("m", DUAL_YAML, node_rank=1)

    for cmd in (head, worker):
        pairs = list(zip(cmd, cmd[1:]))
        assert cmd[:2] == ["docker", "run"]
        assert "sglang-m" in cmd
        assert ("--nnodes", "2") in pairs
        assert ("--tp-size", "2") in pairs
        assert ("--dist-init-addr", "10.100.224.2:25000") in pairs
        assert ("--model-path", "/model") in pairs
        assert ("--served-model-name", "m") in pairs
        assert ("--context-length", "262144") in pairs
        assert ("--mem-fraction-static", "0.8") in pairs
        assert "--device" in cmd and "/dev/infiniband:/dev/infiniband" in cmd
        assert "NCCL_IB_HCA=rocep1s0f1" in cmd
        assert "NCCL_SOCKET_IFNAME=enp1s0f1np1" in cmd
        assert "GLOO_SOCKET_IFNAME=enp1s0f1np1" in cmd
        assert "NCCL_IB_GID_INDEX=3" in cmd
        # vLLM-only flags would make SGLang's argparse reject the launch.
        for banned in ("--gpu-memory-utilization", "--max-model-len",
                       "--tensor-parallel-size", "--headless",
                       "--master-addr", "--enable-auto-tool-choice"):
            assert banned not in cmd, f"{banned} is not an SGLang flag"

    assert "--node-rank" in head and head[head.index("--node-rank") + 1] == "0"
    assert "--node-rank" in worker and worker[worker.index("--node-rank") + 1] == "1"


def test_node_cmd_mounts_each_nodes_own_weights_at_one_path():
    """Host paths differ per node (owner's ~/models vs the peer's NFS view),
    but --model-path is one string, so both must land on /model."""
    rt = SGLangDualRuntime()
    head = rt._node_cmd("m", DUAL_YAML, node_rank=0)
    worker = rt._node_cmd("m", DUAL_YAML, node_rank=1)

    head_mounts = [head[i + 1] for i, a in enumerate(head) if a == "-v"]
    worker_mounts = [worker[i + 1] for i, a in enumerate(worker) if a == "-v"]
    assert any(m.endswith("models-gigabyte/Org/Model-NVFP4:/model:ro")
               for m in head_mounts)
    assert any(m.endswith("models/Org/Model-NVFP4:/model:ro")
               and "models-gigabyte" not in m for m in worker_mounts)
    # ~ must be expanded — docker does not do it for bind mounts.
    assert not any(m.startswith("~") for m in head_mounts + worker_mounts)


def test_start_launches_worker_before_head():
    """The worker has to be waiting on dist_init_addr when the head's
    rendezvous runs, or the TP group never forms."""
    calls = []

    def fake_ssh(host, cmd, **kw):
        calls.append(("ssh", host, list(cmd)))
        return _Result(0)

    def fake_run(cmd, **kw):
        calls.append(("local", None, list(cmd)))
        return _Result(0)

    with mock.patch.object(dual_mod, "_ssh", side_effect=fake_ssh), \
         mock.patch.object(dual_mod.subprocess, "run", side_effect=fake_run), \
         mock.patch.object(SGLangDualRuntime, "_preflight", return_value=None), \
         mock.patch.object(SGLangDualRuntime, "_setup_logging",
                           return_value=Path("/dev/null")), \
         mock.patch.object(dual_mod, "_wait_ready", return_value="ready"), \
         mock.patch.object(checks_mod, "_docker_container",
                           side_effect=[None, "abc123456789", "abc123456789"]):
        state = SGLangDualRuntime().start("m", DUAL_YAML)

    assert state.status == "ready"
    runs = [c for c in calls if "run" in c[2] and "docker" in c[2]]
    assert runs[0][0] == "ssh" and "--node-rank" in runs[0][2]
    assert runs[0][2][runs[0][2].index("--node-rank") + 1] == "1"
    assert runs[1][0] == "local"
    assert runs[1][2][runs[1][2].index("--node-rank") + 1] == "0"


def test_shipped_recipe_builds():
    """The recipe that ships with this runtime must actually parse and build,
    so a typo in it fails here rather than at 2 a.m. on the head node."""
    cfg = _load_yaml(CONF)
    assert cfg["runtime"] == "sglang-dual"
    cmd = SGLangDualRuntime()._node_cmd("qwen3.8-flash-next-nvfp4-dual", cfg,
                                        node_rank=0)
    pairs = list(zip(cmd, cmd[1:]))
    assert ("--tp-size", "2") in pairs
    assert ("--quantization", "modelopt_fp4") in pairs
    assert ("--fp4-gemm-backend", "flashinfer_cutlass") in pairs
    assert ("--page-size", "64") in pairs
    assert ("--mamba-scheduler-strategy", "extra_buffer") in pairs
    assert ("--chunked-prefill-size", "4096") in pairs
    assert ("--max-running-requests", "36") in pairs
    assert "--allow-auto-truncate" in cmd
    assert ("--port", "8888") in pairs


def test_loader_reports_dual_host_and_sglang_type(lmswitch_data_dir):
    """The table must show this as an sglang backend on a "dual" host, or the
    -l/-d view filters and the RAM row misattribute it."""
    (lmswitch_data_dir / "m.yaml").write_text(
        "runtime: sglang-dual\n"
        "image: img\n"
        f"model_path: \"{lmswitch_data_dir}\"\n"
        "port: 8888\n"
    )
    row = next(m for m in load_models() if m["name"] == "m")
    assert row["type"] == "sglang"
    assert row["host"] == "dual"
    assert cli_mod._filter_models([row], view="dual", show_missing=True) == [row]
    assert cli_mod._filter_models([row], view="local", show_missing=True) == []


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_recipe_tp_size_is_not_duplicated():
    """_sglang_args already emits --tp-size when the recipe sets it; a second
    copy from the runtime default would leave the effective config ambiguous."""
    cmd = SGLangDualRuntime()._node_cmd("m", {**DUAL_YAML, "tp_size": 2},
                                        node_rank=0)
    assert cmd.count("--tp-size") == 1
    # Absent from the recipe, the runtime still has to supply it.
    no_tp = {k: v for k, v in DUAL_YAML.items() if k != "tp_size"}
    assert SGLangDualRuntime()._node_cmd("m", no_tp, node_rank=0).count("--tp-size") == 1


def test_over_native_ctx_opts_into_override_env():
    """Without the env var SGLang silently clamps back to the native 262144,
    serving a shorter context than the recipe asked for."""
    long_ctx = {**DUAL_YAML, "ctx": 1048576}
    cmd = SGLangDualRuntime()._node_cmd("m", long_ctx, node_rank=0)
    assert "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1" in cmd
    assert "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1" not in \
        SGLangDualRuntime()._node_cmd("m", DUAL_YAML, node_rank=0)


def test_worker_env_overrides_shared_env():
    """Docker applies the last -e for a key, so worker_env must come after."""
    cfg = {**DUAL_YAML, "env": {"FOO": "shared"},
           "worker_env": {"FOO": "worker-only"}}
    worker = SGLangDualRuntime()._node_cmd("m", cfg, node_rank=1)
    head = SGLangDualRuntime()._node_cmd("m", cfg, node_rank=0)
    assert worker.index("FOO=shared") < worker.index("FOO=worker-only")
    assert "FOO=worker-only" not in head


def test_side_specific_mounts_stay_on_their_own_node():
    cfg = {**DUAL_YAML,
           "head_extra_mounts": ["/h/a:/drafter:ro"],
           "worker_extra_mounts": ["/w/a:/drafter:ro"]}
    head = SGLangDualRuntime()._node_cmd("m", cfg, node_rank=0)
    worker = SGLangDualRuntime()._node_cmd("m", cfg, node_rank=1)
    assert "/h/a:/drafter:ro" in head and "/w/a:/drafter:ro" not in head
    assert "/w/a:/drafter:ro" in worker and "/h/a:/drafter:ro" not in worker


def test_container_namespace_is_sglang_not_vllm():
    """Same model wired for both backends must never collide on a container
    name, and the liveness check has to look under the right prefix."""
    assert checks_mod._container_prefix("sglang-dual") == "sglang"
    assert "sglang-dual" in checks_mod._DOCKER_BACKED_RUNTIMES
    cmd = SGLangDualRuntime()._node_cmd("m", DUAL_YAML, node_rank=0)
    assert cmd[cmd.index("--name") + 1] == "sglang-m"


def test_memory_guard_sizes_from_mem_fraction_static():
    """Sizing a dual sglang recipe from the weights instead would wave through
    a start that grabs ~97Gi of a 121Gi box on each node."""
    with mock.patch.object(memory_mod, "_ram_line", return_value=(121.0, 4.0, 117.0)):
        ok, why = memory_mod._memory_check("m", DUAL_YAML)
        assert ok, why
    with mock.patch.object(memory_mod, "_ram_line", return_value=(121.0, 110.0, 11.0)):
        ok, why = memory_mod._memory_check("m", DUAL_YAML)
        assert not ok
        assert "mem_fraction_static=0.8" in why and "11Gi free" in why


# ---------------------------------------------------------------------------
# Failure cases
# ---------------------------------------------------------------------------

def test_failed_head_launch_tears_down_worker():
    """A live rank 1 holds the peer's GPU; the next start could not bind the
    TP group with a stale worker still up."""
    ssh_calls = []

    def fake_ssh(host, cmd, **kw):
        ssh_calls.append(list(cmd))
        return _Result(0)

    with mock.patch.object(dual_mod, "_ssh", side_effect=fake_ssh), \
         mock.patch.object(dual_mod.subprocess, "run",
                           side_effect=lambda cmd, **kw: _Result(
                               1 if "docker" in cmd and "run" in cmd else 0)), \
         mock.patch.object(SGLangDualRuntime, "_preflight", return_value=None), \
         mock.patch.object(checks_mod, "_docker_container", return_value=None):
        state = SGLangDualRuntime().start("m", DUAL_YAML)

    assert state.status == "dead"
    # start() also issues a pre-launch cleanup rm on the worker, so matching
    # "some rm happened" would pass with the teardown branch deleted. Two rms
    # is the only shape that proves the failure path ran.
    rms = [c for c in ssh_calls if c[:3] == ["docker", "container", "rm"]
           and "sglang-m" in c]
    assert len(rms) == 2, "teardown after failed head launch did not run"


def test_preflight_refuses_missing_fields_and_unreachable_worker():
    rt = SGLangDualRuntime()
    for missing in ("image", "model_path", "worker_host", "master_addr"):
        cfg = {k: v for k, v in DUAL_YAML.items() if k != missing}
        assert rt._preflight("m", cfg) == f"missing required yaml field: {missing}"

    with mock.patch.object(dual_mod, "_ssh", return_value=_Result(255)):
        assert rt._preflight("m", DUAL_YAML) == "worker unreachable over ssh: Gigabyte"


def test_preflight_refuses_when_image_missing_on_a_node():
    rt = SGLangDualRuntime()
    with mock.patch.object(dual_mod, "_ssh", return_value=_Result(0)), \
         mock.patch.object(dual_mod.subprocess, "run", return_value=_Result(1)):
        assert rt._preflight("m", DUAL_YAML) == \
            "image lmsysorg/sglang:test missing on local"
    with mock.patch.object(dual_mod, "_ssh",
                           side_effect=[_Result(0), _Result(1)]), \
         mock.patch.object(dual_mod.subprocess, "run", return_value=_Result(0)):
        assert rt._preflight("m", DUAL_YAML) == \
            "image lmsysorg/sglang:test missing on Gigabyte"


def test_stop_drops_both_ranks():
    ssh_calls = []
    with mock.patch.object(dual_mod, "_ssh",
                           side_effect=lambda h, c, **kw: ssh_calls.append(list(c))), \
         mock.patch.object(checks_mod, "_docker_container", return_value="abc123456789"), \
         mock.patch("lmswitch.runtimes.sglang.subprocess.run", return_value=_Result(0)):
        SGLangDualRuntime().stop("m", DUAL_YAML)
    assert any("sglang-m" in c for c in ssh_calls), "worker rank was left running"
