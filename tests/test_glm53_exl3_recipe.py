"""Contract tests for the GLM-5.3-Flash EXL3 dual recipe.

Pins the port of MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks @ 128da200:
1M ctx after padded slot-share, EXL3 (never marlin), fp8 KV, DFlash2 k=7,
and the four runtime overlays that the local GHCR :exl3 tag still lacks.

Nothing is launched. GPU memory is never touched.
"""

from pathlib import Path

from lmswitch.runtimes.vllm_dual import VLLMDualRuntime
from lmswitch.system.io import _load_yaml


REPO = Path(__file__).resolve().parents[1]
CONF = REPO / "ai-models" / "glm5.3-flash-exl3-dual.yaml"
OVERLAY = REPO / "ai-models" / "glm53-exl3-overlay"

OVERLAY_SCRIPTS = (
    "patch_glm_video_placeholders.py",
    "patch_suppress_stops_in_reasoning.py",
    "patch_scheduler_decode_floor.py",
    "patch_glm5_drafter_group.py",
)


def _recipe() -> dict:
    """Loads the shipped EXL3 dual YAML."""
    return _load_yaml(CONF)


def _cmds() -> tuple[list[str], list[str], dict]:
    """Builds head and worker docker argv from the shipped recipe."""
    yaml = _recipe()
    rt = VLLMDualRuntime()
    return rt._node_cmd("glm5.3-flash-exl3-dual", yaml, 0), rt._node_cmd(
        "glm5.3-flash-exl3-dual", yaml, 1
    ), yaml


# ---------------------------------------------------------------------------
# Expected use
# ---------------------------------------------------------------------------

def test_recipe_tracks_upstream_1m_exl3_dflash2():
    """Expected: 1M ctx, EXL3 packed experts, fp8 KV, DFlash2 k=7, graphs on."""
    yaml = _recipe()
    assert yaml["runtime"] == "vllm-dual"
    assert yaml["ctx"] == 1000000
    assert yaml["enforce_eager"] is False
    assert yaml["gpu_memory_utilization"] == 0.87
    assert yaml["max_num_seqs"] == 4
    assert yaml["tool_call_parser"] == "glm47"
    assert yaml["image"] == "ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3"

    env = yaml["env"]
    assert env["EXL3_FUSED_MOE"] == "1"
    assert env["GLM53_SUPPRESS_STOPS_IN_REASONING"] == "1"
    assert env["GLM53_MIXED_PREFILL_CHUNK"] == "skip"
    assert env["VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"] == "1800"
    assert env["TRITON_CACHE_DIR"] == "/root/.triton/cache"
    assert env["TILELANG_CACHE_DIR"] == "/root/.tilelang/cache"

    args = yaml["extra_args"]
    assert args[args.index("--quantization") + 1] == "exl3"
    assert args[args.index("--kv-cache-dtype") + 1] == "fp8"
    assert args[args.index("--distributed-executor-backend") + 1] == "mp"
    spec = args[args.index("--speculative-config") + 1]
    assert '"method":"dflash"' in spec
    assert '"num_speculative_tokens":7' in spec
    assert '"draft_tensor_parallel_size":1' in spec
    assert '"kv_cache_dtype":"auto"' in spec
    assert "/draft" in spec


def test_dual_cmd_emits_1m_and_runtime_overlays():
    """Expected: both ranks get --max-model-len=1000000, overlay mount, env."""
    head, worker, _ = _cmds()
    for cmd in (head, worker):
        assert "--max-model-len=1000000" in cmd
        assert "--enforce-eager" not in cmd
        assert any(
            a.endswith("glm53-exl3-overlay:/opt/glm53-overlay:ro") for a in cmd
        )
        joined = " ".join(cmd)
        assert "EXL3_FUSED_MOE=1" in joined
        assert "GLM53_MIXED_PREFILL_CHUNK=skip" in joined
        assert "GLM53_SUPPRESS_STOPS_IN_REASONING=1" in joined
        assert "TRITON_CACHE_DIR=/root/.triton/cache" in joined
        script = next(a for a in cmd if "patch_glm5_drafter_group.py" in a)
        for name in OVERLAY_SCRIPTS:
            assert name in script, f"entrypoint missing {name}"


# ---------------------------------------------------------------------------
# Edge
# ---------------------------------------------------------------------------

def test_overlay_dir_is_padded_slot_share_at_pinned_commit():
    """Edge: vendored overlays exist and still contain the 1M-allocating patch."""
    source = (OVERLAY / "SOURCE.txt").read_text()
    assert "128da200b0319c3f11d59e2eeae5d777c5d7be48" in source
    for name in OVERLAY_SCRIPTS:
        path = OVERLAY / name
        assert path.is_file(), f"missing overlay {name}"
        assert path.stat().st_size > 0
    drafter = (OVERLAY / "patch_glm5_drafter_group.py").read_text()
    assert "padded slot-share" in drafter
    assert "page_size_padded=mla_page" in drafter or "page_size_padded" in drafter
    sched = (OVERLAY / "patch_scheduler_decode_floor.py").read_text()
    assert "GLM53_MIXED_PREFILL_CHUNK" in sched


def test_draft_mount_is_per_side_overlay_is_shared():
    """Edge: DFlash2 host path differs per node; overlay path does not."""
    head, worker, yaml = _cmds()
    assert any("models-gigabyte/incoai/GLM-5.3-Flash-DFlash2:/draft:ro" in a
               for a in head)
    assert not any("/models/incoai/GLM-5.3-Flash-DFlash2:/draft:ro" in a
                   and "models-gigabyte" not in a for a in head)
    assert any("/models/incoai/GLM-5.3-Flash-DFlash2:/draft:ro" in a
               for a in worker)
    assert not any("models-gigabyte/incoai/GLM-5.3-Flash-DFlash2" in a
                   for a in worker)
    overlay_spec = "~/utils/lmswitch/ai-models/glm53-exl3-overlay:/opt/glm53-overlay:ro"
    assert overlay_spec in yaml["extra_mounts"]
    for cmd in (head, worker):
        assert any(a.endswith("glm53-exl3-overlay:/opt/glm53-overlay:ro")
                   for a in cmd)


# ---------------------------------------------------------------------------
# Failure
# ---------------------------------------------------------------------------

def test_recipe_rejects_nvfp4_sibling_flags():
    """Failure: marlin / nvfp4 / bf16 KV / TRITON_ATTN would serve the wrong model."""
    yaml = _recipe()
    blob = " ".join(str(x) for x in yaml["extra_args"])
    assert "marlin" not in blob
    assert "nvfp4" not in blob.lower()
    assert "bf16" not in blob.lower()
    assert "TRITON_ATTN" not in blob
    assert yaml["extra_args"][yaml["extra_args"].index("--kv-cache-dtype") + 1] != "bf16"
    head, worker, _ = _cmds()
    for cmd in (head, worker):
        joined = " ".join(cmd)
        assert "--moe-backend" not in joined
        assert "marlin" not in joined
        assert "--kv-cache-dtype nvfp4" not in joined
        assert "--kv-cache-dtype=nvfp4" not in joined
