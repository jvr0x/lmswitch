"""Contract tests for the GLM-5.3-Flash EXL3 dual recipe.

Pins the port of MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks: serving values
re-synced to upstream E3 (2026-09-07/08) - 850k ctx, util 0.85, MNBT 7168,
grouped fat-expert kernels - over overlays vendored @ b5ab809. EXL3 (never
marlin), fp8 KV, DFlash2 k=7 draft TP=2, and the eight runtime overlays that
the local GHCR :exl3-instanttensor tag still lacks.

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
    "patch_hybrid_prefix_hit.py",
    "patch_xgrammar_termination.py",
    "patch_kpool_tail_slotmap.py",
    "patch_adaptive_k.py",
    "patch_ablit.py",
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

def test_recipe_tracks_upstream_e3_exl3_dflash2():
    """Expected: 850k ctx, EXL3 packed experts, fp8 KV, DFlash2 k=7, graphs on."""
    yaml = _recipe()
    assert yaml["runtime"] == "vllm-dual"
    assert yaml["enforce_eager"] is False
    assert yaml["max_num_seqs"] == 4
    assert yaml["tool_call_parser"] == "glm47"
    assert yaml["image"] == "ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor"

    env = yaml["env"]
    assert env["EXL3_FUSED_MOE"] == "1"
    assert env["GLM53_SUPPRESS_STOPS_IN_REASONING"] == "1"
    assert env["GLM53_MIXED_PREFILL_CHUNK"] == "fair"
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
    assert '"draft_tensor_parallel_size":2' in spec
    assert '"kv_cache_dtype":"auto"' in spec
    assert "/draft" in spec


def test_e3_fat_expert_knobs_are_explicit():
    """Expected: E3 tier pinned in env - lmswitch never runs upstream start.sh.

    ctx and gpu_memory_utilization are asserted HERE rather than alongside the
    tier-agnostic keys, so rolling back to E2 on a pre-2026-09-07 image is a
    one-line flip of EXL3_FAT_GROUPED and stays green: E3's 560 MiB fat-row
    scratch is charged to the KV budget, so 1M no longer fits at util <= 0.87.
    """
    yaml = _recipe()
    env = yaml["env"]
    assert env["EXL3_FAT_KERNEL"] == "1"
    assert env["GLM53_INDEXER_WORKSPACE"] in ("rightsize", "stock")
    args = yaml["extra_args"]
    mnbt = args[args.index("--max-num-batched-tokens") + 1]
    if env["EXL3_FAT_GROUPED"] == "1":
        assert yaml["ctx"] == 850000
        assert yaml["gpu_memory_utilization"] == 0.85
        assert env["EXL3_TEMP_ROWS_FUSED"] == "16"
        assert mnbt == "7168"
    else:
        assert yaml["ctx"] == 1000000
        assert yaml["gpu_memory_utilization"] == 0.87
        assert env["EXL3_TEMP_ROWS_FUSED"] == "256"
        assert mnbt == "2048"


def test_adaptive_k_has_its_graph_shapes_and_a_capped_pool():
    """Expected: adaptive-k on, with the capture list and KV cap it requires.

    The drafter still proposes 7; the scheduler verifies a 2/4/7 prefix, so the
    graph list must carry the 3- and 5-token shapes or every decode step falls
    out of its FULL graph. The cap is paired because 15 capture sizes plus the
    image:32 encoder cache are both uncounted against the KV budget.
    """
    yaml = _recipe()
    env = yaml["env"]
    assert env["GLM53_ADAPTIVE_K"] == "ema"
    ks = [int(x) for x in env["GLM53_ADAPTIVE_K_SET"].split(",")]
    assert ks == [2, 4, 7]

    args = yaml["extra_args"]
    start = args.index("--cudagraph-capture-sizes") + 1
    sizes = []
    for a in args[start:]:
        if a.startswith("--"):
            break
        sizes.append(int(a))
    for k in ks:
        assert k + 1 in sizes, f"missing uniform decode shape {k + 1} for k={k}"
    assert args[args.index("--kv-cache-memory-bytes") + 1] == "16106127360"

    head, worker, _ = _cmds()
    for cmd in (head, worker):
        joined = " ".join(cmd)
        assert "GLM53_ADAPTIVE_K=ema" in joined
        assert "patch_adaptive_k.py" in joined


def test_vision_carries_48_images_with_mm_profiling_skipped():
    """Expected: 48 images per prompt, and the init dummy MM profile skipped.

    The encoder cache is not charged against the KV budget, so the max-size
    dummy profile must stay off at this count or init OOMs the UMA.
    """
    yaml = _recipe()
    assert yaml["limit_mm_per_prompt"] == '{"image":48,"video":1}'
    assert "--skip-mm-profiling" in yaml["extra_args"]
    head, worker, _ = _cmds()
    for cmd in (head, worker):
        assert '--limit-mm-per-prompt={"image":48,"video":1}' in cmd
        assert "--skip-mm-profiling" in cmd


def test_dual_cmd_emits_recipe_ctx_and_runtime_overlays():
    """Expected: both ranks get the recipe ctx, overlay mount, E-tier env."""
    head, worker, yaml = _cmds()
    for cmd in (head, worker):
        assert f"--max-model-len={yaml['ctx']}" in cmd
        assert "--enforce-eager" not in cmd
        assert any(
            a.endswith("glm53-exl3-overlay:/opt/glm53-overlay:ro") for a in cmd
        )
        joined = " ".join(cmd)
        assert "EXL3_FUSED_MOE=1" in joined
        for key in ("EXL3_FAT_GROUPED", "EXL3_TEMP_ROWS_FUSED",
                    "GLM53_INDEXER_WORKSPACE"):
            assert f"{key}={yaml['env'][key]}" in joined
        assert "GLM53_MIXED_PREFILL_CHUNK=fair" in joined
        assert "GLM53_SUPPRESS_STOPS_IN_REASONING=1" in joined
        assert "TRITON_CACHE_DIR=/root/.triton/cache" in joined
        script = next(a for a in cmd if "patch_glm5_drafter_group.py" in a)
        for name in OVERLAY_SCRIPTS:
            assert name in script, f"entrypoint missing {name}"


# ---------------------------------------------------------------------------
# Edge
# ---------------------------------------------------------------------------

def test_overlay_dir_is_padded_slot_share_at_pinned_commit():
    """Edge: vendored overlays exist and still contain the slot-share patch.

    The overlays stay pinned at b5ab809 even though the serving values track
    E3: the E3 kernels ship in overlay/exl3.py + a rebuilt exllamav3_ext, both
    COPYd at image build, so no bind mount can deliver them.
    """
    source = (OVERLAY / "SOURCE.txt").read_text()
    assert "b5ab809" in source
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
