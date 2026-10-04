"""Contract tests for the DeepSeek-V4.1-Flash EXL3 dual recipes.

Pins the port of MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks (main @
ef3aa8f, 2026-09-20): EXL3 2.9bpw at 600k over TP=2, file-backed Engram, the
twelve runtime overlays of upstream's start.sh order, and the agent-tuned
defaults that landed with 14ebc39 / 0b5654d.

The regression these lock down: both lanes previously bind-mounted the overlay
directory onto /opt/dsv41, the image's own overlay path, which hid the
librow_store.so the Dockerfile compiles there and killed every boot with
"FileNotFoundError: librow_store.so missing". The overlay must land on a
distinct path and be copied in by the entrypoint.

Nothing is launched. GPU memory is never touched.
"""

from pathlib import Path

import pytest

from lmswitch.runtimes.vllm_dual import VLLMDualRuntime
from lmswitch.system.io import _load_yaml


REPO = Path(__file__).resolve().parents[1]
CONF_DIR = REPO / "ai-models"
OVERLAY = CONF_DIR / "dsv41-exl3-overlay"

LANES = ("deepseek-v4.1-flash-exl3-dual", "deepseek-v4.1-flash-exl3-ablit-dual")

# upstream start.sh, head and worker alike, in order.
OVERLAY_ORDER = (
    "patch_suppress_stops_in_reasoning",
    "patch_scheduler_decode_floor",
    "patch_xgrammar_termination",
    "patch_spinwait",
    "patch_responses_content_types",
    "patch_exl3_packed_names",
    "patch_exl3_lm_head",
    "patch_engram_secondary",
    "patch_engram_file",
    "patch_h2d_stage",
    "patch_sm120_block64",
    "patch_memory_log",
)

# Built or installed by the Dockerfile only, never re-run by start.sh. A
# directory mount over /opt/dsv41 hides all three.
IMAGE_ONLY_ASSETS = ("librow_store.so", "chat_template.jinja", "exl3_k_map.json")


def _recipe(lane: str) -> dict:
    """Loads one shipped EXL3 dual YAML."""
    return _load_yaml(CONF_DIR / f"{lane}.yaml")


def _cmds(lane: str) -> tuple[list[str], list[str], dict]:
    """Builds head and worker docker argv from the shipped recipe."""
    yaml = _recipe(lane)
    rt = VLLMDualRuntime()
    return rt._node_cmd(lane, yaml, 0), rt._node_cmd(lane, yaml, 1), yaml


# ---------------------------------------------------------------------------
# Expected use
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lane", LANES)
def test_recipe_tracks_upstream_exl3_engram_dspark(lane: str):
    """Expected: 600k ctx, EXL3, file-backed Engram, DSpark k=3, graphs on."""
    yaml = _recipe(lane)
    assert yaml["runtime"] == "vllm-dual"
    assert yaml["ctx"] == 600000
    assert yaml["max_num_seqs"] == 2
    assert yaml["enforce_eager"] is False
    assert yaml["port"] == 8888
    assert yaml["image"] == (
        "ghcr.io/miaai-lab/deepseek-v4.1-flash-exl3-2x-dgx-sparks:2.9bpw"
    )

    env = yaml["env"]
    assert env["ENGRAM_MOUNT"] == "/engram-src"
    assert env["DSV41_SOURCE"] == "/engram-src"
    assert env["EXL3_FUSED_MOE"] == "1"
    assert env["EXL3_FAT_GROUPED"] == "1"
    assert env["EXL3_FAT_KERNEL"] == "0"
    assert env["EXL3_TEMP_ROWS_FUSED"] == "16"
    assert env["DSV41_EXL3_SERIAL_STREAMS"] == "1"
    assert env["VLLM_DISABLE_SHARED_EXPERTS_STREAM"] == "1"

    args = yaml["extra_args"]
    assert args[args.index("--quantization") + 1] == "exl3"
    assert args[args.index("--distributed-executor-backend") + 1] == "mp"
    assert args[args.index("--block-size") + 1] == "64"
    assert args[args.index("--kv-cache-memory-bytes") + 1] == "2684354560"
    assert args[args.index("--hf-overrides") + 1] == (
        '{"engram_table_dir":"/engram-src"}'
    )
    spec = args[args.index("--speculative-config") + 1]
    assert '"method":"dspark"' in spec
    assert '"num_speculative_tokens":3' in spec


@pytest.mark.parametrize("lane", LANES)
def test_agent_tuned_defaults_match_ef3aa8f(lane: str):
    """Expected: the 2026-09-20 upstream resync, not the 2026-09-16 port.

    14ebc39 dropped LONG_PREFILL_TOKEN_THRESHOLD (its 1792 figure was measured
    at MNBT=2048, not the 1536 shipped here) and raised DSV41_IO_THREADS to 96;
    0b5654d added the prefix-cache retention interval.
    """
    yaml = _recipe(lane)
    args = yaml["extra_args"]
    assert "--long-prefill-token-threshold" not in args
    assert args[args.index("--max-num-batched-tokens") + 1] == "1536"
    assert yaml["env"]["DSV41_IO_THREADS"] == "96"

    retention = int(args[args.index("--prefix-cache-retention-interval") + 1])
    # vLLM rejects anything that is not a multiple of 128, capped at 1 MiB.
    assert retention == 4096
    assert retention % 128 == 0
    assert 0 < retention <= 1048576


@pytest.mark.parametrize("lane", LANES)
def test_serving_surface_carries_the_v41_parsers(lane: str):
    """Expected: reasoning and tool output are parsed, not served raw."""
    args = _recipe(lane)["extra_args"]
    for flag in ("--tokenizer-mode", "--tool-call-parser", "--reasoning-parser"):
        assert args[args.index(flag) + 1] == "deepseek_v41"
    for flag in (
        "--enable-auto-tool-choice",
        "--enable-prefix-caching",
        "--enable-prompt-tokens-details",
        "--no-enable-flashinfer-autotune",
    ):
        assert flag in args
    assert args[args.index("--chat-template") + 1] == "/opt/dsv41/chat_template.jinja"


@pytest.mark.parametrize("lane", LANES)
def test_vision_stays_on_with_the_dummy_profile_skipped(lane: str):
    """Expected: vision tower on, encoder data-parallel, init profile off.

    8404ac7 made vision the default. The encoder cache is not charged against
    the pinned KV budget, so the max-size dummy profile must stay off at
    image:100 or init OOMs the UMA.
    """
    yaml = _recipe(lane)
    args = yaml["extra_args"]
    assert yaml["env"]["LANGUAGE_MODEL_ONLY"] == "0"
    assert args[args.index("--mm-encoder-tp-mode") + 1] == "data"
    assert args[args.index("--limit-mm-per-prompt") + 1] == '{"image":100}'
    assert "--skip-mm-profiling" in args
    assert "--language-model-only" not in args


@pytest.mark.parametrize("lane", LANES)
def test_entrypoint_runs_upstream_overlay_order(lane: str):
    """Expected: start.sh's twelve patches, in order, over a copied overlay."""
    entry = _recipe(lane)["entrypoint"]
    assert entry[0] == "bash" and entry[1] == "-c"
    script = entry[2]

    assert "cp -f /opt/dsv41-overlay/*.py /opt/dsv41/" in script
    assert "model_executor/layers/quantization/exl3.py" in script
    assert script.endswith('exec vllm serve "$@"')

    positions = [script.index(p) for p in OVERLAY_ORDER]
    assert positions == sorted(positions), "patch order drifted from start.sh"

    # Build-time only in the Dockerfile; start.sh never re-runs them.
    for skipped in (
        "patch_model_overrides",
        "patch_exl3_ext_aarch64",
        "patch_exl3_fat_kernel",
    ):
        assert f"{skipped}.py" not in script


@pytest.mark.parametrize("lane", LANES)
def test_dual_cmd_mounts_engram_per_node_and_overlay_on_both(lane: str):
    """Expected: one engram path per node, identical overlay on both ranks."""
    head, worker, yaml = _cmds(lane)
    for cmd in (head, worker):
        assert f"--max-model-len={yaml['ctx']}" in cmd
        assert "--enforce-eager" not in cmd
        assert any(
            a.endswith("dsv41-exl3-overlay:/opt/dsv41-overlay:ro") for a in cmd
        )
        assert any(a.endswith(":/engram-src:ro") for a in cmd)
        joined = " ".join(cmd)
        assert "DSV41_IO_THREADS=96" in joined
        assert "patch_responses_content_types" in joined

    def engram_src(cmd: list[str]) -> str:
        return next(a for a in cmd if a.endswith(":/engram-src:ro")).split(":")[0]

    assert "models-gigabyte" in engram_src(head)
    assert "models-gigabyte" not in engram_src(worker)


def test_ablit_lane_keeps_its_deltas():
    """Expected: only checkpoint, util and master_port differ from stock."""
    stock, ablit = _recipe(LANES[0]), _recipe(LANES[1])

    assert ablit["model"].endswith("-Abliterated")
    assert ablit["model_path"].endswith("-Abliterated")
    assert ablit["worker_model_path"].endswith("-Abliterated")
    # INSTALL.md caps this lane lower than stock, and a distinct rendezvous
    # port keeps a stale stock peer from being joined by accident.
    assert ablit["gpu_memory_utilization"] == 0.85
    assert stock["gpu_memory_utilization"] == 0.88
    assert ablit["master_port"] == 29523
    assert stock["master_port"] == 29521
    # The ablation edits attention wo_b only; Engram stays the native shards.
    assert ablit["head_extra_mounts"] == stock["head_extra_mounts"]
    assert ablit["worker_extra_mounts"] == stock["worker_extra_mounts"]

    for key in ("image", "entrypoint", "extra_args", "extra_mounts", "ctx", "port"):
        assert ablit[key] == stock[key], f"{key} drifted between the two lanes"


# ---------------------------------------------------------------------------
# Edge case
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lane", LANES)
def test_node_local_caches_are_never_shared(lane: str):
    """Edge: two ranks compiling into one cache dir trample each other.

    The recipe mounts host paths that are identical *by name* on both nodes;
    that is fine because each node has its own ~/.cache. What must not happen
    is a single cache path served over the NFS link.
    """
    yaml = _recipe(lane)
    caches = [m for m in yaml["extra_mounts"] if "/root/.cache" in m or "/root/.triton" in m or "/root/.tilelang" in m]
    assert len(caches) == 3
    for mount in caches:
        host = mount.split(":")[0]
        assert host.startswith("~/.cache/vllm-dsv41-exl3")
        assert "models-gigabyte" not in host
    assert yaml["env"]["VLLM_CACHE_ROOT"] == "/root/.cache/vllm"
    assert yaml["env"]["TRITON_CACHE_DIR"] == "/root/.triton/cache"


@pytest.mark.parametrize("lane", LANES)
def test_every_patch_in_the_order_exists_on_disk(lane: str):
    """Edge: the entrypoint skips a missing patch with a warning.

    A patch present on the head and absent on the worker diverges the ranks
    silently, so the overlay directory must carry all twelve.
    """
    script = _recipe(lane)["entrypoint"][2]
    for patch in OVERLAY_ORDER:
        # The loop names patches bare and appends .py inside the container.
        assert patch in script
        assert (OVERLAY / f"{patch}.py").is_file(), f"{patch}.py missing from overlay"
    assert (OVERLAY / "exl3.py").is_file()
    assert (OVERLAY / "engram_file_backend.py").is_file()


# ---------------------------------------------------------------------------
# Failure case
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lane", LANES)
def test_overlay_never_shadows_the_image_overlay_dir(lane: str):
    """Failure: mounting onto /opt/dsv41 hides the compiled librow_store.so.

    This is the 2026-09-20 boot failure. The image builds librow_store.so from
    row_store.cpp and ships chat_template.jinja and exl3_k_map.json in the same
    directory; a directory mount there replaces all of it and the Engram file
    backend dies at ParallelEngramEmbedding init.
    """
    yaml = _recipe(lane)
    mounts = (
        list(yaml["extra_mounts"])
        + list(yaml["head_extra_mounts"])
        + list(yaml["worker_extra_mounts"])
    )
    for mount in mounts:
        target = mount.split(":")[1]
        assert target != "/opt/dsv41", (
            "overlay mounted over the image's own /opt/dsv41 — this hides "
            f"{', '.join(IMAGE_ONLY_ASSETS)}"
        )

    head, worker, _ = _cmds(lane)
    for cmd in (head, worker):
        for arg in cmd:
            assert ":/opt/dsv41:ro" not in arg
            assert ":/opt/dsv41:rw" not in arg

    # The assets must come from the image, so the host overlay must not be
    # expected to provide them either.
    for asset in IMAGE_ONLY_ASSETS:
        assert not (OVERLAY / asset).exists(), (
            f"{asset} staged on the host — the image's copy is authoritative"
        )


@pytest.mark.parametrize("lane", LANES)
def test_thinking_defaults_off(lane: str):
    args = _recipe(lane)["extra_args"]
    assert args[args.index("--default-chat-template-kwargs") + 1] == '{"enable_thinking": false}'
    head, worker, _ = _cmds(lane)
    for cmd in (head, worker):
        assert cmd[cmd.index("--default-chat-template-kwargs") + 1] == '{"enable_thinking": false}'
