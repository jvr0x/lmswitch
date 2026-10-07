"""RAM checking and memory guard logic."""

from lmswitch.system.io import _model_size_and_present


def _ram_line():
    """Parse /proc/meminfo and return (total, used, avail) in GiB or None."""
    info: dict[str, int] = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                info[k] = int(v.strip().split()[0])
    except Exception:
        return None
    total = info.get("MemTotal", 0) / 1024 ** 2
    avail = info.get("MemAvailable", 0) / 1024 ** 2
    return total, total - avail, avail


def _memory_check(name: str, yaml: dict) -> tuple[bool, str]:
    """Estimates whether a model fits in available RAM before launching it.

    Returns ``(ok, reason)``; ``ok`` is False when the estimated footprint
    exceeds what's free, so the caller can refuse rather than OOM the box.
    """
    ram = _ram_line()
    if not ram:
        return True, ""  # Can't measure (non-Linux); don't block.
    total, _used, avail = ram
    runtime = yaml.get("runtime", "llama")
    # Reason: SGLang pre-allocates a static fraction of memory up front
    # (--mem-fraction-static, 0.95 in the DGX Spark recipe) exactly like
    # vLLM's gpu_memory_utilization — sizing it from the weights instead
    # would wave through a start that grabs ~121Gi of a 128Gi box and
    # hard-resets it.
    if runtime in ("sglang", "sglang-dual"):
        try:
            frac = float(yaml.get("mem_fraction_static", 0.95))
        except (ValueError, TypeError):
            frac = 0.95
        need = frac * total
        what = f"SGLang reserves ~{need:.0f}Gi (mem_fraction_static={frac})"
        return (avail >= need, "" if avail >= need
                else f"{what}, but only {avail:.0f}Gi free")
    # Reason: dual runtimes reserve gpu_memory_utilization on THIS node too
    # (each node holds its TP shard), so the same estimate applies per-node.
    if runtime in ("vllm", "vllm-dual", "vllm-dual-ray"):
        default_util = 0.80 if runtime in ("vllm-dual", "vllm-dual-ray") else 0.15
        try:
            util = float(yaml.get("gpu_memory_utilization", default_util))
        except (ValueError, TypeError):
            util = default_util
        need = util * total
        what = f"vLLM reserves ~{need:.0f}Gi (gpu_memory_utilization={util})"
    else:
        size, _present = _model_size_and_present(yaml.get("model", ""), runtime)
        need = size / 1024 ** 3 * 1.3
        what = f"~{need:.0f}Gi (weights + headroom)"
        # Reason: llama-dual keeps only this node's tensor_split share
        # resident — the rest is pushed to the worker's rpc-server — so the
        # whole-file estimate would refuse every recipe this runtime exists
        # for.
        if runtime == "llama-dual":
            from lmswitch.runtimes.llama_dual import local_share
            share = local_share(yaml)
            need = size / 1024 ** 3 * share * 1.3
            what = (f"~{need:.0f}Gi (local {share:.0%} share of the "
                    f"weights + headroom)")
        # Reason: exllamav3-dual is pipeline-parallel — each rank loads only
        # its own layer slice (plus embed on rank 0, head on rank 1), so
        # size the local slice from `splits`, not the whole checkpoint.
        if runtime == "exllamav3-dual":
            from lmswitch.runtimes.exllamav3_dual import pipeline_share
            from lmswitch.system.io import _dir_size_and_present
            import os
            from pathlib import Path
            mp = yaml.get("model_path", "")
            dsize, _present = _dir_size_and_present(
                Path(os.path.expanduser(os.path.expandvars(str(mp)))))
            # Fall back to the whole-checkpoint estimate when the dir isn't
            # visible yet (e.g. NFS mount still coming up at boot).
            base_gib = dsize / 1024 ** 3 if dsize else size / 1024 ** 3
            share = pipeline_share(yaml)
            need = base_gib * share * 1.3
            what = (f"~{need:.0f}Gi (local {share:.0%} pipeline slice of "
                    f"the weights + headroom)")
    if avail < need:
        return False, f"{what}, but only {avail:.0f}Gi free"
    return True, ""
