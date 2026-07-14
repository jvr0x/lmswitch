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
    # Reason: vllm-dual reserves gpu_memory_utilization on THIS node too (each
    # node holds its TP shard), so the same estimate applies per-node.
    if runtime in ("vllm", "vllm-dual"):
        default_util = 0.80 if runtime == "vllm-dual" else 0.15
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
    if avail < need:
        return False, f"{what}, but only {avail:.0f}Gi free"
    return True, ""
