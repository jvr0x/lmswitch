"""llama-server (GGUF) runtime."""

import shlex
import subprocess
import time
from pathlib import Path

from lmswitch.system.io import RUN_DIR, SCRIPT_DIR
from lmswitch.runtimes.wait import _wait_ready


def _extra_args(yaml: dict) -> list[str]:
    """Raw flags appended verbatim to the server command line.

    Lets ANY llama-server / vllm flag be set from the config, beyond the
    first-class keys above. Accepts either a YAML list (one argv token per
    item) or a single string (split with shell-style quoting), e.g.:

        extra_args: ["-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0"]
        extra_args: "--temp 0.7 --top-p 0.9 --jinja"
    """
    raw = yaml.get("extra_args") or []
    if isinstance(raw, str):
        return shlex.split(raw)
    return [str(x) for x in raw]


def _start_llama_direct(name: str, yaml: dict) -> None:
    models_dir = yaml.get("_models_dir")
    if models_dir is None:
        from lmswitch.system.io import _models_dir
        models_dir = _models_dir()
    model_path = models_dir / yaml["model"]
    port = yaml.get("port", 8081)
    ctx = yaml.get("ctx", 65536)
    gpu_layers = yaml.get("gpu_layers", 99)
    threads = yaml.get("threads", 12)
    batch = yaml.get("batch", 1024)
    ubatch = yaml.get("ubatch", 512)
    alias = yaml.get("alias", name)
    mmproj = yaml.get("mmproj")
    llama_bin = yaml.get("llama_bin",
                          str(SCRIPT_DIR.parent / "llama.cpp" / "build" / "bin" / "llama-server"))

    cmd = [
        llama_bin,
        "--model", str(model_path),
        "--alias", str(alias),
        "--port", str(port),
        "--host", "0.0.0.0",
        "--ctx-size", str(ctx),
        "--n-gpu-layers", str(gpu_layers),
        "--threads", str(threads),
        "--batch-size", str(batch),
        "--ubatch-size", str(ubatch),
    ]
    if mmproj:
        cmd += ["--mmproj", str(models_dir / mmproj)]
    # Reason: llama.cpp's auto memory-fit step calls cudaMemGetInfo, which
    # aborts on some CUDA builds (e.g. GB10/Blackwell). We already pass explicit
    # --n-gpu-layers/--ctx-size, so default it off. Set `fit: none` in the yaml
    # to omit the flag entirely on older llama.cpp builds that lack -fit.
    fit = yaml.get("fit", "off")
    if fit not in (None, "", "none", "skip"):
        cmd += ["-fit", str(fit)]
    # Any other llama-server flag (e.g. -fa on, -ctk q8_0, --temp 0.7) goes
    # through extra_args, appended last so it can override the defaults above.
    cmd += _extra_args(yaml)

    print(f"Starting llama-server {name} on port {port}...")
    print(f"  Model: {model_path}")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RUN_DIR / f"{name}.log"
    log_fh = open(log_path, "wb")
    proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT,
                            start_new_session=True)
    log_fh.close()
    pid_file = RUN_DIR / name
    pid_file.write_text(str(proc.pid))
    print(f"  PID {proc.pid}  (log: {log_path})")
    # Reason: wait until llama-server binds its port (or dies) so a following
    # regen_opencode() sees it; mirrors the vLLM readiness behavior.
    try:
        timeout = int(yaml.get("ready_timeout", 300))
    except (ValueError, TypeError):
        timeout = 300
    status = _wait_ready(name, port, timeout, lambda: proc.poll() is None)
    if status == "ready":
        print(f"  Ready on port {port}")
    elif status == "dead":
        pid_file.unlink(missing_ok=True)
        try:
            tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-15:])
        except Exception:
            tail = "(could not read log)"
        print(f"  ✗ {name} exited during startup (code {proc.returncode}). Last log lines:")
        print(tail)
        print(f"  Full log: {log_path}")
    else:
        print(f"  WARNING: {name} did not become ready in {timeout}s "
              f"(still loading? check {log_path})")
