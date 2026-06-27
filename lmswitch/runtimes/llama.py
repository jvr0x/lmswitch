"""llama-server (GGUF) runtime."""

from __future__ import annotations

import os
import shlex
import subprocess
import time
from pathlib import Path

from lmswitch.system.io import RUN_DIR, SCRIPT_DIR
from lmswitch.runtimes.base import BaseRuntime, RunningState, runtime_registry
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


class LlamaRuntime(BaseRuntime):
    """gguf model runtime using llama-server."""

    def start(self, name: str, yaml: dict) -> RunningState:
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
        fit = yaml.get("fit", "off")
        if fit not in (None, "", "none", "skip"):
            cmd += ["-fit", str(fit)]
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
        return RunningState(status, detail=log_path if status != "ready" else "")

    def stop(self, name: str, yaml: dict) -> None:
        stop_llama_by_pid(name)

    def is_running(self, name: str, runtime_name: str) -> bool:
        from lmswitch.system.checks import _is_running
        return _is_running(name, runtime_name)

    def is_ready(self, name: str, port: int, timeout: int = 300) -> str:
        from lmswitch.system.checks import _is_running
        # Check if process is alive
        pid_file = RUN_DIR / name
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
                alive = lambda: True  # we'll check process below
                status = _wait_ready(name, port, timeout,
                                     lambda: self._proc_alive(pid))
                return status
            except (ValueError, OSError):
                return "dead"
        return "dead"

    @staticmethod
    def _proc_alive(pid: int) -> bool:
        import os
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def stop_llama_by_pid(name: str) -> None:
    """Pure stop function: read PID file and kill process directly.

    Tries killing by PID from file first, then falls back to killing by port
    (finding the process listening on the model's port via ``ss``).

    Args:
        name: Model name (used for PID file path and messages).
    """
    from lmswitch.system.io import RUN_DIR, CONF_DIR, _load_yaml
    pid_file = RUN_DIR / name

    # Try killing by PID from file first
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, 0)
            print(f"Stopping llama-server {name} (PID {pid})...")
            os.kill(pid, 15)
            pid_file.unlink()
            return
        except (ProcessLookupError, ValueError, OSError):
            # PID file is stale — fall through to port-based kill
            pid_file.unlink(missing_ok=True)

    # Fallback: find process by port and kill it
    yaml_path = CONF_DIR / f"{name}.yaml"
    if yaml_path.exists():
        try:
            yaml_cfg = _load_yaml(yaml_path)
            port = int(yaml_cfg.get("port", 0))
        except (ValueError, TypeError):
            port = 0

        if port:
            from lmswitch.system.checks import _listening_ports
            if port in _listening_ports():
                print(f"Stopping llama-server {name} (port {port})...")
                # Use ss to find PIDs listening on the port
                try:
                    import subprocess
                    out = subprocess.check_output(
                        ["ss", "-tlnpH"], text=True, stderr=subprocess.DEVNULL
                    )
                    for line in out.splitlines():
                        if f":{port}" in line:
                            # Extract PID from ss output (format: "users:((""pid""...))")
                            import re
                            m = re.search(r'pid=(\d+)', line)
                            if m:
                                try:
                                    kill_pid = int(m.group(1))
                                    os.kill(kill_pid, 15)
                                    print(f"  ↓ {name} stopped (PID {kill_pid})")
                                    return
                                except (ProcessLookupError, OSError):
                                    pass
                except Exception:
                    pass
                # Fallback: killall by name
                import subprocess as sp
                sp.run(["pkill", "-f", f".*{name}.*"], check=False)
                print(f"  ↓ {name} stopped (port {port})")
                return

    print(f"{name} not running")


# Keep module-level functions for backward compat
def _start_llama_direct(name: str, yaml: dict) -> None:
    LlamaRuntime().start(name, yaml)
