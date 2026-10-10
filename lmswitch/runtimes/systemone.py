"""Decision-model runtime. Serves POST /v1/systemone, not chat completions.

The weights are a Hugging Face directory (safetensors plus the repo's own
code). ``kind`` selects the loader: ``d1`` (Liquid system_one) or ``clef``
(Cloudflare joint schema head).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from lmswitch.runtimes.base import BaseRuntime, RunningState
from lmswitch.runtimes.llama import stop_llama_by_pid
from lmswitch.runtimes.wait import _wait_ready
from lmswitch.system.io import RUN_DIR, SCRIPT_DIR


class SystemOneRuntime(BaseRuntime):
    """Local process that loads a decision model and serves /v1/systemone."""

    def _python(self, yaml: dict) -> str:
        return str(yaml.get(
            "python",
            str(Path.home() / "venvs" / "decision" / "bin" / "python"),
        ))

    def start(self, name: str, yaml: dict) -> RunningState:
        from lmswitch.system.io import _models_dir

        models_dir = yaml.get("_models_dir") or _models_dir()
        model_path = Path(models_dir) / yaml["model"]
        port = int(yaml.get("port", 8170))
        kind = yaml.get("kind", "d1")
        script = SCRIPT_DIR / "lmswitch" / "runtimes" / "systemone_serve.py"
        cmd = [self._python(yaml), str(script)]
        env = os.environ.copy()
        env["SYSTEMONE_MODEL"] = str(model_path)
        env["SYSTEMONE_KIND"] = str(kind)
        env["SYSTEMONE_PORT"] = str(port)
        env["SYSTEMONE_NAME"] = str(yaml.get("alias", name))
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        print(f"Starting systemone {name} ({kind}) on port {port}...")
        print(f"  Model: {model_path}")
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        log_path = RUN_DIR / f"{name}.log"
        log_fh = open(log_path, "wb")
        proc = subprocess.Popen(
            cmd, stdout=log_fh, stderr=subprocess.STDOUT,
            start_new_session=True, env=env,
        )
        log_fh.close()
        pid_file = RUN_DIR / name
        pid_file.write_text(str(proc.pid))
        print(f"  PID {proc.pid}  (log: {log_path})")
        try:
            timeout = int(yaml.get("ready_timeout", 900))
        except (ValueError, TypeError):
            timeout = 900
        status = _wait_ready(name, port, timeout, lambda: proc.poll() is None)
        if status == "ready":
            print(f"  Ready on port {port}")
        elif status == "dead":
            pid_file.unlink(missing_ok=True)
            try:
                tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-20:])
            except Exception:
                tail = "(could not read log)"
            print(f"  x {name} exited during startup. Last log lines:")
            print(tail)
        else:
            print(f"  WARNING: {name} did not become ready in {timeout}s (see {log_path})")
        return RunningState(status, detail="" if status == "ready" else str(log_path), proc=proc)

    def stop(self, name: str, yaml: dict) -> None:
        stop_llama_by_pid(name, port=int(yaml.get("port") or 0))

    def is_running(self, name: str, runtime_name: str) -> bool:
        from lmswitch.system.checks import _is_running
        return _is_running(name, runtime_name)

    def is_ready(self, name: str, port: int, timeout: int = 300) -> str:
        pid_file = RUN_DIR / name
        if not pid_file.exists():
            return "dead"
        try:
            pid = int(pid_file.read_text().strip())
        except (ValueError, OSError):
            return "dead"

        def alive() -> bool:
            try:
                os.kill(pid, 0)
                return True
            except OSError:
                return False

        return _wait_ready(name, port, timeout, alive)
