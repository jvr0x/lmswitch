"""TensorFold dual-Spark runtime.

Wraps MiaAI-Lab's start.sh / stop.sh. That script prepares the image, starts
rank 1 on the worker over ssh, starts rank 0 here, and waits until /v1/models
answers. lmswitch does not rebuild that launcher.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from lmswitch.runtimes.base import BaseRuntime, RunningState

DEFAULT_CONTAINER = "glm53-flash-tf"
DEFAULT_REPO = Path.home() / "utils" / "GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold"


def repo_dir(yaml: dict) -> Path:
    """Returns the TensorFold recipe checkout start.sh lives in."""
    raw = yaml.get("repo", str(DEFAULT_REPO))
    return Path(os.path.expanduser(os.path.expandvars(str(raw))))


def container_name(yaml: dict) -> str:
    """Returns the docker name both ranks use."""
    return str(yaml.get("container", DEFAULT_CONTAINER))


def _path(yaml: dict, key: str) -> str:
    """Expands one host path from the recipe. Empty when the key is absent."""
    raw = yaml.get(key)
    if not raw:
        return ""
    return os.path.expanduser(os.path.expandvars(str(raw)))


def weight_env(yaml: dict) -> dict[str, str]:
    """Maps recipe weight paths onto the variables start.sh bind-mounts.

    A Hub-cache snapshot that is a symlink outside that cache is invisible
    inside the container. ``model_path`` / ``worker_model_path`` are the real
    trees, one path per Spark.
    """
    out: dict[str, str] = {}
    model = _path(yaml, "model_path")
    if model:
        out["MODEL_DIR"] = model
        out["WORKER_MODEL_DIR"] = _path(yaml, "worker_model_path") or model
    drafter = _path(yaml, "drafter_path")
    if drafter:
        out["DFLASH2_DIR"] = drafter
        out["WORKER_DFLASH2_DIR"] = _path(yaml, "worker_drafter_path") or drafter
    return out


def _config_present(path: str, ssh: str = "") -> bool:
    """Returns whether ``path/config.json`` exists, on this host or over ssh."""
    quoted = "'" + path.replace("'", "'\\''") + "/config.json'"
    if ssh:
        cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", ssh, f"test -f {quoted}"]
    else:
        cmd = ["test", "-f", f"{path}/config.json"]
    return subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False).returncode == 0


def missing_weights(yaml: dict) -> str:
    """Names the first weight tree that has no config.json. Empty when all are present."""
    env = weight_env(yaml)
    worker = str(yaml.get("worker") or "")
    checks = (
        ("MODEL_DIR", ""),
        ("WORKER_MODEL_DIR", worker),
        ("DFLASH2_DIR", ""),
        ("WORKER_DFLASH2_DIR", worker),
    )
    for key, ssh in checks:
        path = env.get(key, "")
        if not path:
            continue
        if not _config_present(path, ssh):
            where = f" on {ssh}" if ssh else ""
            return f"{key}={path}{where} has no config.json"
    if yaml.get("model_path") and not env.get("WORKER_MODEL_DIR"):
        return "worker_model_path is empty"
    return ""


def _env(yaml: dict) -> dict[str, str]:
    """Builds the environment start.sh reads. The process environment wins over scripts/local.sh."""
    env = os.environ.copy()
    extra = yaml.get("env") or {}
    if isinstance(extra, dict):
        for key, value in extra.items():
            env[str(key)] = str(value)
    if yaml.get("port"):
        env["PORT"] = str(yaml["port"])
    if yaml.get("worker"):
        env["WORKER"] = str(yaml["worker"])
    if yaml.get("fabric_peer"):
        env["FABRIC_PEER"] = str(yaml["fabric_peer"])
    if yaml.get("master_port"):
        env["MASTER_PORT"] = str(yaml["master_port"])
    env["CONTAINER_NAME"] = container_name(yaml)
    # Recipe paths win over a stale shell and over scripts/local.sh.
    env.update(weight_env(yaml))
    return env


def container_running(name: str) -> bool:
    """Returns whether a container with this exact name is running."""
    try:
        out = subprocess.check_output(
            ["docker", "ps", "--filter", f"name=^{name}$", "--format", "{{.ID}}"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return False
    return bool(out)


class TensorFoldRuntime(BaseRuntime):
    """Serves one TensorFold recipe across spark and gigabyte."""

    def start(self, name: str, yaml: dict) -> RunningState:
        """Runs the recipe's start.sh and waits until it reports the API live."""
        root = repo_dir(yaml)
        script = root / "start.sh"
        if not script.is_file():
            print(f"  ✗ {name}: {script} is missing")
            return RunningState("dead", detail="start.sh missing")
        port = yaml.get("port", 8890)
        print(f"Starting TensorFold {name} on port {port}...")
        print(f"  Recipe: {root}")
        absent = missing_weights(yaml)
        if absent:
            print(f"  ✗ weights not ready: {absent}")
            return RunningState("dead", detail=f"weights not ready: {absent}")
        for key, path in weight_env(yaml).items():
            print(f"  {key}: {path}")
        result = subprocess.run([str(script)], cwd=root, env=_env(yaml), check=False)
        if result.returncode != 0:
            print(f"  ✗ start.sh exited {result.returncode}")
            return RunningState("dead", detail=f"start.sh exit {result.returncode}")
        print(f"  Ready on port {port}")
        return RunningState("ready")

    def stop(self, name: str, yaml: dict) -> None:
        """Stops both ranks through the recipe's stop.sh."""
        root = repo_dir(yaml)
        script = root / "stop.sh"
        if not script.is_file():
            print(f"TensorFold {name}: {script} is missing")
            return
        print(f"Stopping TensorFold {name}...")
        subprocess.run([str(script)], cwd=root, env=_env(yaml), check=False)

    def is_running(self, name: str, runtime_name: str) -> bool:
        """Returns whether rank 0's container is up."""
        return container_running(container_name({}))

    def is_ready(self, name: str, port: int, timeout: int = 300) -> str:
        """Polls the OpenAI models endpoint."""
        from lmswitch.runtimes.wait import _wait_ready
        return _wait_ready(name, port, timeout, lambda: True)
