"""vLLM dual-node Ray runtime (2x DGX Spark over CX7, Ray-based TP).

For models whose distributed launch requires Ray (e.g. MiMo-V2.5's Omni
multimodal path, `--distributed-executor-backend ray`) rather than vLLM's
native multiproc launcher. ``vllm_dual.py`` (runtime: vllm-dual) covers the
Ray-free `--nnodes`/`--node-rank` case — use this one only when the image
genuinely requires Ray for cross-node TP.

Lifecycle is fundamentally different from vllm_dual.py's single
`docker run <image> vllm serve <flags>` per node: containers here start
IDLE (`sleep infinity`), then get driven step by step via `docker exec`:

  1. run head + worker containers idle, each with a workspace dir bind-mounted
     (log file lands there, host-visible, so lmswitch can tail it like any
     other runtime)
  2. `ray start --head` on the head, `ray start --address=<head>` on the
     worker (over ssh)
  3. poll `ray status` on the head until it reports 2.0/2.0 GPU
  4. `docker exec -d` the head to launch `vllm serve` in the background,
     output redirected into the bind-mounted workspace log file

Any launch-time file patches the upstream image needs should be baked into
the image itself (see e.g. Dockerfile.lmswitch in the MiMo build dir) —
this runtime does not apply patches at start() time.

Minimal YAML:

    runtime: vllm-dual-ray
    image: ghcr.io/miaai-lab/mimo-v2.5-vllm-dual-dgx-sparks-lmswitch
    model_path: ~/models/lukealonso/MiMo-V2.5-NVFP4        # this node's copy
    worker_model_path: ~/models-spark/lukealonso/MiMo-V2.5-NVFP4  # worker's
    port: 8888
    worker_host: Gigabyte
    master_addr: 10.100.224.2      # this node's CX7 ip (Ray GCS + NCCL)
    worker_ip: 10.100.224.1        # worker's CX7 ip
    nccl:
      ifname: enp1s0f1np1
      hca: rocep1s0f1
      gid_index: 3
    serve_args:                    # appended after the auto-generated flags
      - "--hf-overrides={\"architectures\":[\"MiMoV2OmniForCausalLM\"]}"
      - ...

Optional: ``ray_port`` (6379), ``ray_object_store_bytes`` (1073741824),
``ray_wait_timeout`` (600 — seconds to wait for 2/2 GPU), ``shm_size``
("16g"), ``gpu_memory_utilization`` (0.80), ``ctx``, ``max_num_seqs`` (3),
``env`` / ``worker_env``, ``extra_mounts`` / ``head_extra_mounts`` /
``worker_extra_mounts``, ``ready_timeout`` (1800).
"""

from __future__ import annotations

import os
import shlex
import subprocess

from lmswitch.system.io import RUN_DIR
from lmswitch.runtimes.base import RunningState
from lmswitch.runtimes.vllm import VLLMRuntime, _extra_mounts, _env_args, _vllm_args
from lmswitch.runtimes.wait import _wait_ready


def _ssh(host: str, cmd: list[str], **kw) -> subprocess.CompletedProcess:
    """Runs ``cmd`` on ``host`` over ssh (BatchMode; never prompts)."""
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host,
         shlex.join(cmd)],
        **kw,
    )


class VLLMDualRayRuntime(VLLMRuntime):
    """Two-node Ray-backed TP vLLM: local head + SSH-launched worker.

    Inherits is_running/is_ready from VLLMRuntime (container-name + port
    poll, same ``vllm-<name>`` naming convention) — only start/stop differ.
    """

    def _workspace(self, name: str) -> str:
        """Host dir bind-mounted at /workspace in both containers — carries
        the vLLM log file so it's tailable from the host like any other
        runtime's log, even though vLLM here runs via `docker exec`, not as
        the container's own PID 1."""
        path = RUN_DIR / f"{name}-ray-workspace"
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    def _container_cmd(self, name: str, yaml: dict, head: bool) -> list[str]:
        """Builds the idle (`sleep infinity`) docker run command for one node."""
        model_path = yaml.get("model_path") if head else (
            yaml.get("worker_model_path") or yaml.get("model_path"))
        model_path = os.path.expanduser(os.path.expandvars(str(model_path)))
        workspace = self._workspace(name)

        cmd = [
            "docker", "run", "-d",
            "--name", f"vllm-{name}",
            "--gpus", "all",
            "--network", "host",
            "--ipc", "host",
            "--shm-size", str(yaml.get("shm_size", "16g")),
            "--ulimit", "memlock=-1",
            "--ulimit", "stack=67108864",
            "--device", "/dev/infiniband:/dev/infiniband",
            "--workdir", "/workspace",
            "-v", f"{model_path}:/model_cache:ro",
            "-v", f"{workspace}:/workspace",
        ]
        cmd += _extra_mounts(yaml)
        side_mounts = yaml.get("head_extra_mounts") if head else yaml.get("worker_extra_mounts")
        cmd += _extra_mounts({"extra_mounts": side_mounts or []})
        cmd += _env_args(yaml)
        if not head:
            cmd += _env_args({"env": yaml.get("worker_env") or {}})
        cmd += [yaml["image"], "sleep", "infinity"]
        return cmd

    def _nccl_env_exports(self, yaml: dict, head: bool, master_addr: str) -> str:
        nccl = yaml.get("nccl") or {}
        gid_index = nccl.get("gid_index", "")
        if not head:
            gid_index = yaml.get("worker_nccl_gid_index", gid_index)
        pairs = {
            "NCCL_IB_DISABLE": "0",
            "NCCL_SOCKET_IFNAME": nccl.get("ifname", ""),
            "NCCL_IB_HCA": nccl.get("hca", ""),
            "NCCL_IB_GID_INDEX": str(gid_index),
            "VLLM_HOST_IP": master_addr if head else yaml.get("worker_ip", ""),
        }
        return " ".join(f"{k}={shlex.quote(str(v))}" for k, v in pairs.items() if v != "")

    def _ray_start_head(self, name: str, yaml: dict, master_addr: str, ray_port: int) -> bool:
        obj_store = yaml.get("ray_object_store_bytes", 1073741824)
        nccl_exports = self._nccl_env_exports(yaml, head=True, master_addr=master_addr)
        script = (
            f"unset RAY_OVERRIDE_NODE_IP_ADDRESS RAY_NODE_IP_ADDRESS; "
            f"export RAY_TMPDIR=/dev/shm/ray; mkdir -p /dev/shm/ray; "
            f"export {nccl_exports}; "
            f"ray stop --force || true; "
            f"ray start --head --port={ray_port} --node-ip-address={master_addr} "
            f"--dashboard-host=0.0.0.0 --num-gpus=1 --object-store-memory={obj_store}"
        )
        r = subprocess.run(["docker", "exec", f"vllm-{name}", "bash", "-lc", script],
                           check=False)
        return r.returncode == 0

    def _ray_start_worker(self, name: str, yaml: dict, worker: str,
                          master_addr: str, ray_port: int) -> bool:
        worker_ip = yaml.get("worker_ip", "")
        obj_store = yaml.get("ray_object_store_bytes", 1073741824)
        nccl_exports = self._nccl_env_exports(yaml, head=False, master_addr=master_addr)
        script = (
            f"unset RAY_OVERRIDE_NODE_IP_ADDRESS RAY_NODE_IP_ADDRESS; "
            f"export RAY_TMPDIR=/dev/shm/ray; mkdir -p /dev/shm/ray; "
            f"export {nccl_exports}; "
            f"ray stop --force || true; "
            f"ray start --address={master_addr}:{ray_port} --node-ip-address={worker_ip} "
            f"--num-gpus=1 --object-store-memory={obj_store}"
        )
        r = _ssh(worker, ["docker", "exec", f"vllm-{name}", "bash", "-lc", script])
        return r.returncode == 0

    def _wait_2_gpus(self, name: str, ray_port: int, master_addr: str, timeout: int) -> bool:
        import time
        elapsed = 0
        while elapsed < timeout:
            r = subprocess.run(
                ["docker", "exec", "-e", f"RAY_ADDRESS={master_addr}:{ray_port}",
                 f"vllm-{name}", "ray", "status"],
                capture_output=True, text=True, check=False,
            )
            if r.returncode == 0 and (
                "2.0/2.0 GPU" in r.stdout or "2.0 GPU" in r.stdout):
                return True
            time.sleep(5)
            elapsed += 5
            if elapsed % 30 == 0:
                print(f"  …waiting for Ray 2/2 GPU ({elapsed}s)")
        return False

    def _serve_cmd(self, name: str, yaml: dict, master_addr: str) -> str:
        """Builds the `vllm serve ...` line run inside the head container."""
        port = yaml.get("port", 8888)
        ctx = yaml.get("ctx", 1000000)
        gpu_mem = yaml.get("gpu_memory_utilization", 0.80)
        max_num_seqs = yaml.get("max_num_seqs", 3)
        vllm_args = _vllm_args(yaml)
        args = [
            "vllm", "serve", "/model_cache",
            f"--served-model-name={name}",
            "--host=0.0.0.0",
            f"--port={port}",
            f"--max-model-len={ctx}",
            "--tensor-parallel-size=2",
            "--pipeline-parallel-size=1",
            "--distributed-executor-backend=ray",
            f"--max-num-seqs={max_num_seqs}",
            f"--gpu-memory-utilization={gpu_mem}",
        ] + vllm_args
        for a in yaml.get("serve_args") or []:
            args.append(str(a))
        # The base image bakes in stale RAY_OVERRIDE_NODE_IP_ADDRESS /
        # RAY_NODE_IP_ADDRESS / VLLM_HOST_IP / NCCL_SOCKET_IFNAME etc.
        # (container-level env, from whatever cluster the image was
        # originally built for — observed VLLM_HOST_IP=10.0.0.5,
        # NCCL_SOCKET_IFNAME auto-detecting to the wrong NIC "enP7s7").
        # vLLM's own Ray worker actors are spawned fresh by THIS exec's
        # process tree (not by the earlier `_ray_start_head`/worker exec,
        # whose env doesn't carry over — only container-level `-e` vars
        # do), so they need the same NCCL wiring re-asserted here or NCCL
        # init fails with "invalid usage" / GLOO "Connection closed by
        # peer" once the head's rank0 process falls over first.
        preamble = (
            "unset RAY_OVERRIDE_NODE_IP_ADDRESS RAY_NODE_IP_ADDRESS; "
            f"export {self._nccl_env_exports(yaml, head=True, master_addr=master_addr)}; "
        )
        return preamble + shlex.join(args) + " > /workspace/vllm.log 2>&1"

    def _preflight(self, name: str, yaml: dict) -> str | None:
        for key in ("image", "worker_host", "master_addr"):
            if not yaml.get(key):
                return f"missing required yaml field: {key}"
        if not yaml.get("model_path"):
            return "missing required yaml field: model_path"
        worker = yaml["worker_host"]
        if _ssh(worker, ["true"], capture_output=True).returncode != 0:
            return f"worker unreachable over ssh: {worker}"
        image = yaml["image"]
        for where, check in (
            ("local", subprocess.run(["docker", "image", "inspect", image],
                                     capture_output=True)),
            (worker, _ssh(worker, ["docker", "image", "inspect", image],
                          capture_output=True)),
        ):
            if check.returncode != 0:
                return f"image {image} missing on {where}"
        return None

    def start(self, name: str, yaml: dict) -> RunningState:
        from lmswitch.system.checks import _docker_container
        if _docker_container(name):
            print(f"vLLM-dual-ray {name} already running")
            return RunningState("ready")

        err = self._preflight(name, yaml)
        if err:
            print(f"  ✗ {err}")
            return RunningState("dead", detail=err)

        worker = yaml["worker_host"]
        master_addr = yaml["master_addr"]
        ray_port = yaml.get("ray_port", 6379)
        port = yaml.get("port", 8888)
        print(f"Starting vLLM-dual-ray {name} on port {port} "
              f"(head=local, worker={worker})...")

        subprocess.run(["docker", "rm", "-f", f"vllm-{name}"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        _ssh(worker, ["docker", "rm", "-f", f"vllm-{name}"],
             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

        # Idle containers on both nodes — worker first, mirroring vllm_dual.py.
        r = _ssh(worker, self._container_cmd(name, yaml, head=False))
        if r.returncode != 0:
            print(f"  ✗ worker docker run failed on {worker} (exit {r.returncode})")
            return RunningState("dead", detail=f"worker exit {r.returncode}")
        r = subprocess.run(self._container_cmd(name, yaml, head=True), check=False)
        if r.returncode != 0:
            print(f"  ✗ head docker run failed (exit {r.returncode}); stopping worker.")
            _ssh(worker, ["docker", "rm", "-f", f"vllm-{name}"],
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            return RunningState("dead", detail=f"head exit {r.returncode}")

        print("  Bringing up Ray cluster (head, then worker)...")
        if not self._ray_start_head(name, yaml, master_addr, ray_port):
            print("  ✗ ray start --head failed")
            self._teardown(name, worker)
            return RunningState("dead", detail="ray start --head failed")
        if not self._ray_start_worker(name, yaml, worker, master_addr, ray_port):
            print("  ✗ ray start (worker) failed")
            self._teardown(name, worker)
            return RunningState("dead", detail="ray start (worker) failed")

        ray_timeout = int(yaml.get("ray_wait_timeout", 600))
        if not self._wait_2_gpus(name, ray_port, master_addr, ray_timeout):
            print(f"  ✗ Ray cluster did not reach 2/2 GPU in {ray_timeout}s")
            self._teardown(name, worker)
            return RunningState("dead", detail="ray 2/2 GPU timeout")
        print("  Ray cluster healthy: 2.0/2.0 GPU")

        serve = self._serve_cmd(name, yaml, master_addr)
        r = subprocess.run(
            ["docker", "exec", "-d", f"vllm-{name}", "bash", "-lc", serve],
            check=False,
        )
        if r.returncode != 0:
            print(f"  ✗ vllm serve exec failed (exit {r.returncode})")
            self._teardown(name, worker)
            return RunningState("dead", detail=f"vllm serve exec exit {r.returncode}")

        log_path = self._setup_log_follower(name)
        RUN_DIR.mkdir(parents=True, exist_ok=True)

        try:
            timeout = int(yaml.get("ready_timeout", 1800))
        except (ValueError, TypeError):
            timeout = 1800
        status = _wait_ready(name, port, timeout,
                             lambda: _docker_container(name) is not None)
        if status == "ready":
            container_id = _docker_container(name)
            (RUN_DIR / name).write_text(container_id or name)
            print(f"  Ready on port {port} (Ray TP=2 across local + {worker})")
            print(f"  Log file:  {log_path}")
        elif status == "dead":
            print(f"  ✗ {name} head container exited during startup — "
                  f"check log: {log_path}")
            self._teardown(name, worker)
        else:
            print(f"  WARNING: {name} not ready in {timeout}s "
                  f"(check {log_path})")
        return RunningState(status)

    def _setup_log_follower(self, name: str):
        """Tail the workspace vllm.log (host-visible via the bind mount)
        into RUN_DIR/<name>.log — vLLM runs via `docker exec`, not as the
        container's own PID 1, so `docker logs` captures nothing useful."""
        import time
        from pathlib import Path
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        src = Path(self._workspace(name)) / "vllm.log"
        # vllm.log doesn't exist until the exec'd process writes its first
        # bytes — wait briefly rather than handing tail a missing file.
        for _ in range(50):
            if src.exists():
                break
            time.sleep(0.1)
        log_path = RUN_DIR / f"{name}.log"
        try:
            if log_path.is_symlink() or log_path.exists():
                os.unlink(log_path)
            log_fh = open(log_path, "wb")
            subprocess.Popen(
                ["tail", "-f", str(src)],
                stdout=log_fh, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            log_fh.close()
        except OSError as exc:
            print(f"  WARNING: could not write log file for {name} ({exc}); "
                  f"logs still available at {src}")
        return log_path

    def _teardown(self, name: str, worker: str) -> None:
        subprocess.run(["docker", "rm", "-f", f"vllm-{name}"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        _ssh(worker, ["docker", "rm", "-f", f"vllm-{name}"],
             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

    def stop(self, name: str, yaml: dict) -> None:
        from lmswitch.system.checks import _docker_container
        cid = _docker_container(name)
        if cid:
            print(f"Stopping vLLM-dual-ray {name} (container {cid[:12]})...")
            subprocess.run(["docker", "exec", f"vllm-{name}", "bash", "-lc",
                            "pkill -f 'vllm serve' 2>/dev/null; ray stop --force 2>/dev/null || true"],
                           check=False)
            subprocess.run(["docker", "rm", "-f", f"vllm-{name}"], check=False)
            (RUN_DIR / name).unlink(missing_ok=True)
        else:
            print(f"vLLM-dual-ray {name} not running")
        worker = yaml.get("worker_host")
        if worker:
            print(f"Stopping vLLM-dual-ray worker on {worker}...")
            _ssh(worker, ["docker", "exec", f"vllm-{name}", "bash", "-lc",
                         "ray stop --force 2>/dev/null || true"], check=False)
            _ssh(worker, ["docker", "rm", "-f", f"vllm-{name}"],
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
