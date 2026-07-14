"""vLLM dual-node (2x DGX Spark over CX7) runtime.

Serves one model tensor-parallel across two Sparks using vLLM's native
multi-node launcher (``--nnodes 2 --node-rank N --headless``), no Ray. The
head container runs locally; the worker container is started over SSH on
``worker_host``. Weights are read from a shared HF cache (``hf_cache``), which
on this cluster is an NFS export living on one node — nothing is duplicated.

Minimal YAML:

    runtime: vllm-dual
    image: vllm-dspark-runtime:dspark-nvfp4-stage-c
    model: deepseek-ai/DeepSeek-V4-Flash-DSpark   # HF id inside hf_cache
    hf_cache: ~/hf-cluster
    port: 8888
    worker_host: Gigabyte          # ssh alias
    master_addr: 10.100.224.2      # this node's CX7 ip
    worker_ip: 10.100.224.1        # worker's CX7 ip (VLLM_HOST_IP there)
    nccl:
      ifname: enp1s0f1np1
      hca: rocep1s0f1
      gid_index: 3

Optional: ``master_port`` (25000), ``shm_size`` ("64g"), ``env`` /
``worker_env`` maps, ``extra_mounts``, ``extra_args``, ``entrypoint``,
``gpu_memory_utilization`` (0.80), ``ctx``, ``max_num_seqs`` (6),
``ready_timeout`` (1800 — TP=2 loads take a while).
"""

from __future__ import annotations

import os
import shlex
import subprocess

from lmswitch.system.io import RUN_DIR
from lmswitch.runtimes.base import RunningState
from lmswitch.runtimes.vllm import VLLMRuntime, _extra_mounts, _env_args, _entrypoint, _vllm_args
from lmswitch.runtimes.wait import _wait_ready


def _ssh(host: str, cmd: list[str], **kw) -> subprocess.CompletedProcess:
    """Runs ``cmd`` on ``host`` over ssh (BatchMode; never prompts)."""
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host,
         shlex.join(cmd)],
        **kw,
    )


def _hf_cache(yaml: dict) -> str:
    """Host path of the shared HF cache mounted at /cache/huggingface."""
    raw = yaml.get("hf_cache", "~/hf-cluster")
    return os.path.expanduser(os.path.expandvars(str(raw)))


class VLLMDualRuntime(VLLMRuntime):
    """Two-node tensor-parallel vLLM: local head + SSH-launched worker.

    Inherits vLLM container conventions (naming, logging, readiness) from
    ``VLLMRuntime``; overrides command building and lifecycle to cover both
    nodes. Container name is ``vllm-<model-id>`` on each node, so all the
    existing ``_docker_container`` checks work unchanged on the head.
    """

    def _node_cmd(self, name: str, yaml: dict, node_rank: int) -> list[str]:
        """Builds the docker run command for one node.

        Args:
            name: Model id (yaml filename stem).
            yaml: Parsed model config.
            node_rank: 0 = head (serves the API), 1 = worker (``--headless``).
        """
        port = yaml.get("port", 8888)
        ctx = yaml.get("ctx", 1048576)
        gpu_mem = yaml.get("gpu_memory_utilization", 0.80)
        image = yaml["image"]
        master_addr = yaml["master_addr"]
        master_port = yaml.get("master_port", 25000)
        nccl = yaml.get("nccl") or {}
        head = node_rank == 0

        cmd = [
            "docker", "run", "-d",
            "--name", f"vllm-{name}",
            "--gpus", "all",
            "--network", "host",
            "--ipc", "host",
            "--shm-size", str(yaml.get("shm_size", "64g")),
            "--ulimit", "memlock=-1",
            "--device", "/dev/infiniband:/dev/infiniband",
            "--log-driver", "json-file",
            "--log-opt", "max-size=10m",
            "--log-opt", "max-file=3",
        ]

        # Weights: either a plain directory (model_path per node — e.g. the
        # owning node's ~/models and the peer's NFS view of it), mounted at a
        # canonical /model so both containers see one path; or an HF repo id
        # resolved inside a shared HF cache mounted at /cache/huggingface.
        model_path = yaml.get("model_path")
        if model_path:
            node_path = model_path if head else yaml.get(
                "worker_model_path", model_path)
            node_path = os.path.expanduser(os.path.expandvars(str(node_path)))
            cmd += ["-v", f"{node_path}:/model:ro"]
            serve_target = "/model"
        else:
            cache = _hf_cache(yaml) if head else os.path.expandvars(
                str(yaml.get("worker_hf_cache", _hf_cache(yaml))))
            cmd += ["-v", f"{cache}:/cache/huggingface",
                    "-e", "HF_HOME=/cache/huggingface"]
            serve_target = str(yaml["model"])

        # NCCL wiring for the CX7 link — identical on both nodes.
        nccl_env = {
            "NCCL_NET": "IB",
            "NCCL_IB_DISABLE": "0",
            "NCCL_IB_HCA": nccl.get("hca", ""),
            "NCCL_SOCKET_IFNAME": nccl.get("ifname", ""),
            # Gloo (torch's CPU process group) picks its own interface unless
            # pinned — on the Spark it grabs a downed NIC and startup dies with
            # "Unable to find address for: enP7s7".
            "GLOO_SOCKET_IFNAME": nccl.get("ifname", ""),
            "TP_SOCKET_IFNAME": nccl.get("ifname", ""),
            "NCCL_IB_GID_INDEX": nccl.get("gid_index", ""),
            "NCCL_CUMEM_ENABLE": "0",
            "VLLM_HOST_IP": master_addr if head else yaml.get("worker_ip", ""),
        }
        for key, val in nccl_env.items():
            if val != "":
                cmd += ["-e", f"{key}={val}"]

        cmd += _extra_mounts(yaml)
        cmd += _env_args(yaml)
        if not head:
            # Worker-only env overrides (after `env:` so they win).
            cmd += _env_args({"env": yaml.get("worker_env") or {}})
        ep_opts, ep_cmd = _entrypoint(yaml)
        cmd += ep_opts

        # _vllm_args re-emits flags that are set in the yaml (gpu mem, seqs),
        # so only add our defaults when it didn't — duplicate keys trip vLLM's
        # argparse warning and make the effective config ambiguous.
        vllm_args = _vllm_args(yaml)
        defaults = []
        if not any(a.startswith("--max-num-seqs") for a in vllm_args):
            defaults.append(f"--max-num-seqs={yaml.get('max_num_seqs', 6)}")
        if not any(a.startswith("--gpu-memory-utilization") for a in vllm_args):
            defaults.append(f"--gpu-memory-utilization={gpu_mem}")

        cmd += [
            image,
            *ep_cmd,
            serve_target,
            f"--served-model-name={name}",
            "--host=0.0.0.0",
            f"--port={port}",
            f"--max-model-len={ctx}",
            "--tensor-parallel-size=2",
            "--nnodes=2",
            f"--node-rank={node_rank}",
            f"--master-addr={master_addr}",
            f"--master-port={master_port}",
        ] + defaults + vllm_args
        if not head:
            cmd += ["--headless"]
        return cmd

    def _preflight(self, name: str, yaml: dict) -> str | None:
        """Returns an error string if the cluster isn't ready to launch."""
        for key in ("image", "model", "worker_host", "master_addr"):
            if not yaml.get(key):
                return f"missing required yaml field: {key}"
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
            print(f"vLLM-dual {name} already running")
            return RunningState("ready")

        err = self._preflight(name, yaml)
        if err:
            print(f"  ✗ {err}")
            return RunningState("dead", detail=err)

        worker = yaml["worker_host"]
        port = yaml.get("port", 8888)
        print(f"Starting vLLM-dual {name} on port {port} "
              f"(head=local, worker={worker})...")

        # Clear stale containers on both nodes before launching.
        subprocess.run(["docker", "rm", "-f", f"vllm-{name}"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       check=False)
        _ssh(worker, ["docker", "rm", "-f", f"vllm-{name}"],
             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

        # Worker first: it must be waiting on master_addr when the head's
        # torch distributed init runs, mirroring the upstream recipes.
        result = _ssh(worker, self._node_cmd(name, yaml, node_rank=1))
        if result.returncode != 0:
            print(f"  ✗ worker docker run failed on {worker} "
                  f"(exit {result.returncode})")
            return RunningState("dead", detail=f"worker exit {result.returncode}")

        result = subprocess.run(self._node_cmd(name, yaml, node_rank=0),
                                check=False)
        if result.returncode != 0:
            print(f"  ✗ head docker run failed (exit {result.returncode}); "
                  f"stopping worker.")
            _ssh(worker, ["docker", "rm", "-f", f"vllm-{name}"],
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                 check=False)
            return RunningState("dead", detail=f"head exit {result.returncode}")

        log_path = self._setup_logging(name)
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
            print(f"  Ready on port {port} (TP=2 across local + {worker})")
            print(f"  Log file:  {log_path}")
        elif status == "dead":
            print(f"  ✗ {name} head container exited during startup — "
                  f"check log: {log_path}")
            _ssh(worker, ["docker", "rm", "-f", f"vllm-{name}"],
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                 check=False)
        else:
            print(f"  WARNING: {name} not ready in {timeout}s "
                  f"(TP=2 load is slow over NFS on first run; check {log_path})")
        return RunningState(status)

    def stop(self, name: str, yaml: dict) -> None:
        # Head via the parent (container + pid-file bookkeeping), then the
        # worker over ssh — a half-stopped pair would hold both GPUs hostage.
        super().stop(name, yaml)
        worker = yaml.get("worker_host")
        if worker:
            print(f"Stopping vLLM-dual worker on {worker}...")
            _ssh(worker, ["docker", "rm", "-f", f"vllm-{name}"],
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                 check=False)
