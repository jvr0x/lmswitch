"""SGLang dual-node (2x DGX Spark over CX7) runtime.

Serves one model tensor-parallel across two Sparks using SGLang's native
multi-node launcher (``--nnodes 2 --node-rank N --dist-init-addr HOST:PORT``).
Both ranks run the same ``sglang.launch_server`` command; only rank 0 binds
the HTTP API, so the head is the node this runs on and the worker is started
over SSH on ``worker_host`` — the same head-local/worker-remote split
``vllm-dual`` uses.

This exists because a checkpoint can be larger than one node's unified memory.
SGLang cannot offload, so a 125.9 GiB checkpoint on a 121 GiB box is not a
tuning problem — it only serves at all when the weights are sharded across
both Sparks.

Minimal YAML:

    runtime: sglang-dual
    image: lmsysorg/sglang:qwen38flashnext
    model_path: "~/models-gigabyte/Org/Model-NVFP4"   # path on THIS node
    worker_model_path: "~/models/Org/Model-NVFP4"     # path on the worker
    port: 8888
    worker_host: Gigabyte          # ssh alias
    master_addr: 10.100.224.2      # this node's CX7 ip
    worker_ip: 10.100.224.1        # worker's CX7 ip
    nccl:
      ifname: enp1s0f1np1
      hca: rocep1s0f1
      gid_index: 3

Both nodes bind their own weights directory at the same canonical ``/model``
inside the container, so ``--model-path`` is identical on each rank even
though the host paths differ (one node owns the files, the peer sees them
over NFS).

Optional: every key the single-node ``sglang`` runtime understands
(``mem_fraction_static``, ``attention_backend``, ``chunked_prefill_size``,
``kv_cache_dtype``, ``max_running_requests``, ``tool_call_parser``,
``reasoning_parser``, ``sampling_defaults``, ``trust_remote_code``,
``extra_args``, ``env``, ``extra_mounts``, ``cpuset``, ``shm_size``,
``json_model_override_args``) plus ``tp_size`` (2), ``master_port`` (25000),
``worker_env``, ``head_extra_mounts`` / ``worker_extra_mounts``,
``ready_timeout`` (2400 — a TP=2 load over NFS is slow), and ``restart`` +
``gate_timeout`` (see runtimes/dual_serve.py).
"""

from __future__ import annotations

import os
import shlex
import subprocess

from lmswitch.system.io import HOME, RUN_DIR
from lmswitch.runtimes.base import RunningState
from lmswitch.runtimes.sglang import (
    NATIVE_CTX,
    SGLangRuntime,
    _cpuset_args,
    _sglang_args,
)
from lmswitch.runtimes.vllm import _extra_mounts, _env_args
from lmswitch.runtimes.wait import _wait_ready

# Canonical in-container path for the weights. The host path differs per node
# (the owning node's ~/models vs the peer's NFS view of it), but --model-path
# is one string shared by both ranks, so the mount target has to be fixed.
CONTAINER_MODEL_PATH = "/model"


def _ssh(host: str, cmd: list[str], **kw) -> subprocess.CompletedProcess:
    """Runs ``cmd`` on ``host`` over ssh (BatchMode; never prompts)."""
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host,
         shlex.join(cmd)],
        **kw,
    )


class SGLangDualRuntime(SGLangRuntime):
    """Two-node tensor-parallel SGLang: local head + SSH-launched worker.

    Inherits the SGLang container conventions (``sglang-<model-id>`` naming,
    log follower, readiness poll) from ``SGLangRuntime`` and overrides command
    building and lifecycle to cover both nodes.
    """

    def _node_cmd(self, name: str, yaml: dict, node_rank: int) -> list[str]:
        """Builds the ``docker run`` command for one node.

        Args:
            name: Model id (yaml filename stem) — names the container and the
                served model.
            yaml: Parsed model config.
            node_rank: 0 = head (binds the HTTP API), 1 = worker.

        Returns:
            The full argv for ``docker run``.
        """
        port = yaml.get("port", 8888)
        ctx = int(yaml.get("ctx", NATIVE_CTX))
        image = yaml["image"]
        master_addr = yaml["master_addr"]
        master_port = yaml.get("master_port", 25000)
        nccl = yaml.get("nccl") or {}
        head = node_rank == 0

        cmd = [
            "docker", "run", "-d",
            "--name", f"sglang-{name}",
            "--gpus", "all",
            "--network", "host",
            "--ipc", "host",
            "--privileged",
            "--shm-size", str(yaml.get("shm_size", "64g")),
            "--ulimit", "memlock=-1",
            # The CX7 link is what carries every TP all-reduce; without the
            # device the container falls back to TCP and the model crawls.
            "--device", "/dev/infiniband:/dev/infiniband",
            *_cpuset_args(yaml),
            "--log-driver", "json-file",
            "--log-opt", "max-size=10m",
            "--log-opt", "max-file=3",
        ]

        # Weights: this node's own path, bound at the canonical container path.
        node_path = yaml["model_path"] if head else yaml.get(
            "worker_model_path", yaml["model_path"])
        node_path = os.path.expanduser(os.path.expandvars(str(node_path)))
        cmd += [
            "-v", f"{node_path}:{CONTAINER_MODEL_PATH}:ro",
            "-v", f"{HOME}/.cache/huggingface:/root/.cache/huggingface",
            "-v", f"{HOME}/.cache/triton:/root/.triton",
            "-e", "HF_HOME=/root/.cache/huggingface",
            "-e", "TRITON_CACHE_DIR=/root/.triton",
        ]

        # NCCL wiring for the CX7 link — identical on both nodes.
        nccl_env = {
            "NCCL_NET": "IB",
            "NCCL_IB_DISABLE": "0",
            "NCCL_IB_HCA": nccl.get("hca", ""),
            "NCCL_SOCKET_IFNAME": nccl.get("ifname", ""),
            # Reason: torch's CPU process group (Gloo) picks its own interface
            # unless pinned — on the Spark it grabs a downed NIC and the
            # rendezvous dies with "Unable to find address for: enP7s7".
            "GLOO_SOCKET_IFNAME": nccl.get("ifname", ""),
            "TP_SOCKET_IFNAME": nccl.get("ifname", ""),
            "NCCL_IB_GID_INDEX": nccl.get("gid_index", ""),
            "NCCL_CUMEM_ENABLE": "0",
        }
        for key, val in nccl_env.items():
            if val != "":
                cmd += ["-e", f"{key}={val}"]

        # Reason: SGLang clamps --context-length back to the checkpoint's
        # derived maximum unless this is set; the recipe still has to supply
        # the YaRN override itself via json_model_override_args.
        if ctx > NATIVE_CTX:
            cmd += ["-e", "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1"]

        cmd += _extra_mounts(yaml)
        # head_extra_mounts / worker_extra_mounts: for a second asset needing a
        # different host path per node (e.g. a speculative-decode drafter) —
        # the same asymmetry model_path/worker_model_path solves for the
        # primary weights. Each list is exclusive to its side; `extra_mounts`
        # stays for genuinely identical-on-both-nodes mounts.
        side_mounts = (yaml.get("head_extra_mounts") if head
                       else yaml.get("worker_extra_mounts"))
        cmd += _extra_mounts({"extra_mounts": side_mounts or []})
        cmd += _env_args(yaml)
        if not head:
            # Worker-only env overrides, emitted after `env:` so they win.
            cmd += _env_args({"env": yaml.get("worker_env") or {}})

        # Reason: the SGLang images carry no `sglang.launch_server`
        # ENTRYPOINT, so pin it instead of inheriting whatever the image
        # declares — same as the single-node runtime.
        cmd += ["--entrypoint", "python3", image,
                "-m", "sglang.launch_server",
                "--model-path", CONTAINER_MODEL_PATH,
                "--served-model-name", name,
                "--context-length", str(ctx),
                "--host", "0.0.0.0",
                "--port", str(port),
                "--nnodes", "2",
                "--node-rank", str(node_rank),
                "--dist-init-addr", f"{master_addr}:{master_port}"]
        if yaml.get("json_model_override_args"):
            cmd += ["--json-model-override-args",
                    str(yaml["json_model_override_args"])]

        # _sglang_args re-emits --tp-size when the recipe sets tp_size, so only
        # add the default when it didn't — a duplicate would leave the
        # effective config ambiguous.
        sgl_args = _sglang_args(yaml)
        if "--tp-size" not in sgl_args:
            cmd += ["--tp-size", "2"]
        cmd += sgl_args
        return cmd

    def _preflight(self, name: str, yaml: dict) -> str | None:
        """Returns an error string if the cluster isn't ready to launch."""
        for key in ("image", "model_path", "worker_host", "master_addr"):
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
        if _docker_container(name, "sglang"):
            print(f"SGLang-dual {name} already running")
            return RunningState("ready")

        err = self._preflight(name, yaml)
        if err:
            print(f"  ✗ {err}")
            return RunningState("dead", detail=err)

        worker = yaml["worker_host"]
        port = yaml.get("port", 8888)
        ctx = int(yaml.get("ctx", NATIVE_CTX))
        print(f"Starting SGLang-dual {name} on port {port} "
              f"(head=local, worker={worker})...")
        print(f"  Image: {yaml['image']}")
        if ctx > NATIVE_CTX and not yaml.get("json_model_override_args"):
            print(f"  WARNING: ctx={ctx} exceeds the native {NATIVE_CTX} but no "
                  f"json_model_override_args (YaRN) is set — SGLang will clamp "
                  f"back to {NATIVE_CTX}.")

        # A container left behind in Created/Exited state keeps the name and
        # makes every retry fail with a name conflict — clear both nodes.
        subprocess.run(["docker", "container", "rm", "-f", f"sglang-{name}"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       check=False)
        _ssh(worker, ["docker", "container", "rm", "-f", f"sglang-{name}"],
             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

        # Worker first: it has to be waiting on dist_init_addr when the head's
        # rendezvous runs, mirroring the vLLM dual runtime and the upstream
        # multi-node recipes.
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
            _ssh(worker, ["docker", "container", "rm", "-f", f"sglang-{name}"],
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                 check=False)
            return RunningState("dead", detail=f"head exit {result.returncode}")

        log_path = self._setup_logging(name)
        RUN_DIR.mkdir(parents=True, exist_ok=True)

        try:
            timeout = int(yaml.get("ready_timeout", 2400))
        except (ValueError, TypeError):
            timeout = 2400
        status = _wait_ready(name, port, timeout,
                             lambda: _docker_container(name, "sglang") is not None)
        if status == "ready":
            container_id = _docker_container(name, "sglang")
            (RUN_DIR / name).write_text(container_id or name)
            print(f"  Ready on port {port} (TP=2 across local + {worker})")
            print(f"  Log file:  {log_path}")
        elif status == "dead":
            print(f"  ✗ {name} head container exited during startup — "
                  f"check log: {log_path}")
            _ssh(worker, ["docker", "container", "rm", "-f", f"sglang-{name}"],
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                 check=False)
        else:
            print(f"  WARNING: {name} not ready in {timeout}s "
                  f"(TP=2 load is slow over NFS on first run; check {log_path})")
        return RunningState(status)

    def stop(self, name: str, yaml: dict) -> None:
        # Head via the parent (container + pid-file bookkeeping), then the
        # worker over ssh — a half-stopped pair holds both GPUs hostage and
        # the next start cannot bind the TP group.
        super().stop(name, yaml)
        worker = yaml.get("worker_host")
        if worker:
            print(f"Stopping SGLang-dual worker on {worker}...")
            _ssh(worker, ["docker", "container", "rm", "-f", f"sglang-{name}"],
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                 check=False)
