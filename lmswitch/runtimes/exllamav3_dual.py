"""exllamav3 dual-node (2x DGX Spark over CX7) job runtime.

Runs one GLM-5/DeepSeek-style EXL3 checkpoint pipeline-parallel across two
Sparks with exllamav3's ``examples/multinode_pipeline.py``: rank 1 starts
over SSH on ``worker_host`` and waits, rank 0 runs locally, drives
tokenization/generation, and both exit when the generation finishes.

This is a JOB runtime, not a server runtime — the pipeline script is
single-shot batch (``--prompt`` in, text out, exit). There is no port, no
``/health``, and no OpenAI endpoint, so recipes set ``port: 0`` (which also
keeps them out of every sync target: opencode/hermes/grok/omp all skip
port-0 entries) and ``lmswitch serve`` (systemd) refuses them. ``lmswitch
on <id>`` blocks for the whole generation, streams rank 0's output, and
tears the worker down on every exit path.

Minimal YAML:

    runtime: exllamav3-dual
    model: vcruz305/RED-SNOW-5.3-FLASH-EXL3-SAGE-4.91bpw  # HF id (label only)
    model_path: "~/models-gigabyte/vcruz305/RED-SNOW-..."  # path on THIS node
    worker_model_path: "~/models/vcruz305/RED-SNOW-..."    # path on the worker
    port: 0
    worker_host: Gigabyte          # ssh alias
    worker_ip: 10.100.224.1        # worker's CX7 ip
    master_addr: 10.100.224.2      # this node's CX7 ip (--addrs rank0 first)

Optional: ``transport`` (tcp, default — bring-up safe; or nccl with the
``nccl:`` block below), ``transport_port`` (29650 — the pipeline's --port;
NCCL rendezvous uses +7, TCP rank 0 listens on +1), ``splits``
("0:23,23:45" — must start each rank on a full indexer layer for
Glm5Next DSA models), ``ctx`` (8192), ``cache_quant`` ("4"),
``exllamav3_repo`` (~/exllamav3), ``pin_commit`` (verified master rev),
``prompt``, ``max_new`` (200), ``job_timeout`` (3600 — how long `on`
waits for rank 0), ``gate_timeout`` (600 — how long to wait for the
worker's "links up").
"""

from __future__ import annotations

import os
import subprocess
import time

from lmswitch.runtimes.base import BaseRuntime, RunningState
from lmswitch.runtimes.llama import _escalate_kill
from lmswitch.system.io import RUN_DIR

DEFAULT_TRANSPORT_PORT = 29650
DEFAULT_SPLITS = "0:23,23:45"
DEFAULT_CTX = 8192
DEFAULT_REPO = "~/exllamav3"
# Pinned merged-master rev from the RED-SNOW DEPLOY.md (multi-node TCP/NCCL
# transports + Glm5Next recurrent path, fail-closed for other families).
DEFAULT_PIN_COMMIT = "764b4ad6c86c59052875ccb1c31e1daaa85813cb"


def _ssh(host: str, cmd: str, **kw) -> subprocess.CompletedProcess:
    """Runs the shell string ``cmd`` on ``host`` over ssh (never prompts)."""
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, cmd],
        **kw,
    )


def _worker_path(raw: str) -> str:
    """Quotes a worker-side path for a remote shell, keeping ``~`` usable.

    ``shlex.quote`` would turn ``~/exllamav3`` into a literal directory named
    ``~``; the worker's shell expands ``$HOME`` inside double quotes instead.
    """
    return '"' + str(raw).replace("~/", "$HOME/", 1).replace('"', r'\"') + '"'


def pipeline_share(yaml: dict) -> float:
    """Fraction of decoder layers owned by rank 0 (this node), from ``splits``.

    An absent key means the default splits apply (the same default
    ``_rank_cmd`` uses); unparseable or degenerate values fall back to an
    even split. The memory guard uses this to size only the local slice —
    the rest is resident on the worker.
    """
    raw = yaml.get("splits", DEFAULT_SPLITS)
    try:
        ranges = []
        for part in str(raw).split(","):
            a, b = part.strip().split(":")
            ranges.append(int(b) - int(a))
        total = sum(ranges)
        if len(ranges) < 2 or total <= 0 or any(r <= 0 for r in ranges):
            return 0.5
        return ranges[0] / total
    except (ValueError, AttributeError):
        return 0.5


def _pkill_cmd(tport: int) -> str:
    """Kills only the pipeline stages bound to ``tport``.

    Scoped by transport port, not by script name: a second exllamav3-dual
    recipe on another port must survive this one starting or stopping — the
    same collision the docker-name-scoped vLLM dual runtimes avoid.
    """
    return f'pkill -f "multinode_pipeline.*--port {tport}( |$)" || true'


def _nccl_env(yaml: dict) -> dict[str, str]:
    """NCCL/RoCE env for the CX7 link, empty unless transport is nccl."""
    if yaml.get("transport", "tcp") != "nccl":
        return {}
    nccl = yaml.get("nccl") or {}
    return {
        "NCCL_SOCKET_IFNAME": str(nccl.get("ifname", "")),
        "NCCL_IB_HCA": str(nccl.get("hca", "")),
        "NCCL_DEBUG": "INFO",
    }


class Exllamav3DualRuntime(BaseRuntime):
    """Two-node exllamav3 pipeline job: SSH-launched rank 1 + local rank 0."""

    def _addrs(self, yaml: dict) -> str:
        """Comma-separated link addresses, rank-0 first (this node)."""
        return f"{yaml['master_addr']},{yaml['worker_ip']}"

    def _rank_cmd(self, yaml: dict, rank: int) -> tuple[list[str], dict[str, str]]:
        """Builds the pipeline argv + env for one rank (rank 0 runs here)."""
        repo = os.path.expanduser(os.path.expandvars(
            str(yaml.get("exllamav3_repo", DEFAULT_REPO))))
        script = os.path.join(repo, "examples", "multinode_pipeline.py")
        head = rank == 0
        node_path = yaml["model_path"] if head else yaml.get(
            "worker_model_path", yaml["model_path"])
        cmd = [
            "python", script,
            "--model", os.path.expanduser(os.path.expandvars(str(node_path))),
            "--rank", str(rank),
            "--addrs", self._addrs(yaml),
            "--splits", str(yaml.get("splits", DEFAULT_SPLITS)),
            "--port", str(yaml.get("transport_port", DEFAULT_TRANSPORT_PORT)),
            "--transport", str(yaml.get("transport", "tcp")),
            "--ctx", str(yaml.get("ctx", DEFAULT_CTX)),
        ]
        if yaml.get("cache_quant") is not None:
            cmd += ["--cache_quant", str(yaml["cache_quant"])]
        if head:
            cmd += ["--prompt", str(yaml.get("prompt", "Explain pipeline parallelism in three sentences.")),
                    "--max_new", str(yaml.get("max_new", 200))]
        return cmd, _nccl_env(yaml)

    def _preflight(self, name: str, yaml: dict) -> str | None:
        """Returns an error string if the pair isn't ready to launch."""
        for key in ("model_path", "worker_host", "worker_ip", "master_addr"):
            if not yaml.get(key):
                return f"missing required yaml field: {key}"
        worker = yaml["worker_host"]
        if _ssh(worker, "true", capture_output=True).returncode != 0:
            return f"worker unreachable over ssh: {worker}"
        repo = yaml.get("exllamav3_repo", DEFAULT_REPO)
        pin = yaml.get("pin_commit", DEFAULT_PIN_COMMIT)
        for where, check in (
            ("local", os.path.isfile(os.path.join(
                os.path.expanduser(os.path.expandvars(str(repo))),
                "examples", "multinode_pipeline.py"))),
            ("worker", _ssh(worker, f"test -f {_worker_path(repo + '/examples/multinode_pipeline.py')}",
                            capture_output=True).returncode == 0),
        ):
            if not check:
                return (f"exllamav3 checkout missing on {where}: {repo} "
                        f"(expected pin {pin})")
        for label, raw in (("local", yaml["model_path"]),
                           ("worker", yaml.get("worker_model_path", yaml["model_path"]))):
            path = os.path.expanduser(os.path.expandvars(str(raw)))
            if label == "local":
                ok = os.path.isdir(path) and any(
                    f.endswith(".safetensors") for f in os.listdir(path))
            else:
                out = _ssh(worker, f"ls {_worker_path(raw)}/*.safetensors >/dev/null 2>&1",
                           capture_output=True)
                ok = out.returncode == 0
            if not ok:
                return f"weights missing on {label}: {raw}"
        return None

    def _start_worker(self, name: str, yaml: dict) -> str | None:
        """Starts rank 1 on the worker; returns an error or None.

        The remote process is detached with ``setsid`` so it outlives the ssh
        session, and its log stays on the worker — the head only polls it
        for the "links up" line.
        """
        import shlex

        worker = yaml["worker_host"]
        tport = int(yaml.get("transport_port", DEFAULT_TRANSPORT_PORT))
        log = f"$HOME/.exllamav3-{name}.log"
        _ssh(worker, _pkill_cmd(tport),
             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        time.sleep(1)

        cmd, env = self._rank_cmd(yaml, rank=1)
        env_prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items() if v)
        remote = f"cd {_worker_path(yaml.get('exllamav3_repo', DEFAULT_REPO))} && "
        if env_prefix:
            remote += env_prefix + " "
        remote += (f"setsid nohup {shlex.join(cmd)} > {log} 2>&1 "
                   f"< /dev/null & echo started")
        result = _ssh(worker, remote, capture_output=True, text=True)
        if result.returncode != 0 or "started" not in (result.stdout or ""):
            return f"rank 1 launch failed on {worker} (exit {result.returncode})"

        try:
            gate = int(yaml.get("gate_timeout", 600))
        except (ValueError, TypeError):
            gate = 600
        elapsed = 0
        while elapsed < gate:
            probe = _ssh(worker, f"grep -c 'links up' {log} 2>/dev/null || true",
                         capture_output=True, text=True)
            if (probe.stdout or "").strip() not in ("", "0"):
                print(f"  Worker rank 1 links up on {worker} "
                      f"(log: {log.replace('$HOME', '~')})")
                return None
            dead = _ssh(worker, f"pgrep -f 'multinode_pipeline.*--port {tport}' >/dev/null",
                        capture_output=True)
            if dead.returncode != 0:
                taillog = _ssh(worker, f"tail -15 {log} 2>/dev/null || echo '(no log yet)'",
                               capture_output=True, text=True)
                return (f"rank 1 on {worker} exited before links came up:\n"
                        f"{(taillog.stdout or '').strip()}")
            time.sleep(5)
            elapsed += 5
            if elapsed % 60 == 0:
                print(f"  …waiting for worker links ({elapsed // 60}m)")
        return f"rank 1 on {worker} never reported links up after {gate}s (see {log})"

    def _stop_worker(self, yaml: dict) -> None:
        """Kills this recipe's rank 1 on the worker, unconditionally."""
        worker = yaml.get("worker_host")
        if not worker:
            return
        tport = yaml.get("transport_port", DEFAULT_TRANSPORT_PORT)
        print(f"Stopping exllamav3 rank 1 on {worker} (port {tport})...")
        _ssh(worker, _pkill_cmd(int(tport)),
             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

    def _pid_alive(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def start(self, name: str, yaml: dict) -> RunningState:
        pid_file = RUN_DIR / name
        if pid_file.exists():
            try:
                if self._pid_alive(int(pid_file.read_text().strip())):
                    print(f"exllamav3-dual {name} already running")
                    return RunningState("ready")
            except (ValueError, OSError):
                pass
            pid_file.unlink(missing_ok=True)

        err = self._preflight(name, yaml)
        if err:
            print(f"  ✗ {err}")
            return RunningState("dead", detail=err)

        print(f"Starting exllamav3-dual {name} "
              f"(rank0=local, rank1={yaml['worker_host']})...")
        err = self._start_worker(name, yaml)
        if err:
            print(f"  ✗ {err}")
            return RunningState("dead", detail=err)

        cmd, env = self._rank_cmd(yaml, rank=0)
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        log_path = RUN_DIR / f"{name}.log"
        full_env = dict(os.environ)
        full_env.update({k: v for k, v in env.items() if v})
        try:
            job_timeout = int(yaml.get("job_timeout", 3600))
        except (ValueError, TypeError):
            job_timeout = 3600
        print(f"  Running rank 0 locally (timeout {job_timeout}s, log: {log_path})...")
        try:
            log_fh = open(log_path, "wb")
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT,
                                    start_new_session=True, env=full_env)
            pid_file.write_text(str(proc.pid))
            assert proc.stdout is not None
            for line in proc.stdout:
                log_fh.write(line)
                log_fh.flush()
                try:
                    print("  │ " + line.decode(errors="replace").rstrip())
                except Exception:
                    pass
            proc.wait(timeout=job_timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            print(f"  ✗ rank 0 timed out after {job_timeout}s; stopping worker.")
            self._stop_worker(yaml)
            return RunningState("timeout", detail=str(log_path))
        finally:
            try:
                log_fh.close()
            except Exception:
                pass
            pid_file.unlink(missing_ok=True)
            self._stop_worker(yaml)

        if proc.returncode == 0:
            print(f"  Job {name} finished (log: {log_path})")
            return RunningState("ready", detail=str(log_path))
        try:
            tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-15:])
        except Exception:
            tail = "(could not read log)"
        print(f"  ✗ rank 0 exited (code {proc.returncode}). Last log lines:")
        print(tail)
        return RunningState("dead", detail=tail)

    def stop(self, name: str, yaml: dict) -> None:
        """Stops the local rank 0, then the worker even if the head was dead.

        A stranded rank 1 holds the worker's layer slice resident, which is
        most of a node's memory for the models this runtime exists to run.
        """
        pid_file = RUN_DIR / name
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
                if self._pid_alive(pid):
                    print(f"Stopping exllamav3 rank 0 {name} (PID {pid})...")
                    _escalate_kill(pid, "multinode_pipeline")
                pid_file.unlink()
            except (ValueError, OSError):
                pid_file.unlink(missing_ok=True)
        else:
            tport = yaml.get("transport_port", DEFAULT_TRANSPORT_PORT)
            subprocess.run(["pkill", "-f",
                            f"multinode_pipeline.*--port {tport}( |$)"],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, check=False)
            print(f"exllamav3-dual {name} not running (no pid file)")
        self._stop_worker(yaml)

    def is_running(self, name: str, runtime_name: str) -> bool:
        from lmswitch.system.checks import _is_running
        return _is_running(name, runtime_name)

    def is_ready(self, name: str, port: int, timeout: int = 300) -> str:
        # Jobs expose no port: running means the generation is in flight.
        return "ready" if self.is_running(name, "exllamav3-dual") else "dead"
