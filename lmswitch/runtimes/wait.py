"""Readiness polling."""

import subprocess
import time


def _wait_ready(name: str, port: int, timeout: int, alive) -> str:
    """Polls until the server answers on ``port``, the backend dies, or timeout.

    Args:
        name: Model id (used only for progress messages).
        port: OpenAI-compatible port to probe.
        timeout: Seconds to wait before giving up.
        alive: Zero-arg callable returning ``False`` once the process or
            container backing the server has exited.

    Returns:
        ``"ready"`` if the port responded, ``"dead"`` if the backend exited, or
        ``"timeout"`` if neither happened within ``timeout`` seconds.
    """
    elapsed = 0
    while elapsed < timeout:
        if not alive():
            return "dead"
        r = subprocess.run(
            ["curl", "-s", "-m", "5", f"http://localhost:{port}/v1/models"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
        )
        if r.returncode == 0:
            return "ready"
        time.sleep(2)
        elapsed += 2
        if elapsed % 10 == 0:
            print(f"  …loading ({elapsed}s)")
    return "timeout"
