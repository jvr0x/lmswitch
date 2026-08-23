"""systemd auto-restart support."""

import subprocess
from pathlib import Path

from lmswitch.system.io import HOME

# Runtimes whose unit has to survive a cold boot: the peer host, the shared
# weights mount and dockerd are all still coming up when the user manager
# reaches default.target.
_DUAL_RUNTIMES = ("vllm-dual", "vllm-dual-ray", "llama-dual")

_SYSTEMD_UNIT = """\
[Unit]
Description=lmswitch serve {name}
After=network.target
{unit_extra}
[Service]
Type=simple
ExecStart=%h/.local/bin/lmswitch serve {name}
Restart={restart}
RestartSec={restart_sec}
TimeoutStartSec=300
StandardOutput={stdout}
StandardError={stderr}

[Install]
WantedBy=default.target
"""

# A user unit cannot order itself after a system mount or network-online, so
# the wait lives in `lmswitch serve` (dual_serve._gate). What the unit must do
# is stop rate-limiting the retries: a peer that takes minutes to boot would
# otherwise burn the default 5-starts-in-10s budget and leave the unit failed.
_DUAL_UNIT_EXTRA = """StartLimitIntervalSec=0

"""


def _start_systemd(name: str, yaml: dict, restart: str) -> None:
    """Writes and enables the user unit that supervises *name*.

    Args:
        name: Model id (the yaml stem).
        yaml: Parsed recipe — its ``runtime`` decides the boot-hardening knobs.
        restart: systemd ``Restart=`` policy from the recipe.
    """
    is_dual = yaml.get("runtime") in _DUAL_RUNTIMES
    unit_name = f"lmswitch@{name}.service"
    unit_path = HOME / ".config" / "systemd" / "user" / unit_name
    unit_path.parent.mkdir(parents=True, exist_ok=True)
    unit_path.write_text(_SYSTEMD_UNIT.format(
        name=name,
        restart=restart,
        unit_extra=_DUAL_UNIT_EXTRA if is_dual else "\n",
        # 30s, not 5: a dual restart re-runs the gate and a TP=2 load, and
        # hammering the peer's ssh every 5s while it boots achieves nothing.
        restart_sec=30 if is_dual else 5,
        # Dual startup is long and can stall on a peer that never appears —
        # keep the gate/preflight lines in the journal so a failed boot is
        # diagnosable. Single-node units stay quiet (they have their own logs).
        stdout="journal" if is_dual else "null",
        stderr="journal" if is_dual else "null",
    ))
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
    subprocess.run(["systemctl", "--user", "enable", "--now", unit_name], check=False)
    print(f"Started {name} via systemd (restart={restart})")
