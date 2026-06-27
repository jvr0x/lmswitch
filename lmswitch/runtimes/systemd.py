"""systemd auto-restart support."""

import subprocess
from pathlib import Path

from lmswitch.system.io import HOME

_SYSTEMD_UNIT = """\
[Unit]
Description=lmswitch serve {name}
After=network.target

[Service]
Type=simple
ExecStart=%h/.local/bin/lmswitch serve {name}
Restart={restart}
RestartSec=5
TimeoutStartSec=300
StandardOutput=null
StandardError=null

[Install]
WantedBy=default.target
"""


def _start_systemd(name: str, yaml: dict, restart: str) -> None:
    unit_name = f"lmswitch@{name}.service"
    unit_path = HOME / ".config" / "systemd" / "user" / unit_name
    unit_path.parent.mkdir(parents=True, exist_ok=True)
    unit_path.write_text(_SYSTEMD_UNIT.format(name=name, restart=restart))
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
    subprocess.run(["systemctl", "--user", "enable", "--now", unit_name], check=False)
    print(f"Started {name} via systemd (restart={restart})")
