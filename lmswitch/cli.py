"""CLI: rendering, commands, interactive TUI, and main entry point."""

import json
import re
import subprocess
import sys
import time
from pathlib import Path

from lmswitch.system.io import (
    TTY,
    CONF_DIR,
    CONFIG_FILE,
    HOME,
    OPENCODE,
    OPENCODE_EXPORT,
    HERMES_CONFIG,
    GROK_CONFIG,
    SPARK_HOST,
    SYNC_OPENCODE,
    SYNC_HERMES,
    SYNC_GROK,
    SCRIPT_DIR,
    _c,
    _read_config,
    _human,
    _load_yaml,
)
from lmswitch.system.checks import _listening_ports, _is_running
from lmswitch.system import _get_sync_targets
from lmswitch.system.memory import _memory_check
from lmswitch.models.loader import load_models
from lmswitch.runtimes import (
    start_model,
    stop_model,
    _start_llama_direct,
    _start_vllm_direct,
    _start_vllm_foreground,
    _start_systemd,
)
from lmswitch.sync import regen_opencode, regen_hermes, regen_grok, regen_all


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _banner() -> str:
    """Returns the lmswitch wordmark (ANSI Shadow block art) for the TUI header."""
    art = "\n".join((
        "██╗     ███╗   ███╗███████╗██╗    ██╗██╗████████╗ ██████╗██╗  ██╗",
        "██║     ████╗ ████║██╔════╝██║    ██║██║╚══██╔══╝██╔════╝██║  ██║",
        "██║     ██╔████╔██║███████╗██║ █╗ ██║██║   ██║   ██║     ███████║",
        "██║     ██║╚██╔╝██║╚════██║██║███╗██║██║   ██║   ██║     ██╔══██║",
        "███████╗██║ ╚═╝ ██║███████║╚███╔███╔╝██║   ██║   ╚██████╗██║  ██║",
        "╚══════╝╚═╝     ╚═╝╚══════╝ ╚══╝╚══╝ ╚═╝   ╚═╝    ╚═════╝╚═╝  ╚═╝",
    ))
    tagline = "  local LLM switcher · GGUF + vLLM"
    return "\n" + _c(art, "36") + "\n" + _c(tagline, "2") + "\n"


def render(models: list[dict]) -> None:
    name_w = max(max((len(m["name"]) for m in models), default=4), 4)
    loaded = [m for m in models if _is_running(m["name"], m["runtime"])]
    total_loaded = sum(m["size"] for m in loaded)
    downloaded = [m for m in models if m["present"]]
    total_disk = sum(m["size"] for m in downloaded)

    print()
    ram = _ram_line()
    if ram:
        total, used, avail = ram
        rows = [
            ("RAM", f"{total:.0f}Gi total   "
                    + _c(f"{used:.0f}Gi used", "33") + "   "
                    + _c(f"{avail:.0f}Gi available", "32")),
            ("Models", _c(f"~{_human(total_loaded)} weights", "1")
                       + f"   {len(loaded)} / {len(models)} loaded"),
            ("Disk", _c(_human(total_disk), "1")
                     + f"   {len(downloaded)} / {len(models)} downloaded"),
        ]
        lbl_w = max(len(lbl) for lbl, _ in rows)
        for lbl, val in rows:
            print(f"  {lbl:<{lbl_w}} {_c('│', '2')} {val}")
    print("  " + _c("● loaded", "32") + "   " + _c("○ stopped", "2")
          + "      " + _c("✓ downloaded", "32") + "   "
          + _c("✗ missing", "31"))
    if any(m.get("restart") for m in models):
        print("  " + _c("R", "36") + " = auto-restart on failure (systemd)")
    print()
    header = (f"  {'#':>2}  S  {'TYPE':<5}  {'NAME':<{name_w}}  "
              f"{'SIZE':>7}  DL  {'PORT':>5}  DISPLAY")
    print(_c(header, "1"))

    disp_w = max((len(m["display"]) for m in models), default=7)
    rule_w = name_w + disp_w + 37
    cur_family = None
    for i, m in enumerate(models, 1):
        running = _is_running(m["name"], m["runtime"])
        if m["family"] != cur_family:
            cur_family = m["family"]
            label = f" {cur_family} "
            print(_c("  " + label + "─" * (rule_w - len(label)), "36"))
        dot = _c("●", "32") if running else _c("○", "2")
        dl = _c("✓", "32") if m["present"] else _c("✗", "31")
        port = str(m["port"]) if m["port"] else "-"
        size = _human(m["size"])
        name = f"{m['name']:<{name_w}}"
        disp = m["display"]
        if running:
            name = _c(name, "1")
            disp = _c(disp, "32")
        elif not m["present"]:
            name = _c(name, "2")
            disp = _c(disp, "2")
        restart = "R" if m.get("restart") else " "
        print(f"  {i:>2}  {dot}{restart}  {m['type']:<5}  {name}  "
              f"{size:>7}  {dl}  {port:>5}  {disp}")
    print()


def _ram_line():
    """Wrapper for system._ram_line that can be stubbed in tests."""
    from lmswitch.system.memory import _ram_line as _rl
    return _rl()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_init() -> None:
    CONF_DIR.mkdir(exist_ok=True)
    from lmswitch.system.io import RUN_DIR
    RUN_DIR.mkdir(exist_ok=True)

    try:
        if not CONFIG_FILE.exists():
            default_models = str(HOME / "models")
            ans = input(f"Where are your models stored? [{default_models}] ").strip()
            models_dir = ans or default_models
            CONFIG_FILE.write_text(f'MODELS_DIR="{models_dir}"\n')
            print(f"  Config written to {CONFIG_FILE}")
    except KeyboardInterrupt:
        print("\nAborted.")
        return

    bin_dir = HOME / ".local" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    link = bin_dir / "lmswitch"
    # Write a small wrapper that runs `python -m lmswitch`
    link.write_text(
        "#!/usr/bin/env python3\n"
        'import sys\n'
        'from lmswitch.cli import main\n'
        "main()\n"
    )
    link.chmod(0o755)
    print(f"  Installed: {link} (python -m lmswitch wrapper)")

    existing_cfg = _read_config()
    sync_cfg_lines: list[str] = []

    try:
        has_hermes = HERMES_CONFIG.exists()
        has_grok = GROK_CONFIG.exists()
        has_opencode = OPENCODE.exists()

        if has_opencode:
            val = existing_cfg.get(SYNC_OPENCODE, "true")
            if val.lower() not in ("true", "1", "yes"):
                val = "true"
            print(f"  Sync to opencode.json [y/N] ({'yes' if val.lower() in ('true','1','yes') else 'no'})")
            ans = input("  > ").strip().lower()
            if ans == "y":
                sync_cfg_lines.append(f'{SYNC_OPENCODE}=true')
            else:
                sync_cfg_lines.append(f'{SYNC_OPENCODE}=false')
        else:
            sync_cfg_lines.append(f'{SYNC_OPENCODE}=false')
            print("  opencode.json not found — sync disabled.")

        if has_hermes:
            print(f"  Sync to hermes config [Y/n] (hermes detected)")
            ans = input("  > ").strip().lower()
            if ans == "n":
                sync_cfg_lines.append(f'{SYNC_HERMES}=false')
            else:
                sync_cfg_lines.append(f'{SYNC_HERMES}=true')
        else:
            sync_cfg_lines.append(f'{SYNC_HERMES}=false')
            print("  hermes config not found — sync disabled.")

        if has_grok:
            print(f"  Sync to grok config [Y/n] (grok detected)")
            ans = input("  > ").strip().lower()
            if ans == "n":
                sync_cfg_lines.append(f'{SYNC_GROK}=false')
            else:
                sync_cfg_lines.append(f'{SYNC_GROK}=true')
        else:
            sync_cfg_lines.append(f'{SYNC_GROK}=false')
            print("  grok config not found — sync disabled.")

        if sync_cfg_lines:
            sync_cfg_content = "\n".join(sync_cfg_lines) + "\n"
            cfg_text = CONFIG_FILE.read_text()
            for key in (SYNC_OPENCODE, SYNC_HERMES, SYNC_GROK):
                cfg_text = re.sub(
                    rf'^{key}=.+$', '', cfg_text, flags=re.MULTILINE
                )
            cfg_text = cfg_text.rstrip() + "\n" + sync_cfg_content
            CONFIG_FILE.write_text(cfg_text)
            print(f"  Sync targets written to {CONFIG_FILE}")
    except KeyboardInterrupt:
        print("\nAborted.")
        return

    print(f"\nDone. Config dir: {CONF_DIR}")
    print(f"Add model configs to {CONF_DIR}/ as YAML files.")
    print(f"Run `lmswitch add <name>` to create one interactively.")


def cmd_list() -> None:
    models = load_models()
    if not models:
        print(f"No models found in {CONF_DIR}")
        return
    render(models)


def _resolve(name_or_idx: str) -> str | None:
    if name_or_idx.isdigit():
        idx = int(name_or_idx)
        models = load_models()
        if 1 <= idx <= len(models):
            return models[idx - 1]["name"]
        sys.exit(f"No model #{name_or_idx} (1-{len(models)})")
    if (CONF_DIR / f"{name_or_idx}.yaml").exists():
        return name_or_idx
    sys.exit(f"Unknown model: {name_or_idx}")


def cmd_on(target: str) -> None:
    name = _resolve(target)
    yaml_path = CONF_DIR / f"{name}.yaml"
    yaml = _load_yaml(yaml_path)
    runtime = yaml.get("runtime", "llama")
    start_model(name, yaml)
    regen_all()


def cmd_off(target: str) -> None:
    name = _resolve(target)
    yaml_path = CONF_DIR / f"{name}.yaml"
    yaml = _load_yaml(yaml_path)
    runtime = yaml.get("runtime", "llama")
    if yaml.get("restart"):
        unit = f"lmswitch@{name}.service"
        subprocess.run(["systemctl", "--user", "stop", unit], check=False)
        print(f"Stopped {name} (systemd)")
    else:
        stop_model(name, runtime)
    regen_all()


def cmd_sync() -> None:
    targets = _get_sync_targets()
    any_changed = False
    if "opencode" in targets:
        if regen_opencode():
            any_changed = True
    if "hermes" in targets:
        if regen_hermes():
            any_changed = True
    if "grok" in targets:
        if regen_grok():
            any_changed = True
    if any_changed:
        print(f"Synced {', '.join(targets)} to currently-serving models.")
    else:
        print(f"{', '.join(targets)} already in sync.")


def cmd_serve(name: str) -> None:
    """Foreground serve — used by systemd for restart-managed models."""
    yaml_path = CONF_DIR / f"{name}.yaml"
    if not yaml_path.exists():
        sys.exit(f"Config not found: {yaml_path}")
    yaml = _load_yaml(yaml_path)
    runtime = yaml.get("runtime", "llama")
    if runtime == "vllm":
        _start_vllm_foreground(name, yaml)
    else:
        _start_llama_direct(name, yaml)
        while True:
            time.sleep(1)


def cmd_add(name: str) -> None:
    path = CONF_DIR / f"{name}.yaml"
    if path.exists():
        sys.exit(f"Config already exists: {path}")

    try:
        print(f"Creating config for '{name}'")
        runtime = input("  Runtime [llama/vllm] (default: llama): ").strip() or "llama"
        model = input("  Model path (relative to models dir): ").strip()
        port = input("  Port (default: 8081): ").strip() or "8081"
        ctx = input("  Context length (default: 65536): ").strip() or "65536"
        display = input(f"  Display name (default: {name}): ").strip() or name

        lines = [
            f"runtime: {runtime}",
            f"model: {model}",
            f"port: {port}",
            f"ctx: {ctx}",
            f'display_name: "{display}"',
        ]
        if runtime == "vllm":
            gpu = input("  GPU memory utilization (default: 0.15): ").strip() or "0.15"
            lines.append(f"gpu_memory_utilization: {gpu}")
            tp = input("  Tool call parser (e.g. hermes, qwen3_coder, empty=none): ").strip()
            if tp:
                lines.append(f'tool_call_parser: "{tp}"')
            rp = input("  Reasoning parser (e.g. deepseek_r1, qwen3, empty=none): ").strip()
            if rp:
                lines.append(f'reasoning_parser: "{rp}"')
            extra = input("  Extra vllm args (space-separated, empty=none): ").strip()
            if extra:
                lines.append("extra_args:")
                for arg in extra.split():
                    lines.append(f'  - "{arg}"')
            restart = input("  Restart on failure? [y/N]: ").strip().lower()
            if restart == "y":
                lines.append('restart: "on-failure"')

        path.write_text("\n".join(lines) + "\n")
        print(f"\n  Written: {path}")
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_HELP = """\
lmswitch — list and toggle local LLMs from YAML configs.

Configs live in <lmswitch-dir>/ai-models/*.yaml. GGUF models run llama-server
as a background process; vLLM models run in Docker. Per-model restart: on-failure
opts into systemd-managed auto-restart.

On toggle, lmswitch syncs the list of currently-serving models to any tools
you opted into during `lmswitch init` (opencode.json, hermes config.yaml, grok
config.toml). Toggle `sync` to re-sync on demand.

Usage:
  lmswitch                  interactive TUI: list models, type numbers to toggle
  lmswitch list             print the model table (read-only)
  lmswitch on  <name|#>     start a model
  lmswitch off <name|#>     stop a model
  lmswitch sync             regenerate all synced configs from currently-serving models
  lmswitch init             bootstrap ai-models/ dir, config, symlink, and sync targets
  lmswitch add  <name>      create a new YAML config interactively
  lmswitch serve <name>     run a model in the foreground (for systemd)
  lmswitch -h, --help       show this help
"""


def main() -> None:
    try:
        args = sys.argv[1:]
        if not args:
            show()
            return
        if args[0] in ("-h", "--help"):
            print(_HELP)
            return
        if args[0] in ("list", "status", "ls"):
            cmd_list()
        elif args[0] in ("on", "start") and len(args) == 2:
            cmd_on(args[1])
        elif args[0] in ("off", "stop") and len(args) == 2:
            cmd_off(args[1])
        elif args[0] == "sync":
            cmd_sync()
        elif args[0] == "init":
            cmd_init()
        elif args[0] == "add" and len(args) == 2:
            cmd_add(args[1])
        elif args[0] == "serve" and len(args) == 2:
            cmd_serve(args[1])
        else:
            sys.exit(_HELP)
    except KeyboardInterrupt:
        print("\nAborted.")
        raise SystemExit(130)


def show() -> None:
    models = load_models()
    if not TTY:
        if not models:
            print(f"No models found in {CONF_DIR}")
            print("Run `lmswitch add <name>` to create a model config.")
            return
        render(models)
        return
    print(_banner())
    # Interactive: always enter the loop so the user sees the table/headers
    # and can discover commands even when no models are configured yet.
    while True:
        if models:
            render(models)
        else:
            print()
            print(f"  No models found in {CONF_DIR}")
            print("  Run `lmswitch add <name>` to create a model config.")
        try:
            choice = input("  Toggle # (space/comma separated, enter or q to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if choice.lower() in ("q", "quit", "exit", ""):
            return
        if not models:
            print("  (no models to toggle — use `lmswitch add <name>` first)\n")
            continue
        nums = [int(t) for t in re.split(r"[^0-9]+", choice) if t]
        if not nums or any(not (1 <= n <= len(models)) for n in nums):
            print(f"  ? enter 1-{len(models)} (one or more, e.g. 8 9 24) or q\n")
            continue
        for n in nums:
            m = models[n - 1]
            toggle(m["name"], "off" if _is_running(m["name"], m["runtime"]) else "on")
        models = load_models()
        print()


def toggle(target: str, action: str) -> None:
    name = _resolve(target)
    yaml_path = CONF_DIR / f"{name}.yaml"
    yaml = _load_yaml(yaml_path)
    runtime = yaml.get("runtime", "llama")
    verb = "start" if action == "on" else "stop"
    print(f"{verb.capitalize()}ing {name} ({runtime}) on port "
          f"{yaml.get('port', '?')} ...")
    if action == "on":
        start_model(name, yaml)
    else:
        if yaml.get("restart"):
            unit = f"lmswitch@{name}.service"
            subprocess.run(["systemctl", "--user", "stop", unit], check=False)
            print(f"  ↓ {name} stopped (systemd)")
        else:
            stop_model(name, runtime)
            print(f"  ↓ {name} stopped.")
    # Reason: keep all configured sync targets in sync with the TUI toggle,
    # matching the behavior of the non-interactive cmd_on / cmd_off commands.
    regen_all()
