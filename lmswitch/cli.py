"""CLI: rendering, commands, interactive TUI, and main entry point."""

import json
import re
import shutil
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
    _cluster_hosts,
)
from lmswitch.system.checks import _listening_ports, _is_running
from lmswitch.system import _get_sync_targets
from lmswitch.system import usage as usage_mod
from lmswitch.system.memory import _memory_check
from lmswitch.models.loader import load_models
from lmswitch.models.cluster import gather_cluster_models
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


def _annotate_running(models: list[dict]) -> list[dict]:
    """Fills m["running"] for local models; remote entries arrive pre-filled."""
    for m in models:
        if "running" not in m:
            m["running"] = _is_running(m["name"], m["runtime"])
    return models


def _gather_cluster_models() -> list[dict]:
    """Peers' models (see ``models.cluster.gather_cluster_models``), excluding
    anything with a YAML in this node's own ``CONF_DIR``."""
    local_names = {p.stem for p in CONF_DIR.glob("*.yaml")} if CONF_DIR.is_dir() else set()
    return gather_cluster_models(local_names)


def load_cluster_models() -> list[dict]:
    """Local models (running-annotated) + peers' models, family-sorted.

    Deduplicates: when the same model name appears both locally and remotely,
    keeps the running entry (local preferred), drops the stale stopped local
    entry so the table doesn't show unreachable duplicates.
    """
    local = _annotate_running(load_models())
    remote = _gather_cluster_models()
    _annotate_running(remote)

    # Build lookup by name; prefer running over stopped, local over remote.
    best: dict[str, dict] = {}
    for m in remote:
        name = m["name"]
        if name not in best:
            best[name] = m
        else:
            existing = best[name]
            # Prefer running over stopped; if tie, prefer local (no remote_host).
            if m["running"] and not existing["running"]:
                best[name] = m
            elif m["running"] == existing["running"] and not existing.get("remote_host"):
                best[name] = m  # local wins over remote at same running state

    for m in local:
        name = m["name"]
        if name not in best:
            best[name] = m
        else:
            existing = best[name]
            # Local always wins if it's running. If local is stopped but remote
            # is running, keep remote (the local entry would be unreachable).
            if m["running"]:
                best[name] = m
            elif not existing["running"] and m.get("remote_host"):
                best[name] = m  # local stopped beats remote stopped

    models = list(best.values())
    models.sort(key=lambda m: (m.get("fam_order", 99), m["name"]))
    return models


def render(models: list[dict]) -> None:
    _annotate_running(models)
    name_w = max(max((len(m["name"]) for m in models), default=4), 4)
    loaded = [m for m in models if m["running"]]
    total_loaded = sum(m["size"] for m in loaded)
    downloaded = [m for m in models if m["present"]]
    total_disk = sum(m["size"] for m in downloaded)

    print()
    ram = _ram_line()
    if ram:
        total, used, avail = ram
        # GGUF weights are mmap'd file-backed pages, so /proc/meminfo counts
        # them as reclaimable cache and "used" looks tiny with a 36G model
        # loaded. Treat local loaded GGUF weights as used — evicting them
        # would thrash inference. (vLLM CUDA allocations already show as used;
        # peers' models live in the other box's RAM.)
        gguf_gib = sum(m["size"] for m in loaded
                       if m["runtime"] == "llama"
                       and not m.get("remote_host")) / 1024 ** 3
        used = min(used + gguf_gib, total)
        avail = max(avail - gguf_gib, 0.0)
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
    # HOST column only appears on cluster setups (a peer's model or a dual
    # model is in the table) so single-box output stays byte-identical.
    show_host = any(m.get("host") not in (None, "") and
                    (m.get("remote_host") or m.get("host") == "dual")
                    for m in models) or bool(_cluster_hosts())
    host_w = max((len(str(m.get("host", ""))) for m in models), default=4)
    host_w = max(host_w, 4)
    host_hdr = f"  {'HOST':<{host_w}}" if show_host else ""
    header = (f"  {'#':>2}  S  {'TYPE':<5}  {'NAME':<{name_w}}  "
              f"{'SIZE':>7}  DL  {'PORT':>5}{host_hdr}  DISPLAY")
    print(_c(header, "1"))

    disp_w = max((len(m["display"]) for m in models), default=7)
    rule_w = name_w + disp_w + 37 + ((host_w + 2) if show_host else 0)
    cur_family = None
    for i, m in enumerate(models, 1):
        running = m["running"]
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
        host_col = ""
        if show_host:
            host = str(m.get("host", "-") or "-")
            color = "35" if host == "dual" else ("2" if m.get("remote_host") else "36")
            host_col = "  " + _c(f"{host:<{host_w}}", color)
        print(f"  {i:>2}  {dot}{restart}  {m['type']:<5}  {name}  "
              f"{size:>7}  {dl}  {port:>5}{host_col}  {disp}")
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

    # `pip install -e .` already provides a `lmswitch` console script bound to
    # the right interpreter. Only fall back to a hand-written wrapper if no
    # entry point is reachable on PATH, and pin its shebang to the interpreter
    # that actually has the package importable (this one) — `/usr/bin/env
    # python3` would resolve to system python, which usually can't import the
    # editable install and would break the command.
    if shutil.which("lmswitch"):
        print("  Console script already installed (pip entry point); skipping wrapper.")
    else:
        bin_dir = HOME / ".local" / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        link = bin_dir / "lmswitch"
        link.write_text(
            f"#!{sys.executable}\n"
            "from lmswitch.cli import main\n"
            "main()\n"
        )
        link.chmod(0o755)
        print(f"  Installed wrapper: {link}")

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


def cmd_list(as_json: bool = False) -> None:
    if as_json:
        # Machine-readable dump for cluster peers: local YAMLs only (no
        # recursion into other hosts) with running state resolved here.
        import socket
        models = _annotate_running(load_models())
        print(json.dumps({
            "host": socket.gethostname(),
            "serve_host": SPARK_HOST,
            "models": models,
        }))
        return
    models = load_cluster_models()
    if not models:
        print(f"No models found in {CONF_DIR}")
        return
    render(models)


def _resolve(name_or_idx: str) -> tuple[str, str | None]:
    """Resolves a name or table index to ``(name, remote_host)``.

    ``remote_host`` is the SSH alias of the peer owning the model, or None
    for local (and dual) models. Numeric indexes match the merged cluster
    table the user just looked at.
    """
    if name_or_idx.isdigit():
        idx = int(name_or_idx)
        models = load_cluster_models()
        if 1 <= idx <= len(models):
            m = models[idx - 1]
            return m["name"], m.get("remote_host")
        sys.exit(f"No model #{name_or_idx} (1-{len(models)})")
    if (CONF_DIR / f"{name_or_idx}.yaml").exists():
        return name_or_idx, None
    for m in _gather_cluster_models():
        if m["name"] == name_or_idx:
            return name_or_idx, m["remote_host"]
    sys.exit(f"Unknown model: {name_or_idx}")


def _delegate(host: str, action: str, name: str) -> None:
    """Runs ``lmswitch <action> <name>`` on a peer node over SSH."""
    print(f"→ {host}: lmswitch {action} {name}")
    subprocess.run(["ssh", "-o", "BatchMode=yes", host,
                    "$HOME/.local/bin/lmswitch", action, name], check=False)


def cmd_on(target: str) -> None:
    name, remote = _resolve(target)
    if remote:
        _delegate(remote, "on", name)
        return
    yaml_path = CONF_DIR / f"{name}.yaml"
    yaml = _load_yaml(yaml_path)
    start_model(name, yaml)
    regen_all()


def cmd_off(target: str) -> None:
    name, remote = _resolve(target)
    if remote:
        _delegate(remote, "off", name)
        return
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
    """Foreground serve — used by systemd for restart-managed models.

    Must actually exit when the backing process dies, so systemd's
    ``Restart=always`` can recover it. A supervisor that only sleeps forever
    defeats that entirely: this wrapper's own PID stays alive even after
    llama-server crashes underneath it, so systemd sees a healthy unit and
    never restarts anything — the model silently never comes back until
    someone notices and restarts the unit by hand.
    """
    yaml_path = CONF_DIR / f"{name}.yaml"
    if not yaml_path.exists():
        sys.exit(f"Config not found: {yaml_path}")
    yaml = _load_yaml(yaml_path)
    runtime = yaml.get("runtime", "llama")
    if runtime == "vllm":
        _start_vllm_foreground(name, yaml)
    else:
        state = _start_llama_direct(name, yaml)
        if state.status != "ready":
            # Startup itself failed — exit now so systemd retries right
            # away instead of idling in a fake "up" state.
            sys.exit(f"{name} failed to start ({state.status})")
        while True:
            time.sleep(2)
            # proc.poll() (not a PID-liveness check) — it's the only thing
            # that both detects AND reaps the child, so a crash surfaces
            # immediately instead of leaving a zombie behind while this
            # wrapper sleeps on forever, unnoticed by systemd.
            if state.proc is not None and state.proc.poll() is not None:
                sys.exit(f"{name} process exited (code {state.proc.returncode}); "
                        f"handing back to systemd for restart")


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


def _human_sec(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = minutes / 60
    if hours < 24:
        return f"{hours:.1f}h"
    days = hours / 24
    return f"{days:.1f}d"


def _human_bytes(n: int) -> str:
    """Format bytes into human-readable size."""
    if n == 0:
        return "-"
    gb = n / 1024 ** 3
    if gb >= 1000:
        return f"{gb / 1024:.1f}T"
    return f"{gb:.1f}G"


def cmd_stats() -> None:
    """Display usage statistics from the JSONL events file."""
    from datetime import datetime as _dt

    events = usage_mod.query_events()
    if not events:
        print("No usage events recorded yet.")
        return

    total_starts = 0
    total_stops = 0
    by_model: dict[str, dict] = {}
    by_runtime: dict[str, int] = {}
    start_times: dict[str, float] = {}

    # Track token counts
    total_prompt_tokens = 0
    total_gen_tokens = 0
    total_all_tokens = 0

    for ev in events:
        action = ev.get("action", "")
        model = ev.get("model", "unknown")
        runtime = ev.get("runtime", "unknown")
        ts_str = ev.get("ts", "")
        try:
            ts = _dt.fromisoformat(ts_str).timestamp()
        except (ValueError, TypeError):
            ts = 0.0
        duration = float(ev.get("duration", 0) or 0)
        size = int(ev.get("size", 0) or 0)
        prompt_tok = int(ev.get("prompt_tokens", 0) or 0)
        gen_tok = int(ev.get("generation_tokens", 0) or 0)
        total_tok = int(ev.get("total_tokens", 0) or 0)

        if action == "start":
            total_starts += 1
            start_times[model] = ts
            if model not in by_model:
                by_model[model] = {
                    "starts": 0, "stops": 0, "total_duration": 0.0,
                    "total_size": 0, "total_prompt_tokens": 0,
                    "total_gen_tokens": 0, "total_tokens": 0,
                }
            by_model[model]["starts"] += 1
            by_model[model]["total_size"] += size
        elif action == "stop":
            total_stops += 1
            total_prompt_tokens += prompt_tok
            total_gen_tokens += gen_tok
            total_all_tokens += total_tok
            if model not in by_model:
                by_model[model] = {
                    "starts": 0, "stops": 0, "total_duration": 0.0,
                    "total_size": 0, "total_prompt_tokens": 0,
                    "total_gen_tokens": 0, "total_tokens": 0,
                }
            by_model[model]["stops"] += 1
            by_model[model]["total_duration"] += duration
            by_model[model]["total_size"] += size
            by_model[model]["total_prompt_tokens"] += prompt_tok
            by_model[model]["total_gen_tokens"] += gen_tok
            by_model[model]["total_tokens"] += total_tok

        if runtime:
            by_runtime[runtime] = by_runtime.get(runtime, 0) + 1

    total_duration = sum(m.get("total_duration", 0) for m in by_model.values())

    # ---- Display: TUI dashboard ----
    _print_stats_dashboard(
        total_starts, total_stops, total_duration, total_prompt_tokens,
        total_gen_tokens, total_all_tokens, len(by_model), by_model,
        by_runtime, events,
    )


def _bar(value: int, max_val: int, width: int = 20, color: str = "32") -> str:
    """Render a simple ASCII bar chart segment."""
    if max_val == 0:
        return "─" * width
    filled = round(value / max_val * width)
    return _c("█" * filled, color) + "─" * (width - filled)


def _format_tokens(n: int) -> str:
    """Format token count with K/M suffix."""
    if n == 0:
        return "0"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _print_stats_dashboard(
    total_starts: int,
    total_stops: int,
    total_duration: float,
    total_prompt_tokens: int,
    total_gen_tokens: int,
    total_all_tokens: int,
    model_count: int,
    by_model: dict[str, dict],
    by_runtime: dict[str, int],
    events: list[dict],
) -> None:
    """Render the usage statistics dashboard."""
    print()
    print(_c("  Usage Statistics", "1"))
    print()

    # Summary table
    print(_c("  Summary", "1"))
    print()
    print(f"  {'Starts:':<15} {total_starts}")
    print(f"  {'Stops:':<15} {total_stops}")
    print(f"  {'Total uptime:':<15} {_human_sec(total_duration)}")
    print(f"  {'Models tracked:':<15} {model_count}")
    print()

    # Token stats
    if total_all_tokens > 0 or total_stops > 0:
        print(_c("  Token Counts", "1"))
        print()
        print(f"  {'Prompt tokens:':<15} {_format_tokens(total_prompt_tokens)}")
        print(f"  {'Generated tokens:':<15} {_format_tokens(total_gen_tokens)}")
        print(f"  {'Total tokens:':<15} {_format_tokens(total_all_tokens)}")
        print()

        # Per-model token bars (only if we have token data)
        if by_model and any(
            m.get("total_tokens", 0) > 0 for m in by_model.values()
        ):
            print(_c("  Token Breakdown", "1"))
            print()

            # Find max token count for bar scaling
            max_tokens = max(
                (m.get("total_tokens", 0) for m in by_model.values()),
                default=1,
            )
            if max_tokens == 0:
                max_tokens = 1

            for model in sorted(by_model):
                info = by_model[model]
                tok = info.get("total_tokens", 0)
                prompt_tok = info.get("total_prompt_tokens", 0)
                gen_tok = info.get("total_gen_tokens", 0)
                display = model
                for ev in events:
                    if ev.get("model") == model and ev.get("action") == "start":
                        display = ev.get("display_name", model)
                        break
                dot = _c("●", "32") if info["starts"] > 0 else _c("○", "2")
                bar_total = _bar(tok, max_tokens, width=25, color="32")
                bar_prompt = _bar(prompt_tok, max_tokens, width=12, color="33")
                bar_gen = _bar(gen_tok, max_tokens, width=12, color="34")
                print(f"  {dot} {display}")
                print(f"      total : {bar_total} {_format_tokens(tok)}")
                print(f"      prompt: {bar_prompt} {_format_tokens(prompt_tok)}")
                print(f"      gen   : {bar_gen} {_format_tokens(gen_tok)}")
            print()

    # Runtime breakdown with bars
    if by_runtime:
        print(_c("  By Runtime", "1"))
        print()
        max_events = max(by_runtime.values(), default=1)
        for rt in sorted(by_runtime):
            count = by_runtime[rt]
            bar = _bar(count, max_events, width=25, color="36")
            print(f"  {rt:<12} {bar} {count} events")
        print()

    # Latest events table
    print(_c("  Latest Events", "1"))
    print()
    print(f"  {'ACTION':>6}  {'DISPLAY':<20}  {'RUNTIME':<8}  {'PROMPT':>8}  {'GEN':>8}  {'TOTAL':>8}  {'TIME'}")
    recent = events[-15:] if len(events) > 15 else events
    for ev in recent:
        ts = ev.get("ts", "")
        action = ev.get("action", "?")
        model = ev.get("model", "?")
        runtime = ev.get("runtime", "?")
        display = ev.get("display_name", model)
        prompt_tok = int(ev.get("prompt_tokens", 0) or 0)
        gen_tok = int(ev.get("generation_tokens", 0) or 0)
        total_tok = int(ev.get("total_tokens", 0) or 0)
        dot = _c("●", "32") if action == "start" else _c("○", "2")
        act_label = action.upper()[:6].rjust(6)
        time_str = ts.split("T")[1][:8] if "T" in ts else ts
        print(
            f"  {dot} {act_label}  {display:<20}  {runtime:<8}  "
            f"{_format_tokens(prompt_tok):>8}  {_format_tokens(gen_tok):>8}  "
            f"{_format_tokens(total_tok):>8}  {time_str}"
        )
    print()


def cmd_stats_clear() -> None:
    """Clear all usage statistics."""
    usage_mod.clear_events()
    print("Usage statistics cleared.")


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
  lmswitch stats            show usage statistics (starts, stops, uptime, per-model)
  lmswitch stats-clear      clear all usage statistics
  lmswitch -h, --help       show this help
  lmswitch -v, --version    show the version
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
        if args[0] in ("-v", "--version"):
            from lmswitch import __version__
            print(f"lmswitch {__version__}")
            return
        if args[0] in ("list", "status", "ls"):
            cmd_list(as_json="--json" in args[1:])
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
        elif args[0] == "stats":
            cmd_stats()
        elif args[0] == "stats-clear":
            cmd_stats_clear()
        else:
            sys.exit(_HELP)
    except KeyboardInterrupt:
        print("\nAborted.")
        raise SystemExit(130)


def show() -> None:
    models = load_cluster_models()
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
        if choice.lower() == "stats":
            cmd_stats()
            continue
        if not models:
            print("  (no models to toggle — use `lmswitch add <name>` first)\n")
            continue
        nums = [int(t) for t in re.split(r"[^0-9]+", choice) if t]
        if not nums or any(not (1 <= n <= len(models)) for n in nums):
            print(f"  ? enter 1-{len(models)} (one or more, e.g. 8 9 24), or 'stats' (or q)\n")
            continue
        for n in nums:
            m = models[n - 1]
            action = "off" if m.get("running") else "on"
            if m.get("remote_host"):
                # Peer-owned model: delegate the whole toggle to that node's
                # lmswitch so its runtime + sync logic apply there.
                _delegate(m["remote_host"], action, m["name"])
            else:
                toggle(m["name"], action)
        models = load_cluster_models()
        print()


def toggle(target: str, action: str) -> None:
    name, remote = _resolve(target)
    if remote:
        _delegate(remote, action, name)
        return
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
