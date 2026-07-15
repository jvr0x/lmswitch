"""Config sync: regenerate opencode.json, hermes config.yaml, grok config.toml."""

import json
import re
from pathlib import Path

from lmswitch.system.io import (
    CONF_DIR,
    OPENCODE,
    OPENCODE_EXPORT,
    HERMES_CONFIG,
    GROK_CONFIG,
    SPARK_HOST,
    SYNC_OPENCODE,
    SYNC_HERMES,
    SYNC_GROK,
)
from lmswitch.system import _get_sync_targets
from lmswitch.system.checks import _listening_ports
from lmswitch.models.loader import load_models
from lmswitch.models.cluster import gather_cluster_models


def _host_label(hostname: str) -> str:
    """Short display label for a host string, e.g. ``spark.local`` -> ``Spark``,
    ``dual`` -> ``Dual``."""
    return (hostname or "?").split(".")[0].capitalize()


def _cluster_running_models() -> list[dict]:
    """Peers' currently-running models, each already carrying ``serve_host``
    (where its API listens) and ``remote_host`` (the SSH alias). Best-effort:
    an unreachable peer or no ``CLUSTER_HOSTS`` configured just yields
    nothing, so single-box sync is unaffected. See
    ``models.cluster.gather_cluster_models``.
    """
    local_names = {p.stem for p in CONF_DIR.glob("*.yaml")} if CONF_DIR.is_dir() else set()
    return [m for m in gather_cluster_models(local_names)
            if m.get("running") and m.get("port")]


def regen_opencode() -> bool:
    if not OPENCODE.parent.exists():
        return False
    cfg: dict = {}
    if OPENCODE.exists():
        try:
            cfg = json.loads(OPENCODE.read_text())
        except Exception:
            cfg = {}
    cfg["$schema"] = "https://opencode.ai/config.json"
    cfg["plugin"] = ["@warp-dot-dev/opencode-warp"]

    ports = _listening_ports()
    local = [(m, f"http://{SPARK_HOST}:{m['port']}/v1", _host_label(SPARK_HOST))
             for m in load_models() if m["port"] != 0 and m["port"] in ports]
    remote = [(m, f"http://{m['serve_host']}:{m['port']}/v1",
              _host_label(m.get("host") or m["remote_host"]))
              for m in _cluster_running_models()]

    providers: dict = {}
    for m, base, label in local + remote:
        try:
            ctx = int(m["ctx"]) if m["ctx"] else 65536
        except (ValueError, TypeError):
            ctx = 65536
        providers[m["name"]] = {
            "npm": "@ai-sdk/openai-compatible",
            "name": f"{m['display']} ({label})",
            "options": {"baseURL": base},
            "models": {
                m["name"]: {
                    "name": m["display"],
                    "limit": {"context": ctx, "output": min(8192, ctx)},
                }
            },
        }
    cfg["provider"] = providers
    cfg["permission"] = {
        "read": {"*": "allow", "*.env": "deny", "*.env.*": "deny",
                 "*.env.example": "allow", "*.env.mocknet": "allow"},
        "edit": "allow", "glob": "allow", "grep": "allow", "bash": "ask",
        "task": "allow", "external_directory": "allow", "todowrite": "allow",
        "question": "allow", "webfetch": "allow", "websearch": "allow",
        "lsp": "allow", "skill": "allow",
    }
    payload = json.dumps(cfg, indent=2) + "\n"
    if OPENCODE.exists() and OPENCODE.read_text() == payload:
        return False
    tmp = OPENCODE.with_suffix(".tmp")
    tmp.write_text(payload)
    tmp.replace(OPENCODE)
    if OPENCODE_EXPORT.parent.exists():
        try:
            OPENCODE_EXPORT.write_text(payload)
        except Exception:
            pass
    return True


def regen_hermes() -> bool:
    if "hermes" not in _get_sync_targets():
        return False
    if not HERMES_CONFIG.exists():
        return False

    try:
        import yaml as _yaml
    except ImportError:
        return False

    try:
        cfg = _yaml.safe_load(HERMES_CONFIG.read_text()) or {}
    except Exception:
        cfg = {}

    if not isinstance(cfg.get("model"), dict):
        cfg["model"] = {}

    ports = _listening_ports()
    all_models = load_models()
    running = [m for m in all_models if m["port"] and m["port"] in ports]
    by_name = {m["name"]: m for m in running}

    def _is_vision(m: dict) -> bool:
        name = m["name"].lower()
        return "vision" in name or re.search(r"\bvl\b", name) is not None

    non_vision = [m for m in running if not _is_vision(m)]
    vision = [m for m in running if _is_vision(m)]

    # hermes holds a single active model in `model:` and a single vision
    # endpoint in `auxiliary.vision:`. Keep the user's current choice sticky:
    # only switch when the configured model is no longer being served. This
    # avoids yanking the active model out from under hermes whenever an
    # unrelated model is toggled.
    cur_default = cfg["model"].get("default")
    primary_model = by_name.get(cur_default)
    if primary_model is None:
        primary_model = non_vision[0] if non_vision else (running[0] if running else None)

    aux_cfg = cfg.get("auxiliary")
    cur_vision = (aux_cfg or {}).get("vision", {}).get("model") if isinstance(aux_cfg, dict) else None
    vision_model = by_name.get(cur_vision) if (cur_vision in by_name and _is_vision(by_name.get(cur_vision, {}))) else None
    if vision_model is None and vision:
        vision_model = vision[0]

    def _ctx(m: dict) -> int:
        try:
            return int(m["ctx"]) if m["ctx"] else 65536
        except (ValueError, TypeError):
            return 65536

    def _set(d: dict, key, value) -> bool:
        if d.get(key) != value:
            d[key] = value
            return True
        return False

    changed = False
    if primary_model:
        base = f"http://{SPARK_HOST}:{primary_model['port']}/v1"
        changed |= _set(cfg["model"], "default", primary_model["name"])
        changed |= _set(cfg["model"], "provider", "custom")
        changed |= _set(cfg["model"], "base_url", base)
        changed |= _set(cfg["model"], "api_key", "none")
        changed |= _set(cfg["model"], "context_length", _ctx(primary_model))

    if vision_model:
        base = f"http://{SPARK_HOST}:{vision_model['port']}/v1"
        if not isinstance(cfg.get("auxiliary"), dict):
            cfg["auxiliary"] = {}
        if not isinstance(cfg["auxiliary"].get("vision"), dict):
            cfg["auxiliary"]["vision"] = {}
        aux = cfg["auxiliary"]["vision"]
        changed |= _set(aux, "provider", "custom")
        changed |= _set(aux, "model", vision_model["name"])
        changed |= _set(aux, "base_url", base)
        changed |= _set(aux, "api_key", "none")
        changed |= _set(aux, "context_length", _ctx(vision_model))

    # Register every serving model as a named custom provider so they all show
    # up in hermes' `/model` picker (the single `model:` block only holds the
    # active one). Each model is its own llama-server on its own port, so each
    # needs its own provider entry — hermes resolves base_url per provider, not
    # per model within a provider. We own entries whose name matches one of our
    # managed models (running or stopped); any other custom_providers the user
    # added by hand are preserved. Ownership is keyed on name rather than the
    # base_url's SPARK_HOST substring so a host rename can't orphan stale
    # entries into looking "foreign" and duplicating on next sync.
    # discover_models: false is essential. Otherwise hermes' /model picker runs
    # live `/v1/models` discovery against every entry on each open — with an
    # api_key set (even "none") that probe always fires, so a stopped/loading
    # endpoint or slow mDNS makes the picker hang for the full HTTP timeout. We
    # already know each server's one model, so discovery is pointless here.
    cluster_running = _cluster_running_models()
    desired_entries = [
        {
            "name": m["name"],
            "base_url": f"http://{SPARK_HOST}:{m['port']}/v1",
            "api_key": "none",
            "model": m["name"],
            "context_length": _ctx(m),
            "models": {m["name"]: {}},
            "discover_models": False,
        }
        for m in running
    ] + [
        {
            "name": m["name"],
            "base_url": f"http://{m['serve_host']}:{m['port']}/v1",
            "api_key": "none",
            "model": m["name"],
            "context_length": _ctx(m),
            "models": {m["name"]: {}},
            "discover_models": False,
        }
        for m in cluster_running
    ]
    # A remote model is only "managed" (safe to overwrite/prune) once it's
    # actually reachable and included above — an unreachable peer must not
    # cause its entries to be treated as foreign-and-preserved one sync then
    # silently dropped the next; scoping to cluster_running keeps this stable
    # either way since gather_cluster_models is best-effort per call.
    managed_names = {m["name"] for m in all_models} | {m["name"] for m in cluster_running}
    existing_cps = cfg.get("custom_providers")
    existing_cps = existing_cps if isinstance(existing_cps, list) else []
    foreign = [
        e for e in existing_cps
        if not (isinstance(e, dict) and e.get("name") in managed_names)
    ]
    new_cps = foreign + desired_entries
    if existing_cps != new_cps:
        # Drop the key entirely when there's nothing to register, rather than
        # writing an empty list.
        if new_cps:
            cfg["custom_providers"] = new_cps
        elif "custom_providers" in cfg:
            del cfg["custom_providers"]
        changed = True

    if not changed:
        return False

    # width is large so hermes' long strings (environment_hint, personalities)
    # aren't re-wrapped into a noisy diff on every sync.
    payload = _yaml.safe_dump(
        cfg, default_flow_style=False, sort_keys=False, allow_unicode=True, width=4096
    )
    existing = HERMES_CONFIG.read_text() if HERMES_CONFIG.exists() else ""
    if existing == payload:
        return False
    tmp = HERMES_CONFIG.with_suffix(".tmp")
    tmp.write_text(payload)
    tmp.replace(HERMES_CONFIG)
    return True


def regen_grok() -> bool:
    if "grok" not in _get_sync_targets():
        return False
    if not GROK_CONFIG.parent.exists():
        return False

    ports = _listening_ports()
    local = [(m, f"http://{SPARK_HOST}:{m['port']}/v1", "")
             for m in load_models() if m["port"] != 0 and m["port"] in ports]
    remote = [(m, f"http://{m['serve_host']}:{m['port']}/v1",
              f" [{_host_label(m.get('host') or m['remote_host'])}]")
              for m in _cluster_running_models()]

    sections: list[str] = []
    for m, base, tag in local + remote:
        name_key = m["name"].replace(".", "-").replace("_", "-")
        try:
            ctx = int(m["ctx"]) if m["ctx"] else 65536
        except (ValueError, TypeError):
            ctx = 65536
        sections.append(
            f'[model.{name_key}]\n'
            f'model = "{m["name"]}"\n'
            f'base_url = "{base}"\n'
            f'name = "{m["display"]}{tag}"\n'
            f'context_window = {ctx}'
        )

    if not sections:
        return False

    existing = ""
    if GROK_CONFIG.exists():
        existing = GROK_CONFIG.read_text()

    lines = existing.splitlines() if existing else []

    # Parse into sections: keep non-model sections with their content,
    # drop all [model.*] sections.
    sections_out: list[list[str]] = []
    cur: list[str] | None = None
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("[model."):
            # Skip this entire model section.
            i += 1
            while i < len(lines):
                i += 1
                if i < len(lines) and lines[i].strip().startswith("["):
                    break
            continue
        if stripped.startswith("["):
            # Start of a non-model section — begin capturing it.
            cur = [line]
            sections_out.append(cur)
            i += 1
            while i < len(lines):
                if lines[i].strip().startswith("["):
                    break
                cur.append(lines[i])
                i += 1
        else:
            # Leading content before any section header.
            if cur is None:
                cur = []
                sections_out.append(cur)
            cur.append(line)
            i += 1

    if not sections_out:
        sections_out = [[]]

    # Repair a dangling `[models] default`: grok keeps the last-used model as
    # default, but if that model is no longer being served the reference
    # points at nothing and grok shows a dead model. Keep the user's choice
    # sticky while it serves; otherwise fall back to the first live model
    # (mirrors regen_hermes).
    serving = [m["name"] for m, _base, _tag in local + remote]
    for sec in sections_out:
        if not (sec and sec[0].strip() == "[models]"):
            continue
        for j, line in enumerate(sec):
            mt = re.match(r'\s*default\s*=\s*"([^"]*)"', line)
            if mt and mt.group(1) not in serving and serving:
                sec[j] = f'default = "{serving[0]}"'

    # Build output: all existing sections, then new model sections at the end.
    # Strip trailing blank lines from each captured section to avoid double-spacing.
    new_lines: list[str] = []
    for sec in sections_out:
        while sec and not sec[-1].strip():
            sec.pop()
        new_lines.extend(sec)
        # Always end with exactly one blank line separator.
        while new_lines and new_lines[-1].strip() == "":
            new_lines.pop()
        new_lines.append("")

    for sec in sections:
        # Ensure single blank line before each model section.
        while new_lines and new_lines[-1].strip() == "":
            new_lines.pop()
        new_lines.append("")
        new_lines.extend(sec.splitlines())

    payload = "\n".join(new_lines)
    if not payload.endswith("\n"):
        payload += "\n"
    if existing == payload:
        return False
    tmp = GROK_CONFIG.with_suffix(".tmp")
    tmp.write_text(payload)
    tmp.replace(GROK_CONFIG)
    return True


def regen_all() -> bool:
    any_changed = False
    if "opencode" in _get_sync_targets():
        if regen_opencode():
            any_changed = True
    if "hermes" in _get_sync_targets():
        if regen_hermes():
            any_changed = True
    if "grok" in _get_sync_targets():
        if regen_grok():
            any_changed = True
    return any_changed
