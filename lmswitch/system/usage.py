"""Usage tracking — record and query model run history.

Usage events are persisted as JSONL (one JSON per line) in
``<CONF_DIR>/lmswitch-usage.json`` so they survive restarts.

Each event has:
    - ``ts``: ISO-8601 timestamp
    - ``action``: ``"start"`` | ``"stop"``
    - ``model``: model name (filename stem)
    - ``display_name``: human-readable display name
    - ``runtime``: ``"llama"`` | ``"vllm"`` | …
    - ``port``: serving port
    - ``ctx``: context length
    - ``size``: weight file size in bytes
    - ``duration``: seconds (only on ``stop``, 0 on ``start``)
    - ``config``: the full model YAML dict (on ``start``)

Events are appended to a single file.  Reading queries returns all events
sorted by timestamp.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lmswitch.system.io import CONF_DIR

USAGE_FILE = CONF_DIR / "lmswitch-usage.json"


def _get_usage_file() -> Path:
    """Return the usage file path, ensuring parent dirs exist."""
    # Ensure CONF_DIR exists so the file can be created
    CONF_DIR.mkdir(parents=True, exist_ok=True)
    return USAGE_FILE


def record_event(event: dict[str, Any]) -> None:
    """Append a single usage event to the JSONL file.

    Args:
        event: Dict with keys ``ts``, ``action``, ``model``, ``runtime``,
               ``port``, ``ctx``, ``size``, and optionally ``duration``,
               ``display_name``, ``config``.
    """
    path = _get_usage_file()
    line = json.dumps(event, default=str) + "\n"
    with open(path, "a") as f:
        f.write(line)


def record_start(
    name: str,
    yaml: dict,
) -> None:
    """Record a model start event.

    Args:
        name: Model name (filename stem).
        yaml: Parsed YAML config dict.
    """
    runtime = yaml.get("runtime", "llama")
    port = yaml.get("port", 0)
    ctx = yaml.get("ctx", 65536)
    display = yaml.get("display_name", name)
    try:
        ctx_int = int(ctx) if ctx else 65536
    except (ValueError, TypeError):
        ctx_int = 65536

    from lmswitch.system.io import _model_size_and_present

    rel = yaml.get("model", "")
    size, _present = _model_size_and_present(rel, runtime)

    event: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "action": "start",
        "model": name,
        "display_name": display,
        "runtime": runtime,
        "port": port,
        "ctx": ctx_int,
        "size": size,
        "duration": 0,
        "config": {k: v for k, v in yaml.items() if k != "config"},
    }
    record_event(event)


def record_stop(
    name: str,
    duration_sec: float,
) -> None:
    """Record a model stop event.

    Args:
        name: Model name.
        duration_sec: Approximate uptime in seconds.
    """
    # TODO (perf): This calls query_events() which reads the full file from disk
    # to look up the prior start event. For typical usage (hundreds to ~10k events)
    # this is fine. It becomes a problem if the file grows to ~100k+ events — at that
    # point stop() will be slow (~50-200ms full file parse on every call), and
    # frequent toggling will add up. Mitigation: replace with a small JSON index
    # file (e.g. ``lmswitch-index.json``) keyed by ``{"model": <name>, "last_start": <event>}``
    # that is maintained in-memory at module level or appended to on each start. The
    # index can be read lazily on stop (one seek to the end) instead of scanning the
    # entire log. For now, the simple approach keeps the implementation at ~200 lines.
    events = query_events()
    model_event = None
    for ev in reversed(events):
        if ev["model"] == name and ev["action"] == "start":
            model_event = ev
            break

    display_name = model_event.get("display_name", name) if model_event else name
    runtime = model_event.get("runtime", "unknown") if model_event else "unknown"
    port = model_event.get("port", 0) if model_event else 0
    ctx = model_event.get("ctx", 65536) if model_event else 65536
    size = model_event.get("size", 0) if model_event else 0

    event: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "action": "stop",
        "model": name,
        "display_name": display_name,
        "runtime": runtime,
        "port": port,
        "ctx": ctx,
        "size": size,
        "duration": duration_sec,
    }
    record_event(event)


def query_events(
    model: str | None = None,
    runtime: str | None = None,
    action: str | None = None,
    since: str | None = None,
    until: str | None = None,
    limit: int = 0,
) -> list[dict[str, Any]]:
    """Query usage events from the JSONL file.

    Args:
        model: Filter by model name.
        runtime: Filter by runtime type.
        action: Filter by action ("start" | "stop").
        since: ISO-8601 timestamp — only events after this time.
        until: ISO-8601 timestamp — only events before this time.
        limit: Max events to return (0 = no limit).

    Returns:
        Sorted list of matching events (newest last).
    """
    path = _get_usage_file()
    if not path.exists():
        return []

    events: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            events.append(ev)

    # Apply filters
    if model:
        events = [e for e in events if e.get("model") == model]
    if runtime:
        events = [e for e in events if e.get("runtime") == runtime]
    if action:
        events = [e for e in events if e.get("action") == action]
    if since:
        events = [e for e in events if e.get("ts", "") >= since]
    if until:
        events = [e for e in events if e.get("ts", "") <= until]

    # Sort by timestamp (oldest first)
    events.sort(key=lambda e: e.get("ts", ""))

    if limit > 0:
        events = events[-limit:]

    return events


def clear_events() -> None:
    """Remove the usage file entirely."""
    path = _get_usage_file()
    if path.exists():
        path.unlink()
