"""Shared paths, constants, and I/O helpers."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — all relative to the script's own directory so the tool is portable.
#
# LMSWITCH_DATA_DIR env var (if set) overrides the default ai-models/ location.
# This lets pytest and the shipped console entry point share the same isolated
# fixture without patching individual module-level names.
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent

_DATA_ROOT = Path(os.environ.get("LMSWITCH_DATA_DIR", ""))
if not _DATA_ROOT:
    _DATA_ROOT = SCRIPT_DIR / "ai-models"

CONF_DIR = _DATA_ROOT
RUN_DIR = CONF_DIR / "running"
CONFIG_FILE = CONF_DIR / ".lmswitch"

HOME = Path.home()
OPENCODE = HOME / ".config" / "opencode" / "opencode.json"
OPENCODE_EXPORT = HOME / ".local" / "share" / "opencode-export" / "opencode.json"
HERMES_CONFIG = HOME / ".hermes" / "config.yaml"
GROK_CONFIG = HOME / ".grok" / "config.toml"
SPARK_HOST = "spark-8912.local"

SYNC_OPENCODE = "SYNC_OPENCODE"
SYNC_HERMES = "SYNC_HERMES"
SYNC_GROK = "SYNC_GROK"

DEFAULT_SYNC_TARGETS = [SYNC_OPENCODE, SYNC_HERMES, SYNC_GROK]

TTY = sys.stdout.isatty()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if TTY else text


def _read_config() -> dict:
    cfg: dict = {}
    if CONFIG_FILE.exists():
        for line in CONFIG_FILE.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            cfg[k.strip()] = v.strip().strip('"').strip("'")
    return cfg


def _models_dir() -> Path:
    cfg = _read_config()
    raw = cfg.get("MODELS_DIR", str(HOME / "models"))
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (SCRIPT_DIR / p).resolve()
    return p


def _human(nbytes: int) -> str:
    if nbytes <= 0:
        return "-"
    gb = nbytes / 1024 ** 3
    if gb >= 1000:
        return f"{gb / 1024:.1f}T"
    return f"{gb:.1f}G"


# ---------------------------------------------------------------------------
# YAML loading — prefer pyyaml, fall back to a simple built-in parser.
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict:
    """Load a YAML file. Prefer pyyaml; fall back to a simple parser."""
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        pass
    return _parse_yaml_simple(path.read_text())


def _parse_yaml_simple(text: str) -> dict:
    """Minimal YAML parser for our config subset: scalars, lists, mappings."""
    result: dict = {}
    stack = [(result, -1)]
    in_list = False
    list_buf: list = []

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        raw = line.rstrip("\n\r")
        stripped = raw.lstrip()
        indent = len(raw) - len(stripped)

        if not stripped or stripped.startswith("#"):
            i += 1
            continue

        while len(stack) > 1 and indent <= stack[-1][1]:
            container, _ = stack.pop()
            if in_list:
                stack[-1][0].append(list_buf)
                list_buf = []
                in_list = False

        if stripped.startswith("- "):
            val = stripped[2:].strip()
            if val:
                list_buf.append(_yaml_scalar(val))
            in_list = True
            i += 1
            continue

        if in_list:
            stack[-1][0].append(list_buf)
            list_buf = []
            in_list = False

        if ":" in stripped:
            key, _, val = stripped.partition(":")
            key = key.strip()
            val = val.strip()
            if val:
                stack[-1][0][key] = _yaml_scalar(val)
            else:
                new_dict: dict = {}
                stack[-1][0][key] = new_dict
                stack.append((new_dict, indent))
        i += 1

    if in_list:
        stack[-1][0].append(list_buf)

    return result


def _yaml_scalar(s: str):
    if s in ("true", "True", "yes", "on"):
        return True
    if s in ("false", "False", "no", "off"):
        return False
    if s in ("null", "~", ""):
        return None
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


# ---------------------------------------------------------------------------
# Model size detection helpers
# ---------------------------------------------------------------------------

FAMILY_RULES = [
    ("Qwen", ("qwen", "qwopus")),
    ("Gemma", ("diffusiongemma", "gemma")),
    ("DeepSeek", ("deepseek",)),
    ("Nemotron", ("nemotron",)),
    ("Kimi", ("kimi",)),
    ("GPT-OSS", ("gpt-oss", "gpt")),
    ("GLM", ("glm",)),
    ("Nex", ("nex",)),
    ("Step", ("step",)),
    ("Ornith", ("ornith",)),
]


def _family(name: str) -> tuple[int, str]:
    low = name.lower()
    for i, (label, keys) in enumerate(FAMILY_RULES):
        if any(k in low for k in keys):
            return i, label
    return len(FAMILY_RULES), "Other"


def _model_size_and_present(rel: str, runtime: str) -> tuple[int, bool]:
    models_dir = _models_dir()
    full = models_dir / rel
    if runtime == "vllm":
        if not full.is_dir():
            return 0, False
        total = 0
        has_weights = False
        for dp, _, files in os.walk(full):
            for fn in files:
                if fn.endswith(".safetensors"):
                    has_weights = True
                fp = Path(dp) / fn
                if fp.is_file() and not fp.is_symlink():
                    total += fp.stat().st_size
                elif fp.is_symlink() and fp.exists():
                    total += fp.resolve().stat().st_size
        return total, has_weights
    # GGUF
    if not full.exists():
        return 0, False
    m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", str(full))
    if m:
        pattern = re.sub(r"-\d{5}-of-\d{5}\.gguf$",
                         f"-*-of-{m.group(2)}.gguf", str(full))
        files = [f for f in full.parent.glob(f"*-of-{m.group(2)}.gguf") if Path(f).is_file()]
        return sum(Path(f).stat().st_size for f in files), True
    return full.stat().st_size, True


# ---------------------------------------------------------------------------
# Sync target resolution
# ---------------------------------------------------------------------------

def _get_sync_targets() -> list[str]:
    cfg = _read_config()
    targets = []
    if cfg.get(SYNC_OPENCODE, "true").lower() in ("true", "1", "yes"):
        targets.append("opencode")
    if cfg.get(SYNC_HERMES, "true").lower() in ("true", "1", "yes"):
        targets.append("hermes")
    if cfg.get(SYNC_GROK, "true").lower() in ("true", "1", "yes"):
        targets.append("grok")
    if not targets:
        targets = ["opencode"]
    return targets
