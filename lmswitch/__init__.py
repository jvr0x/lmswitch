"""lmswitch — list and toggle local LLMs from YAML configs."""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    # Single source of truth: the version declared in pyproject.toml.
    __version__ = _pkg_version("lmswitch")
except PackageNotFoundError:  # not installed (e.g. running from a source checkout)
    __version__ = "0.0.0+dev"
