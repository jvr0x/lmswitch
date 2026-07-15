"""Abstract base class for model server runtimes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class RunningState:
    """Result of a start() call."""

    __slots__ = ("status", "detail", "proc")

    def __init__(self, status: str, detail: str = "", proc=None) -> None:
        """
        Args:
            status: One of ``"ready"``, ``"dead"``, ``"timeout"``.
            detail: Human-readable message (e.g. last log lines).
            proc: The backing ``subprocess.Popen``, for runtimes that started
                one directly in-process (llama-server). Lets a foreground
                supervisor (``cli.cmd_serve``) detect and reap its exit via
                ``.poll()`` — a plain PID-liveness check (``os.kill(pid, 0)``)
                cannot tell a zombie from a live process, since a zombie
                keeps its PID valid until its parent reaps it. None for
                runtimes with no local child to hold (vLLM containers,
                remote workers).
        """
        self.status = status
        self.detail = detail
        self.proc = proc

    def __repr__(self) -> str:
        return f"RunningState({self.status!r}, {self.detail!r})"


class BaseRuntime(ABC):
    """Abstract interface for a model server runtime.

    Each runtime (llama-server, vLLM, sglang, etc.) implements this
    interface. Adding a new runtime means writing one new file that
    subclasses ``BaseRuntime`` — no changes to existing code.
    """

    @abstractmethod
    def start(self, name: str, yaml: dict) -> RunningState:
        """Start the model server.

        Args:
            name: Model name (used for container/pid naming, logs).
            yaml: Parsed YAML config dict for this model.

        Returns:
            ``RunningState`` with status ``"ready"``, ``"dead"``, or ``"timeout"``.
        """

    @abstractmethod
    def stop(self, name: str, yaml: dict) -> None:
        """Stop the model server.

        Args:
            name: Model name.
            yaml: Parsed YAML config dict.
        """

    @abstractmethod
    def is_running(self, name: str, runtime_name: str) -> bool:
        """Check if the model server is currently running.

        Args:
            name: Model name.
            runtime_name: Runtime type string (e.g. "llama", "vllm").

        Returns:
            True if the server is running.
        """

    @abstractmethod
    def is_ready(self, name: str, port: int, timeout: int = 300) -> str:
        """Poll until the server is ready, dead, or timeout.

        Args:
            name: Model name (for progress messages).
            port: Server port to probe.
            timeout: Seconds to wait.

        Returns:
            ``"ready"``, ``"dead"``, or ``"timeout"``.
        """


class RuntimeRegistry:
    """Registry mapping runtime names to their implementation classes.

    Usage:
        from lmswitch.runtimes.base import runtime_registry

        runtime_cls = runtime_registry.lookup("llama")
        runtime = runtime_cls()
        state = runtime.start("my_model", config)
    """

    _registry: dict[str, type[BaseRuntime]] = {}

    @classmethod
    def register(cls, name: str, runtime_cls: type[BaseRuntime]) -> None:
        """Register a runtime class under a name."""
        cls._registry[name] = runtime_cls

    @classmethod
    def lookup(cls, name: str) -> type[BaseRuntime]:
        """Return the runtime class for *name*, or the default (llama)."""
        return cls._registry.get(name, cls._registry.get("llama", BaseRuntime))

    @classmethod
    def list_runtimes(cls) -> list[str]:
        """Return all registered runtime names."""
        return sorted(cls._registry.keys())


# Singleton registry instance
runtime_registry = RuntimeRegistry()
