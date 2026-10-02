"""In-process Python adapter: the agent is a class that implements AgentAdapter."""

from __future__ import annotations

import importlib

from agentoscopy.adapters.base import AgentAdapter


class AdapterLoadError(Exception):
    """The configured entrypoint could not be imported or instantiated."""


def load_python_adapter(entrypoint: str) -> AgentAdapter:
    """Instantiate `package.module:ClassName`."""
    module_name, _, class_name = entrypoint.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise AdapterLoadError(f"cannot import {module_name!r}: {exc}") from exc
    adapter_class = getattr(module, class_name, None)
    if adapter_class is None:
        raise AdapterLoadError(f"{module_name!r} has no attribute {class_name!r}")
    try:
        return adapter_class()
    except Exception as exc:  # arbitrary user code; report it as a load failure
        raise AdapterLoadError(f"cannot instantiate {entrypoint!r}: {exc}") from exc
