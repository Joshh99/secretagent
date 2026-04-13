"""Utility helpers shared across implementation factories."""

import importlib
from typing import Any, Callable

from secretagent.core import Interface, all_interfaces


def resolve_dotted(name: str) -> Any:
    """Resolve a dotted name like 'module.func' to the actual object.
    """
    parts = name.split('.')
    obj = importlib.import_module(parts[0])
    for part in parts[1:]:
        obj = getattr(obj, part)
    return obj


def resolve_tools(interface: Interface, tools) -> list[Callable]:
    """Resolve a tools specification into a list of callables.

    The tools parameter can be:
      - None or [] → no tools (returns [])
      - '__all__' → all implemented interfaces except the given one
      - a list where each element is:
          - a callable (used as-is)
          - an Interface (resolved to its implementing function)
          - a string (resolved via resolve_dotted)
    """
    if tools == '__all__':
        tools = [iface for iface in all_interfaces()
                 if iface is not interface and iface.implementation is not None]
    tools = tools or []
    resolved = []
    for tool in tools:
        if isinstance(tool, str):
            tool = resolve_dotted(tool)
        if isinstance(tool, Interface):
            if tool.implementation is None:
                raise ValueError(f'Interface {tool.name!r} has no implementation')
            resolved.append(tool.implementation.implementing_fn)
        else:
            if not callable(tool):
                raise ValueError(
                    f'Tool {tool!r} is not callable')
            resolved.append(tool)
    return resolved
