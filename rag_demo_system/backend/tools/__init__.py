from __future__ import annotations

from typing import Any

from .base import ToolDefinition


def get_tool_schemas() -> list[dict[str, Any]]:
    """Return all tool schemas for the LLM tools=[] parameter."""
    return [tool.schema() for tool in _TOOLS.values()]


def get_tool(name: str) -> ToolDefinition:
    """Get a tool instance by name. Raises KeyError if not found."""
    return _TOOLS[name]


def get_all_tools() -> dict[str, ToolDefinition]:
    """Return the full tool registry."""
    return dict(_TOOLS)


# Registry populated as tools are implemented.
_TOOLS: dict[str, ToolDefinition] = {}


def init_tools(settings: Any) -> None:
    """Initialize tools with settings. Called once at app startup."""
    pass  # Tool imports added in later tasks
