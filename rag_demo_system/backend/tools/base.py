from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ToolDefinition(ABC):
    @abstractmethod
    def schema(self) -> dict[str, Any]:
        """OpenAI-compatible function schema for the tools=[] parameter."""

    @abstractmethod
    def defaults(self) -> dict[str, Any]:
        """Default parameter values. Keys must match schema property names."""

    def fill_defaults(self, params: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        """Merge user-provided params with defaults.
        Returns (filled_params, list_of_defaulted_field_names).
        """
        result = dict(params)
        defaulted: list[str] = []
        for key, value in self.defaults().items():
            if key not in result:
                result[key] = value
                defaulted.append(key)
        return result, defaulted

    @abstractmethod
    def execute(self, params: dict[str, Any], session_context: dict[str, Any]) -> dict[str, Any]:
        """Execute the tool (synchronous). Returns structured result dict.
        Called via asyncio.to_thread() from the async streaming pipeline."""

    @abstractmethod
    def format_voice_summary(self, result: dict[str, Any]) -> str:
        """Concise Russian text for LLM to base spoken response on."""

    def format_sms_body(self, result: dict[str, Any]) -> str | None:
        """Full text for SMS delivery. None if tool does not support SMS."""
        return None
