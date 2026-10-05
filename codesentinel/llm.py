"""Thin Groq client wrapper with per-instance token accounting.

A client instance is created per user session (never a process-wide global), so
token counts and API keys are never mixed between Streamlit sessions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from groq import Groq

from codesentinel.config import DEFAULT_MODEL


class MissingAPIKeyError(RuntimeError):
    """Raised when an LLM call is attempted without a Groq API key."""


@dataclass
class UsageTracker:
    """Counts LLM calls and tokens, grouped by label (agent, ai_judge, ...)."""

    entries: dict[str, dict[str, int]] = field(default_factory=dict)

    def record(self, label: str, usage) -> None:
        entry = self.entries.setdefault(label, {"calls": 0, "tokens": 0})
        entry["calls"] += 1
        entry["tokens"] += getattr(usage, "total_tokens", 0) or 0

    @property
    def total_calls(self) -> int:
        return sum(e["calls"] for e in self.entries.values())

    @property
    def total_tokens(self) -> int:
        return sum(e["tokens"] for e in self.entries.values())

    def summary(self) -> str:
        lines = [f"{label:<14}{e['calls']:>4} calls{e['tokens']:>9} tokens" for label, e in self.entries.items()]
        lines.append(f"{'TOTAL':<14}{self.total_calls:>4} calls{self.total_tokens:>9} tokens")
        return "\n".join(lines)


class LLMClient:
    """Groq chat-completions client that records usage for every call."""

    def __init__(self, api_key: str | None, model: str = DEFAULT_MODEL, max_retries: int = 6):
        if not api_key:
            raise MissingAPIKeyError(
                "No Groq API key found. Set GROQ_API_KEY (env var, .env or Streamlit secrets)."
            )
        # The free tier has tight rate limits; the SDK waits and retries on 429s.
        self._client = Groq(api_key=api_key, max_retries=max_retries)
        self.model = model
        self.usage = UsageTracker()

    def chat(self, messages: list[dict[str, Any]], *, label: str, model: str | None = None, **kwargs: Any) -> Any:
        response = self._client.chat.completions.create(model=model or self.model, messages=messages, **kwargs)  # type: ignore
        if response.usage is not None:
            self.usage.record(label, response.usage)
        return response
