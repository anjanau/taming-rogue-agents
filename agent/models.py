"""Claude model factory for the native LangGraph demo."""

from __future__ import annotations

import os


def model_for_tier(tier: str = "strong") -> str:
    """Return the configured Claude model for a strength tier."""
    tier = tier.lower()
    if tier not in {"strong", "weak"}:
        raise ValueError(f"Unsupported model tier: {tier!r}; use 'strong' or 'weak'")

    default = "claude-sonnet-4-5" if tier == "strong" else "claude-haiku-4-5"
    return os.getenv(f"ANTHROPIC_{tier.upper()}_MODEL", default)


def build_chat_model(model: str, temperature: float = 0.0):
    """Build a native LangChain Claude chat model."""
    from langchain_anthropic import ChatAnthropic

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise ValueError("ANTHROPIC_API_KEY is required")
    return ChatAnthropic(model=model, temperature=temperature)
