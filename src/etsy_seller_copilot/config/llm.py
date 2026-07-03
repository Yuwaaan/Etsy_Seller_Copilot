"""Provider-agnostic LLM loading for the Business Analyst Agent.

Reads ``MODEL_PROVIDER`` / ``MODEL_NAME`` / ``OPENAI_API_KEY`` /
``ANTHROPIC_API_KEY`` from the environment and returns a ready-to-use chat
model -- or ``None`` if nothing is configured. The whole point of returning
``None`` instead of raising is that every caller (the chat agent, the AI
Insights health report) is expected to degrade to a deterministic fallback
rather than crash when no API key is set.
"""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

load_dotenv()

# Small, fast, tool-calling-capable models -- cheap enough for a portfolio
# project, sufficient for reasoning over a handful of pre-computed metrics.
_DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5",
}


def _detect_provider() -> str | None:
    """Pick a provider from ``MODEL_PROVIDER``, or infer it from whichever API key is set."""
    explicit = os.environ.get("MODEL_PROVIDER", "").strip().lower()
    if explicit in _DEFAULT_MODELS:
        return explicit

    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return None


def get_chat_model() -> BaseChatModel | None:
    """Return a chat model for the configured provider, or ``None`` if unconfigured.

    Never raises: construction failures (bad model name, missing optional
    provider dependency, etc.) are treated the same as "not configured" so
    callers can uniformly fall back to the deterministic engine.
    """
    provider = _detect_provider()
    if provider is None:
        return None

    model_name = os.environ.get("MODEL_NAME", _DEFAULT_MODELS[provider])
    try:
        return init_chat_model(model_name, model_provider=provider, temperature=0)
    except Exception:
        return None
