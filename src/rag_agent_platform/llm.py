"""Chat-model factory. Provider SDKs are imported lazily so offline mode needs no keys."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage

from .config import Settings
from .errors import ConfigError


def build_chat_model(settings: Settings) -> BaseChatModel | None:
    """Return a chat model for live mode, or ``None`` in offline mode."""
    if not settings.is_live:
        return None
    key = settings.api_key.get_secret_value() if settings.api_key else ""
    if settings.provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=settings.model, api_key=key, temperature=0, max_retries=0)
    if settings.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=settings.model, api_key=key, temperature=0, max_retries=0)
    raise ConfigError(f"Unknown provider: {settings.provider}")


def message_text(message: BaseMessage) -> str:
    """Flatten a message's content (string or list of content blocks) to plain text."""
    content = message.content
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts)
