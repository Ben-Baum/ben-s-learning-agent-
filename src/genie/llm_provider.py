"""LLM provider interface: OpenAI (default) and Gemini (stub)."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Optional


class LLMProvider(ABC):
    """Abstract LLM provider for CrewAI-compatible calls."""

    @abstractmethod
    def get_model_name(self) -> str:
        pass

    @abstractmethod
    def get_api_key(self) -> str:
        pass

    @abstractmethod
    def get_base_url(self) -> Optional[str]:
        """Optional base URL for API (e.g. OpenAI-compatible endpoints)."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI API provider. Works with CrewAI via OPENAI_API_KEY."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini") -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._model = model

    def get_model_name(self) -> str:
        return self._model

    def get_api_key(self) -> str:
        return self._api_key

    def get_base_url(self) -> Optional[str]:
        return None


class GeminiProvider(LLMProvider):
    """Gemini provider via LiteLLM (used by CrewAI). Requires GOOGLE_API_KEY env var."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini/gemini-2.5-flash") -> None:
        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        self._model = model

    def get_model_name(self) -> str:
        return self._model

    def get_api_key(self) -> str:
        return self._api_key

    def get_base_url(self) -> Optional[str]:
        return None


def get_llm_provider() -> LLMProvider:
    """Return provider based on ENV LLM_PROVIDER=openai|gemini."""
    provider = os.environ.get("LLM_PROVIDER", "openai").lower()
    if provider == "gemini":
        return GeminiProvider()
    return OpenAIProvider()
