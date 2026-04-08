"""
LLM providers for semantic interpretation – abstract interface + implementations.

Switch LLM models by passing a different provider:

    from tracescope.providers.llm import OpenAILLM, AnthropicLLM
    llm = OpenAILLM(api_key="sk-...", model="gpt-5-mini")
    llm = AnthropicLLM(api_key="sk-ant-...", model="claude-sonnet-4-20250514")
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod

log = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_BACKOFF = (2, 5, 10)  # seconds between retries


class LLMProvider(ABC):
    """Abstract base class for LLM interpretation providers."""

    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""
        ...

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Send a prompt and return the text response."""
        ...


class OpenAILLM(LLMProvider):
    """OpenAI chat completions (GPT-5-mini, GPT-4o, etc.)."""

    def __init__(self, api_key: str, model: str = "gpt-5-mini"):
        from openai import OpenAI
        self._model = model
        self._client = OpenAI(api_key=api_key)

    def model_name(self) -> str:
        return self._model

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        for attempt in range(_MAX_RETRIES):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt == _MAX_RETRIES - 1:
                    raise
                wait = _RETRY_BACKOFF[attempt]
                log.warning("OpenAI request failed (attempt %d/%d): %s — "
                            "retrying in %ds", attempt + 1, _MAX_RETRIES, e, wait)
                time.sleep(wait)


class AnthropicLLM(LLMProvider):
    """Anthropic Claude via the official SDK."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        import anthropic

        self._model = model
        self._client = anthropic.Anthropic(api_key=api_key)

    def model_name(self) -> str:
        return self._model

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        for attempt in range(_MAX_RETRIES):
            try:
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=16000,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                return response.content[0].text.strip()
            except Exception as e:
                if attempt == _MAX_RETRIES - 1:
                    raise
                wait = _RETRY_BACKOFF[attempt]
                log.warning("Anthropic request failed (attempt %d/%d): %s — "
                            "retrying in %ds", attempt + 1, _MAX_RETRIES, e, wait)
                time.sleep(wait)
