# SPDX-License-Identifier: Apache-2.0
"""LLM provider abstraction — a single seam every agent calls through.

Per the v0.1 design, the user supplies their own API key. The library
never owns LLM credentials. Provider is Anthropic by default; the
abstraction lets us swap to Ollama / OpenAI / etc. without touching
extraction code.

Designed for easy mocking: tests inject a fake `LlmClient` that returns
canned JSON. Real Anthropic calls only fire when explicitly wired.
"""

from __future__ import annotations

import os
from typing import Protocol


class LlmClient(Protocol):
    """The minimal contract every provider implements.

    `chat(prompt, *, model, max_tokens, temperature) -> str` returns
    the assistant's text content (the first `TextBlock`'s `.text` from
    Anthropic, or equivalent from other providers).
    """

    def chat(
        self,
        prompt: str,
        *,
        model: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> str: ...


class AnthropicClient:
    """Default `LlmClient` implementation.

    Reads `ANTHROPIC_API_KEY` from env. Lazily imports the SDK so the
    library can be imported without it (useful for tests that mock the
    client entirely, and for environments that pin a different
    provider).
    """

    def __init__(self, api_key: str | None = None) -> None:
        # Lazy import — the anthropic SDK only loads when Anthropic is
        # actually used.
        from anthropic import Anthropic

        self._client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def chat(
        self,
        prompt: str,
        *,
        model: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> str:
        from anthropic.types import TextBlock

        msg = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        # Anthropic returns a list of content blocks; we want the first
        # text block's text. Tool-use / thinking blocks are skipped.
        for block in msg.content:
            if isinstance(block, TextBlock):
                return block.text
        return ""


def default_client() -> LlmClient:
    """Construct the default Anthropic client. Convenience for callers
    that want one line of setup."""
    return AnthropicClient()
