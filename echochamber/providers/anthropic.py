"""Anthropic Claude provider."""

import os
from typing import Optional

from .base import LLMProvider, LLMResponse, Message


class AnthropicProvider(LLMProvider):
    """Provider for Anthropic's Claude models."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY env var or pass api_key."
            )

        # Lazy import to avoid requiring anthropic if not used
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("Please install anthropic: pip install anthropic")

    def complete(
        self,
        messages: list[Message],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        # Convert to Anthropic format
        anthropic_messages = []
        for msg in messages:
            if msg.role == "system":
                # Anthropic handles system separately
                continue
            anthropic_messages.append({
                "role": msg.role,
                "content": msg.content,
            })

        # Build request
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": anthropic_messages,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = self.client.messages.create(**kwargs)

        return LLMResponse(
            content=response.content[0].text,
            model=response.model,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            raw_response=response.model_dump(),
        )

    @property
    def name(self) -> str:
        return f"anthropic/{self.model}"
