"""Together.AI provider."""

import os
from typing import Optional

from .base import LLMProvider, LLMResponse, Message


class TogetherProvider(LLMProvider):
    """Provider for Together.AI models."""

    def __init__(
        self,
        model: str = "meta-llama/Llama-3-70b-chat-hf",
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("TOGETHER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Together API key required. Set TOGETHER_API_KEY env var or pass api_key."
            )

        try:
            from openai import OpenAI
            # Together uses OpenAI-compatible API
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.together.xyz/v1",
            )
        except ImportError:
            raise ImportError("Please install openai: pip install openai")

    def complete(
        self,
        messages: list[Message],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        # Convert to OpenAI format (Together uses OpenAI-compatible API)
        together_messages = []

        if system_prompt:
            together_messages.append({
                "role": "system",
                "content": system_prompt,
            })

        for msg in messages:
            together_messages.append({
                "role": msg.role,
                "content": msg.content,
            })

        response = self.client.chat.completions.create(
            model=self.model,
            messages=together_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        choice = response.choices[0]
        return LLMResponse(
            content=choice.message.content,
            model=response.model,
            usage={
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            } if response.usage else None,
            raw_response=response.model_dump(),
        )

    @property
    def name(self) -> str:
        return f"together/{self.model}"
