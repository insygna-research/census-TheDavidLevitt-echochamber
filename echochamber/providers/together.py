"""Together.AI provider (OpenAI-compatible API)."""

import os
from typing import Optional

from .openai_compat import OpenAICompatProvider


class TogetherProvider(OpenAICompatProvider):
    """Provider for Together.AI models."""

    def __init__(
        self,
        model: str = "meta-llama/Llama-3-70b-chat-hf",
        api_key: Optional[str] = None,
    ):
        api_key = api_key or os.environ.get("TOGETHER_API_KEY")
        if not api_key:
            raise ValueError(
                "Together API key required. Set TOGETHER_API_KEY env var or pass api_key."
            )

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Please install openai: pip install openai")

        super().__init__(
            model=model,
            client=OpenAI(api_key=api_key, base_url="https://api.together.xyz/v1"),
        )

    @property
    def name(self) -> str:
        return f"together/{self.model}"
