"""OpenAI provider."""

import os
from typing import Optional

from .openai_compat import OpenAICompatProvider


class OpenAIProvider(OpenAICompatProvider):
    """Provider for OpenAI models."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
    ):
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key."
            )

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Please install openai: pip install openai")

        super().__init__(model=model, client=OpenAI(api_key=api_key))

    @property
    def name(self) -> str:
        return f"openai/{self.model}"
