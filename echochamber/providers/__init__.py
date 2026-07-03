"""LLM Providers for EchoChamber."""

from .base import LLMProvider, LLMResponse, Message, ToolCall, ToolDef
from .retry import call_with_retries
from .anthropic import AnthropicProvider
from .openai import OpenAIProvider
from .together import TogetherProvider
from .lmstudio import LMStudioProvider
from .gemini import GeminiProvider


def create_provider(
    provider: str,
    model: str | None = None,
    **kwargs,
) -> LLMProvider:
    """
    Factory function to create an LLM provider.

    Args:
        provider: Provider name ("anthropic", "openai", "together", "lmstudio", "gemini")
        model: Model name (optional, uses provider default if not specified)
        **kwargs: Additional provider-specific arguments

    Returns:
        Configured LLMProvider instance
    """
    providers = {
        "anthropic": AnthropicProvider,
        "openai": OpenAIProvider,
        "together": TogetherProvider,
        "lmstudio": LMStudioProvider,
        "gemini": GeminiProvider,
    }

    if provider not in providers:
        raise ValueError(
            f"Unknown provider: {provider}. Available: {list(providers.keys())}"
        )

    provider_class = providers[provider]
    if model:
        kwargs["model"] = model

    return provider_class(**kwargs)


__all__ = [
    "LLMProvider",
    "LLMResponse",
    "Message",
    "ToolCall",
    "ToolDef",
    "call_with_retries",
    "AnthropicProvider",
    "OpenAIProvider",
    "TogetherProvider",
    "LMStudioProvider",
    "GeminiProvider",
    "create_provider",
]
