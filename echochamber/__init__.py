"""
EchoChamber: Multi-LLM Courtroom Debate Environment

A framework for adversarial debate between LLM agents with
moderator-controlled proceedings.
"""

from .core import (
    Agent,
    Role,
    create_agent,
    CourtSession,
    SessionConfig,
    SessionResult,
    Transcript,
)
from .providers import (
    LLMProvider,
    create_provider,
    AnthropicProvider,
    OpenAIProvider,
    TogetherProvider,
    LMStudioProvider,
)

__version__ = "0.1.0"

__all__ = [
    # Core
    "Agent",
    "Role",
    "create_agent",
    "CourtSession",
    "SessionConfig",
    "SessionResult",
    "Transcript",
    # Providers
    "LLMProvider",
    "create_provider",
    "AnthropicProvider",
    "OpenAIProvider",
    "TogetherProvider",
    "LMStudioProvider",
]
