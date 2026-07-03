"""Base LLM Provider interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ToolDef:
    """A tool the model may call, described provider-agnostically."""
    name: str
    description: str
    parameters: dict  # JSON Schema for the arguments object


@dataclass
class ToolCall:
    """A tool invocation requested by the model."""
    id: str
    name: str
    arguments: dict


@dataclass
class Message:
    """A single message in a conversation.

    Roles: "system", "user", "assistant", or "tool" (a tool result being
    returned to the model; set tool_call_id to the ToolCall it answers).
    """
    role: str
    content: str
    name: Optional[str] = None  # Speaker name for multi-agent contexts
    tool_calls: list = field(default_factory=list)  # ToolCalls on assistant messages
    tool_call_id: Optional[str] = None  # Set on role="tool" result messages


@dataclass
class LLMResponse:
    """Normalized response from any LLM provider."""
    content: str
    model: str
    usage: Optional[dict] = None  # token counts if available
    raw_response: Optional[dict] = None  # provider-specific data
    tool_calls: list = field(default_factory=list)  # ToolCalls requested by the model
    latency_ms: Optional[float] = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    # Whether complete() honors the tools/tool_choice arguments. Providers
    # that leave this False are driven via the [SEARCH:]/sentinel fallback.
    supports_tools: bool = False

    @abstractmethod
    def complete(
        self,
        messages: list[Message],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        tools: Optional[list[ToolDef]] = None,
        tool_choice: Optional[str] = None,
    ) -> LLMResponse:
        """
        Generate a completion from the LLM.

        Args:
            messages: Conversation history
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            tools: Tools the model may call (ignored if supports_tools is False)
            tool_choice: Name of a tool the model must call

        Returns:
            Normalized LLMResponse
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging."""
        pass
