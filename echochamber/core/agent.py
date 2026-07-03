"""Agent class for courtroom participants."""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from ..providers import LLMProvider, Message, LLMResponse


class Role(Enum):
    """Courtroom roles."""
    PROSECUTION = "prosecution"
    DEFENSE = "defense"
    MODERATOR = "moderator"
    JUROR = "juror"  # For Phase 5


@dataclass
class Agent:
    """
    An LLM agent that plays a role in the courtroom.

    Attributes:
        name: Display name for the agent
        role: Courtroom role (prosecution, defense, moderator)
        provider: LLM provider instance
        system_prompt: Base system prompt for the agent
        temperature: Sampling temperature
        max_tokens: Maximum response tokens
    """
    name: str
    role: Role
    provider: LLMProvider
    system_prompt: str = ""
    temperature: float = 0.7
    max_tokens: int = 1024
    _conversation_history: list[Message] = field(default_factory=list, repr=False)

    def respond(
        self,
        conversation: list[Message],
        additional_context: Optional[str] = None,
    ) -> str:
        """
        Generate a response given the conversation history.

        Args:
            conversation: Full conversation history visible to this agent
            additional_context: Optional context to prepend to system prompt

        Returns:
            Agent's response text
        """
        # Build effective system prompt
        system = self.system_prompt
        if additional_context:
            system = f"{additional_context}\n\n{system}"

        response = self.provider.complete(
            messages=conversation,
            system_prompt=system,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return response.content

    def __str__(self) -> str:
        return f"{self.name} ({self.role.value}) - {self.provider.name}"


@dataclass
class ModeratorDecision:
    """Result of moderator evaluation."""
    continue_debate: bool
    winner: Optional[str] = None  # "prosecution", "defense", or None
    reasoning: str = ""


def create_agent(
    name: str,
    role: Role,
    provider: LLMProvider,
    custom_prompt: Optional[str] = None,
    **kwargs,
) -> Agent:
    """
    Factory to create an agent with role-appropriate defaults.

    Args:
        name: Agent display name
        role: Courtroom role
        provider: LLM provider
        custom_prompt: Override default role prompt
        **kwargs: Additional Agent arguments

    Returns:
        Configured Agent instance
    """
    from ..roles import get_role_prompt

    system_prompt = custom_prompt or get_role_prompt(role)

    return Agent(
        name=name,
        role=role,
        provider=provider,
        system_prompt=system_prompt,
        **kwargs,
    )
