"""Core components for EchoChamber."""

from .agent import Agent, Role, ModeratorDecision, create_agent
from .transcript import Transcript, TranscriptEntry
from .session import CourtSession, SessionConfig, SessionResult, TerminationReason
from .evidence import Evidence, EvidenceStore, create_case_folder
from .preprocessor import (
    ContextStrategy,
    PreprocessorConfig,
    DocumentPreprocessor,
    ProcessedEvidenceStore,
    create_summarizer_from_provider,
)
from .costs import estimate_run_cost, format_cost_estimate, get_model_pricing
from .runner import DebateSpec, DebateOutcome, run_debate
from .turns import run_agent_turn
from .usage import UsageEvent, UsageMeter

__all__ = [
    "DebateSpec",
    "DebateOutcome",
    "run_debate",
    "run_agent_turn",
    "UsageEvent",
    "UsageMeter",
    "Agent",
    "Role",
    "ModeratorDecision",
    "create_agent",
    "Transcript",
    "TranscriptEntry",
    "CourtSession",
    "SessionConfig",
    "SessionResult",
    "TerminationReason",
    "Evidence",
    "EvidenceStore",
    "create_case_folder",
    "ContextStrategy",
    "PreprocessorConfig",
    "DocumentPreprocessor",
    "ProcessedEvidenceStore",
    "create_summarizer_from_provider",
    "estimate_run_cost",
    "format_cost_estimate",
    "get_model_pricing",
]
