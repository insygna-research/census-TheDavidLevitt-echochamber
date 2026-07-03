"""Court session orchestrator."""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable

from typing import Union

from .agent import Agent, Role
from .transcript import Transcript
from .evidence import EvidenceStore
from .preprocessor import ProcessedEvidenceStore, ContextStrategy
from ..providers import Message


class TerminationReason(Enum):
    """Why the session ended."""
    MAX_ROUNDS = "max_rounds"
    CONCESSION = "concession"
    MODERATOR_DECISION = "moderator_decision"
    ERROR = "error"


@dataclass
class SessionConfig:
    """Configuration for a court session."""
    max_rounds: int = 5
    allow_concession: bool = True
    allow_conviction: bool = False  # If True, agents can be convinced by opposing arguments
    require_moderator_approval: bool = True
    verbose: bool = True


@dataclass
class SessionResult:
    """Result of a completed court session."""
    transcript: Transcript
    termination_reason: TerminationReason
    winner: Optional[str] = None
    rounds_completed: int = 0


class CourtSession:
    """
    Orchestrates a courtroom debate between LLM agents.

    The session alternates between prosecution and defense,
    with the moderator evaluating after each round.
    """

    def __init__(
        self,
        prosecution: Agent,
        defense: Agent,
        moderator: Agent,
        config: Optional[SessionConfig] = None,
        evidence: Optional[Union[EvidenceStore, ProcessedEvidenceStore]] = None,
        on_turn: Optional[Callable[[str, str, str], None]] = None,
        search_tool: Optional[object] = None,
        max_moderator_searches: int = 0,
    ):
        """
        Initialize a court session.

        Args:
            prosecution: Agent arguing for the proposition
            defense: Agent arguing against
            moderator: Agent controlling flow and deciding outcome
            config: Session configuration
            evidence: Optional evidence store (raw or processed) with case documents
            on_turn: Callback(speaker, role, content) after each turn
            search_tool: Optional WebSearchTool for moderator searches during final ruling
            max_moderator_searches: Maximum searches allowed for moderator (0 = disabled)
        """
        if prosecution.role != Role.PROSECUTION:
            raise ValueError("Prosecution agent must have PROSECUTION role")
        if defense.role != Role.DEFENSE:
            raise ValueError("Defense agent must have DEFENSE role")
        if moderator.role != Role.MODERATOR:
            raise ValueError("Moderator agent must have MODERATOR role")

        self.prosecution = prosecution
        self.defense = defense
        self.moderator = moderator
        self.config = config or SessionConfig()
        self.evidence = evidence
        self.on_turn = on_turn
        self.search_tool = search_tool
        self.max_moderator_searches = max_moderator_searches

        # Track if using processed evidence for RAG queries
        self._is_processed = isinstance(evidence, ProcessedEvidenceStore)
        self._use_rag = (
            self._is_processed and
            evidence.config.strategy == ContextStrategy.RAG
        )

        self.transcript: Optional[Transcript] = None
        self._running = False

    def run(self, topic: str, prosecution_position: str) -> SessionResult:
        """
        Run the full court session.

        Args:
            topic: The subject of debate
            prosecution_position: The position the prosecution will argue for

        Returns:
            SessionResult with transcript and outcome
        """
        self._running = True
        self.transcript = Transcript(case_topic=topic)

        # Opening context for all parties
        case_context = f"""
CASE TOPIC: {topic}

PROSECUTION POSITION: {prosecution_position}

DEFENSE POSITION: Argue against the prosecution's position.

The debate will proceed in rounds. Each round:
1. Prosecution presents arguments
2. Defense responds
3. Moderator evaluates

The debate ends when:
- Maximum rounds ({self.config.max_rounds}) reached
- A party concedes
- The moderator decides further debate is unnecessary
"""

        # Add shared evidence context if available
        shared_evidence = ""
        if self.evidence:
            shared_evidence = self.evidence.get_shared_context()
            if shared_evidence:
                case_context += f"\n{shared_evidence}"

        self._log(f"Starting court session on: {topic}")
        self._log(f"Prosecution argues: {prosecution_position}")
        if self.evidence:
            self._log(self.evidence.summary())
        self._log("-" * 50)

        # Moderator opens the session
        opening = self._get_moderator_opening(case_context)
        self._record_turn(self.moderator.name, "moderator", opening)

        round_num = 0
        termination_reason = TerminationReason.MAX_ROUNDS
        winner = None

        try:
            for round_num in range(1, self.config.max_rounds + 1):
                self.transcript.total_rounds = round_num
                self._log(f"\n=== ROUND {round_num} ===\n")

                # Prosecution argues
                prosecution_response = self._get_prosecution_argument(case_context, round_num)
                self._record_turn(
                    self.prosecution.name, "prosecution", prosecution_response, round_num
                )

                # Check for concession
                if self.config.allow_concession and self._check_concession(prosecution_response):
                    termination_reason = TerminationReason.CONCESSION
                    winner = "defense"
                    break

                # Defense responds
                defense_response = self._get_defense_argument(case_context, round_num)
                self._record_turn(
                    self.defense.name, "defense", defense_response, round_num
                )

                # Check for concession
                if self.config.allow_concession and self._check_concession(defense_response):
                    termination_reason = TerminationReason.CONCESSION
                    winner = "prosecution"
                    break

                # Moderator evaluates
                if self.config.require_moderator_approval:
                    should_continue, mod_winner, reasoning = self._get_moderator_evaluation(
                        case_context, round_num
                    )
                    self._record_turn(
                        self.moderator.name, "moderator", reasoning, round_num
                    )

                    if not should_continue:
                        termination_reason = TerminationReason.MODERATOR_DECISION
                        winner = mod_winner
                        break

        except Exception as e:
            self._log(f"Error during session: {e}")
            termination_reason = TerminationReason.ERROR
            self.transcript.add_entry(
                speaker="SYSTEM",
                role="system",
                content=f"Session terminated due to error: {e}",
            )

        # Get final ruling (may update winner if not already set)
        final_ruling, final_winner = self._get_final_ruling(case_context, termination_reason, winner)
        self._record_turn(self.moderator.name, "moderator", final_ruling)

        # Use updated winner from final ruling if available
        if final_winner and not winner:
            winner = final_winner

        # Finalize transcript
        outcome_str = f"{termination_reason.value}"
        if winner:
            outcome_str += f" - {winner} wins"
        self.transcript.finalize(outcome=outcome_str, winner=winner)

        self._running = False

        return SessionResult(
            transcript=self.transcript,
            termination_reason=termination_reason,
            winner=winner,
            rounds_completed=round_num,
        )

    def _get_evidence_context_for_role(
        self,
        role: Role,
        base_context: str,
        query: Optional[str] = None,
    ) -> str:
        """
        Get case context with role-specific evidence appended.

        For RAG mode, uses query to retrieve relevant chunks.
        """
        if not self.evidence:
            return base_context

        if self._is_processed:
            # ProcessedEvidenceStore
            role_str = role.value if hasattr(role, 'value') else str(role)
            if self._use_rag and query:
                role_evidence = self.evidence.get_context_for_role(role_str, query=query)
            else:
                role_evidence = self.evidence.get_context_for_role(role_str)
        else:
            # Raw EvidenceStore
            role_evidence = self.evidence.get_context_for_role(role)

        if role_evidence:
            return f"{base_context}\n{role_evidence}"
        return base_context

    def _get_moderator_opening(self, case_context: str) -> str:
        """Get the moderator's opening statement."""
        context = self._get_evidence_context_for_role(Role.MODERATOR, case_context)
        messages = [
            Message(
                role="user",
                content=f"{context}\n\nPlease open this court session with a brief statement about the case and rules.",
            )
        ]
        return self.moderator.respond(messages)

    def _get_prosecution_argument(self, case_context: str, round_num: int) -> str:
        """Get prosecution's argument for this round."""
        history = self.transcript.get_conversation_history()

        # For RAG, use recent context as query
        rag_query = None
        if self._use_rag and history:
            # Use last few exchanges as context for retrieval
            recent = [m.content for m in history[-4:]]
            rag_query = " ".join(recent)[:1000]

        context = self._get_evidence_context_for_role(Role.PROSECUTION, case_context, query=rag_query)

        concession_note = ""
        if self.config.allow_concession:
            concession_note = '\nIf you believe your position is untenable, you may concede by explicitly stating "I CONCEDE" in your response.'

        instruction = f"""
{context}

This is round {round_num}. Present your arguments for your position.
{"This is your opening argument. Establish your main points." if round_num == 1 else "Build on your previous arguments and respond to the defense's points."}
{concession_note}
"""
        messages = history + [Message(role="user", content=instruction)]
        return self.prosecution.respond(messages)

    def _get_defense_argument(self, case_context: str, round_num: int) -> str:
        """Get defense's argument for this round."""
        history = self.transcript.get_conversation_history()

        # For RAG, use recent context as query
        rag_query = None
        if self._use_rag and history:
            recent = [m.content for m in history[-4:]]
            rag_query = " ".join(recent)[:1000]

        context = self._get_evidence_context_for_role(Role.DEFENSE, case_context, query=rag_query)

        concession_note = ""
        if self.config.allow_concession:
            concession_note = '\nIf you believe your position is untenable, you may concede by explicitly stating "I CONCEDE" in your response.'

        instruction = f"""
{context}

This is round {round_num}. Present your counter-arguments.
{"This is your opening response. Address the prosecution's main points and establish your defense." if round_num == 1 else "Continue your defense and address new points raised by prosecution."}
{concession_note}
"""
        messages = history + [Message(role="user", content=instruction)]
        return self.defense.respond(messages)

    def _get_moderator_evaluation(
        self, case_context: str, round_num: int
    ) -> tuple[bool, Optional[str], str]:
        """
        Get moderator's evaluation of the round.

        Returns:
            (should_continue, winner_if_decided, reasoning)
        """
        context = self._get_evidence_context_for_role(Role.MODERATOR, case_context)
        history = self.transcript.get_conversation_history()

        instruction = f"""
{context}

Round {round_num} has concluded. Evaluate the arguments presented.

You must respond in this exact format:
CONTINUE: [YES/NO]
WINNER: [PROSECUTION/DEFENSE/NONE]
REASONING: [Your evaluation of the arguments]

Set CONTINUE to NO only if:
- One side has clearly won and further debate would be pointless
- Both sides are repeating arguments without progress
- The debate has reached a natural conclusion

Set WINNER to the side that has made the stronger case so far, or NONE if it's too close to call.
"""
        messages = history + [Message(role="user", content=instruction)]
        response = self.moderator.respond(messages)

        # Parse the response
        should_continue = True
        winner = None

        if "CONTINUE: NO" in response.upper() or "CONTINUE:NO" in response.upper():
            should_continue = False

        if "WINNER: PROSECUTION" in response.upper() or "WINNER:PROSECUTION" in response.upper():
            winner = "prosecution"
        elif "WINNER: DEFENSE" in response.upper() or "WINNER:DEFENSE" in response.upper():
            winner = "defense"

        return should_continue, winner, response

    def _get_final_ruling(
        self,
        case_context: str,
        termination_reason: TerminationReason,
        winner: Optional[str],
    ) -> tuple[str, Optional[str]]:
        """
        Get the moderator's final ruling.

        If search is enabled, allows the moderator to perform searches before ruling.

        Returns:
            (ruling_text, winner) - winner may be updated from the ruling
        """
        context = self._get_evidence_context_for_role(Role.MODERATOR, case_context)
        history = self.transcript.get_conversation_history()

        # If no winner yet, ask moderator to decide
        if winner:
            winner_instruction = f"Winner: {winner}"
        else:
            winner_instruction = """No winner has been determined yet. You must now decide.

In your ruling, you MUST include one of these exact phrases:
- "FINAL VERDICT: PROSECUTION WINS" if prosecution made the stronger case
- "FINAL VERDICT: DEFENSE WINS" if defense made the stronger case
- "FINAL VERDICT: DRAW" if neither side clearly prevailed"""

        # Add search reminder if search is enabled
        search_reminder = ""
        if self.search_tool and self.max_moderator_searches > 0:
            search_reminder = """

IMPORTANT: Before making your final ruling, you SHOULD use web search to verify legal claims.
Include [SEARCH: your query] to search for relevant laws, precedents, or legal analysis.
Your ruling should be grounded in verified legal principles."""

        instruction = f"""
{context}

The debate has concluded.
Termination reason: {termination_reason.value}
{winner_instruction}
{search_reminder}
Please provide your final ruling and summary of the proceedings.
"""
        messages = history + [Message(role="user", content=instruction)]
        ruling = self.moderator.respond(messages)

        # Process any search requests in the ruling
        if self.search_tool and self.max_moderator_searches > 0:
            ruling = self._process_ruling_searches(ruling, messages)

        # Extract winner from ruling if not already set
        final_winner = winner
        if not final_winner:
            ruling_upper = ruling.upper()
            if "FINAL VERDICT: PROSECUTION WINS" in ruling_upper or "PROSECUTION WINS" in ruling_upper:
                final_winner = "prosecution"
            elif "FINAL VERDICT: DEFENSE WINS" in ruling_upper or "DEFENSE WINS" in ruling_upper:
                final_winner = "defense"
            elif "FINAL VERDICT: DRAW" in ruling_upper:
                final_winner = "draw"

        return ruling, final_winner

    def _process_ruling_searches(self, ruling: str, messages: list[Message]) -> str:
        """
        Process search requests in the moderator's ruling.

        If the ruling contains [SEARCH: query] tags, perform the searches
        and ask the moderator to provide a final ruling with the results.
        """
        from ..tools.search import extract_search_queries

        queries = extract_search_queries(ruling)
        if not queries:
            return ruling

        # Limit searches
        queries = queries[:self.max_moderator_searches]

        # Perform searches
        search_results = []
        self._log(f"\n  [Moderator searching: {', '.join(queries)}]")
        for query in queries:
            results = self.search_tool.search_formatted(query, max_results=3)
            search_results.append(results)
            self._log(f"\n{results}")

        # Ask moderator to finalize ruling with search results
        search_context = "\n\n".join(search_results)
        followup = f"""
Based on your search results:

{search_context}

Now provide your FINAL ruling. Remember to include one of:
- "FINAL VERDICT: PROSECUTION WINS"
- "FINAL VERDICT: DEFENSE WINS"
- "FINAL VERDICT: DRAW"

Give your complete final ruling:
"""
        messages_with_search = messages + [
            Message(role="assistant", content=ruling),
            Message(role="user", content=followup),
        ]
        final_ruling = self.moderator.respond(messages_with_search)
        return final_ruling

    def _check_concession(self, response: str) -> bool:
        """Check if the response contains a concession."""
        concession_patterns = [
            r"\bI\s+CONCEDE\b",
            r"\bI\s+CONCEDE\s+THE\s+POINT\b",
            r"\bI\s+CONCEDE\s+THIS\s+DEBATE\b",
        ]
        for pattern in concession_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                return True
        return False

    def _record_turn(
        self,
        speaker: str,
        role: str,
        content: str,
        round_number: Optional[int] = None,
    ) -> None:
        """Record a turn in the transcript and notify callback."""
        self.transcript.add_entry(
            speaker=speaker,
            role=role,
            content=content,
            round_number=round_number,
        )

        if self.on_turn:
            self.on_turn(speaker, role, content)

        if self.config.verbose:
            self._log(f"\n[{speaker} ({role})]:\n{content}\n")

    def _log(self, message: str) -> None:
        """Log a message if verbose mode is on."""
        if self.config.verbose:
            print(message)
