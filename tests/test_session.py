"""Session orchestration: verdicts, concessions, structured moderator decisions."""

from helpers import FakeProvider, tool_call_response

from echochamber.core.agent import Agent, Role
from echochamber.core.session import CourtSession, SessionConfig, TerminationReason


def build_session(pros_responses, def_responses, mod_responses,
                  mod_supports_tools=False, **config_kwargs):
    prosecution = Agent(
        name="Prosecution", role=Role.PROSECUTION,
        provider=FakeProvider(pros_responses),
    )
    defense = Agent(
        name="Defense", role=Role.DEFENSE,
        provider=FakeProvider(def_responses),
    )
    moderator = Agent(
        name="Moderator", role=Role.MODERATOR,
        provider=FakeProvider(mod_responses, supports_tools=mod_supports_tools),
    )
    config = SessionConfig(verbose=False, **config_kwargs)
    return CourtSession(prosecution, defense, moderator, config=config)


def test_text_protocol_moderator_decides():
    session = build_session(
        pros_responses=["prosecution opening"],
        def_responses=["defense response"],
        mod_responses=[
            "welcome to the court",
            "CONTINUE: NO\nWINNER: PROSECUTION\nREASONING: stronger case",
            "Final ruling: prosecution prevailed.",
        ],
        max_rounds=3,
    )
    result = session.run("Test topic", "Test position")

    assert result.termination_reason == TerminationReason.MODERATOR_DECISION
    assert result.winner == "prosecution"
    assert result.rounds_completed == 1
    roles = [e.role for e in result.transcript.entries]
    assert roles == ["moderator", "prosecution", "defense", "moderator", "moderator"]


def test_text_protocol_final_verdict_extracted():
    session = build_session(
        pros_responses=["argument"],
        def_responses=["counter"],
        mod_responses=[
            "opening",
            "CONTINUE: YES\nWINNER: NONE\nREASONING: too close",
            "Summary of proceedings. FINAL VERDICT: DEFENSE WINS",
        ],
        max_rounds=1,
    )
    result = session.run("Topic", "Position")

    assert result.termination_reason == TerminationReason.MAX_ROUNDS
    assert result.winner == "defense"


def test_concession_ends_debate():
    session = build_session(
        pros_responses=["On reflection, I CONCEDE this debate."],
        def_responses=[],
        mod_responses=["opening", "The prosecution conceded; defense wins."],
        max_rounds=3,
        allow_concession=True,
    )
    result = session.run("Topic", "Position")

    assert result.termination_reason == TerminationReason.CONCESSION
    assert result.winner == "defense"


def test_structured_moderator_via_tools():
    session = build_session(
        pros_responses=["argument"],
        def_responses=["counter"],
        mod_responses=[
            "opening",
            tool_call_response("submit_evaluation", {
                "continue_debate": False,
                "winner": "defense",
                "reasoning": "defense dismantled the case",
            }),
            tool_call_response("submit_verdict", {
                "winner": "defense",
                "reasoning": "final ruling for the defense",
            }),
        ],
        mod_supports_tools=True,
        max_rounds=3,
    )
    result = session.run("Topic", "Position")

    assert result.termination_reason == TerminationReason.MODERATOR_DECISION
    assert result.winner == "defense"
    # The tool-call reasoning is what lands in the transcript
    assert "dismantled" in result.transcript.entries[3].content
    assert "final ruling for the defense" in result.transcript.entries[-1].content


def test_structured_verdict_decides_draw():
    session = build_session(
        pros_responses=["argument"],
        def_responses=["counter"],
        mod_responses=[
            "opening",
            tool_call_response("submit_evaluation", {
                "continue_debate": True,
                "winner": "none",
                "reasoning": "even so far",
            }),
            tool_call_response("submit_verdict", {
                "winner": "draw",
                "reasoning": "neither side prevailed",
            }),
        ],
        mod_supports_tools=True,
        max_rounds=1,
    )
    result = session.run("Topic", "Position")

    assert result.termination_reason == TerminationReason.MAX_ROUNDS
    assert result.winner == "draw"
