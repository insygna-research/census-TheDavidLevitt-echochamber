"""Token budgets, cancellation, and status callbacks."""

import pytest
from helpers import FakeProvider

from echochamber.core.agent import Agent, Role
from echochamber.core.session import CourtSession, SessionConfig, TerminationReason
from echochamber.core.usage import TokenBudgetExceeded, UsageMeter


def build_session(mod_responses, pros_responses=("argument",), def_responses=("counter",),
                  meter=None, **session_kwargs):
    def agent(name, role, responses):
        return Agent(name=name, role=role, provider=FakeProvider(list(responses)), meter=meter)

    return CourtSession(
        agent("Prosecution", Role.PROSECUTION, pros_responses),
        agent("Defense", Role.DEFENSE, def_responses),
        agent("Moderator", Role.MODERATOR, list(mod_responses)),
        config=SessionConfig(verbose=False, max_rounds=1),
        **session_kwargs,
    )


def test_meter_raises_when_budget_crossed():
    meter = UsageMeter(hard_limit_tokens=100)
    with pytest.raises(TokenBudgetExceeded):
        meter.record(module="m", provider="p", model="gpt-4o",
                     usage={"input_tokens": 90, "output_tokens": 20})
    # The crossing call is still recorded
    assert meter.total_tokens() == (90, 20)


def test_session_halts_gracefully_on_budget():
    # FakeProvider reports 150 tokens/call; budget of 200 trips on call #2
    # (prosecution turn), so no defense/eval/ruling calls happen.
    meter = UsageMeter(hard_limit_tokens=200)
    session = build_session(
        mod_responses=["opening"],  # only the opening should be consumed
        meter=meter,
    )
    result = session.run("Topic", "Position")

    assert result.termination_reason == TerminationReason.TOKEN_BUDGET
    assert "halted" in result.transcript.entries[-1].content
    # Ruling was synthetic — no extra provider calls after the budget tripped
    assert meter.total_tokens() == (200, 100)  # exactly 2 calls recorded


def test_should_stop_cancels_before_next_phase():
    calls = {"n": 0}

    def stop_after_first_phase():
        calls["n"] += 1
        return calls["n"] > 1  # allow opening, stop the prosecution phase

    session = build_session(
        mod_responses=["opening"],
        should_stop=stop_after_first_phase,
    )
    result = session.run("Topic", "Position")

    assert result.termination_reason == TerminationReason.CANCELLED
    assert "halted" in result.transcript.entries[-1].content


def test_on_status_reports_stage_and_agent():
    seen = []
    session = build_session(
        mod_responses=[
            "opening",
            "CONTINUE: NO\nWINNER: DEFENSE\nREASONING: done",
            "ruling",
        ],
        on_status=lambda stage, agent: seen.append((stage, agent.name)),
    )
    session.run("Topic", "Position")

    assert seen == [
        ("opening", "Moderator"),
        ("round 1: prosecution", "Prosecution"),
        ("round 1: defense", "Defense"),
        ("round 1: evaluation", "Moderator"),
        ("final ruling", "Moderator"),
    ]
