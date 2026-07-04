"""Text streaming: provider deltas flow through turns and sessions."""

from helpers import FakeProvider

from echochamber.core.agent import Agent, Role
from echochamber.core.session import CourtSession, SessionConfig
from echochamber.core.turns import run_agent_turn
from echochamber.providers import Message


def test_turn_streams_deltas_that_join_to_final_text():
    provider = FakeProvider(["a dense three point argument"])
    agent = Agent(name="P", role=Role.PROSECUTION, provider=provider)
    fragments = []

    text = run_agent_turn(
        agent, [Message(role="user", content="go")], on_delta=fragments.append
    )

    assert text == "a dense three point argument"
    assert len(fragments) > 1  # actually streamed, not one blob
    assert "".join(fragments) == text


def test_session_streams_with_speaker_attribution():
    prosecution = Agent(name="Prosecution", role=Role.PROSECUTION,
                        provider=FakeProvider(["tabs are better"]))
    defense = Agent(name="Defense", role=Role.DEFENSE,
                    provider=FakeProvider(["spaces are better"]))
    moderator = Agent(name="Moderator", role=Role.MODERATOR,
                      provider=FakeProvider([
                          "welcome to court",
                          "CONTINUE: NO\nWINNER: DEFENSE\nREASONING: done",
                          "final ruling",
                      ]))
    deltas = []
    session = CourtSession(
        prosecution, defense, moderator,
        config=SessionConfig(verbose=False, max_rounds=1),
        on_delta=lambda speaker, role, frag: deltas.append((speaker, role, frag)),
    )
    session.run("Topic", "Position")

    speakers = {(s, r) for s, r, _ in deltas}
    assert ("Moderator", "moderator") in speakers  # opening streamed
    assert ("Prosecution", "prosecution") in speakers
    assert ("Defense", "defense") in speakers
    # Each speaker's fragments reassemble into their recorded turn
    pros_text = "".join(f for s, _, f in deltas if s == "Prosecution")
    assert pros_text == "tabs are better"


def test_no_streaming_callback_is_fine():
    provider = FakeProvider(["plain"])
    agent = Agent(name="P", role=Role.PROSECUTION, provider=provider)
    assert run_agent_turn(agent, [Message(role="user", content="go")]) == "plain"
    assert provider.calls[0]["on_delta"] is None
