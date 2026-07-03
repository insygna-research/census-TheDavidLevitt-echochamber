"""Token-economy prompts and per-role custom instructions."""

from helpers import FakeProvider

from echochamber.core.agent import Role
from echochamber.core.runner import DebateSpec, run_debate
from echochamber.roles import get_role_prompt


def test_advocates_are_told_to_skip_salutations():
    for role in (Role.PROSECUTION, Role.DEFENSE):
        prompt = get_role_prompt(role)
        assert "TOKEN ECONOMY" in prompt
        assert "salutations" in prompt
        assert "Esteemed Moderator" in prompt  # the named anti-pattern


def test_judge_evaluates_logic_not_verbosity():
    prompt = get_role_prompt(Role.MODERATOR)
    assert "Do NOT reward verboseness" in prompt
    assert "Logical consistency" in prompt
    assert "case law" in prompt


def test_custom_instructions_reach_system_prompts(tmp_path):
    fakes = [
        FakeProvider(["p arg"]),
        FakeProvider(["d arg"]),
        FakeProvider([
            "opening",
            "CONTINUE: NO\nWINNER: PROSECUTION\nREASONING: done",
            "ruling",
        ]),
    ]
    queue = list(fakes)

    spec = DebateSpec(
        topic="T", position="P",
        prosecution_instructions="Cite only peer-reviewed sources.",
        defense_instructions="Attack methodology first.",
        moderator_instructions="Penalize unsourced claims.",
        max_rounds=1, enable_search=False, verbose=False,
        transcript_dir=str(tmp_path),
    )
    run_debate(spec, provider_factory=lambda p, m=None, **kw: queue.pop(0))

    pros_sys = fakes[0].calls[0]["system_prompt"]
    def_sys = fakes[1].calls[0]["system_prompt"]
    mod_sys = fakes[2].calls[0]["system_prompt"]
    assert "Cite only peer-reviewed sources." in pros_sys
    assert "Attack methodology first." in def_sys
    assert "Penalize unsourced claims." in mod_sys
    # Instructions land under the ADDITIONAL INSTRUCTIONS banner
    assert "ADDITIONAL INSTRUCTIONS" in pros_sys
