"""APA recommendation loading and role mapping."""

import json

from echochamber.recommendations import (
    load_apa_data,
    provider_for_model,
    recommend_for_debate,
    recommend_for_role,
)


def test_provider_inference():
    assert provider_for_model("claude-sonnet-4-20250514") == "anthropic"
    assert provider_for_model("gemini-2.5-pro") == "gemini"
    assert provider_for_model("gpt-4o") == "openai"
    assert provider_for_model("llama-4-maverick") == "together"
    assert provider_for_model("grok-4.3") is None  # no EchoChamber provider


def test_bundled_sample_recommends_all_roles():
    data = load_apa_data()
    recs = recommend_for_debate(data)

    assert set(recs) == {"prosecution", "defense", "moderator"}
    for role, items in recs.items():
        assert len(items) == 2, f"{role} should have top-2"
        assert items[0].rank == 1 and items[1].rank == 2
        assert items[0].justification
        assert items[0].price_in is not None

    # Judge maps to the reasoning role; advocates to daily
    assert recs["moderator"][0].apa_role == "reasoning"
    assert recs["prosecution"][0].apa_role == "daily"


def test_unrunnable_winners_are_skipped(tmp_path):
    board = {
        "roles": {
            "reasoning": {
                "primary": "AA Intelligence",
                "winner": "grok-4.3",  # no provider → skipped
                "fallbacks": ["claude-opus-4-20250514", "gpt-4o"],
            },
        },
        "prices": {"claude-opus-4": {"in": 15.0, "out": 75.0}},
        "cutoffs": {"reasoning": {"min": 50, "why": "hard analysis"}},
    }
    path = tmp_path / "board.json"
    path.write_text(json.dumps(board))

    data = load_apa_data(str(path))
    recs = recommend_for_role("moderator", data)

    assert [r.model for r in recs] == ["claude-opus-4-20250514", "gpt-4o"]
    assert recs[0].rank == 1  # promoted after the unrunnable winner
    assert recs[0].benchmark == "AA Intelligence ≥ 50"


def test_dashboard_dir_format(tmp_path):
    (tmp_path / "apa-roles.json").write_text(json.dumps({
        "roles": {"daily": {"primary": "AA Intelligence",
                            "winner": "gemini-2.5-flash", "fallbacks": ["gpt-4o-mini"]}},
    }))
    (tmp_path / "apa-state.json").write_text(json.dumps({
        "prices": {"gemini-2.5-flash": {"in": 0.3, "out": 2.5}},
        "cutoffs": {"daily": {"min": 33, "why": "solid comprehension"}},
    }))

    data = load_apa_data(str(tmp_path))
    recs = recommend_for_role("prosecution", data)

    assert recs[0].model == "gemini-2.5-flash"
    assert recs[0].provider == "gemini"
    assert recs[0].price_out == 2.5
    assert "solid comprehension" in recs[0].justification
