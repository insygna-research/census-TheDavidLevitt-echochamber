"""Footprint estimation, verdict headlines, and APA credit notes."""

import json

from echochamber.core.costs import estimate_debate_footprint
from echochamber.recommendations import (
    credit_note_for_provider,
    credits_callout,
    load_apa_data,
    recommend_for_role,
)
from echochamber.ui import estimate_md, verdict_headline


def test_footprint_scales_with_iterations():
    one = estimate_debate_footprint("gpt-4o", "gpt-4o", "gpt-4o", max_rounds=2)
    five = estimate_debate_footprint("gpt-4o", "gpt-4o", "gpt-4o", max_rounds=2, iterations=5)
    assert five["tokens"] == one["tokens"] * 5
    assert five["cost_max"] == one["cost_max"] * 5
    assert five["calls"] == one["calls"] * 5 == (2 + 3 * 2) * 5
    assert one["cost_min"] > 0 and one["seconds"] > 0


def test_footprint_free_for_local():
    fp = estimate_debate_footprint("local-model", "local-model", "local-model", max_rounds=3)
    assert fp["cost_max"] == 0.0
    assert fp["tokens"] > 0


def test_estimate_md_colors_follow_funding_class():
    # Gemini bills against the sample credit pool → green "credits"
    credit = estimate_md("gemini", "gemini-2.5-flash", "gemini", "gemini-2.5-flash",
                         "gemini", "gemini-2.5-flash", 1, 1, 200_000)
    assert "🟢" in credit
    assert "color:#4ade80" in credit and "credits" in credit
    assert "color:#f87171" not in credit     # no paid-API money in this config
    assert "rough approximation" in credit   # disclaimer present
    assert "hard token budget is strongly recommended" in credit

    # Anthropic has no pool in the sample → red "paid API"; big config → 🔴
    pricey = estimate_md("anthropic", "claude-opus-4-20250514", "anthropic",
                         "claude-opus-4-20250514", "anthropic", "claude-opus-4-20250514",
                         8, 10, 20_000_000)
    assert "🔴" in pricey
    assert "color:#f87171" in pricey and "paid API" in pricey


def test_estimate_md_mixed_classes_shown_side_by_side():
    mixed = estimate_md("gemini", "gemini-2.5-flash", "gemini", "gemini-2.5-flash",
                        "anthropic", "claude-sonnet-4-20250514", 2, 1, 200_000)
    assert "color:#4ade80" in mixed and "credits" in mixed        # advocates on credits
    assert "color:#f87171" in mixed and "paid API" in mixed       # judge on cash
    assert mixed.index("paid API") < mixed.index("credits")       # real listed first


def test_funding_class_mapping():
    from echochamber.recommendations import funding_class
    sample = load_apa_data()
    assert funding_class("gemini", sample.credits) == "credit"
    assert funding_class("anthropic", sample.credits) == "real"
    assert funding_class("lmstudio", sample.credits) == "included"
    # Monthly/subscription-style pools classify as included
    subs = {"claude-sub": {"name": "Claude subscription", "total": 20, "period": "month"}}
    assert funding_class("anthropic", subs) == "included"


def test_estimate_md_warns_when_budget_below_estimate():
    warned = estimate_md("gemini", "gemini-2.5-flash", "gemini", "gemini-2.5-flash",
                         "gemini", "gemini-2.5-flash", 8, 5, 10_000)
    assert "Warning: estimated token burn exceeds the hard limit" in warned
    assert "verdict may not be rendered" in warned
    fine = estimate_md("gemini", "gemini-2.5-flash", "gemini", "gemini-2.5-flash",
                       "gemini", "gemini-2.5-flash", 1, 1, 500_000)
    assert "Warning" not in fine


def test_credit_backed_models_are_promoted_to_rank_one():
    data = load_apa_data()  # bundled sample: gemini pool active
    for role in ("prosecution", "defense", "moderator"):
        recs = recommend_for_role(role, data)
        assert recs[0].provider == "gemini", f"{role} #1 should be credit-backed"
        assert recs[0].rank == 1
        assert "Promoted to #1" in recs[0].justification
    callout = credits_callout(data)
    assert "AgentStable" in callout


def test_verdict_headline_single_and_aggregate():
    single = verdict_headline(["defense"], "Tabs are superior to spaces")
    assert "DEFENSE" in single
    assert "position rejected" in single
    assert "Tabs are superior to spaces" in single

    agg = verdict_headline(["defense", "defense", "prosecution"], "P")
    assert "DEFENSE 2/3" in agg
    assert "position rejected" in agg


def test_bundled_sample_carries_credit_note():
    data = load_apa_data()
    note = credit_note_for_provider("gemini", data.credits)
    assert "$200" in note and "2026-07-15" in note
    assert credit_note_for_provider("together", data.credits) == ""

    callout = credits_callout(data)
    assert "💰" in callout and "gemini" in callout

    # The note rides on gemini recommendations
    recs = recommend_for_role("prosecution", data)
    gemini_recs = [r for r in recs if r.provider == "gemini"]
    assert gemini_recs and all("$200" in r.funding_note for r in gemini_recs)


def test_dashboard_credits_file_is_read(tmp_path):
    (tmp_path / "apa-roles.json").write_text(json.dumps({
        "roles": {"daily": {"primary": "AA", "winner": "gemini-2.5-flash", "fallbacks": []}},
    }))
    (tmp_path / "apa-state.json").write_text(json.dumps({"prices": {}, "cutoffs": {}}))
    (tmp_path / "credits.json").write_text(json.dumps({
        "gcp-credits": {"name": "GCP credits (Gemini/Vertex)", "total": 300,
                        "until": "2026-07-16", "note": "Free Trial: $281 REMAINING — expires soon"},
    }))
    data = load_apa_data(str(tmp_path))
    note = credit_note_for_provider("gemini", data.credits)
    assert "$300" in note and "2026-07-16" in note
    assert "$281 REMAINING" in note  # detail before the em-dash survives
