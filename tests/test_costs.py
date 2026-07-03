"""Pricing table loading and cost estimation."""

from echochamber.core.costs import MODEL_PRICING, estimate_run_cost, get_model_pricing


def test_pricing_loads_from_data_file():
    assert "claude-sonnet-4-20250514" in MODEL_PRICING
    pricing = MODEL_PRICING["claude-sonnet-4-20250514"]
    assert pricing.input_per_million == 3.0
    assert pricing.output_per_million == 15.0


def test_partial_match_resolves_model_variants():
    pricing = get_model_pricing("gpt-4o-2024-08-06")
    assert pricing.input_per_million == 2.5


def test_unknown_model_defaults_to_free():
    pricing = get_model_pricing("some/unknown-model")
    assert pricing.input_per_million == 0.0
    assert pricing.output_per_million == 0.0


def test_estimate_run_cost_positive_for_paid_models():
    min_cost, max_cost, breakdown = estimate_run_cost(
        "gpt-4o", "gpt-4o", "claude-sonnet-4-20250514", max_rounds=3
    )
    assert 0 < min_cost < max_cost
    assert set(breakdown) == {"prosecution", "defense", "moderator"}
    assert all(d["cost"] > 0 for d in breakdown.values())


def test_estimate_run_cost_free_for_local():
    min_cost, max_cost, _ = estimate_run_cost(
        "local-model", "local-model", "local-model", max_rounds=3
    )
    assert min_cost == 0.0 and max_cost == 0.0
