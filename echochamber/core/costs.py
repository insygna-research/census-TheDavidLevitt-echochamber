"""Cost estimation for LLM API calls.

Prices live in data/pricing.json (a human-editable data file) rather than
in code, so updating them never touches logic.
"""

import json
from dataclasses import dataclass
from importlib import resources
from typing import Optional


@dataclass
class ModelPricing:
    """Pricing per million tokens."""
    input_per_million: float
    output_per_million: float
    name: str = ""


def _load_pricing() -> dict[str, ModelPricing]:
    raw = json.loads(
        resources.files("echochamber.data").joinpath("pricing.json").read_text()
    )
    return {
        model: ModelPricing(
            input_per_million=float(p["in"]),
            output_per_million=float(p["out"]),
            name=p.get("name", model),
        )
        for model, p in raw["models"].items()
    }


MODEL_PRICING = _load_pricing()


def get_model_pricing(model: str) -> Optional[ModelPricing]:
    """Get pricing for a model, with fallback matching."""
    # Exact match
    if model in MODEL_PRICING:
        return MODEL_PRICING[model]

    # Partial match (for model variants)
    model_lower = model.lower()
    for key, pricing in MODEL_PRICING.items():
        if key.lower() in model_lower or model_lower in key.lower():
            return pricing

    # Default to free (assume local)
    return ModelPricing(0.0, 0.0, model)


def estimate_run_cost(
    prosecution_model: str,
    defense_model: str,
    moderator_model: str,
    max_rounds: int,
    evidence_tokens: int = 0,
) -> tuple[float, float, dict]:
    """
    Estimate the cost of a single debate run.

    Args:
        prosecution_model: Model name for prosecution
        defense_model: Model name for defense
        moderator_model: Model name for moderator
        max_rounds: Maximum number of debate rounds
        evidence_tokens: Estimated tokens in evidence context

    Returns:
        Tuple of (min_cost, max_cost, breakdown_dict)
    """
    # Estimate tokens per turn
    # These are rough estimates based on typical debate patterns
    TOKENS_PER_OPENING = 500
    TOKENS_PER_ARGUMENT = 800
    TOKENS_PER_REBUTTAL = 600
    TOKENS_PER_MODERATION = 400
    TOKENS_PER_FINAL_RULING = 1000

    # Get pricing
    pros_pricing = get_model_pricing(prosecution_model)
    def_pricing = get_model_pricing(defense_model)
    mod_pricing = get_model_pricing(moderator_model)

    # Calculate input tokens per agent per round
    # Input grows as conversation history accumulates
    base_system_tokens = 500  # System prompt
    context_tokens = evidence_tokens

    breakdown = {
        "prosecution": {"model": prosecution_model, "input_tokens": 0, "output_tokens": 0, "cost": 0.0},
        "defense": {"model": defense_model, "input_tokens": 0, "output_tokens": 0, "cost": 0.0},
        "moderator": {"model": moderator_model, "input_tokens": 0, "output_tokens": 0, "cost": 0.0},
    }

    # Opening statements
    # Moderator opening
    breakdown["moderator"]["input_tokens"] += base_system_tokens + context_tokens
    breakdown["moderator"]["output_tokens"] += TOKENS_PER_MODERATION

    # Prosecution opening
    breakdown["prosecution"]["input_tokens"] += base_system_tokens + context_tokens + TOKENS_PER_MODERATION
    breakdown["prosecution"]["output_tokens"] += TOKENS_PER_OPENING

    # Defense opening
    history_so_far = TOKENS_PER_MODERATION + TOKENS_PER_OPENING
    breakdown["defense"]["input_tokens"] += base_system_tokens + context_tokens + history_so_far
    breakdown["defense"]["output_tokens"] += TOKENS_PER_OPENING

    # Each round
    history_tokens = history_so_far + TOKENS_PER_OPENING
    for round_num in range(max_rounds):
        # Prosecution argument
        breakdown["prosecution"]["input_tokens"] += base_system_tokens + context_tokens + history_tokens
        breakdown["prosecution"]["output_tokens"] += TOKENS_PER_ARGUMENT
        history_tokens += TOKENS_PER_ARGUMENT

        # Defense rebuttal
        breakdown["defense"]["input_tokens"] += base_system_tokens + context_tokens + history_tokens
        breakdown["defense"]["output_tokens"] += TOKENS_PER_REBUTTAL
        history_tokens += TOKENS_PER_REBUTTAL

        # Moderator evaluation
        breakdown["moderator"]["input_tokens"] += base_system_tokens + context_tokens + history_tokens
        breakdown["moderator"]["output_tokens"] += TOKENS_PER_MODERATION
        history_tokens += TOKENS_PER_MODERATION

    # Final ruling
    breakdown["moderator"]["input_tokens"] += base_system_tokens + context_tokens + history_tokens
    breakdown["moderator"]["output_tokens"] += TOKENS_PER_FINAL_RULING

    # Calculate costs
    for agent, data in breakdown.items():
        if agent == "prosecution":
            pricing = pros_pricing
        elif agent == "defense":
            pricing = def_pricing
        else:
            pricing = mod_pricing

        input_cost = (data["input_tokens"] / 1_000_000) * pricing.input_per_million
        output_cost = (data["output_tokens"] / 1_000_000) * pricing.output_per_million
        data["cost"] = input_cost + output_cost

    total_cost = sum(d["cost"] for d in breakdown.values())

    # Min/max estimates (actual could vary by 50%)
    min_cost = total_cost * 0.5
    max_cost = total_cost * 1.5

    return min_cost, max_cost, breakdown


def estimate_debate_footprint(
    prosecution_model: str,
    defense_model: str,
    moderator_model: str,
    max_rounds: int,
    iterations: int = 1,
    evidence_tokens: int = 0,
    seconds_per_call: float = 8.0,
) -> dict:
    """
    Rough total footprint for a debate configuration: tokens, dollars,
    model calls, and wall-clock seconds, scaled by iterations.

    Clock time is a heuristic (API models average ~8s/call; local models
    vary wildly) — treat it as an order of magnitude, not a promise.
    """
    iterations = max(1, int(iterations))
    min_cost, max_cost, breakdown = estimate_run_cost(
        prosecution_model, defense_model, moderator_model, max_rounds, evidence_tokens
    )
    tokens_per_run = sum(
        d["input_tokens"] + d["output_tokens"] for d in breakdown.values()
    )
    # opening + per round (prosecution, defense, evaluation) + final ruling
    calls_per_run = 2 + 3 * max_rounds
    return {
        "iterations": iterations,
        "tokens": tokens_per_run * iterations,
        "cost_min": min_cost * iterations,
        "cost_max": max_cost * iterations,
        "calls": calls_per_run * iterations,
        "seconds": calls_per_run * iterations * seconds_per_call,
    }


def format_cost_estimate(
    min_cost: float,
    max_cost: float,
    breakdown: dict,
    num_runs: int = 1,
) -> str:
    """Format a cost estimate for display."""
    lines = []

    if num_runs > 1:
        lines.append(f"Estimated cost for {num_runs} runs:")
        lines.append(f"  Total: ${min_cost * num_runs:.2f} - ${max_cost * num_runs:.2f}")
        lines.append("")
        lines.append("Per run breakdown:")
    else:
        lines.append("Estimated cost:")

    lines.append(f"  Range: ${min_cost:.4f} - ${max_cost:.4f}")
    lines.append("")
    lines.append("  By agent:")
    for agent, data in breakdown.items():
        model_short = data["model"].split("/")[-1][:30]
        lines.append(f"    {agent.capitalize()}: ${data['cost']:.4f} ({model_short})")
        lines.append(f"      Input: {data['input_tokens']:,} tokens, Output: {data['output_tokens']:,} tokens")

    return "\n".join(lines)
