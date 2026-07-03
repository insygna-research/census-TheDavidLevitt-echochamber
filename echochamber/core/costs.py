"""Cost estimation for LLM API calls."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelPricing:
    """Pricing per million tokens."""
    input_per_million: float
    output_per_million: float
    name: str = ""


# Pricing as of early 2025 (USD per million tokens)
# Update these as prices change
MODEL_PRICING = {
    # Anthropic
    "claude-opus-4-20250514": ModelPricing(15.0, 75.0, "Claude Opus 4"),
    "claude-sonnet-4-20250514": ModelPricing(3.0, 15.0, "Claude Sonnet 4"),
    "claude-haiku-3-5-20241022": ModelPricing(0.25, 1.25, "Claude Haiku 3.5"),

    # OpenAI
    "gpt-4o": ModelPricing(2.5, 10.0, "GPT-4o"),
    "gpt-4o-mini": ModelPricing(0.15, 0.60, "GPT-4o Mini"),
    "gpt-4-turbo": ModelPricing(10.0, 30.0, "GPT-4 Turbo"),

    # Together.AI (approximate)
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": ModelPricing(3.5, 3.5, "Llama 3.1 405B"),
    "meta-llama/Llama-3-70b-chat-hf": ModelPricing(0.9, 0.9, "Llama 3 70B"),
    "deepseek-ai/DeepSeek-R1": ModelPricing(3.0, 7.0, "DeepSeek R1"),
    "deepseek-ai/DeepSeek-V3": ModelPricing(0.5, 1.0, "DeepSeek V3"),

    # Google Gemini
    "gemini-2.0-flash": ModelPricing(0.10, 0.40, "Gemini 2.0 Flash"),
    "gemini-1.5-pro": ModelPricing(1.25, 5.0, "Gemini 1.5 Pro"),
    "gemini-1.5-flash": ModelPricing(0.075, 0.30, "Gemini 1.5 Flash"),

    # Local (free)
    "local-model": ModelPricing(0.0, 0.0, "Local Model"),
}


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
