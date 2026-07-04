"""APA model recommendations for debate roles.

APA (Agent Procurement Agent) is a companion system that continuously
benchmarks models and maintains, per use-case role, a current winner and
ordered fallbacks plus scraped prices and quality cutoffs. This module maps
APA's roles onto EchoChamber's:

    moderator (judge)        → APA "reasoning"  (evaluating arguments is hard analysis)
    prosecution / defense    → APA "daily"      (structured argumentation, not frontier depth)

and surfaces the top 2 models per debate role with justification, price,
and the benchmark bar they cleared.

Data sources, in order:
  1. explicit path argument
  2. $ECHOCHAMBER_APA_DATA (a directory with apa-roles.json + apa-state.json,
     or a single merged JSON file {"roles": ..., "prices": ..., "cutoffs": ...})
  3. the bundled sample (data/apa_board_sample.json) so the feature works
     without an APA deployment
"""

import json
import os
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Optional

# Debate role → APA use-case role
ROLE_MAP = {
    "prosecution": "daily",
    "defense": "daily",
    "moderator": "reasoning",
}

# Model-name prefix → EchoChamber provider. Models whose family has no
# provider here (e.g. grok) are skipped — we can't run them anyway.
_PROVIDER_PREFIXES = [
    ("claude", "anthropic"),
    ("gemini", "gemini"),
    ("gpt", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
    ("llama", "together"),
    ("meta-llama", "together"),
    ("deepseek", "together"),
    ("qwen", "together"),
    ("glm", "together"),
    ("mistral", "together"),
]


def provider_for_model(model: str) -> Optional[str]:
    """Infer the EchoChamber provider for a model id, or None if unrunnable."""
    m = model.lower()
    for prefix, provider in _PROVIDER_PREFIXES:
        if m.startswith(prefix) or f"/{prefix}" in m:
            return provider
    return None


# Which credit-pool descriptions apply to which EchoChamber provider —
# matched against pool keys/names from an APA credits table.
_CREDIT_KEYWORDS = {
    "gemini": ("gcp", "gemini", "vertex", "google"),
    "anthropic": ("anthropic", "claude"),
    "openai": ("openai", "gpt"),
    "together": ("together",),
}


@dataclass
class Recommendation:
    """One recommended model for a debate role."""
    model: str
    provider: str
    rank: int  # 1 = APA's current winner, 2 = first fallback
    apa_role: str
    justification: str
    price_in: Optional[float] = None   # $/1M input tokens
    price_out: Optional[float] = None  # $/1M output tokens
    benchmark: str = ""                # the quality bar this role screens on
    funding_note: str = ""             # source-of-funds: credits this model bills against

    def label(self) -> str:
        price = (
            f"${self.price_in:g}/${self.price_out:g} per 1M"
            if self.price_in is not None
            else "price unknown"
        )
        tag = "APA winner" if self.rank == 1 else "APA fallback"
        return f"{self.model} ({self.provider}) — {tag}, {price}"


@dataclass
class ApaData:
    """Parsed APA export: role selections, prices, cutoffs, credit pools."""
    roles: dict = field(default_factory=dict)
    prices: dict = field(default_factory=dict)
    cutoffs: dict = field(default_factory=dict)
    credits: dict = field(default_factory=dict)  # pool -> {name,total,until,note}
    source: str = ""


def _read_merged(path: Path) -> ApaData:
    raw = json.loads(path.read_text())
    return ApaData(
        roles=raw.get("roles", {}),
        prices=raw.get("prices", {}),
        cutoffs=raw.get("cutoffs", {}),
        credits=raw.get("credits", {}),
        source=str(path),
    )


def _read_dashboard_dir(path: Path) -> ApaData:
    roles = json.loads((path / "apa-roles.json").read_text()).get("roles", {})
    state = json.loads((path / "apa-state.json").read_text())
    credits = {}
    credits_file = path / "credits.json"
    if credits_file.exists():
        credits = json.loads(credits_file.read_text())
    return ApaData(
        roles=roles,
        prices=state.get("prices", {}),
        cutoffs=state.get("cutoffs", {}),
        credits=credits,
        source=str(path),
    )


def load_apa_data(path: Optional[str] = None) -> ApaData:
    """Load APA data from a path, $ECHOCHAMBER_APA_DATA, or the bundled sample."""
    candidate = path or os.environ.get("ECHOCHAMBER_APA_DATA")
    if candidate:
        p = Path(candidate).expanduser()
        if p.is_dir():
            return _read_dashboard_dir(p)
        return _read_merged(p)

    sample = resources.files("echochamber.data").joinpath("apa_board_sample.json")
    data = json.loads(sample.read_text())
    return ApaData(
        roles=data.get("roles", {}),
        prices=data.get("prices", {}),
        cutoffs=data.get("cutoffs", {}),
        credits=data.get("credits", {}),
        source="bundled sample",
    )


def credit_note_for_provider(provider: str, credits: dict) -> str:
    """Source-of-funds note if this provider bills against a credit pool."""
    keywords = _CREDIT_KEYWORDS.get(provider, ())
    for key, pool in credits.items():
        if not isinstance(pool, dict) or not pool.get("total"):
            continue
        haystack = f"{key} {pool.get('name', '')}".lower()
        if any(k in haystack for k in keywords):
            name = pool.get("name") or key
            note = f"{name}: ${pool['total']:g} pool"
            if pool.get("until"):
                note += f", expires {pool['until']}"
            detail = str(pool.get("note", "")).split("—")[0].strip()
            if detail:
                note += f" — {detail[:110]}"
            return note
    return ""


def credits_callout(data: ApaData) -> str:
    """One attention-grabbing line per credit pool relevant to a runnable provider."""
    lines = []
    seen = set()
    for provider in _CREDIT_KEYWORDS:
        note = credit_note_for_provider(provider, data.credits)
        if note and note not in seen:
            seen.add(note)
            lines.append(f"💰 **{note}** → prefer {provider} models: they bill against credits, not your card.")
    return "\n\n".join(lines)


def _price_of(model: str, prices: dict) -> tuple[Optional[float], Optional[float]]:
    """Substring-match a model against the APA price table."""
    m = model.lower()
    for key, p in prices.items():
        k = key.lower()
        if k in m or m in k:
            return p.get("in"), p.get("out")
    return None, None


def recommend_for_role(debate_role: str, data: ApaData, top_n: int = 2) -> list[Recommendation]:
    """Top-N model recommendations for one debate role."""
    apa_role = ROLE_MAP.get(debate_role)
    if not apa_role or apa_role not in data.roles:
        return []

    role_cfg = data.roles[apa_role]
    cutoff = data.cutoffs.get(apa_role, {})
    primary = role_cfg.get("primary", "")
    min_score = cutoff.get("min")
    benchmark = (
        f"{primary} ≥ {min_score}" if primary and min_score is not None else primary
    )

    role_why = cutoff.get("why", "")
    candidates = [role_cfg.get("winner")] + list(role_cfg.get("fallbacks", []))

    recommendations = []
    for model in candidates:
        if not model:
            continue
        provider = provider_for_model(model)
        if not provider:
            continue  # no EchoChamber provider can run it
        rank = len(recommendations) + 1
        position = (
            f"APA's current winner for its '{apa_role}' role"
            if rank == 1
            else f"APA's ranked fallback for '{apa_role}'"
        )
        price_in, price_out = _price_of(model, data.prices)
        recommendations.append(Recommendation(
            model=model,
            provider=provider,
            rank=rank,
            apa_role=apa_role,
            justification=f"{position}. {role_why}".strip(),
            price_in=price_in,
            price_out=price_out,
            benchmark=benchmark,
            funding_note=credit_note_for_provider(provider, data.credits),
        ))
        if len(recommendations) == top_n:
            break
    return recommendations


def recommend_for_debate(
    data: Optional[ApaData] = None, top_n: int = 2
) -> dict[str, list[Recommendation]]:
    """Top-N recommendations for every debate role."""
    if data is None:
        data = load_apa_data()
    return {
        role: recommend_for_role(role, data, top_n)
        for role in ("prosecution", "defense", "moderator")
    }


def format_recommendations(recs: dict[str, list[Recommendation]], source: str = "") -> str:
    """Human-readable recommendation summary (used by CLI/GUI)."""
    lines = []
    if source:
        lines.append(f"APA recommendations (source: {source})")
    for role, items in recs.items():
        lines.append(f"\n{role.capitalize()}:")
        if not items:
            lines.append("  (no APA data for this role)")
        for r in items:
            lines.append(f"  {r.rank}. {r.label()}")
            if r.benchmark:
                lines.append(f"     Bar: {r.benchmark}")
            lines.append(f"     {r.justification}")
    return "\n".join(lines)
