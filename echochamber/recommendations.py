"""AgentStable model recommendations for debate roles.

AgentStable is a companion cost/performance system that continuously
benchmarks models and maintains, per use-case role, a current winner and
ordered fallbacks plus scraped prices, quality cutoffs, and credit pools
(source-of-funds). This module maps its roles onto EchoChamber's:

    moderator (judge)        → "reasoning"  (evaluating arguments is hard analysis)
    prosecution / defense    → "daily"      (structured argumentation, not frontier depth)

and surfaces the top 2 models per debate role with justification, price,
and the benchmark bar they cleared. Models that bill against an active
credit pool are promoted to #1 — spending expiring credits beats spending
cash for equal-quality work.

Data sources, in order:
  1. explicit path argument
  2. $ECHOCHAMBER_AGENTSTABLE_DATA (or legacy $ECHOCHAMBER_APA_DATA) — a
     directory with apa-roles.json + apa-state.json (+ credits.json), or a
     single merged JSON file {"roles", "prices", "cutoffs", "credits"}
  3. the bundled sample (data/apa_board_sample.json) so the feature works
     without an AgentStable deployment
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
    """Load AgentStable data from a path, env var, or the bundled sample."""
    candidate = (
        path
        or os.environ.get("ECHOCHAMBER_AGENTSTABLE_DATA")
        or os.environ.get("ECHOCHAMBER_APA_DATA")  # legacy name
    )
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


def funding_class(provider: str, credits: dict) -> str:
    """Source-of-funds class for a provider: 'real' | 'credit' | 'included'.

    Mirrors AgentStable's costClass idea: not how much a call costs, but
    whose money it spends. Local models and subscription-style monthly
    pools are 'included'; finite credit pools are 'credit'; everything
    else is out-of-pocket 'real'.
    """
    if provider == "lmstudio":
        return "included"
    keywords = _CREDIT_KEYWORDS.get(provider, ())
    for key, pool in credits.items():
        if not isinstance(pool, dict) or not pool.get("total"):
            continue
        haystack = f"{key} {pool.get('name', '')}".lower()
        if any(k in haystack for k in keywords):
            if pool.get("period") == "month" or "sub" in key.lower():
                return "included"
            return "credit"
    return "real"


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
            lines.append(
                f"💰 **AgentStable: {note}** → {provider} models are ranked first below: "
                f"they bill against credits, not your card."
            )
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

    # Collect runnable candidates in AgentStable's benchmark order
    entries = []  # (benchmark_position, model, provider, funding_note)
    for position, model in enumerate(m for m in candidates if m):
        provider = provider_for_model(model)
        if not provider:
            continue  # no EchoChamber provider can run it
        entries.append((
            position, model, provider,
            credit_note_for_provider(provider, data.credits),
        ))

    # Source-of-funds beats benchmark order: models billing against an
    # active credit pool are promoted ahead of cash models.
    entries.sort(key=lambda e: (0 if e[3] else 1, e[0]))

    recommendations = []
    for position, model, provider, funding_note in entries[:top_n]:
        rank = len(recommendations) + 1
        origin = (
            f"AgentStable's benchmark winner for '{apa_role}'"
            if position == 0
            else f"AgentStable's ranked fallback for '{apa_role}'"
        )
        promoted = funding_note and rank == 1 and position > 0
        prefix = (
            "Promoted to #1 — bills against your credit pool, not cash. "
            if promoted
            else ""
        )
        price_in, price_out = _price_of(model, data.prices)
        recommendations.append(Recommendation(
            model=model,
            provider=provider,
            rank=rank,
            apa_role=apa_role,
            justification=f"{prefix}{origin}. {role_why}".strip(),
            price_in=price_in,
            price_out=price_out,
            benchmark=benchmark,
            funding_note=funding_note,
        ))
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
