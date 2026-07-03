"""Usage metering — records every provider call with tokens, latency, and cost.

Events use the agent-stable normalized shape
({type, at, host, module, model, input, output, costUsd, latencyMs, note})
so a JSONL usage log can be ingested directly by agent-stable's sinks.
fundingClass is deliberately left to the ingesting host, which owns the
source-of-funds policy.
"""

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from .costs import get_model_pricing


@dataclass
class UsageEvent:
    """One metered provider call."""
    at: str
    module: str  # e.g. "debate.prosecution"
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: Optional[float] = None

    def to_stable_event(self, host: str = "echochamber") -> dict:
        """Serialize in agent-stable's normalized usage-event shape."""
        return {
            "type": "usage",
            "at": self.at,
            "host": host,
            "module": self.module,
            "model": self.model,
            "input": self.input_tokens,
            "output": self.output_tokens,
            "costUsd": round(self.cost_usd, 6),
            "latencyMs": round(self.latency_ms) if self.latency_ms is not None else None,
            "note": self.provider,
        }


@dataclass
class UsageMeter:
    """Accumulates usage events across a debate run. Thread-safe."""
    host: str = "echochamber"
    events: list = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(
        self,
        module: str,
        provider: str,
        model: str,
        usage: Optional[dict],
        latency_ms: Optional[float] = None,
    ) -> None:
        """Record one provider call. Safe to call with usage=None."""
        input_tokens = int((usage or {}).get("input_tokens") or 0)
        output_tokens = int((usage or {}).get("output_tokens") or 0)
        pricing = get_model_pricing(model)
        cost = (
            input_tokens / 1_000_000 * pricing.input_per_million
            + output_tokens / 1_000_000 * pricing.output_per_million
        )
        event = UsageEvent(
            at=datetime.now().astimezone().isoformat(),
            module=module,
            model=model,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
        )
        with self._lock:
            self.events.append(event)

    def total_cost(self) -> float:
        return sum(e.cost_usd for e in self.events)

    def summary(self) -> str:
        """Human-readable per-module cost/token summary."""
        if not self.events:
            return "No usage recorded."
        by_module: dict[str, dict] = {}
        for e in self.events:
            agg = by_module.setdefault(
                e.module, {"model": e.model, "calls": 0, "input": 0, "output": 0, "cost": 0.0}
            )
            agg["calls"] += 1
            agg["input"] += e.input_tokens
            agg["output"] += e.output_tokens
            agg["cost"] += e.cost_usd

        lines = ["Usage summary:"]
        for module, agg in sorted(by_module.items()):
            lines.append(
                f"  {module}: {agg['calls']} calls, "
                f"{agg['input']:,} in / {agg['output']:,} out tokens, "
                f"${agg['cost']:.4f} ({agg['model']})"
            )
        lines.append(f"  TOTAL: ${self.total_cost():.4f}")
        return "\n".join(lines)

    def write_jsonl(self, path: str | Path) -> Path:
        """Append events as JSONL in agent-stable's event shape."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            for e in self.events:
                f.write(json.dumps(e.to_stable_event(self.host)) + "\n")
        return path
