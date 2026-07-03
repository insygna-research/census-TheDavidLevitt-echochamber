#!/usr/bin/env python3
"""
EchoChamber Evals - use debates to benchmark one model against another.

The meta idea: a debate win under a fixed judge is a capability signal.
A candidate model argues against an incumbent across N topics, taking each
side of every topic once (so side bias cancels), with the same judge model
throughout. The result is a win-rate, a cost figure, and optionally an
APA-shaped "finding" that a procurement agent can ingest.

Usage:
    python -m echochamber.evals \\
      --candidate-provider gemini --candidate gemini-2.5-pro \\
      --incumbent-provider gemini --incumbent gemini-2.5-flash \\
      --judge-provider gemini --judge gemini-2.5-pro \\
      --topics 3 --rounds 1 --max-total-tokens 30000
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from importlib import resources
from pathlib import Path
from typing import Callable, Optional

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from .core.runner import DebateSpec, run_debate
from .core.session import TerminationReason
from .core.costs import get_model_pricing
from .providers import create_provider


def default_topics() -> list[dict]:
    raw = json.loads(
        resources.files("echochamber.data").joinpath("eval_topics.json").read_text()
    )
    return raw["topics"]


@dataclass
class EvalConfig:
    """One candidate-vs-incumbent evaluation."""
    candidate_provider: str
    candidate_model: str
    incumbent_provider: str
    incumbent_model: str
    judge_provider: str
    judge_model: str
    topics: list = field(default_factory=list)  # [{topic, position}]; default set if empty
    num_topics: int = 3          # how many default topics to use if topics is empty
    rounds: int = 1
    enable_search: bool = False  # off by default: cheaper and a purer model comparison
    max_total_tokens_per_debate: Optional[int] = 50_000
    transcript_dir: str = "./transcripts/evals"
    verbose: bool = False


@dataclass
class Matchup:
    """One debate in the eval grid."""
    topic: str
    candidate_side: str  # "prosecution" | "defense"
    winner: Optional[str]
    candidate_result: str  # "win" | "loss" | "draw"
    termination: str
    cost_usd: float
    tokens: int
    transcript: str


def _spec_for(config: EvalConfig, topic: dict, candidate_side: str) -> DebateSpec:
    if candidate_side == "prosecution":
        pros_provider, pros_model = config.candidate_provider, config.candidate_model
        def_provider, def_model = config.incumbent_provider, config.incumbent_model
    else:
        pros_provider, pros_model = config.incumbent_provider, config.incumbent_model
        def_provider, def_model = config.candidate_provider, config.candidate_model

    return DebateSpec(
        topic=topic["topic"],
        position=topic["position"],
        prosecution_provider=pros_provider,
        prosecution_model=pros_model,
        defense_provider=def_provider,
        defense_model=def_model,
        moderator_provider=config.judge_provider,
        moderator_model=config.judge_model,
        max_rounds=config.rounds,
        enable_search=config.enable_search,
        max_total_tokens=config.max_total_tokens_per_debate,
        verbose=config.verbose,
        transcript_dir=config.transcript_dir,
    )


def _score(winner: Optional[str], candidate_side: str) -> str:
    if winner == candidate_side:
        return "win"
    if winner in ("prosecution", "defense"):
        return "loss"
    return "draw"  # draw, undecided, or halted


def run_eval(
    config: EvalConfig,
    provider_factory: Callable = create_provider,
    log: Callable[[str], None] = print,
    on_progress: Optional[Callable[[int, int, "Matchup"], None]] = None,
) -> dict:
    """
    Run the full eval grid and return a report dict.

    Every topic is debated twice (candidate on each side) under the fixed
    judge. Score = wins + 0.5 * draws over all matchups.
    """
    topics = config.topics or default_topics()[: config.num_topics]
    total = len(topics) * 2
    matchups: list[Matchup] = []
    done = 0

    log(f"EchoChamber eval: {config.candidate_model} vs {config.incumbent_model} "
        f"(judge: {config.judge_model}), {len(topics)} topics x 2 sides")

    for topic in topics:
        for side in ("prosecution", "defense"):
            spec = _spec_for(config, topic, candidate_side=side)
            try:
                outcome = run_debate(spec, provider_factory=provider_factory)
                in_tok, out_tok = outcome.usage.total_tokens()
                matchup = Matchup(
                    topic=topic["topic"],
                    candidate_side=side,
                    winner=outcome.result.winner,
                    candidate_result=_score(outcome.result.winner, side),
                    termination=outcome.result.termination_reason.value,
                    cost_usd=outcome.usage.total_cost(),
                    tokens=in_tok + out_tok,
                    transcript=str(outcome.transcript_path or ""),
                )
            except Exception as e:
                matchup = Matchup(
                    topic=topic["topic"],
                    candidate_side=side,
                    winner=None,
                    candidate_result="draw",
                    termination=f"error: {e}",
                    cost_usd=0.0,
                    tokens=0,
                    transcript="",
                )
            matchups.append(matchup)
            done += 1
            log(f"  [{done}/{total}] {matchup.topic} (candidate as {side}): "
                f"{matchup.candidate_result} ({matchup.termination}, "
                f"{matchup.tokens:,} tok, ${matchup.cost_usd:.4f})")
            if on_progress:
                on_progress(done, total, matchup)

    wins = sum(1 for m in matchups if m.candidate_result == "win")
    losses = sum(1 for m in matchups if m.candidate_result == "loss")
    draws = sum(1 for m in matchups if m.candidate_result == "draw")
    score = (wins + 0.5 * draws) / len(matchups) if matchups else 0.0

    return {
        "kind": "echochamber-eval",
        "at": datetime.now().astimezone().isoformat(),
        "candidate": f"{config.candidate_provider}/{config.candidate_model}",
        "incumbent": f"{config.incumbent_provider}/{config.incumbent_model}",
        "judge": f"{config.judge_provider}/{config.judge_model}",
        "rounds_per_debate": config.rounds,
        "matchups": [vars(m) for m in matchups],
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "score": round(score, 3),
        "total_cost_usd": round(sum(m.cost_usd for m in matchups), 4),
        "total_tokens": sum(m.tokens for m in matchups),
    }


def to_apa_finding(report: dict) -> dict:
    """Shape an eval report as an APA finding.

    Matches APA's considerFinding() input:
    {kind, lab, model, headline, url, why, priceIn, priceOut}.
    """
    provider, _, model = report["candidate"].partition("/")
    pricing = get_model_pricing(model)
    n = len(report["matchups"])
    return {
        "kind": "echochamber-eval",
        "lab": provider,
        "model": model,
        "headline": (
            f"Debate eval: {report['wins']}W/{report['losses']}L/{report['draws']}D "
            f"vs {report['incumbent']} (score {report['score']})"
        ),
        "url": "",
        "why": (
            f"Won {report['wins']} of {n} adversarial debates against the incumbent "
            f"under judge {report['judge']} (side-balanced, {report['rounds_per_debate']} "
            f"round(s) each). Eval cost ${report['total_cost_usd']}."
        ),
        "priceIn": pricing.input_per_million,
        "priceOut": pricing.output_per_million,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark a candidate model against an incumbent via debates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--candidate-provider", required=True)
    parser.add_argument("--candidate", required=True, metavar="MODEL")
    parser.add_argument("--incumbent-provider", required=True)
    parser.add_argument("--incumbent", required=True, metavar="MODEL")
    parser.add_argument("--judge-provider", required=True)
    parser.add_argument("--judge", required=True, metavar="MODEL")
    parser.add_argument("--topics", type=int, default=3,
                        help="Number of default topics to debate (each is run twice; default: 3)")
    parser.add_argument("--topics-file", type=str,
                        help="JSON file with custom topics: {\"topics\": [{\"topic\", \"position\"}]}")
    parser.add_argument("--rounds", type=int, default=1,
                        help="Rounds per debate (default: 1)")
    parser.add_argument("--enable-search", action="store_true",
                        help="Allow web search during eval debates (default: off)")
    parser.add_argument("--max-total-tokens", type=int, default=50_000,
                        help="Token budget per debate (default: 50000)")
    parser.add_argument("--output", type=str, default="eval_report.json",
                        help="Where to write the JSON report (default: eval_report.json)")
    parser.add_argument("--apa-findings", type=str, metavar="PATH",
                        help="Append an APA-shaped finding (JSONL) for procurement ingestion")
    parser.add_argument("--transcript-dir", type=str, default="./transcripts/evals")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    topics = []
    if args.topics_file:
        topics = json.loads(Path(args.topics_file).read_text())["topics"]

    config = EvalConfig(
        candidate_provider=args.candidate_provider,
        candidate_model=args.candidate,
        incumbent_provider=args.incumbent_provider,
        incumbent_model=args.incumbent,
        judge_provider=args.judge_provider,
        judge_model=args.judge,
        topics=topics,
        num_topics=args.topics,
        rounds=args.rounds,
        enable_search=args.enable_search,
        max_total_tokens_per_debate=args.max_total_tokens,
        transcript_dir=args.transcript_dir,
        verbose=args.verbose,
    )

    # Rough cost preview before spending anything
    n_debates = (len(topics) or args.topics) * 2
    print(f"Planned: {n_debates} debates, {args.rounds} round(s) each, "
          f"budget {args.max_total_tokens:,} tokens/debate")

    report = run_eval(config)

    print(f"\nRESULT: {report['candidate']} scored {report['score']} vs {report['incumbent']}")
    print(f"  {report['wins']}W / {report['losses']}L / {report['draws']}D")
    print(f"  Total: {report['total_tokens']:,} tokens, ${report['total_cost_usd']}")

    Path(args.output).write_text(json.dumps(report, indent=2))
    print(f"  Report: {args.output}")

    if args.apa_findings:
        finding = to_apa_finding(report)
        path = Path(args.apa_findings)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(finding) + "\n")
        print(f"  APA finding appended to: {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
