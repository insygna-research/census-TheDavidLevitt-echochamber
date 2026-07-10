"""Evidence-salience ablation: scenarios, exclusion, resume, salience math."""

import json

from helpers import FakeProvider, tool_call_response

from echochamber.ablation import (
    AblationConfig,
    Scenario,
    auto_scenarios,
    run_ablation,
    signed_margin,
)
from echochamber.core.runner import DebateSpec, run_debate
from echochamber.ui import parse_grid


def make_case(tmp_path):
    case = tmp_path / "case"
    (case / "shared").mkdir(parents=True)
    (case / "prosecution").mkdir()
    (case / "shared" / "alpha.txt").write_text("ALPHA-FACT")
    (case / "shared" / "beta.txt").write_text("BETA-FACT")
    (case / "prosecution" / "gamma.txt").write_text("GAMMA-FACT")
    return case


def test_auto_scenarios_baseline_plus_loo_with_protection():
    names = [("alpha.txt", "shared"), ("beta.txt", "shared"), ("gamma.txt", "prosecution")]
    scenarios = auto_scenarios(names, protect=["beta.txt"], runs=7)
    assert [s.name for s in scenarios] == ["baseline", "minus alpha.txt", "minus gamma.txt"]
    assert all(s.runs == 7 for s in scenarios)
    assert scenarios[1].exclude == ["alpha.txt"]


def test_signed_margin_convention():
    assert signed_margin("prosecution", 80) == 80.0
    assert signed_margin("defense", 60) == -60.0
    assert signed_margin("draw", 40) == 0.0
    assert signed_margin("prosecution", None) is None


def test_excluded_evidence_never_reaches_agents(tmp_path):
    case = make_case(tmp_path)
    fakes = [
        FakeProvider(["p arg"]), FakeProvider(["d arg"]),
        FakeProvider(["opening",
                      "CONTINUE: NO\nWINNER: PROSECUTION\nREASONING: r",
                      "ruling"]),
    ]
    queue = list(fakes)
    spec = DebateSpec(
        topic="T", position="P", case_folder=str(case),
        exclude_evidence=["beta.txt"], context_strategy="full",
        max_rounds=1, enable_search=False, verbose=False,
        transcript_dir=str(tmp_path / "t"),
    )
    run_debate(spec, provider_factory=lambda p, m=None, **kw: queue.pop(0))

    all_text = " ".join(
        m.content for call in fakes[0].calls for m in call["messages"]
    )
    assert "ALPHA-FACT" in all_text
    assert "BETA-FACT" not in all_text  # ablated document is really gone


def test_campaign_runs_resumes_and_reports(tmp_path):
    case = make_case(tmp_path)

    def factory(provider, model=None, **kw):
        if provider == "judge":
            return FakeProvider([
                "opening",
                tool_call_response("submit_evaluation", {
                    "continue_debate": False, "winner": "prosecution",
                    "reasoning": "done",
                }),
                tool_call_response("submit_verdict", {
                    "winner": "prosecution", "strength": 70, "reasoning": "r",
                }),
            ], supports_tools=True)
        return FakeProvider(["argument"])

    config = AblationConfig(
        case_folder=str(case), topic="T", position="P",
        provider="adv", judge_provider="judge",
        rounds=1, runs=2, protect=["beta.txt", "gamma.txt"],
        output_dir=str(tmp_path / "out"), verbose=False,
    )
    report = run_ablation(config, provider_factory=factory, log=lambda s: None)

    # baseline + minus alpha (beta/gamma protected), 2 runs each
    assert set(report["scenarios"]) == {"baseline", "minus alpha.txt"}
    assert report["total_debates"] == 4
    stats = report["scenarios"]["minus alpha.txt"]
    assert stats["n"] == 2
    assert stats["prosecution_win_rate"] == 1.0
    assert stats["mean_margin"] == 70.0
    assert stats["salience_margin_delta"] == 0.0  # same outcome everywhere

    # Resume: everything already in the manifest → zero new runs
    progress = []
    report2 = run_ablation(
        config, provider_factory=factory, log=lambda s: None,
        on_progress=lambda n, t, r: progress.append(r),
    )
    assert progress == []
    assert report2["total_debates"] == 4

    # Changing the experiment against the same output dir is refused
    config2 = AblationConfig(**{**config.__dict__, "topic": "DIFFERENT"})
    try:
        run_ablation(config2, provider_factory=factory, log=lambda s: None)
        raise AssertionError("expected ValueError for changed fingerprint")
    except ValueError as e:
        assert "different experiment" in str(e)

    # Manifest is valid JSONL
    lines = (tmp_path / "out" / "runs.jsonl").read_text().splitlines()
    assert len(lines) == 4
    assert all(json.loads(l)["strength"] == 70 for l in lines)


def test_parse_grid_matches_spec():
    rows = [
        ["Scenario name", "baseline", "no-alpha", "", "lean"],
        ["# runs", "10", "5", "9", "0"],
        ["alpha.txt", "", "x", "", "x"],
        ["beta.txt", "", "", "x", "x"],
    ]
    scenarios = parse_grid(rows)
    # Empty-name and zero-run columns are skipped
    assert [(s.name, s.runs) for s in scenarios] == [("baseline", 10), ("no-alpha", 5)]
    assert scenarios[0].exclude == []
    assert scenarios[1].exclude == ["alpha.txt"]


def test_unknown_exclusion_is_rejected(tmp_path):
    case = make_case(tmp_path)
    config = AblationConfig(
        case_folder=str(case), topic="T", position="P",
        scenarios=[Scenario(name="bad", runs=1, exclude=["nope.txt"])],
        output_dir=str(tmp_path / "out2"),
    )
    try:
        run_ablation(config, provider_factory=lambda *a, **k: None, log=lambda s: None)
        raise AssertionError("expected ValueError for unknown evidence")
    except ValueError as e:
        assert "unknown evidence" in str(e)
