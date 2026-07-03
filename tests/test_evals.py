"""The debate-based model eval harness."""

from helpers import FakeProvider

from echochamber.evals import EvalConfig, default_topics, run_eval, to_apa_finding


def scripted_factory():
    """Factory producing fakes for every eval debate.

    Judge always rules PROSECUTION WINS, so with side-swapping the candidate
    should land exactly 1 win + 1 loss per topic.
    """
    def factory(provider, model=None, **kwargs):
        if provider == "judge":
            return FakeProvider([
                "opening",
                "CONTINUE: YES\nWINNER: NONE\nREASONING: continue",
                "FINAL VERDICT: PROSECUTION WINS",
            ], model=model or "judge-model")
        return FakeProvider(["an argument"], model=model or "advocate-model")
    return factory


def make_config(num_topics=1):
    return EvalConfig(
        candidate_provider="cand", candidate_model="candidate-x",
        incumbent_provider="inc", incumbent_model="incumbent-y",
        judge_provider="judge", judge_model="judge-z",
        num_topics=num_topics,
        rounds=1,
        verbose=False,
        transcript_dir="",  # not saved
    )


def run_quiet(config):
    return run_eval(config, provider_factory=scripted_factory(), log=lambda s: None)


def test_side_balanced_scoring(tmp_path):
    config = make_config(num_topics=2)
    config.transcript_dir = str(tmp_path)
    report = run_quiet(config)

    # Judge always picks prosecution: candidate wins as prosecution,
    # loses as defense → 2W/2L over 2 topics, score 0.5.
    assert report["wins"] == 2
    assert report["losses"] == 2
    assert report["draws"] == 0
    assert report["score"] == 0.5
    assert len(report["matchups"]) == 4
    sides = [m["candidate_side"] for m in report["matchups"]]
    assert sides == ["prosecution", "defense", "prosecution", "defense"]

    # Verboseness: FakeProvider emits 50 output tokens/call for both sides
    v = report["verbosity"]
    assert v["candidate_output_tokens"] == v["incumbent_output_tokens"] == 200
    assert v["ratio"] == 1.0
    assert v["candidate_tokens_per_win"] == 100  # 200 tokens over 2 wins
    assert v["incumbent_tokens_per_win"] == 100


def test_apa_finding_shape(tmp_path):
    config = make_config()
    config.transcript_dir = str(tmp_path)
    finding = to_apa_finding(run_quiet(config))

    assert set(finding) == {"kind", "lab", "model", "headline", "url", "why", "priceIn", "priceOut"}
    assert finding["kind"] == "echochamber-eval"
    assert finding["lab"] == "cand"
    assert finding["model"] == "candidate-x"
    assert "1W/1L" in finding["headline"]
    assert "Verbosity" in finding["why"]


def test_default_topics_are_well_formed():
    topics = default_topics()
    assert len(topics) >= 5
    assert all({"topic", "position"} <= set(t) for t in topics)
