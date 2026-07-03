"""Usage metering and agent-stable event export."""

import json

from echochamber.core.usage import UsageMeter


def test_meter_computes_cost_from_pricing_table():
    meter = UsageMeter()
    meter.record(
        module="debate.prosecution",
        provider="fake/gpt-4o",
        model="gpt-4o",
        usage={"input_tokens": 1_000_000, "output_tokens": 1_000_000},
        latency_ms=123.4,
    )
    assert meter.total_cost() == 12.5  # $2.5 in + $10 out


def test_meter_tolerates_missing_usage():
    meter = UsageMeter()
    meter.record(module="debate.defense", provider="fake", model="gpt-4o", usage=None)
    assert meter.total_cost() == 0.0
    assert meter.events[0].input_tokens == 0


def test_jsonl_events_match_agent_stable_shape(tmp_path):
    meter = UsageMeter()
    meter.record(
        module="debate.moderator",
        provider="fake/gpt-4o",
        model="gpt-4o",
        usage={"input_tokens": 100, "output_tokens": 50},
        latency_ms=99.9,
    )
    path = meter.write_jsonl(tmp_path / "usage.jsonl")

    event = json.loads(path.read_text().splitlines()[0])
    assert event["type"] == "usage"
    assert event["host"] == "echochamber"
    assert event["module"] == "debate.moderator"
    assert event["model"] == "gpt-4o"
    assert event["input"] == 100
    assert event["output"] == 50
    assert event["costUsd"] > 0
    assert event["latencyMs"] == 100
    assert "at" in event


def test_summary_mentions_totals():
    meter = UsageMeter()
    meter.record(
        module="debate.prosecution", provider="p", model="gpt-4o",
        usage={"input_tokens": 10, "output_tokens": 10},
    )
    summary = meter.summary()
    assert "debate.prosecution" in summary
    assert "TOTAL" in summary
