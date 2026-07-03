"""Retry layer: transient-error detection and backoff behavior."""

import pytest

from echochamber.providers.retry import call_with_retries, is_retryable


@pytest.mark.parametrize("message", [
    "Server disconnected without sending a response.",  # Vertex mid-eval, seen live
    "429 Your prepayment credits are depleted.",
    "Rate limit exceeded, retry after 3s",
    "Connection reset by peer",
    "503 Service Unavailable",
    "Request timed out",
])
def test_transient_errors_are_retryable(message):
    assert is_retryable(RuntimeError(message))


@pytest.mark.parametrize("message", [
    "Invalid API key provided",
    "404 model not found",
    "permission denied for project",
])
def test_permanent_errors_are_not_retryable(message):
    assert not is_retryable(RuntimeError(message))


def test_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("Server disconnected without sending a response.")
        return "ok"

    assert call_with_retries(flaky, attempts=3) == "ok"
    assert attempts["n"] == 3


def test_permanent_error_fails_fast(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    attempts = {"n": 0}

    def broken():
        attempts["n"] += 1
        raise RuntimeError("Invalid API key provided")

    with pytest.raises(RuntimeError):
        call_with_retries(broken, attempts=3)
    assert attempts["n"] == 1  # no pointless retries
