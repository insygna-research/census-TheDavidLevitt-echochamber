"""Centralized retry with exponential backoff for provider API calls.

Kept SDK-agnostic: transient failures are recognized by sniffing the
exception's type name and message rather than importing every provider's
exception hierarchy.
"""

import time
from typing import Callable, Optional, TypeVar

T = TypeVar("T")

# Substrings that mark an error as transient (rate limits, overload, network).
RETRYABLE_MARKERS = (
    "rate limit",
    "rate_limit",
    "ratelimit",
    "overloaded",
    "429",
    "500",
    "502",
    "503",
    "529",
    "timeout",
    "timed out",
    "connection",
    "temporarily",
    "service unavailable",
    "internal server error",
    "apiconnection",
)


def is_retryable(exc: Exception) -> bool:
    """Heuristically decide whether an API error is worth retrying."""
    haystack = f"{type(exc).__name__} {exc}".lower()
    return any(marker in haystack for marker in RETRYABLE_MARKERS)


def call_with_retries(
    fn: Callable[[], T],
    attempts: int = 3,
    base_delay: float = 1.0,
    on_retry: Optional[Callable[[int, float, Exception], None]] = None,
) -> T:
    """
    Call fn, retrying transient failures with exponential backoff.

    Args:
        fn: Zero-argument callable performing the API request
        attempts: Total attempts including the first
        base_delay: Delay before the first retry; doubles each retry
        on_retry: Optional callback(attempt_number, delay_seconds, exception)

    Returns:
        fn's return value

    Raises:
        The last exception if all attempts fail, or immediately for
        non-retryable errors (auth, bad request, ...).
    """
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as e:
            if attempt == attempts - 1 or not is_retryable(e):
                raise
            delay = base_delay * (2 ** attempt)
            if on_retry:
                on_retry(attempt + 1, delay, e)
            time.sleep(delay)
    raise RuntimeError("unreachable")  # for type-checkers
