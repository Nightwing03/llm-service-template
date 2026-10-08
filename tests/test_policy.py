"""Backoff and rate limiting, driven by a fake clock."""

import pytest

from llm_service.policy import RateLimiter, RetryPolicy


class FakeClock:
    """A clock that only moves when something sleeps.

    Lets the limiter's blocking behaviour be asserted exactly, in microseconds
    of real time. A test that genuinely waits 60 seconds is a test that gets
    skipped, and rung 5's CI gate depends on this suite staying fast.
    """

    def __init__(self) -> None:
        self.t = 0.0
        self.waits: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.waits.append(seconds)
        self.t += seconds


def test_backoff_doubles_from_the_base():
    policy = RetryPolicy(base_delay=0.5, max_delay=8.0, jitter=False)
    assert [policy.delay_before(n) for n in (1, 2, 3, 4)] == [0.5, 1.0, 2.0, 4.0]


def test_backoff_is_capped():
    policy = RetryPolicy(base_delay=0.5, max_delay=2.0, jitter=False)
    assert [policy.delay_before(n) for n in (1, 2, 3, 9)] == [0.5, 1.0, 2.0, 2.0]


def test_jitter_stays_within_half_the_nominal_delay():
    """Jitter must desynchronise callers without abandoning the backoff curve."""
    policy = RetryPolicy(base_delay=1.0, jitter=True)
    samples = [policy.delay_before(3) for _ in range(200)]

    assert all(2.0 <= s <= 4.0 for s in samples)
    assert len(set(samples)) > 1  # actually random, not a constant


def test_limiter_without_a_limit_never_blocks():
    clock = FakeClock()
    limiter = RateLimiter(None, now=clock.now, sleep=clock.sleep)

    for _ in range(100):
        assert limiter.acquire() == 0.0
    assert clock.waits == []


def test_limiter_allows_the_full_quota_without_waiting():
    clock = FakeClock()
    limiter = RateLimiter(3, now=clock.now, sleep=clock.sleep)

    for _ in range(3):
        assert limiter.acquire() == 0.0
    assert clock.waits == []


def test_limiter_blocks_once_the_window_is_full():
    clock = FakeClock()
    limiter = RateLimiter(2, now=clock.now, sleep=clock.sleep)

    limiter.acquire()
    limiter.acquire()
    waited = limiter.acquire()

    # Third call waits until the first leaves the 60s window.
    assert waited == pytest.approx(60.0, abs=0.01)
    assert clock.t == pytest.approx(60.0, abs=0.01)


def test_limiter_recovers_as_calls_age_out():
    clock = FakeClock()
    limiter = RateLimiter(2, now=clock.now, sleep=clock.sleep)

    limiter.acquire()
    limiter.acquire()
    clock.t += 61.0  # both fall out of the window

    assert limiter.acquire() == 0.0
    assert clock.waits == []


def test_limiters_are_independent():
    """One throttled backend must not stall a healthy one."""
    clock = FakeClock()
    busy = RateLimiter(1, now=clock.now, sleep=clock.sleep)
    idle = RateLimiter(1, now=clock.now, sleep=clock.sleep)

    busy.acquire()
    assert idle.acquire() == 0.0