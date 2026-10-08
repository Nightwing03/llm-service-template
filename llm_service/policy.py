"""Retry and rate-limiting policy.

Both live here rather than in any adapter. A provider makes one call and
reports what happened; deciding whether to wait, retry or move on is the
client's job, and keeping that split clean is what lets a new adapter be
written without touching any control flow.
"""

import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable


@dataclass
class RetryPolicy:
    """How hard the client tries one provider before abandoning it.

    ``jitter`` is on by default because synchronised retries across a batch
    run are how a rate limit turns into a thundering herd. Tests should switch
    it off to get deterministic delays.
    """

    max_attempts: int = 3
    base_delay: float = 0.5
    max_delay: float = 8.0
    jitter: bool = True

    def delay_before(self, attempt: int) -> float:
        """Seconds to wait before attempt number ``attempt``.

        ``attempt`` is 1-based and counts attempts already made, so the wait
        before the second attempt is ``delay_before(1)``.
        """
        raw = min(self.base_delay * (2 ** (attempt - 1)), self.max_delay)
        if not self.jitter:
            return raw
        # Decorrelated within [50%, 100%] of the nominal delay: still backs
        # off monotonically, but two callers never wake together.
        return raw * (0.5 + random.random() / 2)


class RateLimiter:
    """Sliding-window limiter. One instance per provider, not one globally.

    Free tiers are per-provider, so a shared limiter would throttle a healthy
    backend because a different one is busy.

    ``now`` and ``sleep`` are injected so tests can drive the clock instead of
    waiting on it. A test that genuinely sleeps is a test nobody runs.
    """

    def __init__(
        self,
        requests_per_minute: int | None,
        now: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.requests_per_minute = requests_per_minute
        self._now = now
        self._sleep = sleep
        self._calls: deque[float] = deque()

    def acquire(self) -> float:
        """Block until a request may proceed. Returns seconds spent waiting.

        The returned wait is excluded from reported latency: time spent
        respecting our own limiter is not the provider being slow.
        """
        if self.requests_per_minute is None:
            return 0.0

        waited = 0.0
        while True:
            cutoff = self._now() - 60.0
            while self._calls and self._calls[0] <= cutoff:
                self._calls.popleft()

            if len(self._calls) < self.requests_per_minute:
                self._calls.append(self._now())
                return waited

            # Wait just past the point where the oldest call leaves the window.
            pause = self._calls[0] - cutoff + 0.001
            self._sleep(pause)
            waited += pause