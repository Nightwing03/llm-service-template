"""The client: chain, retry, rate limiting, accounting.

Everything that decides lives here. Providers only translate.
"""

import time
from typing import Callable, Sequence

from pydantic import BaseModel

from llm_service.errors import (
    AllProvidersFailedError,
    InvalidRequestError,
    ProviderUnavailableError,
    RateLimitedError,
    TransientError,
)
from llm_service.policy import RateLimiter, RetryPolicy
from llm_service.providers.base import Provider
from llm_service.types import LLMResult


class LLMClient:
    """Calls an ordered chain of providers until one succeeds.

    The dispatch is deliberately flat: one ``except`` clause per exception
    type, each doing exactly what that type's docstring promises. If a new
    branch is ever needed here, the taxonomy is wrong -- fix it there rather
    than adding a special case.

    ``sleep`` is injected so tests exercise backoff without waiting for it.
    """

    def __init__(
        self,
        providers: Sequence[Provider],
        policy: RetryPolicy | None = None,
        requests_per_minute: dict[str, int] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not providers:
            raise ValueError("at least one provider is required")
        self.providers = list(providers)
        self.policy = policy or RetryPolicy()
        self._sleep = sleep
        limits = requests_per_minute or {}
        self._limiters = {
            p.name: RateLimiter(limits.get(p.name), sleep=sleep) for p in self.providers
        }

    def complete(
        self,
        *,
        user: str,
        system: str | None = None,
        response_model: type[BaseModel] | None = None,
        temperature: float | None = None,
    ) -> LLMResult:
        """Return the first successful completion from the chain.

        Raises:
            InvalidRequestError: The request is malformed. Raised before any
                provider is contacted, and never swallowed if an adapter
                raises it mid-chain.
            AllProvidersFailedError: Every provider was tried and failed.
        """
        if not user or not user.strip():
            raise InvalidRequestError("user prompt must be non-empty")

        providers_tried: list[str] = []
        failures: dict[str, Exception] = {}

        for provider in self.providers:
            attempts = 0
            while attempts < self.policy.max_attempts:
                attempts += 1
                try:
                    self._limiters[provider.name].acquire()

                    started = time.perf_counter()
                    response = provider.complete(
                        system=system,
                        user=user,
                        response_model=response_model,
                        temperature=temperature,
                    )
                    latency_ms = (time.perf_counter() - started) * 1000.0

                    return LLMResult(
                        value=response.value,
                        provider=provider.name,
                        model=provider.model,
                        latency_ms=latency_ms,
                        attempts=attempts,
                        providers_tried=providers_tried,
                        input_tokens=response.input_tokens,
                        output_tokens=response.output_tokens,
                        cost_usd=response.cost_usd,
                    )

                except InvalidRequestError:
                    # No provider would accept this. Walking the chain only
                    # makes the identical failure slower.
                    raise

                except ProviderUnavailableError as exc:
                    # Retrying cannot change the outcome. Next provider.
                    failures[provider.name] = exc
                    break

                except RateLimitedError as exc:
                    failures[provider.name] = exc
                    if attempts >= self.policy.max_attempts:
                        break
                    # Honour the server's instruction when it gave one; it
                    # knows when the window reopens and we are guessing.
                    wait = exc.retry_after
                    if wait is None:
                        wait = self.policy.delay_before(attempts)
                    self._sleep(wait)

                except TransientError as exc:
                    failures[provider.name] = exc
                    if attempts >= self.policy.max_attempts:
                        break
                    self._sleep(self.policy.delay_before(attempts))

            providers_tried.append(provider.name)

        raise AllProvidersFailedError(
            f"all {len(self.providers)} provider(s) failed", errors=failures
        )