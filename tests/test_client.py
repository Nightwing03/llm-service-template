"""Client behaviour, proven against fakes. No network, no sleeping."""

import pytest
from pydantic import BaseModel

from llm_service.client import LLMClient
from llm_service.errors import (
    AllProvidersFailedError,
    InvalidRequestError,
    ProviderUnavailableError,
    RateLimitedError,
    TransientError,
)
from llm_service.policy import RetryPolicy
from llm_service.providers.base import Provider
from tests.fakes import FakeProvider

# Jitter off so delays are exact; a test that asserts on a random number is
# a test that fails one morning for no reason.
POLICY = RetryPolicy(max_attempts=3, base_delay=0.5, max_delay=8.0, jitter=False)


@pytest.fixture
def slept():
    """Collects every sleep the client asks for, without performing any."""
    return []


def make_client(providers, slept, **kwargs):
    return LLMClient(providers, policy=POLICY, sleep=slept.append, **kwargs)


def test_fake_satisfies_the_protocol():
    assert isinstance(FakeProvider("a", "m", ["x"]), Provider)


def test_first_provider_success_short_circuits(slept):
    second = FakeProvider("b", "m", ["never"])
    result = make_client([FakeProvider("a", "m", ["ok"]), second], slept).complete(user="hi")

    assert result.value == "ok"
    assert result.provider == "a"
    assert result.attempts == 1
    assert result.providers_tried == []
    assert second.call_count == 0
    assert slept == []


def test_transient_errors_retry_the_same_provider(slept):
    provider = FakeProvider("a", "m", [TransientError(""), TransientError(""), "ok"])
    result = make_client([provider], slept).complete(user="hi")

    assert result.value == "ok"
    assert result.attempts == 3
    assert provider.call_count == 3
    assert result.providers_tried == []
    # Exponential: 0.5 then 1.0. No sleep after the successful attempt.
    assert slept == [0.5, 1.0]


def test_retries_stop_at_the_attempt_limit(slept):
    provider = FakeProvider("a", "m", [TransientError("always")])
    with pytest.raises(AllProvidersFailedError):
        make_client([provider], slept).complete(user="hi")

    assert provider.call_count == POLICY.max_attempts
    assert len(slept) == POLICY.max_attempts - 1


def test_fallback_moves_to_the_next_provider(slept):
    result = make_client(
        [FakeProvider("a", "m", [ProviderUnavailableError("no key")]),
         FakeProvider("b", "m", ["from-b"])],
        slept,
    ).complete(user="hi")

    assert result.value == "from-b"
    assert result.provider == "b"
    assert result.providers_tried == ["a"]


def test_provider_unavailable_is_not_retried(slept):
    """The classic bug: burning the full backoff on an error that cannot clear."""
    provider = FakeProvider("a", "m", [ProviderUnavailableError("bad credentials")])
    with pytest.raises(AllProvidersFailedError):
        make_client([provider], slept).complete(user="hi")

    assert provider.call_count == 1
    assert slept == []


def test_invalid_request_aborts_the_whole_chain(slept):
    second = FakeProvider("b", "m", ["never reached"])
    with pytest.raises(InvalidRequestError):
        make_client(
            [FakeProvider("a", "m", [InvalidRequestError("bad schema")]), second], slept
        ).complete(user="hi")

    assert second.call_count == 0
    assert slept == []


def test_rate_limit_prefers_server_retry_after(slept):
    """The server knows when its window reopens; our backoff is a guess."""
    provider = FakeProvider("a", "m", [RateLimitedError("429", retry_after=1.25), "ok"])
    result = make_client([provider], slept).complete(user="hi")

    assert result.attempts == 2
    assert slept == [1.25]


def test_rate_limit_falls_back_to_backoff_without_a_header(slept):
    provider = FakeProvider("a", "m", [RateLimitedError("429"), "ok"])
    make_client([provider], slept).complete(user="hi")

    assert slept == [0.5]


def test_exhausted_chain_reports_every_provider(slept):
    with pytest.raises(AllProvidersFailedError) as caught:
        make_client(
            [FakeProvider("a", "m", [TransientError("down")]),
             FakeProvider("b", "m", [RateLimitedError("throttled")])],
            slept,
        ).complete(user="hi")

    errors = caught.value.errors
    assert set(errors) == {"a", "b"}
    assert isinstance(errors["a"], TransientError)
    assert isinstance(errors["b"], RateLimitedError)
    # The message must name the causes; an opaque one costs an evening.
    assert "TransientError" in str(caught.value)
    assert "RateLimitedError" in str(caught.value)


@pytest.mark.parametrize("prompt", ["", "   ", "\n"])
def test_empty_prompt_rejected_before_any_provider_is_called(prompt, slept):
    provider = FakeProvider("a", "m", ["x"])
    with pytest.raises(InvalidRequestError):
        make_client([provider], slept).complete(user=prompt)

    assert provider.call_count == 0


def test_arguments_reach_the_provider_unchanged(slept):
    class Schema(BaseModel):
        x: int

    provider = FakeProvider("a", "m", ["ok"])
    make_client([provider], slept).complete(
        user="u", system="s", response_model=Schema, temperature=0.7
    )

    assert provider.last_call == {
        "system": "s",
        "user": "u",
        "response_model": Schema,
        "temperature": 0.7,
    }


def test_result_carries_provider_metadata(slept):
    result = make_client([FakeProvider("a", "model-x", ["ok"])], slept).complete(user="hi")

    assert result.model == "model-x"
    assert (result.input_tokens, result.output_tokens) == (11, 7)
    assert result.latency_ms >= 0


def test_client_requires_at_least_one_provider():
    with pytest.raises(ValueError):
        LLMClient([])