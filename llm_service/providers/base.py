"""The contract every provider adapter implements."""

from typing import Protocol, runtime_checkable
 
from pydantic import BaseModel
 
from llm_service.types import ProviderResponse

@runtime_checkable
class Provider(Protocol):
    """A single LLM backend.
 
    An adapter's entire job is to make one call and translate the outcome --
    including failure -- into this package's vocabulary. It performs no
    retries, never sleeps, and knows nothing about any other provider. Those
    are the client's concerns.
 
    **The contract:** ``complete`` raises only the types defined in
    :mod:`llm_service.errors`. An SDK exception, an ``httpx`` exception or a
    bare ``ValueError`` escaping an adapter is a bug in that adapter, because
    the client dispatches on exception type alone and has no way to interpret
    anything else.
 
    Declared as a ``Protocol`` rather than an ABC so that test doubles satisfy
    it structurally, without importing or inheriting anything.
 
    ``runtime_checkable`` verifies only that the named members exist -- it does
    not check signatures. Useful in a test assertion; not a guarantee.
    """

    name: str
    """Identifier recorded in ``LLMResult.provider`` and ``LLMResult.providers_tried``."""

    model: str
    """Model identifier as this backend knows it."""

    def complete(
        self,
        *,
        system: str | None,
        user: str,
        response_model: type[BaseModel] | None = None,
        temperature: float | None = None,
    ) -> ProviderResponse:
        """Make exactly one call. No retries, no fallback, no sleeping.
 
        Args:
            system: System prompt, or ``None`` when the task needs no framing.
            user: The user-turn prompt. Must be non-empty.
            response_model: When given, the return value must be a validated
                instance of this model. When ``None``, return raw text.
            temperature: ``None`` means use whatever this adapter was
                constructed with.
 
        Returns:
            A :class:`ProviderResponse` carrying the value and, where the
            backend reports them, token counts.
 
        Raises:
            TransientError: Timeout, connection reset, 5xx -- worth retrying.
            RateLimitedError: HTTP 429, optionally carrying a retry-after.
            ProviderUnavailableError: This backend cannot serve the request,
                but another might -- bad credentials, unknown model, outage,
                or a prompt exceeding *this* model's context window.
            InvalidRequestError: The request is malformed and no backend would
                accept it.
 
        Keyword-only by design: four parameters, two of them optional, is
        precisely where positional calls start being misread.
        """
        ...
