"""Value types passed between providers, the client, and callers."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProviderResponse:
    """What a single provider can honestly report about one call.

    Deliberately narrower than :class:`LLMResult`. A provider does not know how
    many times it was retried, which providers were tried before it, or whether
    it was the primary -- so it reports none of those. Returning ``LLMResult``
    here would force every adapter to invent ``attempts=1`` and
    ``providers_tried=[]``, values the client would immediately overwrite.

    The client times the call, counts attempts, tracks the chain, and assembles
    the :class:`LLMResult` from this plus what it knows itself.
    """

    value: Any
    """Validated response model instance, or ``str`` when no schema was requested."""

    input_tokens: int | None = None
    """``None`` when the provider does not report usage."""

    output_tokens: int | None = None
    """``None`` when the provider does not report usage."""

    cost_usd: float | None = None
    """Computed by the adapter, which knows its own pricing. ``0.0`` for a local
    model; ``None`` when unknown. The client does not hold a price table."""


@dataclass
class LLMResult:
    """A successful completion and everything worth recording about it.

    A plain dataclass rather than a pydantic model: ``value`` may itself be a
    validated pydantic instance, but this container carries no untrusted input
    and so needs no validation of its own.

    ``attempts`` and ``providers_tried`` are not decoration. They are the only
    way a test can assert that retry and fallback actually behaved, which is
    why they are returned rather than merely logged.
    """

    value: Any
    """Validated response model instance, or ``str`` when no schema was requested."""

    provider: str
    """Name of the adapter that served the call."""

    model: str
    """Model identifier as the serving provider knows it."""

    latency_ms: float
    """Wall clock for the successful attempt only, excluding earlier failed ones."""

    attempts: int
    """Attempts against the serving provider. ``1`` means it succeeded first try."""

    providers_tried: list[str] = field(default_factory=list)
    """Providers attempted and abandoned before this one. Empty when the primary served it."""

    input_tokens: int | None = None
    """``None`` when the provider does not report usage."""

    output_tokens: int | None = None
    """``None`` when the provider does not report usage."""

    cost_usd: float | None = None
    """``None`` means unknown, not free. A local model should record ``0.0``."""