
"""Exception taxonomy for LLM provider calls.
 
Each exception type maps to exactly one client behaviour. If two types would
cause the client to do the same thing, one of them is redundant and should be
removed rather than kept "for clarity".
 
Provider adapters translate their SDK's exceptions into these types. The client
then decides what to do based solely on the type it caught, which is what keeps
the client free of provider-specific error handling.
"""
 
 
class LLMServiceError(Exception):
    """Base for every error raised by this package.
 
    Client behaviour: none directly. Exists so callers can catch the whole
    family in a single clause, and so no error from this package escapes as a
    bare ``Exception``.
    """
 
 
class TransientError(LLMServiceError):
    """A failure expected to clear on its own: timeout, connection reset, 5xx.
 
    Client behaviour: retry the same provider with exponential backoff and
    jitter, up to the configured attempt limit. Move to the next provider only
    once that limit is exhausted.
    """
 
 
class RateLimitedError(LLMServiceError):
    """The provider refused the call because a rate limit was hit (HTTP 429).
 
    Client behaviour: sleep for ``retry_after`` when the provider supplied it,
    otherwise fall back to the standard backoff, then retry the same provider.
    Move on once the attempt limit is exhausted.
 
    ``retry_after`` is in seconds and is frequently ``None`` -- not every API
    returns a Retry-After header, so the client must never assume it is set.
    """
 
    def __init__(self, message: str = "", retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after
 
 
class ProviderUnavailableError(LLMServiceError):
    """This provider cannot serve the request, but another one might.
 
    Raised for bad credentials, an unknown or withdrawn model, a provider
    outage, or a prompt exceeding *this* model's context window.
 
    Client behaviour: do not retry this provider -- retrying cannot change the
    outcome. Record it in ``providers_tried`` and move to the next provider in
    the chain immediately.
    """
 
 
class InvalidRequestError(LLMServiceError):
    """The request itself is malformed; no provider will accept it.
 
    Raised for an empty prompt, an unusable response schema, or arguments that
    violate this package's own contract.
 
    Client behaviour: abort the entire chain immediately. No retry, no
    fallback -- every provider would fail identically, and walking the chain
    only makes the same failure slower.
    """