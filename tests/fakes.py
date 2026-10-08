"""Test doubles. No network, no model, fully deterministic."""
 
from typing import Any
 
from pydantic import BaseModel
 
from llm_service.types import ProviderResponse
 
 
class FakeProvider:
    """A scripted provider, for testing client behaviour without a backend.
 
    ``script`` is consumed one entry per call. An ``Exception`` instance is
    raised; anything else is returned as ``ProviderResponse.value``. Once only
    one entry remains it repeats indefinitely, so a provider that should always
    fail needs a single-entry script and one that should fail twice then
    succeed needs three entries.
 
    Satisfies :class:`~llm_service.providers.base.Provider` structurally,
    without importing or inheriting it -- which is the reason that contract is
    a ``Protocol`` rather than an ABC.
 
    Examples:
        Always fails::
 
            FakeProvider("a", "m", [TransientError("down")])
 
        Fails twice, then succeeds::
 
            FakeProvider("a", "m", [TransientError(""), TransientError(""), "ok"])
    """
 
    def __init__(self, name: str, model: str, script: list[Any]) -> None:
        if not script:
            raise ValueError("script must contain at least one entry")
        self.name = name
        self.model = model
        self._script = list(script)
        self.call_count = 0
        self.last_call: dict[str, Any] | None = None
        """Arguments from the most recent call, so tests can assert pass-through."""
 
    def complete(
        self,
        *,
        system: str | None,
        user: str,
        response_model: type[BaseModel] | None = None,
        temperature: float | None = None,
    ) -> ProviderResponse:
        self.call_count += 1
        self.last_call = {
            "system": system,
            "user": user,
            "response_model": response_model,
            "temperature": temperature,
        }
 
        # Hold the final entry once the script is down to one, so "always
        # fails" and "always succeeds" need no special casing in tests.
        entry = self._script.pop(0) if len(self._script) > 1 else self._script[0]
 
        if isinstance(entry, Exception):
            raise entry
 
        # Fixed token counts: deterministic, and non-equal so a test that
        # transposes input and output tokens fails visibly.
        return ProviderResponse(value=entry, input_tokens=11, output_tokens=7)