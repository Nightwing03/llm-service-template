"""Groq adapter: an OpenAI-shaped chat completions API.

Groq constrains output with ``response_format: {"type": "json_object"}``, which
guarantees *valid JSON* but not *your schema*. So unlike Ollama, this adapter
has to describe the schema in the prompt and then validate what comes back.
That asymmetry is exactly what an adapter is for: the client asks for a
validated model and never learns how each backend got there.
"""

import json
import os
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from llm_service.errors import (
    InvalidRequestError,
    ProviderUnavailableError,
    RateLimitedError,
    TransientError,
)
from llm_service.types import ProviderResponse

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"

_SCHEMA_INSTRUCTION = (
    "Respond with a single JSON object and nothing else -- no prose, no code "
    "fences. It must validate against this JSON Schema:\n{schema}"
)


class GroqProvider:
    """One model on Groq's free or paid tier.

    Reads its key from the environment rather than taking it as an argument,
    so a key cannot be accidentally committed inside a provider construction.
    """

    def __init__(
        self,
        model: str = "openai/gpt-oss-20b",
        api_key_env: str = "GROQ_API_KEY",
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 60.0,
        temperature: float = 0.0,
        cost_per_1k_input: float | None = None,
        cost_per_1k_output: float | None = None,
        name: str = "groq",
    ) -> None:
        self.name = name
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self._api_key = os.environ.get(api_key_env)
        self._api_key_env = api_key_env
        self._cost_in = cost_per_1k_input
        self._cost_out = cost_per_1k_output

    def complete(
        self,
        *,
        system: str | None,
        user: str,
        response_model: type[BaseModel] | None = None,
        temperature: float | None = None,
    ) -> ProviderResponse:
        if not self._api_key:
            # Not fatal to the chain -- a local provider may still serve this.
            raise ProviderUnavailableError(f"{self._api_key_env} is not set")

        system_text = system
        if response_model is not None:
            try:
                schema = json.dumps(response_model.model_json_schema())
            except Exception as exc:  # pragma: no cover - malformed model class
                raise InvalidRequestError(f"unusable response model: {exc}") from exc
            instruction = _SCHEMA_INSTRUCTION.format(schema=schema)
            system_text = f"{system}\n\n{instruction}" if system else instruction

        messages: list[dict[str, str]] = []
        if system_text:
            messages.append({"role": "system", "content": system_text})
        messages.append({"role": "user", "content": user})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
        }
        if response_model is not None:
            payload["response_format"] = {"type": "json_object"}

        data = self._post(payload)

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise TransientError("groq returned no choices") from exc

        if not content:
            raise TransientError("empty completion from groq")

        value: Any = content
        if response_model is not None:
            value = self._parse(content, response_model)

        usage = data.get("usage") or {}
        return ProviderResponse(
            value=value,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            cost_usd=self._cost(usage),
        )

    # -- internals -------------------------------------------------------

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        try:
            with httpx.Client(timeout=self.timeout) as http:
                response = http.post(
                    f"{self.base_url}/chat/completions", json=payload, headers=headers
                )
        except httpx.TimeoutException as exc:
            raise TransientError(f"groq timed out after {self.timeout}s") from exc
        except httpx.HTTPError as exc:
            raise TransientError(f"groq transport error: {exc}") from exc

        status = response.status_code

        if status == 429:
            raise RateLimitedError(
                "groq rate limit", retry_after=_retry_after(response)
            )
        if status in (401, 403):
            raise ProviderUnavailableError("groq rejected the credentials")
        if status == 404:
            raise ProviderUnavailableError(f"model '{self.model}' not found on groq")
        if status >= 500:
            raise TransientError(f"groq returned {status}")
        if status == 400:
            # OpenAI-shaped APIs return 400 for context overflow as well as for
            # genuinely malformed requests. Only the latter is fatal to the
            # chain -- a model with a larger window may still accept this.
            body = response.text[:400]
            if "context" in body.lower() and "length" in body.lower():
                raise ProviderUnavailableError(f"prompt exceeds {self.model} context")
            raise InvalidRequestError(f"groq rejected the request: {body}")
        if status >= 400:
            raise ProviderUnavailableError(f"groq returned {status}")

        try:
            return response.json()
        except ValueError as exc:
            raise TransientError("groq returned a non-JSON body") from exc

    @staticmethod
    def _parse(content: str, response_model: type[BaseModel]) -> BaseModel:
        """Validate the model's output; both failure modes are retryable."""
        try:
            parsed = json.loads(content)
        except ValueError as exc:
            raise TransientError(f"model did not return JSON: {content[:200]}") from exc
        try:
            return response_model.model_validate(parsed)
        except ValidationError as exc:
            raise TransientError(f"output failed schema validation: {exc}") from exc

    def _cost(self, usage: dict[str, Any]) -> float | None:
        """``None`` unless prices were supplied -- a guess is worse than nothing."""
        if self._cost_in is None or self._cost_out is None:
            return None
        return (
            usage.get("prompt_tokens", 0) / 1000 * self._cost_in
            + usage.get("completion_tokens", 0) / 1000 * self._cost_out
        )


def _retry_after(response: httpx.Response) -> float | None:
    """Read Retry-After when present. The server knows; we would be guessing."""
    raw = response.headers.get("retry-after")
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None