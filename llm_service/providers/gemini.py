"""Gemini adapter: Google's generateContent API.

Shaped differently from the OpenAI convention in three ways that matter here:
the system prompt is a separate ``systemInstruction`` rather than a message,
generation settings live under ``generationConfig``, and structured output is
constrained by ``responseSchema`` -- which accepts an OpenAPI-flavoured subset
of JSON Schema, not the full document pydantic emits.

None of that reaches the client. The protocol holds; the translation cost is
paid here, which is the point of having a protocol at all.
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

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

# Keys pydantic emits that Gemini's schema dialect rejects or ignores.
_UNSUPPORTED_SCHEMA_KEYS = {
    "title",
    "default",
    "additionalProperties",
    "$defs",
    "$schema",
    "examples",
    "discriminator",
}


class GeminiProvider:
    """One Gemini model on the free or paid tier."""

    def __init__(
        self,
        model: str = "gemini-3.8-flash",
        api_key_env: str = "GEMINI_API_KEY",
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 60.0,
        temperature: float = 0.0,
        cost_per_1k_input: float | None = None,
        cost_per_1k_output: float | None = None,
        name: str = "gemini",
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
            raise ProviderUnavailableError(f"{self._api_key_env} is not set")

        generation: dict[str, Any] = {
            "temperature": self.temperature if temperature is None else temperature
        }

        if response_model is not None:
            try:
                raw = response_model.model_json_schema()
            except Exception as exc:  # pragma: no cover - malformed model class
                raise InvalidRequestError(f"unusable response model: {exc}") from exc
            generation["responseMimeType"] = "application/json"
            generation["responseSchema"] = to_gemini_schema(raw)

        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": generation,
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        data = self._post(payload)

        candidates = data.get("candidates") or []
        if not candidates:
            # Safety blocks and recitation stops land here. Retrying the same
            # prompt will not help, but another backend may accept it.
            reason = (data.get("promptFeedback") or {}).get("blockReason", "no candidates")
            raise ProviderUnavailableError(f"gemini returned no output: {reason}")

        try:
            content = candidates[0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as exc:
            raise TransientError("gemini returned a candidate with no text") from exc

        if not content:
            raise TransientError("empty completion from gemini")

        value: Any = content
        if response_model is not None:
            value = self._parse(content, response_model)

        usage = data.get("usageMetadata") or {}
        return ProviderResponse(
            value=value,
            input_tokens=usage.get("promptTokenCount"),
            output_tokens=usage.get("candidatesTokenCount"),
            cost_usd=self._cost(usage),
        )

    # -- internals -------------------------------------------------------

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/models/{self.model}:generateContent"
        try:
            with httpx.Client(timeout=self.timeout) as http:
                response = http.post(
                    url,
                    json=payload,
                    headers={"x-goog-api-key": self._api_key or ""},
                )
        except httpx.TimeoutException as exc:
            raise TransientError(f"gemini timed out after {self.timeout}s") from exc
        except httpx.HTTPError as exc:
            raise TransientError(f"gemini transport error: {exc}") from exc

        status = response.status_code

        if status == 429:
            raise RateLimitedError(
                "gemini rate limit", retry_after=_retry_after(response)
            )
        if status in (401, 403):
            raise ProviderUnavailableError("gemini rejected the credentials")
        if status == 404:
            raise ProviderUnavailableError(f"model '{self.model}' not found on gemini")
        if status >= 500:
            raise TransientError(f"gemini returned {status}")
        if status == 400:
            body = response.text[:400]
            lowered = body.lower()
            # A schema Gemini's dialect won't accept is our bug, not the
            # caller's -- but no retry will fix it, so move to the next backend
            # rather than aborting a chain that might still succeed.
            if "schema" in lowered or "token" in lowered:
                raise ProviderUnavailableError(f"gemini rejected the request: {body}")
            raise InvalidRequestError(f"gemini rejected the request: {body}")
        if status >= 400:
            raise ProviderUnavailableError(f"gemini returned {status}")

        try:
            return response.json()
        except ValueError as exc:
            raise TransientError("gemini returned a non-JSON body") from exc

    @staticmethod
    def _parse(content: str, response_model: type[BaseModel]) -> BaseModel:
        try:
            parsed = json.loads(content)
        except ValueError as exc:
            raise TransientError(f"model did not return JSON: {content[:200]}") from exc
        try:
            return response_model.model_validate(parsed)
        except ValidationError as exc:
            raise TransientError(f"output failed schema validation: {exc}") from exc

    def _cost(self, usage: dict[str, Any]) -> float | None:
        if self._cost_in is None or self._cost_out is None:
            return None
        return (
            usage.get("promptTokenCount", 0) / 1000 * self._cost_in
            + usage.get("candidatesTokenCount", 0) / 1000 * self._cost_out
        )


def to_gemini_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert a pydantic JSON Schema into Gemini's OpenAPI-flavoured subset.

    Inlines ``$ref`` pointers into ``$defs`` and drops keys Gemini rejects.
    Module-level and named rather than private because it is the one piece of
    this adapter worth testing directly -- it is pure, and it is where a nested
    model will break first.

    Known limitation: recursive models (a type referencing itself) would
    recurse forever and are not supported. Documented rather than handled,
    because nothing in this project needs one.
    """
    defs = schema.get("$defs", {})

    def convert(node: Any) -> Any:
        if isinstance(node, list):
            return [convert(item) for item in node]
        if not isinstance(node, dict):
            return node

        if "$ref" in node:
            ref = node["$ref"]
            key = ref.rsplit("/", 1)[-1]
            if key not in defs:
                raise InvalidRequestError(f"unresolvable schema reference: {ref}")
            return convert(defs[key])

        # anyOf carrying a null branch is how pydantic spells Optional[T].
        # Gemini has no union support, so take the first non-null branch and
        # mark it nullable.
        if "anyOf" in node:
            branches = [b for b in node["anyOf"] if b.get("type") != "null"]
            if not branches:
                raise InvalidRequestError("schema field admits only null")
            converted = convert(branches[0])
            if len(branches) < len(node["anyOf"]):
                converted["nullable"] = True
            return converted

        return {
            key: convert(value)
            for key, value in node.items()
            if key not in _UNSUPPORTED_SCHEMA_KEYS
        }

    return convert(schema)


def _retry_after(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after")
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None