"""Ollama adapter: a local model over Ollama's HTTP API.

Uses httpx directly rather than the ollama package. One dependency covers the
other adapters too, and the wire format stays visible -- which matters when a
response comes back malformed and you need to see what actually arrived.
"""

import json
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from llm_service.errors import (
    InvalidRequestError,
    ProviderUnavailableError,
    TransientError,
)
from llm_service.types import ProviderResponse

DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaProvider:
    """One local model served by Ollama.

    Makes exactly one call and never retries -- including when the model
    returns unparseable JSON, which is raised as :class:`TransientError` so
    the client's policy decides how many times to try again. An adapter that
    retried internally would make the client's ``attempts`` count a lie.
    """

    def __init__(
        self,
        model: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 120.0,
        temperature: float = 0.0,
        name: str = "ollama",
    ) -> None:
        self.name = name
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature

    def complete(
        self,
        *,
        system: str | None,
        user: str,
        response_model: type[BaseModel] | None = None,
        temperature: float | None = None,
    ) -> ProviderResponse:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature if temperature is None else temperature
            },
        }

        if response_model is not None:
            try:
                # Ollama accepts a JSON Schema here and constrains decoding to it.
                payload["format"] = response_model.model_json_schema()
            except Exception as exc:  # pragma: no cover - malformed model class
                raise InvalidRequestError(f"unusable response model: {exc}") from exc

        data = self._post(payload)

        content = data.get("message", {}).get("content", "")
        if not content:
            raise TransientError("empty completion from ollama")

        value: Any = content
        if response_model is not None:
            value = self._parse(content, response_model)

        return ProviderResponse(
            value=value,
            input_tokens=data.get("prompt_eval_count"),
            output_tokens=data.get("eval_count"),
            # Local inference: genuinely zero, not merely unknown.
            cost_usd=0.0,
        )

    # -- internals -------------------------------------------------------

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST to /api/chat, translating every failure into the taxonomy."""
        try:
            with httpx.Client(timeout=self.timeout) as http:
                response = http.post(f"{self.base_url}/api/chat", json=payload)
        except httpx.ConnectError as exc:
            # Ollama is not running. Another provider may still work.
            raise ProviderUnavailableError(f"cannot reach ollama at {self.base_url}") from exc
        except httpx.TimeoutException as exc:
            raise TransientError(f"ollama timed out after {self.timeout}s") from exc
        except httpx.HTTPError as exc:
            raise TransientError(f"ollama transport error: {exc}") from exc

        status = response.status_code
        if status == 404:
            # Model not pulled on this host; a different backend may have it.
            raise ProviderUnavailableError(f"model '{self.model}' not available")
        if status == 400:
            raise InvalidRequestError(f"ollama rejected the request: {response.text[:200]}")
        if status >= 500:
            raise TransientError(f"ollama returned {status}")
        if status >= 400:
            raise ProviderUnavailableError(f"ollama returned {status}")

        try:
            return response.json()
        except ValueError as exc:
            raise TransientError("ollama returned a non-JSON body") from exc

    @staticmethod
    def _parse(content: str, response_model: type[BaseModel]) -> BaseModel:
        """Validate the model's output.

        Both failures here are transient by design: generation is stochastic,
        so the same prompt often parses on the next attempt. The client's
        attempt limit stops this becoming an infinite loop, and the chain
        moves on once that limit is reached.
        """
        try:
            parsed = json.loads(content)
        except ValueError as exc:
            raise TransientError(f"model did not return JSON: {content[:200]}") from exc

        try:
            return response_model.model_validate(parsed)
        except ValidationError as exc:
            raise TransientError(f"output failed schema validation: {exc}") from exc