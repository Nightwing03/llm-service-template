"""FastAPI wrapper.

The library is the product; this exists so the package runs and can be probed
as a service. Structured output is deliberately library-only -- see ``/complete``.
"""

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from llm_service.client import LLMClient
from llm_service.errors import AllProvidersFailedError, InvalidRequestError
from llm_service.providers.gemini import GeminiProvider
from llm_service.providers.groq import GroqProvider
from llm_service.providers.ollama import OllamaProvider

app = FastAPI(title="llm-service-template", version="0.1.0")


def build_chain() -> list[Any]:
    """Order the chain fastest-first, ending with the local model.

    Ollama is last on purpose: it is the slowest but the only one that cannot
    be rate limited or have its credentials revoked, which makes it the right
    thing to fall back *to* rather than start from.
    """
    return [
        GroqProvider(model=os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")),
        GeminiProvider(model=os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")),
        OllamaProvider(
            model=os.environ.get("OLLAMA_MODEL", "qwen2.5"),
            # Inside a container, localhost is the container. Ollama runs on
            # the host, so a containerised deployment must point at
            # host.docker.internal (Docker Desktop) or the host's LAN address.
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        ),
    ]


client = LLMClient(
    build_chain(),
    requests_per_minute={"groq": 25, "gemini": 12},
)


class CompleteRequest(BaseModel):
    user: str = Field(min_length=1)
    system: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)


class CompleteResponse(BaseModel):
    value: str
    provider: str
    model: str
    latency_ms: float
    attempts: int
    providers_tried: list[str]
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness plus configuration, not a probe of the backends.

    Deliberately makes no upstream call: a health check that spends rate-limit
    quota is one you cannot afford to poll, and a green light bought with a
    real request is a green light you stop trusting. ``key_present`` reports
    whether credentials exist, never whether they work.
    """
    return {
        "status": "ok",
        "providers": [
            {
                "name": p.name,
                "model": p.model,
                "key_present": getattr(p, "_api_key", "local") is not None,
            }
            for p in client.providers
        ],
    }


@app.post("/complete", response_model=CompleteResponse)
def complete(request: CompleteRequest) -> CompleteResponse:
    """Free-text completion across the provider chain.

    Structured output is not exposed here. It takes a pydantic *class*, which
    has no faithful representation over HTTP -- accepting a raw JSON Schema
    instead would mean validating untrusted schemas and returning unvalidated
    objects, which is the opposite of what this package is for. Import the
    client directly when you need it.
    """
    try:
        result = client.complete(
            user=request.user,
            system=request.system,
            temperature=request.temperature,
        )
    except InvalidRequestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except AllProvidersFailedError as exc:
        # 502: this service is fine, everything it depends on is not.
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return CompleteResponse(
        value=result.value,
        provider=result.provider,
        model=result.model,
        latency_ms=result.latency_ms,
        attempts=result.attempts,
        providers_tried=result.providers_tried,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost_usd=result.cost_usd,
    )