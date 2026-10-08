# llm-service-template

Typed LLM calls with retries, cross-provider fallback, rate limiting and cost
accounting. Three backends behind one interface, with the differences between
them absorbed by adapters rather than leaked to callers.

Built as the foundation for a document-analysis pipeline, extracted because
the concerns are the same everywhere.

```python
from pydantic import BaseModel
from llm_service.client import LLMClient
from llm_service.providers.groq import GroqProvider
from llm_service.providers.ollama import OllamaProvider

class Verdict(BaseModel):
    sentiment: str
    confidence: float

client = LLMClient(
    [GroqProvider(model="openai/gpt-oss-20b"), OllamaProvider(model="qwen2.5")],
    requests_per_minute={"groq": 25},
)

result = client.complete(
    system="Classify sentiment.",
    user="The product shipped late and support ignored me.",
    response_model=Verdict,
)

result.value            # Verdict(sentiment='negative', confidence=0.95)
result.provider         # 'groq'
result.attempts         # 1
result.providers_tried  # []
result.input_tokens     # 199
```

## Measured

One prompt, one schema, three backends. Free tiers, September 2026.

| Provider | Latency | Input tokens | Output tokens |
|---|---:|---:|---:|
| groq (llama-3.1-8b-instant) | 611 ms | 199 | 78 |
| gemini (2.0-flash) | 3167 ms | 14 | 15 |
| ollama (qwen2.5, local M-series) | 5491 ms | 26 | 21 |

The input-token gap is the finding. Groq constrains output to *valid JSON* but
not to *your schema*, so its adapter appends the schema to the system prompt
and validates afterwards. Gemini accepts the schema natively and never pays
for it in tokens. Same result, 14× the input cost.

That matters for batch work: on a free tier with per-minute token caps, the
faster provider can be the wrong one.

## Design

**Providers translate. The client decides.** An adapter makes one call, never
retries, never sleeps, and knows nothing about any other backend. Retry,
backoff, fallback, rate limiting and accounting live in the client. A new
backend is one file and touches no control flow.

**Each exception type maps to exactly one client behaviour.** If two types
would produce the same response, one is redundant.

| Raised | Client does |
|---|---|
| `TransientError` | Retry the same provider with backoff |
| `RateLimitedError` | Sleep `retry_after` if given, else backoff, then retry |
| `ProviderUnavailableError` | Do not retry. Move to the next provider |
| `InvalidRequestError` | Abort the chain. No provider would accept this |
| `AllProvidersFailedError` | Raised, not caught — the chain is exhausted |

The distinction that matters most is the third versus the fourth. Bad
credentials on one backend should not kill a chain that still has others; a
malformed schema should stop everything rather than fail identically three
times. A prompt exceeding one model's context window is
`ProviderUnavailableError`, not `InvalidRequestError`, because a larger model
may still accept it.

**Structured output, three ways.** All hidden behind one method.

| | System prompt | Schema |
|---|---|---|
| Ollama | a message | JSON Schema in `format`, constrains decoding |
| Groq | a message | `json_object` only — schema goes in the prompt |
| Gemini | separate `systemInstruction` | `responseSchema`, OpenAPI subset |

**Metadata is returned, not just logged.** `attempts` and `providers_tried`
exist so tests can assert that retry and fallback actually behaved. Logging
them would make those assertions scrape log output.

**`cost_usd = None` means unknown, not free.** Local inference records `0.0`.
Adapters compute their own cost; the client holds no price table.

## Install

```bash
pip install -r requirements.txt
```

Keys come from the environment — `GROQ_API_KEY`, `GEMINI_API_KEY` — and are
never constructor arguments, so one cannot be committed inside a provider
construction. Ollama needs no key.

## Service

```bash
uvicorn llm_service.app:app --reload      # or: docker build -t llm-svc . && docker run -p 8000:8000 llm-svc
```

`GET /health` reports liveness and configuration. It makes **no** upstream
call: a health check that spends rate-limit quota is one you cannot afford to
poll, and `key_present` reports that credentials exist, never that they work.

`POST /complete` does free-text completion. Structured output is not exposed
over HTTP — it takes a pydantic class, and accepting a raw JSON Schema instead
would mean validating untrusted schemas and returning unvalidated objects.
Import the client directly for that.

## Tests

```bash
pip install -r requirements-dev.txt && python -m pytest -q
```

24 tests, ~0.3 s, no network and no real sleeping. `sleep` and `now` are
injected into the client and the rate limiter, so a 60-second throttle is
asserted exactly, in microseconds. A suite needing API keys could not gate a
pull request from a fork; a slow suite gets skipped.

## Limitations

1. **Groq cannot enforce your schema.** It guarantees valid JSON only. Schema
   compliance is achieved by prompt instruction plus post-validation, so a
   malformed response is retried rather than prevented.
2. **Gemini's schema dialect is a subset.** `to_gemini_schema` inlines `$defs`
   and drops unsupported keys. `Optional[T]` becomes `nullable`, since Gemini
   has no unions. **Recursive models are not supported** and would recurse
   forever.
3. **Ollama's schema support is version-dependent.** Older builds accept only
   `format: "json"` and will reject a full JSON Schema with a 400.
4. **Groq model ids go stale.** Models are retired regularly; the default here
   will eventually 404. Query `/v1/models` for a live id.
5. **Rate limits are client-side only.** A limiter configured above the real
   quota will still see 429s — handled, but not prevented.
6. **Synchronous by design.** Free-tier rate limits, not I/O concurrency, are
   the bottleneck, so async would add complexity without adding throughput.
7. **Cost is only as good as the prices you supply.** Unsupplied means `None`.
   No price table is bundled, because a stale one is worse than no number.