# 0020. OpenAI + OpenRouter LLM providers via a parameterized OpenAICompatibleProvider

Date: 2026-06-17
Status: Accepted

## Context

The ``LLMProvider`` Protocol is the universal contract every
narrative-generation backend implements - ``AnthropicProvider`` being
the reference. Supporting model choice beyond Anthropic raised the
question of how to add OpenAI + OpenRouter, given both speak the
IDENTICAL OpenAI Chat Completions wire protocol (as do Together, Groq,
vLLM, LM Studio, Fireworks). Writing two fully-separate classes would
duplicate the entire request / parse / usage-map body.

## Decision

A two-tier provider design.

**Tier 1 (unchanged):** the ``LLMProvider`` Protocol -
``generate(system, user) -> LLMResponse``. Any API shape implements
this directly. ``AnthropicProvider`` does, via the native top-level
``system=`` kwarg on the Messages API.

**Tier 2 (new):** ``OpenAICompatibleProvider(LLMProvider)`` -
**public**, no leading underscore - holds the single ``generate()`` for
the OpenAI chat-completions protocol, parameterized by:

- ``base_url`` (None lets the SDK use OpenAI's default)
- ``api_key`` (explicit) + ``api_key_env_var`` (fallback source)
- ``default_headers`` (None means omit the kwarg)
- ``model``, ``max_tokens`` (required; unbounded max_tokens is a cost footgun)

``OpenAIProvider`` and ``OpenRouterProvider`` are thin keyword-only
subclasses presetting endpoint + env var (+ OpenRouter's optional
``HTTP-Referer`` / ``X-Title`` headers for app attribution). The bare
``OpenAICompatibleProvider(base_url=..., api_key_env_var=...)`` is
usable directly for any other OpenAI-compatible endpoint with zero new
code.

**Key behaviors:**

- EAGER key resolution in ``__init__``:
  ``api_key or os.environ[api_key_env_var]``; ``ValueError`` naming the
  env var if missing. Fail-fast at construction.
- LAZY ``openai`` SDK import in ``generate()``: ``ImportError`` naming
  the ``openai`` pip extra. Matches ``AnthropicProvider``'s pattern.
- ``system + user`` packed as
  ``messages=[{"role": "system", ...}, {"role": "user", ...}]`` - the
  boundary that differs from Anthropic's top-level ``system=``.
- ``usage`` mapping: ``prompt_tokens -> input``, ``completion_tokens
  -> output``, ``total`` recomputed as ``input + output`` (ADR 0005
  3-key contract). ``tokens_used = None`` when ``response.usage`` is
  None (some self-hosted endpoints omit it).
- ``model_name = response.model`` (the API echo - e.g.
  ``"gpt-4o-mini"`` resolves to ``"gpt-4o-mini-2024-07-18"``).
- ``message.content or ""`` for the tool-call None-content case.

## Rationale

- **Shared base justified by two concrete users NOW.** OpenAI and
  OpenRouter together satisfy the CLAUDE.md ">=2 implementations"
  rule - the same one that justified ``_substitution.py`` in step 2a.
  Precisely: the perplexity-side ``OpenAICompatibleEchoProvider`` is a
  DIFFERENT protocol (``compute(text) -> float``, not ``generate()``) -
  it is evidence the OpenAI wire format is widespread, NOT a third
  user of this base.
- **``OpenAICompatibleProvider`` is public** (no leading underscore)
  because it IS the documented extension point - "add an
  OpenAI-compatible LLM" needs no new class, just a constructor call.
- **Eager key / lazy SDK.** Key resolution is cheap and pure, so
  fail-fast at construction is friendlier than a delayed failure on
  the first ``generate()`` call. The SDK is the heavy optional dep,
  so it is deferred to first use (no SDK touch in ``__init__``).
- **Explicit ``api_key_env_var``** (not the SDK's silent
  ``OPENAI_API_KEY`` auto-read). OpenRouter wants
  ``OPENROUTER_API_KEY``, not ``OPENAI_API_KEY`` - the explicit var
  name prevents the confusing collision (and makes the source of the
  key obvious in tracebacks and code review). Mirrors the
  perplexity-side ADR 0015 pattern.
- **New ``openai`` pip extra added ALONGSIDE ``perplexity-api``** (not
  renamed). Two extras pinning the same SDK is fine; renaming would
  break existing install commands users have in their docs and CI.

## Consequences

- Choosing the model is choosing the provider:
  ``Explainer(generator=LLMNarrativeGenerator(llm=OpenAIProvider(model="gpt-4o"), prompt_template=...))``.
  Zero ``Explainer`` changes; the abstraction from ADR 0018 pays off
  again.
- Adding Together / Groq / vLLM / LM Studio / Fireworks / any other
  OpenAI-compatible endpoint needs zero new code:
  ``OpenAICompatibleProvider(base_url=..., api_key_env_var=..., model=..., max_tokens=...)``.
  A genuinely different API (a new shape with non-chat-completions
  endpoints, or custom auth, or proprietary streaming) implements the
  tier-1 ``LLMProvider`` Protocol directly, like ``AnthropicProvider``.
- All providers are now top-level importable:
  ``from xainarratives import AnthropicProvider, OpenAIProvider, OpenRouterProvider, OpenAICompatibleProvider, MockLLMProvider, LLMProvider, LLMResponse``.
  Uniform public API; the older single-module-only convention for
  ``AnthropicProvider`` is brought up to parity.

## Validated against live endpoints

``OpenAIProvider`` (``gpt-4o-mini``), ``OpenRouterProvider``
(``meta-llama/llama-3.3-70b-instruct``), ``AnthropicProvider``
(``claude-haiku-4-5``), and the bare ``OpenAICompatibleProvider``
pointed at Together (``https://api.together.xyz/v1``,
``TOGETHER_API_KEY``, ``meta-llama/Llama-3.3-70B-Instruct-Turbo``) all
returned text + correct ``model`` echo + the 3-key ``tokens_used``
dict on real calls. The extension point is demonstrated, not just
designed.

## References

- The ``LLMProvider`` Protocol + ``LLMResponse`` in
  ``src/xainarratives/providers/base.py`` - the tier-1 contract.
- ADR 0005 - the ``{input, output, total}`` tokens contract.
- ADR 0015 - the explicit ``api_key_env_var`` pattern (perplexity
  sibling).
- ``src/xainarratives/providers/openai_compatible.py`` - the new
  module: ``OpenAICompatibleProvider`` base + ``OpenAIProvider`` and
  ``OpenRouterProvider`` presets.
- ``tests/integration/test_providers_openai_compatible.py`` - 11
  mock-spy tests pinning the wire format, key resolution, presets,
  edge cases, and the ImportError-on-missing-SDK path.
