# 0009. Perplexity providers: two concretes, no shared base

Date: 2026-05-11
Status: Accepted

## Context

PR 6 shipped the seven paper narrativity metrics gated through
``PerplexityProvider``. At that point the codebase had two providers:

* ``DisabledProvider`` — always returns ``None``.
* ``APIPerplexityProvider`` — abstract callable wrapper; took a
  ``perplexity_fn: Callable[[str], float | None]`` and forwarded
  ``compute`` to it. Zero callers in the codebase, no concrete subclasses,
  no tests beyond its own self-test.

Neither produces real perplexity. Without a real provider, the seven
narrativity metrics that depend on ``r`` (decay constant from cumulative
perplexity) never return non-None values. PR 7 fills that gap and
revisits the abstraction.

## Decision

Two concrete providers, no shared base class. ``APIPerplexityProvider``
deleted.

1. ``HuggingFacePerplexityProvider`` (``perplexity_hf.py``) — local
   autoregressive model via ``transformers`` + ``torch``. Eager-loads
   tokenizer and model in ``__init__``. Constructor:
   ``(model_name="gpt2", device=None, max_length=None)``. Computes
   perplexity as ``exp(cross_entropy_loss)`` over a single forward pass
   with ``labels=input_ids``. Auto-detects CUDA; truncates with one
   ``UserWarning`` per instance.
2. ``OpenAICompatibleEchoProvider`` (``perplexity_api.py``) — hits any
   OpenAI-compatible ``/v1/completions`` endpoint with
   ``echo=True, logprobs=1, max_tokens=1``. Constructor:
   ``(base_url, api_key, model, timeout=30.0)``. Computes perplexity as
   ``exp(-mean(token_logprobs))`` from the echoed prompt.

``DisabledProvider`` and the ``PerplexityProvider`` Protocol are
unchanged. ``APIPerplexityProvider`` is removed.

## Rationale

### Why two, not one or three

One provider isn't enough — paper replication needs local model
reproduction; production scoring needs hosted API access. The two are
operationally distinct (local weight management vs HTTP auth + rate
limits), and a single provider can't span both cleanly.

Three or more providers would over-fit to vendor-specific APIs:
Together, OpenAI, Anthropic, vLLM, TGI all expose perplexity through
different surfaces. The OpenAI-compatible ``/v1/completions`` wire
format already covers Together, vLLM, TGI's OpenAI shim, and OpenAI's
legacy completions. One API provider, configured per ``base_url``, beats
four vendor-specific ones.

### Why composition, not a shared base class

``APIPerplexityProvider`` wrapped a callable. It carried no shared init
logic, no shared compute logic, and no shared state with anything else.
Zero callers. Per CLAUDE.md's "abstractions need ≥2 concrete
implementations" rule, it was a placeholder for an abstraction that
never materialized.

``HuggingFacePerplexityProvider`` and ``OpenAICompatibleEchoProvider``
have disjoint shapes:

* HF eager-loads weights to a device and runs forward passes;
* API constructs a lazy SDK client and makes HTTP calls.

A shared base would carry only the ``Protocol.compute`` signature, which
is already the Protocol. The Protocol is the shared interface; class
inheritance would add false signal.

### Why TGI's ``decoder_input_details`` was rejected

Hugging Face Text Generation Inference exposes per-token logprobs
through a vendor-specific ``decoder_input_details`` field on
``/generate``. Three costs:

* Locked to TGI; vLLM, Together, OpenAI legacy don't expose this shape.
* TGI's OpenAI-compatible shim already speaks ``/v1/completions``, so
  ``OpenAICompatibleEchoProvider`` works against TGI without a separate
  provider.
* The shape would force vendor sniffing in ``compute``.

The OpenAI-compatible echo pattern is the portable answer.

### Why ``OpenAIPerplexityProvider`` against actual OpenAI was rejected

OpenAI's legacy ``/v1/completions`` is on a deprecation path —
``gpt-3.5-turbo-instruct`` is the only model still on it. Modern
``/v1/chat/completions`` does not return token logprobs for input
tokens; only generated-token logprobs are available with
``logprobs=true``, and those don't let us derive prompt perplexity.

The compatible-third-party ecosystem (Together, vLLM, TGI) has longer
runway than OpenAI's own legacy endpoint.
``OpenAICompatibleEchoProvider`` points at any of them via ``base_url``.

### Paper-replication notes

Cedro & Martens 2026 uses ``meta-llama/Llama-3.1-8B`` (base, not
Instruct) locally for paper figures. Two practical paths:

* **Replicate locally:**
  ``HuggingFacePerplexityProvider(model_name="meta-llama/Llama-3.1-8B")``,
  subject to Hugging Face's gating for that model. Matches the paper
  methodology exactly.
* **Approximate via API:** Together's serverless catalog does not
  include the Llama-3.1-8B *base* model (only Instruct variants). Users
  wanting serverless paper-adjacent numbers should pick a current
  catalog model (``meta-llama/Meta-Llama-3-8B-Instruct-Lite``,
  ``Qwen/Qwen2.5-7B-Instruct-Turbo``, ``openai/gpt-oss-20b``) and note
  the model mismatch when reporting results.

### Dual-shape response parsing

Discovered during cassette recording: Together returns
``/v1/completions`` echo logprobs in two shapes depending on the model:

* **Shape A (OpenAI standard):**
  ``response.choices[0].logprobs.token_logprobs``.
* **Shape B (Together extension on prompt echo):**
  ``response.prompt[0].logprobs.token_logprobs``.

Shape B is semantically more correct — the logprobs are attached to the
echoed prompt tokens, not the (1-token, temperature=0) generated
continuation — and Together returns it for Llama and Qwen variants.
Shape A is the OpenAI-standard return shape used by
``openai/gpt-oss-20b`` and others.

``OpenAICompatibleEchoProvider.compute`` probes Shape B first via
``response.model_dump()`` (the openai SDK's Pydantic Completion model
doesn't expose ``prompt`` as a typed attribute), then falls back to
Shape A. Empty / unparseable in both → ``None``.

## Consequences

- Two new optional extras: ``perplexity-hf``, ``perplexity-api``.
- ``APIPerplexityProvider`` removed. Breaking, but pre-1.0 with no
  shipped release; listed under CHANGELOG ``Removed``.
- ``HuggingFacePerplexityProvider`` default ``model_name="gpt2"`` triggers
  a ~500 MB cached download on first use. Docstring warns prominently
  and points paper-replication callers at Llama-3.1-8B; production
  callers should pin an explicit model.
- The seven narrativity metrics from PR 6 now produce real numbers when
  paired with either of the new providers.
- Live API tests (``@pytest.mark.live``) gated behind
  ``TOGETHER_API_KEY`` env var, skipped in default CI. Cassette tests
  run offline against the recorded YAML.

## Rejected alternatives

- **Keep ``APIPerplexityProvider`` as a callable wrapper.** Rejected:
  no callers, fails the ≥2-implementations rule, contributes no shared
  behavior. Users with a callable can pass it through their own ad-hoc
  class — the Protocol is structural, so they don't need a wrapper class
  from us.
- **Single combined provider** with ``mode: Literal["hf", "api"]``.
  Rejected: forces every constructor argument into one shape; HF and API
  have disjoint config (weights vs URLs). Two narrow classes beat one
  branchy class.
- **Use httpx directly instead of the openai SDK.** Rejected: the
  ``OpenAICompatibleEchoProvider`` name signals SDK compatibility; the
  SDK handles auth, retries, error mapping, and timeout consistently
  across vendors. The dependency cost is minimal (openai SDK is small
  relative to torch).
- **TGI ``decoder_input_details`` path.** Rejected — see Rationale.
- **OpenAI legacy ``/v1/completions`` as a first-class target.**
  Rejected — see Rationale.

## References

- Cedro, M., & Martens, D. (2026). *On the Importance and Evaluation of
  Narrativity in Natural Language AI Explanations.* arXiv preprint
  arXiv:2604.18311.
- ADR 0001: Scope boundary — post-hoc verbalizer only (dependency
  discipline).
- ADR 0005: ``LLMResponse.tokens_used`` schema.
- ADR 0008: Narrativity metrics — paper-faithful composition over
  Protocol changes.
