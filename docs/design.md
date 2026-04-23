# xainarratives — Design

This document captures the architecture and the reasoning behind it. It's the
single source of truth for "why is it shaped this way?" questions. Short-form
decisions live in [`decisions/`](decisions/) as ADRs.

## 1. Purpose

Take pre-computed XAI outputs (attributions, counterfactuals) plus a dataset
schema and a model prediction, and produce a natural-language explanation for
a human audience. Also evaluate the quality of that verbalization.

## 2. Scope (What's In, What's Out)

**In.**
- Instance-level explanations (not global).
- Four modalities: tabular, text, image, graph (GNN).
- Factual, contrastive ("why class A, not B"), and counterfactual explanations.
- Evaluation: verbalization fidelity (LLM-as-judge), readability, narrativity
  metrics (incl. perplexity via pluggable providers).

**Out (permanently, by design).**
- Model training, inference.
- Attribution computation (SHAP, LIME, Captum, GNNExplainer, permutation, …).
- Counterfactual search (DiCE, Alibi, Wachter, …).
- Global / dataset-level explanations.
- UI / dashboards.

## 3. Core Data Model

Three layers:

1. **Schema** (per model) — `DatasetSchema` with `FeatureSchema` /
   `TargetSchema` / per-modality `*Spec`. Set once; reused for every instance.
2. **Request** (per instance) — polymorphic `*ExplanationRequest` discriminated
   by `modality`. Contains features, prediction, contributions, optional
   counterfactuals.
3. **Config** (per call / per deployment) — `ExplanationConfig` controls
   audience, language, length, tone, mode (factual / contrastive /
   counterfactual / auto).

Polymorphism is via pydantic v2 discriminated unions. See
[`decisions/0003-data-model.md`](decisions/0003-data-model.md).

## 4. Contribution Hierarchy

A single `Contribution` tagged-union covers all modalities:

| Type                  | Used by          | Shape (key fields)                     |
| --------------------- | ---------------- | -------------------------------------- |
| `TabularContribution` | tabular          | `name`, `value`, `importance`          |
| `TokenContribution`   | text             | `token`, `span`, `importance`          |
| `RegionContribution`  | image            | `region_id`, `bbox?`, `description?`   |
| `NodeContribution`    | graph            | `node_id`, `features`, `label?`        |
| `EdgeContribution`    | graph            | `src`, `dst`, `edge_type?`             |

Each request subclass narrows its `contributions` field to the compatible
types. A `TabularExplanationRequest` with `NodeContribution`s fails at
validation, not at inference.

## 5. Counterfactuals

Accepted as a **list** of pre-computed instances (length ≥ 1). Single-CF is
the degenerate case. See
[`decisions/0004-counterfactual-payload-shape.md`](decisions/0004-counterfactual-payload-shape.md).

Polymorphic in the same way requests are: `TabularCounterfactual`,
`TextCounterfactual`, `ImageCounterfactual`, `GraphCounterfactual`.

The library never searches for counterfactuals. If the user provides one
whose `predicted_class` equals the factual's `predicted_class`, the
`Explainer` warns loudly — it is almost certainly a user error.

## 6. LLM Provider Abstraction

A minimal `Protocol`:

```python
class LLMProvider(Protocol):
    def generate(self, system: str, user: str) -> LLMResponse: ...
```

Concrete implementations in v0: `MockLLMProvider`. Planned: `AnthropicProvider`,
`OpenAIProvider`, `LiteLLMProvider`, `OllamaProvider`. Per the Abstraction
Rule in CLAUDE.md the Protocol stays because at least two concrete providers
are committed for subsequent PRs.

## 7. Prompt Template Abstraction

ABC `PromptTemplate.render(request, schema, config) -> (system, user)`.

Concrete in v0: `EchoPromptTemplate` (renders full request as JSON; for
testing). Planned: `FactualTabularPromptTemplate`, `ContrastivePromptTemplate`,
`CounterfactualPromptTemplate`, plus modality-specific variants.

## 8. Explainer (Orchestrator)

Single class. Sync only. Responsibilities:

1. Validate request modality matches schema modality.
2. Resolve mode (factual / contrastive / counterfactual) from config + request.
3. Render prompt via template.
4. Call LLM via provider.
5. Assemble `ExplanationResult` with audit metadata (prompt, raw response,
   model name, tokens).

Not in v0: guardrails, caching, retries, streaming, evaluation pipeline. Each
gets its own PR and ADR when added.

## 9. Integrations (Planned, Not in v0)

Adapter pattern. Named by **input shape**, not by upstream library:

- `from_feature_importance(...)` → accepts SHAP, LIME, sklearn
  `feature_importances_`, permutation importance, integrated gradients,
  anything reducible to a signed scalar per feature.
- `from_token_importance(...)` → text.
- `from_region_importance(...)` → image.
- `from_graph_importance(...)` → graph (node mask + edge mask).
- `from_counterfactual_set(...)` → DiCE / Alibi / manual counterfactual bundles.

Each adapter lives in its own module in `xainarratives/integrations/` and
imports its upstream dep lazily with a clear `ImportError` if missing.

## 10. Evaluation (Planned, Not in v0)

Two axes:

- **Verbalization fidelity** (does the text match the provided attributions?).
  Rule-based checks first (every mentioned feature exists in schema; sign of
  claimed effect matches importance sign; predicted class named in text
  matches `prediction.predicted_class`). LLM-as-judge as a second, more
  expensive layer.
- **Narrativity** (readability, perplexity, type-token ratio, sentence-length
  variance).

Perplexity is the only metric that needs a language model. It goes behind a
`PerplexityProvider` Protocol with at least:

- `APILogprobProvider` — uses an LLM API's token-logprobs (no download).
- `LocalHFProvider` — local causal LM (optional extra).
- `ONNXProvider` — pre-quantized int8 model, CPU-fast, no torch dep.
- `DisabledProvider` — returns `None`; evaluator degrades gracefully.

CPU-only / air-gapped users pick `APILogprobProvider` or `DisabledProvider`.

## 11. Testing Strategy

- `tests/unit/` — fast, no network, `MockLLMProvider`.
- `tests/integration/` — cassette-based (recorded LLM responses).
- `tests/live/` — real APIs, `@pytest.mark.live`, opt-in.
- `tests/eval/` — evaluator self-tests on a small benchmark set.

Every public function is tested. Every pydantic model has a valid-case and an
invalid-case test. Property tests (hypothesis) for guardrails once those
exist.

## 12. Versioning and Stability

`0.y.z` until the API has been used by a non-author on a non-toy problem.
Breaking changes in minor versions are allowed while `0.y.z`, but noted in
`CHANGELOG.md` and ideally preceded by a `DeprecationWarning` cycle.
