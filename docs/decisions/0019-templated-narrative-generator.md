# 0019. TemplatedNarrativeGenerator: LLM-free feature-importance narratives

Date: 2026-06-16
Status: Accepted

## Context

The ``NarrativeGenerator`` abstraction in ADR 0018 was deliberately
designed to admit a second generation strategy without rewiring the
``Explainer``. This ADR delivers that second strategy: a templated
generator that verbalizes a ranked list of feature-importance
contributions as prose with NO LLM call.

This kind of templated explanation is the standard baseline that the
LLM-narrative literature benchmarks against - e.g., Cedro 2026's
templated comparator. Having both strategies in one library, evaluable
by the same fidelity / coverage / narrativity metrics, is the point of
the abstraction; this ADR is what makes that direct comparison
possible.

## Decision

``TemplatedNarrativeGenerator`` (in
``src/xainarratives/generation/templated.py``) implements
``NarrativeGenerator.generate(request, schema, config) -> GenerationResult``
with the following shape:

**Output structure.** A lead-in sentence followed by one clause per
top-ranked contribution, joined by single spaces into continuous prose.
Defaults:

```
lead:   "The model predicts {prediction}."
clause: "The {ordinal} important feature is {name} ({value}), with a {method}contribution of {importance}."
```

**Ranking.** ``sorted(contributions, key=lambda c: abs(c.importance), reverse=True)``,
sliced to ``config.top_k_features``. Identical to the LLM template's
ranking - method-of-generation does not change which features get
mentioned.

**Ordinals.** ``rank=1 -> "most"``, ``rank=2 -> "second most"``,
``rank=3 -> "third most"``, ``rank>=4 -> f"{rank}th most"``. Word forms
for the top three; digit form from four onward (matches Cedro 2026's
shape and reads naturally in English).

**Editable templates.** ``lead_template`` and ``clause_template`` are
keyword-only constructor args defaulted to ``DEFAULT_LEAD_TEMPLATE`` /
``DEFAULT_CLAUSE_TEMPLATE``. Substitution goes through
``xainarratives._substitution.substitute()`` - the same one-pass
primitive used by ``FeatureImportanceTabularPromptTemplate``. Available
placeholders: ``{prediction}`` (lead); ``{ordinal}``, ``{name}``,
``{value}``, ``{importance}``, ``{method}`` (clause).

**method param.** ``method: str | None = None``. Resolves to
``f"{method} "`` (with trailing space) inside ``{method}`` when set,
or ``""`` when unset. **Default is method-agnostic** ("with a
contribution of ..."); ``method="SHAP"`` yields "with a SHAP
contribution of ..." to reproduce Cedro 2026's wording. The library
hardcodes no attribution method - users pick the word that names what
they fed in.

**Values.** ``{value}`` is ``str(request.features[name])`` - the raw
dataset value for this instance. No thousands separator, no unit
bracket, no ``:g`` rounding. Users who want different formatting
override the clause template (the editability surface from ADR 0017
extends here).

**Strict missing-feature check.** A contribution naming a feature
that is not in ``request.features`` raises ``ValueError`` naming the
feature. No silent fallback to ``contribution.value`` - the templated
path is method-agnostic about how attributions were computed, but
strict about pulling values from the canonical request source.

**Tabular-only.** Non-tabular requests raise ``TypeError``. The
guardrail mirrors ``FeatureImportanceTabularPromptTemplate``'s
modality check. Other modalities will get their own generators when
their templates are designed.

**GenerationResult shape.** ``text`` and ``latency_ms`` are populated.
``prompt``, ``model_name``, ``raw_llm_response``, ``tokens_used`` are
all ``None`` (the ``str | None`` widening from ADR 0018 is now
load-bearing). ``guardrails`` is ``None``: the ``class_name_mentioned``
guardrail lives in ``LLMNarrativeGenerator`` (ADR 0018); it is an
LLM-output concern. Templated text is deterministic from structured
data, so there is no LLM output to guard.

**LLM-free generation, LLM grading.** The ``Explainer`` runs
``judge_llm``-based narrative extraction on the templated text just
as it does on LLM-generated text. ``judge_llm`` is still required
when ``extract_narrative=True`` (the fail-fast from ADR 0018). This
is the LLM-free-generation, LLM-graded path the abstraction was
built for.

## Rationale

- **Editable templates over hardcoded strings.** Consistent with ADR
  0017 - prompts and templated wording are user-editable text in this
  library. Researchers iterate on wording; we expose strings, not
  hidden behavior.
- **Method-agnostic default.** The library scope (README) is
  attribution-source-agnostic: it accepts SHAP, LIME, sklearn
  feature_importances_, permutation importance, anything signed
  per-feature. Hardcoding "SHAP" in the default narrative would
  contradict that scope; making it a one-arg knob preserves the
  general case and reproduces the SHAP-specific Cedro wording with
  one parameter.
- **Raw values from ``request.features``.** The contribution's own
  ``.value`` field is the value as the attribution pipeline saw it
  (potentially one-hot-encoded, scaled, or otherwise transformed).
  For user-facing prose, ``request.features[name]`` is the value the
  user has in their dataset - the right thing to render.
- **Strict raise on missing feature, not silent fallback.**
  Faithfulness: the verbalization tells the truth about what the
  library used. If a contribution names a feature the request did
  not provide a raw value for, the right behavior is to say "I
  cannot faithfully verbalize this," not to pick a different value
  source.
- **substitute() now has a second user.** ADR 0017 deferred the
  question of extracting the substitution helper to a shared
  location until a second user appeared. This is that user. The
  helper now lives in ``xainarratives._substitution`` and is
  imported by both ``FeatureImportanceTabularPromptTemplate`` and
  ``TemplatedNarrativeGenerator``. The CLAUDE.md ">=2
  implementations" rule for promoting an abstraction is satisfied
  exactly when the second implementation arrives, not earlier.

## Consequences

- ``TemplatedNarrativeGenerator`` slots into
  ``Explainer(generator=...)`` with zero ``Explainer`` changes -
  the ADR 0018 abstraction paying off in concrete form. Templated
  explanations flow through the same modality validation, the same
  mode check, the same judge-based narrative extraction, and the
  same ``ExplanationResult`` packaging as LLM-generated explanations.
- Templated and LLM explanations are directly comparable: same
  fidelity metrics (``sign_faithfulness``, ``value_faithfulness``,
  ``rank_correlation``), same coverage / hallucination counts, same
  narrativity metrics, same ``ExplanationResult`` schema. Choose the
  generator; everything downstream is identical.
- ``DEFAULT_LEAD_TEMPLATE`` and ``DEFAULT_CLAUSE_TEMPLATE`` are
  re-exported from ``xainarratives.generation`` (mirroring how the
  LLM template's defaults live in ``xainarratives.prompts``). The
  class ``TemplatedNarrativeGenerator`` is also re-exported at the
  top level (``xainarratives.TemplatedNarrativeGenerator``), in
  parallel with ``LLMNarrativeGenerator``.
- The ``str | None`` widening of
  ``ExplanationResult.{prompt, model_name, raw_llm_response}`` from
  ADR 0018 is now actively used: a templated explanation populates
  none of those fields. Callers reading these fields must None-check
  if they consume both paths.

## Reproducing prior work

To reproduce Cedro 2026's templated baseline wording, pass
``method="SHAP"`` and use ``config.top_k_features=5`` (the paper's
top-5 setting):

```python
Explainer(
    schema=schema,
    generator=TemplatedNarrativeGenerator(method="SHAP"),
    judge_llm=llm,
    config=ExplanationConfig(mode="feature_importance", top_k_features=5),
).explain(request)
```

The output reads "The model predicts <label>. The most important
feature is <name> (<value>), with a SHAP contribution of
<+/-importance>. ..." - the paper's exact shape, modulo value
typesetting (the paper formats "3,632 EUR"; this library renders raw
``str(value)`` and lets the user override the clause template if they
want unit brackets or thousands separators).

## References

- ADR 0018 (NarrativeGenerator abstraction) - the seam this ADR fills.
- ADR 0017 (editable prompt templates) - the editability pattern this
  ADR extends from LLM prompts to templated narratives.
- ``src/xainarratives/generation/templated.py`` - the concrete generator.
- ``src/xainarratives/_substitution.py`` - the shared substitute()
  primitive, now serving its second user (resolves ADR 0017's
  deferred-extraction note).
- ``tests/unit/test_generation_templated.py`` - 8 new tests
  including 2 byte-exact regression anchors
  (``test_default_output_byte_exact_method_agnostic`` and
  ``test_method_shap_injects_into_each_clause``).
