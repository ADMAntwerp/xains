# 0018. NarrativeGenerator abstraction + Explainer(generator=) signature

Date: 2026-06-15
Status: Accepted

## Context

Producing a narrative from a request had exactly one path: render a
prompt with a ``PromptTemplate``, then call an ``LLMProvider``. A second
path is on the way - generate a narrative directly from structured
inputs (schema + contributions + counterfactuals), with no LLM call
between input and output. The pre-refactor ``Explainer`` hard-wired the
LLM flow: ``prompt_template`` and ``llm`` were stored as direct fields,
and ``explain()`` rendered + called inline. There was no seam for a
second strategy.

The post-generation steps - judge-based narrative extraction and
grading - depend only on the produced **text**, not on how it was
produced. They are the natural place to draw the seam.

## Decision

Introduce ``NarrativeGenerator`` (ABC) with a single contract:

```python
def generate(
    self,
    request: ExplanationRequest,
    schema: DatasetSchema,
    config: ExplanationConfig,
) -> GenerationResult: ...
```

``GenerationResult`` is a frozen dataclass carrying the generated
``text`` plus the audit envelope around it (``prompt``, ``model_name``,
``raw_llm_response``, ``tokens_used``, ``latency_ms``, ``guardrails``).
The four LLM-specific fields (``prompt``, ``model_name``,
``raw_llm_response``, ``tokens_used``) are ``str | None`` /
``dict | None``: a templated generator leaves them ``None``. ``text``,
``latency_ms``, and ``guardrails`` are generator-agnostic and always
populated where meaningful.

``Explainer.__init__`` now takes ``generator`` instead of
``prompt_template`` + ``llm``:

```python
Explainer(
    schema: DatasetSchema,
    generator: NarrativeGenerator,
    config: ExplanationConfig | None = None,
    judge_llm: LLMProvider | None = None,
)
```

``LLMNarrativeGenerator`` wraps the prompt_template + llm flow. It also
owns the ``class_name_mentioned`` guardrail - that check operates on the
LLM output text (a generation-side concern) and ships in
``GenerationResult.guardrails``. ``extract_narrative_claims`` and
grading **stay in ``Explainer``**: they run on the produced text
regardless of how it was generated.

``ExplanationResult.{prompt, model_name, raw_llm_response}`` widen to
``str | None`` to reflect that the templated path will produce
narratives without an LLM call.

### judge_llm: optional, no fallback

``judge_llm`` is optional at construction. There is **no fallback** -
``Explainer`` no longer sees an LLM directly, and introspecting
``generator.llm`` would reintroduce the generator-type coupling this
abstraction removes. ``explain()`` raises ``ValueError`` when
``config.extract_narrative=True`` and ``judge_llm is None``. Reason:
extraction needs a judge regardless of how the narrative was generated;
the templated path will need the same judge, supplied the same way.

### CLEAN BREAK

No dual-signature shim. Pre-1.0, consistent with this session's
established BREAKING changes (ADRs 0010, 0012, 0014, 0015, 0016). A
back-compat path detecting ``prompt_template=`` / ``llm=`` versus
``generator=`` would carry two ways to do one thing and obscure the
abstraction.

17 in-repo construction sites migrated mechanically:

```python
# old
Explainer(schema=S, llm=L, prompt_template=P, [config=C], [judge_llm=J])
# new
Explainer(
    schema=S,
    generator=LLMNarrativeGenerator(prompt_template=P, llm=L),
    [config=C,]
    judge_llm=J if specified else L,  # preserve the old self.llm fallback explicitly
)
```

One site (the new
``test_explain_raises_when_extract_narrative_true_and_judge_llm_none``)
is exempt: it deliberately omits ``judge_llm`` to provoke the new
``ValueError``.

## Consequences

- The LLM path is byte-identical to pre-refactor on every existing test
  (283 unchanged assertions + 4 new tests pinning the new surface). The
  four ``Explainer`` internal validators (``_validate_modality``,
  ``_validate_mode``, ``_check_explicit_mode``,
  ``_warn_if_counterfactual_does_not_flip``) are byte-identical.
- Prompt-preview is intrinsically an LLM-generator concept. The
  ``LLMNarrativeGenerator`` exposes a read-only ``prompt_template``
  property (``@property`` returning ``self._prompt_template``); the
  quickstart notebook's preview cell reads
  ``explainer.generator.prompt_template.render(request, schema,
  explainer.config)``. A templated generator has no prompt and no such
  property; a templated-path notebook would not have a prompt-preview
  cell.
- Callers that did ``result.prompt.upper()`` or similar without a
  ``None`` check on ``ExplanationResult`` need to add one. In-repo, one
  assertion in ``test_tabular_round_trip`` was widened to
  ``assert result.prompt is not None and "SYSTEM:" in result.prompt and
  "USER:" in result.prompt`` - a type-narrowing clause that does not
  change what the test verifies (the LLM path always populates the
  field).
- Documented in CHANGELOG under ``### Changed (BREAKING)``.

## Module placement

New subpackage ``src/xainarratives/generation/``:

```
src/xainarratives/generation/
    __init__.py            # re-exports NarrativeGenerator, LLMNarrativeGenerator, GenerationResult
    base.py                # NarrativeGenerator ABC + GenerationResult
    llm.py                 # LLMNarrativeGenerator
```

Phase-2's ``TemplatedNarrativeGenerator`` lands as a sibling
``templated.py`` in the same subpackage. The substitution helper from
ADR 0017 (``_substitute`` + ``_PLACEHOLDER_RE`` in
``feature_importance_tabular.py``) is the precedent for what the
templated generator will reuse; promoting it to a shared location
becomes appropriate once that second user exists (>=2-impl rule from
CLAUDE.md).

Top-level re-exports added to ``xainarratives/__init__.py``:
``NarrativeGenerator``, ``LLMNarrativeGenerator``, ``GenerationResult``
(each added to ``__all__`` in alphabetical position).

## Rejected alternatives

- **A ``TemplatedProvider`` implementing ``LLMProvider``.** The
  ``LLMProvider`` interface receives the rendered prompt (a
  ``(system, user)`` string pair) and returns text. A templated
  generator needs structured inputs (schema, contributions,
  counterfactuals) - it has no use for a rendered prompt and would have
  to parse data back out of the prompt string. It also wouldn't fit the
  ``class_name_mentioned`` guardrail wiring, which assumes an LLM-output
  flow. Mismatched abstraction; rejected.
- **Overloading ``prompt_template`` + ``llm=None`` to mean
  "templated".** ``PromptTemplate.render()`` promises ``(system, user)``
  - a prompt to send. A templated path produces a final narrative, not
  a prompt. Same name, different semantics; rejected.
- **Standalone ``templated_narrative()`` function reusing the grading
  API.** Smaller diff but creates two entry points
  (``Explainer.explain()`` for the LLM path, a free function for the
  templated path). Splits validation, judging, and audit packaging
  across two code paths. Rejected in favor of a single
  ``Explainer.explain()`` entry point that takes whichever generator.
- **Introspecting ``generator.llm`` as the ``judge_llm`` fallback.**
  Special-cases ``LLMNarrativeGenerator`` from within ``Explainer``,
  defeating the abstraction. The cost (one extra explicit ``judge_llm=``
  per construction site) is small and the resulting API is honest about
  a real dependency.

## References

- ADR 0017 (editable templates) - the ``prompt_template`` that
  ``LLMNarrativeGenerator`` now holds (and exposes via the new
  read-only property).
- ``src/xainarratives/generation/base.py`` - the ABC + dataclass.
- ``src/xainarratives/generation/llm.py`` - the LLM concrete + the
  ``prompt_template`` property.
- ``src/xainarratives/explainer.py`` - the new signature + the
  ``extract_narrative=True`` + ``judge_llm is None`` ``ValueError``.
- ``tests/unit/test_generation_llm.py`` - 3 new tests for
  ``LLMNarrativeGenerator.generate()`` shape and behavior.
- ``tests/unit/test_explainer.py::test_explain_raises_when_extract_narrative_true_and_judge_llm_none``
  - the judge-required fail-fast guard.
- CLAUDE.md "Abstraction Rule" (>=2 implementations) - applied to defer
  promoting the substitution helper to a shared location until the
  templated generator becomes the second user.
