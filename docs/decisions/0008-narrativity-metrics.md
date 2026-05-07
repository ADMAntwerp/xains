# 0008. Narrativity metrics: paper-faithful composition over Protocol changes

Date: 2026-05-07
Status: Accepted

## Context

PR 6 adds the seven narrativity metrics from Cedro & Martens 2026 ("On the
Importance and Evaluation of Narrativity in Natural Language AI
Explanations", arXiv:2604.18311):

- **Continuous structure:** CSR, DCPR, CCPR.
- **Cause-effect:** CECPR.
- **Linguistic fluency:** FDR.
- **Lexical diversity:** TTCPR, VCPR.

Several of these need cumulative perplexity over sentence prefixes,
exponential-decay curve fitting, NLTK sentence/POS tagging, and two
vendored lexicons (Das et al. 2018 connectives, paper Appendix A
cause-effect markers). Three design questions surfaced:

1. Where do nltk and scipy live in the dependency graph?
2. Do we extend the ``PerplexityProvider`` Protocol so providers can
   compute cumulative perplexity natively, or compose at the metric
   layer with N calls?
3. How do we surface partial failures (NaN fits, missing optional deps,
   provider returning None mid-stream)?

## Decision

**Optional extras.** Both nltk and scipy land under
``[project.optional-dependencies].narrativity``. Core install (``pip
install xainarratives``) keeps a single runtime dep (``pydantic``); paper
metrics are opt-in (``pip install "xainarratives[narrativity]"``). This
preserves the dependency-discipline contract from ADR 0001.

**Composition at the metric layer.** The ``PerplexityProvider`` Protocol
keeps its single-method shape (``compute(text) -> float | None``). The
new ``cumulative_perplexity`` helper in
``xainarratives.metrics._internal.perplexity_utils`` calls
``provider.compute()`` on growing sentence prefixes, short-circuiting on
the first ``None``. Providers do not need to implement a new method.

**Strict ``None`` semantics.** Every metric and primitive returns
``float | None`` (``hallucination_count`` and ``coverage`` from PR 5 are
exceptions; they're always defined). ``None`` propagates through dependent
metrics: missing PPL → no ``r`` → all r-based metrics ``None``. No
fallback heuristics, no curve-fit retries, no synthetic perplexities.

**Separate ``NarrativityScores`` model.** Narrativity scoring lives in its
own pydantic model (``NarrativityScores``) and orchestrator
(``score_narrativity``). It does not extend ``ExtractionScores`` from PR 5.
The two have different inputs (extraction + request vs. raw narrative
text), different cost profiles (PR 5 needs no provider for most metrics;
PR 6 needs O(N) provider calls), and different paper origins (Ichmoukhamedov
et al. vs. Cedro & Martens). They're orthogonal.

**Vendored lexicons.** The 142-entry connectives lexicon (Das et al. 2018)
and 19-entry cause-effect lexicon (paper Appendix A) ship as JSON under
``src/xainarratives/metrics/_internal/data/``. Loaders assert the expected
counts at load time so a corrupt commit fails loud rather than silently
producing wrong scores.

## Rationale

- **Keeping the Protocol minimal.** Adding ``compute_cumulative`` to
  ``PerplexityProvider`` would force every existing implementation
  (including ``DisabledProvider``) to grow a method. The N-call
  composition is the same number of inferences and reuses the existing
  surface — the only cost is callers can't batch across a provider that
  natively supports prefix scoring. None of our v0 providers do, so the
  cost is zero.
- **No fallbacks.** A heuristic that "fixes" a failed curve fit (e.g.
  using mean PPL as ``r``) would silently mask exactly the inputs the
  paper would call non-narrative. Returning ``None`` makes the
  degradation visible.
- **Separate score model.** ``ExtractionScores`` is reachable from
  ``Explainer.explain`` (or will be in a later PR). ``NarrativityScores``
  is purely post-hoc, called manually on a generated narrative, and
  doesn't need to ride alongside an ``ExplanationResult``. Merging them
  would force every caller to pay for narrativity scoring's O(N)
  provider calls even when they only want the PR 5 fidelity numbers.

## Consequences

- ``pyproject.toml`` gains a ``narrativity`` extra
  (``nltk>=3.9,<4``, ``scipy>=1.13,<2``) plus mypy overrides for both
  (neither ships with type stubs that satisfy ``disallow_any_unimported``).
- ``src/xainarratives/metrics/_internal/`` is a new private subpackage
  housing tokenize, curve_fit, lexicons, perplexity_utils, plus the
  vendored ``data/`` JSON files. Underscore prefix marks it as not part
  of the public API.
- Standalone metric functions raise ``ImportError`` when their optional
  dep is missing (matching the ``readability``/``AnthropicProvider``
  pattern). ``score_narrativity`` catches ``ImportError`` at the
  orchestrator boundary and degrades affected fields to ``None``.
- ``score_narrativity`` makes ``N + 1`` provider calls for an ``N``-
  sentence narrative (``N`` cumulative + 1 shuffled). Callers who can't
  afford that pay the standalone ``readability`` price (zero provider
  calls) and skip ``score_narrativity``.
- ``NarrativityScores`` carries 7 derived metrics + 9 auxiliary primitives
  (``ppl_ordered``, ``ppl_shuffled``, ``decay_constant``, ``dist2``,
  ``ttr``, ``vr``, ``cr``, ``cer``, ``n_sentences``). Auxiliaries are
  free-of-charge (already computed for the derived metrics) and let
  paper-replicating callers inspect the building blocks.
- The curve fitter rejects non-decreasing inputs before calling scipy,
  uses a tighter ``b ∈ [-5, 5]`` bound, and returns ``None`` on any of:
  N<3 prefixes, constant series, scipy raise, fitted ``b ≤ 0``.

## Rejected alternatives

- **Extending the ``PerplexityProvider`` Protocol with
  ``compute_cumulative``.** Rejected: makes every existing provider grow
  a method, doesn't speed up our v0 providers (none batch internally),
  and adds a Protocol method that only one metric layer consumes.
- **Computing fallback perplexities when the provider returns ``None``.**
  Rejected: silently masks the exact failure mode that
  ``DisabledProvider`` exists to model. ``None`` is the answer.
- **Folding narrativity scoring into ``ExtractionScores``.** Rejected:
  forces every PR 5 caller to pay PR 6's per-call cost; couples two
  independent paper origins; and inflates a model that's already shipped.
- **Hard-failing on missing nltk/scipy at import time.** Rejected: the
  metrics package is imported by ``xainarratives.__init__``; a hard fail
  at the import boundary breaks the core install for users who never
  call ``score_narrativity``.
- **Computing connectives / cause-effect markers via embeddings or LLM.**
  Rejected: paper-faithfulness requires the exact Das et al. 2018 +
  Appendix A lexicons. An embedding-based proxy would not reproduce the
  paper's numbers.

## References

- Cedro, M., & Martens, D. (2026). *On the Importance and Evaluation of
  Narrativity in Natural Language AI Explanations.* arXiv preprint
  arXiv:2604.18311.
- Das, D., Stede, M., & Taboada, M. (2018). *Constructing a Lexicon of
  English Discourse Connectives.* In Proceedings of the 19th Annual
  SIGdial Meeting on Discourse and Dialogue.
  https://aclanthology.org/W18-5042/
- ADR 0001: Scope boundary — post-hoc verbalizer only (dependency
  discipline).
- ADR 0005: ``LLMResponse.tokens_used`` schema.
- ADR 0006 (revised by 0007): Guardrails and narrative extraction layer.
- ADR 0007: Resolution at extraction time.

```bibtex
@article{cedro2026narrativity,
  title={On the Importance and Evaluation of Narrativity in Natural Language AI Explanations},
  author={Cedro, Mateusz and Martens, David},
  journal={arXiv preprint arXiv:2604.18311},
  year={2026}
}
```
