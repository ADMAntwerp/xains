# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

# Project: xains

Project-specific rules. These take precedence over the generic guidelines above where they conflict. Read `docs/design.md` before making non-trivial changes.

## Scope (Hard Boundary)

`xains` is a **post-hoc explanation verbalizer**. It takes pre-computed inputs and produces natural-language explanations plus verbalization-quality metrics.

**It never:**
- Trains models
- Runs inference on models
- Computes attributions (SHAP / LIME / GNNExplainer / Captum / sklearn feature_importances_ / etc.)
- Searches for counterfactuals (DiCE / Wachter / Alibi / etc.)

If a feature would require any of the above, it does not belong in this library. Point the user at the upstream tool.

## Dependency Discipline

Core runtime dependency: `pydantic` only. Add another only with an ADR justifying it.

**Never import in core** (`src/xains/` outside of `integrations/` and `eval/` subpackages):

- `torch`, `tensorflow`, `jax`
- `sklearn`, `xgboost`, `lightgbm`
- `shap`, `lime`, `dice-ml`, `alibi`, `captum`
- `transformers`, `sentence-transformers`
- `numpy`, `pandas`

These belong in optional extras, imported lazily inside `xains/integrations/*` or `xains/eval/*` adapters with `try: import x` guards that raise a clear `ImportError` telling the user the right `pip install xains[extra]` command.

## Abstraction Rule

Every `Protocol` / ABC must have **≥2 concrete implementations**, or — during skeleton phase — one concrete plus a committed plan (in an ADR or issue) for the second. Abstractions without a second implementation in sight get inlined.

This rule exists to reconcile "production quality / SOLID" with "Simplicity First" above. SOLID is not a license to abstract pre-emptively.

## Vocabulary (Use Precisely)

- **Attribution faithfulness** — whether the upstream attribution reflects the model. Out of scope for this library.
- **Verbalization fidelity** — whether the generated text reflects the *provided* attributions. This is what we measure.
- Never write "faithfulness" unqualified.

## Modalities

Tabular, text, image, graph. All four use polymorphic request / contribution / counterfactual types discriminated by a `modality` / `type` field. No modality-specific branches in the core `Explainer` — variation lives in schema, request type, and prompt template.

Image modality requires either a vision-capable LLM **or** pre-computed region descriptions. Incompatible combinations must fail at `Explainer.__init__`, never at inference time.

## API Style

Sync is the canonical API. Do not design for async in v0. If async becomes necessary later, add it as a separate thin wrapper.

## Counterfactuals

The library accepts a **list** of pre-computed counterfactual instances (length ≥ 1). Single-CF is the degenerate case. Counterfactuals are never computed internally.

## Integration Naming

Adapters in `xains/integrations/` are named by the **shape of the input**, not the upstream tool. E.g., `from_feature_importance` (accepts SHAP, LIME, sklearn feature_importances_, permutation importance, any signed-per-feature scalar), not `from_shap`.

## Testing

- Default LLM provider in tests is `MockLLMProvider`. Unit tests never hit a real API.
- Tests that call real APIs are marked `@pytest.mark.live` and skipped in default CI.
- Every new pydantic model gets at least one valid-case and one invalid-case test.
- A new public function without a test does not merge.

## Python Version

Floor is 3.11. Use native PEP 604 (`X | Y`) and PEP 585 (`list[X]`) syntax. Do not add `from __future__ import annotations` — it silently disables runtime type introspection that pydantic and `typing.get_type_hints` rely on.
