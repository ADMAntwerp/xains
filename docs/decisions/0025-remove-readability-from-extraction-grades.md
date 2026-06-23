# 0025. Remove readability from ExtractionGrades

Date: 2026-06-19
Status: Accepted

## Context
After ADR 0023 removed `perplexity` from the verbalization-fidelity aggregate, `ExtractionGrades` still carried a `readability: float | None` field that was populated inside `grade_extraction` by calling the standalone `readability(extraction, narrative_text)` helper (Flesch reading-ease via `textstat`). `readability` is, by design.md section 10, a narrativity property - it is grouped with perplexity, type-token ratio, and sentence-length variance under "Narrativity", not under "Verbalization fidelity". ADR 0008 made `NarrativityGrades` and `ExtractionGrades` orthogonal; the `readability` field on the fidelity aggregate crossed that boundary in the same direction the `perplexity` field did before ADR 0023 removed it. Separately, Cedro & Martens 2026 argue that classic readability metrics are a weak fit for XAI narratives (short, structured text with non-prose elements), so a Flesch score is a noisy default to ship inside the fidelity record.

## Decision
Drop the `readability` field from `ExtractionGrades` and the corresponding computation (the `try/except ImportError` block plus `readability=` return argument) from `grade_extraction`. The `readability()` function in `xains.metrics.narrativity`, the `textstat` optional extra, and the public re-exports (`xains.readability`, `xains.metrics.readability`) remain. Callers who want a Flesch score call `readability(extraction, narrative_text)` directly. The `"readability": "↑"` entry is removed from `EXTRACTION_GRADE_DIRECTIONS` (resumed in the arrows commit). `ExtractionGrades` now carries only `sign_faithfulness`, `value_faithfulness`, `rank_correlation`, `coverage`, `hallucination_count`, and `prompt_version`.

## Rationale
- Mirrors ADR 0023 on the same orthogonality grounds: narrativity-shaped concepts belong on `NarrativityGrades` or as opt-in standalone helpers, not on the verbalization-fidelity aggregate (ADR 0008).
- Removes a noisy default. Cedro & Martens 2026 argue Flesch reading-ease is poorly suited to XAI narratives; shipping it as a default field invited callers to read it as a quality signal it does not reliably provide. Making it opt-in puts the choice in the caller's hands.
- Pre-1.0 hard cutover (no shim, no deprecation cycle) matches the precedent set by ADR 0009 (`APIPerplexityProvider` removal), ADR 0014 (`score_*` to `grade_*` rename), and ADR 0023 (perplexity removal from `ExtractionGrades`).
- The function is still useful for paper-replication baselines, so we keep it; only the aggregate field goes.

## Consequences
- `ExtractionGrades` becomes six fields instead of seven. `extra='forbid'` rejects any caller passing `readability=...` to the constructor.
- `grade_extraction(extraction, request, schema, narrative_text, k=10)` signature is unchanged (`narrative_text` stays in the parameter list to preserve the API for callers; it is currently unused inside the function).
- The README sample output for `ExtractionGrades` loses its `readability=` token.
- The quickstart notebook Step 6 markdown loses the "Readability" bullet and the summary loses "readability" from the axes list.
- The grader's missing-textstat-graceful test in `tests/unit/test_metrics_grader.py` is removed (no longer relevant). `readability()` itself keeps direct coverage in `tests/unit/test_metrics_narrativity.py` (returns-float, returns-none-on-empty, raises-clean-error-on-missing-textstat).
- `tests/unit/test_metrics_render.py` was added during the paused arrows path and still references `readability`; it will be regenerated when the arrows commit resumes (it has a pre-existing `ImportError` on `render_grades` and is excluded from this commit's chain check).

## Rejected alternatives
- **Keep `readability` on `ExtractionGrades` and document it as a weak default.** Rejected: a documented caveat inside an aggregate field is still a default; ADR 0008 and design.md section 10 both locate readability under narrativity.
- **Delete the `readability()` function and the `textstat` extra outright.** Rejected: the function remains useful as an opt-in baseline for paper comparison (Cedro & Martens 2026 report Flesch among other metrics). Removing the field is the boundary fix; removing the helper would be a separate, larger decision.
- **Move `readability` to `NarrativityGrades`.** Rejected for this commit: would re-cross the boundary in the opposite direction. `NarrativityGrades` ships the paper's seven metrics plus its auxiliary primitives; bolting Flesch onto it without a paper-faithfulness motivation is scope creep. Callers who want both call `readability()` alongside `grade_narrativity()`.
- **Deprecation cycle (warn for one release, then remove).** Rejected: pre-1.0, no external users on PyPI under `xains`, no compatibility cost. Matches the ADR 0009 / 0014 / 0023 precedent of hard cutover.

## References
- ADR 0008 - narrativity metrics: `NarrativityGrades` is orthogonal to `ExtractionGrades`.
- ADR 0023 - removed `perplexity` from `ExtractionGrades` on the same orthogonality grounds.
- `docs/design.md` section 10 - readability is listed under narrativity.
- Cedro, M., & Martens, D. (2026). *On the Importance and Evaluation of Narrativity in Natural Language AI Explanations.* arXiv:2604.18311. (Argues classic readability metrics are a weak fit for XAI narratives.)
- `src/xains/metrics/grader.py` - the field, computation, import, and direction-dict entry removal.
- `src/xains/metrics/narrativity.py` - `readability()` retained.
