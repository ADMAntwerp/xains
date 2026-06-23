# 0026. scored_only flag on render_grades

Date: 2026-06-19
Status: Accepted

## Context
ADR 0024 introduced `render_grades(extraction=None, narrativity=None) -> str` to print grade aggregates with desired-direction arrows. By default it iterates `model_dump()` in declaration order and renders every field, attaching an arrow only when the field appears in the direction dict. For `ExtractionGrades` that is exactly the readable summary callers want. For `NarrativityGrades` it produces sixteen lines: the seven scored Cedro & Martens 2026 metrics with arrows, plus the nine auxiliary primitives (`ppl_ordered`, `ppl_shuffled`, `decay_constant`, `dist2`, `ttr`, `vr`, `cr`, `cer`, `n_sentences`) without arrows. The auxiliaries are valuable for paper replication but noisy for everyday display - README and notebook narrativity samples want the seven-line scored view. The choice is presentation, not data.

## Decision
Add a single keyword argument `scored_only: bool = False` to `render_grades`. When `True`, `_render_section` skips any field whose name is not a key in the section's direction dict. `prompt_version` is still omitted by name in either mode. The default is `False`, which preserves prior behaviour exactly. The README narrativity sample switches to `render_grades(narrativity=narrativity, scored_only=True)`; the quickstart notebook Step 7 display does the same. Step 6 (extraction) does not pass the flag - it has no observable effect there, because every `ExtractionGrades` field is scored.

## Rationale
- The direction dict is already the source of truth for "which fields are scored". Reusing membership in that dict as the filter avoids introducing a second list of scored field names.
- One flag is enough. Two-mode behaviour (full / scored-only) is what callers want; intermediate modes (scored + some auxiliaries) would be speculative and lack a known caller.
- Default `False` preserves the paper-replication path. Researchers reading the auxiliaries directly should not have to opt in to a behaviour that has been in place since ADR 0024.
- `scored_only` on `ExtractionGrades` is a deliberate no-op (every field is in the direction dict). Callers do not need to know which grade their input is; a uniform flag keeps the surface small.

## Consequences
- `render_grades` and `_render_section` gain one keyword-only parameter each. The helper takes `scored_only` as keyword-only (`*, scored_only: bool`) because it has no other call site than the public function.
- Four new tests in `tests/unit/test_metrics_render.py` pin: scored-only hides the nine auxiliaries; scored-only keeps the seven scored metrics with arrows; default `False` still renders auxiliaries; `scored_only=True` on `ExtractionGrades` is byte-identical to the default.
- The README narrativity block now shows the seven-metric arrowed view (`csr ↑: 0.27`, `dcpr ↓: 1.3`, ...) instead of the prior full `model_dump()` dump of all sixteen narrativity fields.
- The quickstart notebook Step 7 display reads as the same seven-metric view (auxiliaries available on the model instance for callers who want them).
- No change to the grade models themselves; the flag is render-time only.

## Rejected alternatives
- **Change the default to `scored_only=True`.** Rejected: hides the diagnostics paper-replicators want by default. The auxiliaries are cheap to capture and shipped on `NarrativityGrades` precisely so callers can inspect them; the default render should match that intent.
- **A separate `render_scored_grades` function.** Rejected: two functions for one boolean argument is API bloat. One function with one keyword argument carries the same information with half the surface, and the flag composes cleanly with the existing `extraction=` / `narrativity=` arguments.
- **A `scored: list[str] | None` filter on `_render_section` taking explicit field names.** Rejected: the direction dict is the source of truth for which fields are scored; a parallel filter list duplicates that information and risks drift when a new metric is added.
- **Expose the scored field set as a separate public attribute** (e.g. `EXTRACTION_SCORED_FIELDS = list(EXTRACTION_GRADE_DIRECTIONS)`). Rejected: same information, three lookup points instead of one. The dict's `keys()` already answers the question.

## References
- ADR 0008 - narrativity metrics: defines the seven scored metrics and nine auxiliaries.
- ADR 0024 - `render_grades` and the direction dicts (the source of truth this flag filters by).
- CLAUDE.md - the no-features-beyond-asked and abstraction rules drive the one-flag, single-function decision.
- `src/xains/metrics/render.py` - the flag.
- `tests/unit/test_metrics_render.py` - the four new pinning tests.
