# 0024. Render grades with desired-direction arrows

Date: 2026-06-19
Status: Accepted

## Context
`ExtractionGrades` and `NarrativityGrades` carry a mix of metrics with opposite improvement directions: `coverage` and `sign_faithfulness` are higher-is-better, `hallucination_count` is lower-is-better, narrativity's `csr` / `fdr` are higher-is-better, the various `*cpr` metrics are lower-is-better, and the nine narrativity auxiliaries are diagnostic primitives with no "good direction" interpretation. Calling `print(grades)` produces the pydantic repr - flat `field=value` pairs with no signal about which direction is desirable for which metric. A reader has to look up the equation tags in the metric docstrings or the paper to know whether `dcpr=1.3` is encouraging or worrying. CLAUDE.md prohibits over-engineering: the goal is a small affordance that surfaces direction, not a formatting framework.

## Decision
Direction is plain data, stored as a module-level `dict[str, str]` next to each grade model: `EXTRACTION_GRADE_DIRECTIONS` in `xains.metrics.grader` and `NARRATIVITY_GRADE_DIRECTIONS` in `xains.metrics.narrativity`. The mapping is field-name to arrow glyph (`↑` or `↓`). A single public `render_grades(extraction=None, narrativity=None) -> str` in `xains.metrics.render` produces a grouped text block with two headers (`Verbalization fidelity` and `Narrativity`). Each scored metric renders as `name <arrow>: value`; auxiliary primitives render as `name: value` (no arrow); `prompt_version` is omitted. Either argument can be `None`. Both grades are iterated via `model_dump()` in declaration order. The whole thing is one function plus one private `_render_section` helper. Both direction dicts are exported from `xains.metrics`; `render_grades` is exported from both `xains.metrics` and top-level `xains`.

## Rationale
- Direction is data about the metric, not a property of the value or the field name. Encoding it as a dict keeps the metric definitions unchanged and lets future tooling (notebooks, CLIs, third-party dashboards) read the same mapping without re-deriving it from paper text.
- One concrete renderer is enough. Per CLAUDE.md's abstraction rule, a Protocol or ABC needs at least two concrete implementations in sight; we have one (text), no committed plan for a second (HTML / Markdown / Rich would each be a separate decision with its own ADR), so the renderer is a plain function.
- Auxiliary primitives stay un-arrowed because they are not "scored" - `ppl_ordered` / `ttr` / `n_sentences` are inputs to derived metrics, not optimization targets. ADR 0008 captured them as cheap-to-store diagnostics; arrowing them would invite misreading them as scores.
- `prompt_version` is metadata (a string identifier, not a metric), so it is excluded from the render entirely rather than printed without an arrow.
- The directions for the seven narrativity metrics match the equation tags already documented in each metric's docstring; we do not re-derive them here.

## Consequences
- New public module `xains.metrics.render` with one public function (`render_grades`) and one private helper (`_render_section`).
- Two new public dicts: `EXTRACTION_GRADE_DIRECTIONS` (5 entries) and `NARRATIVITY_GRADE_DIRECTIONS` (7 entries). They are the source of truth for desired-direction-per-metric.
- `xains.metrics.__all__` gains three names; `xains.__all__` gains one (`render_grades`).
- The README `ExtractionGrades` sample now uses `render_grades(extraction=grades)` and shows the arrowed output.
- Tests in `tests/unit/test_metrics_render.py` pin: scored fields render with arrows; auxiliaries render without; None values render as `None`; two headers are present in the right order; `prompt_version` is omitted; empty input returns `""`; single-grade calls omit the unused header.
- No change to the grade models themselves; arrows are render-time only.

## Rejected alternatives
- **Bake the arrow into the field name** (e.g. `sign_faithfulness_up`). Rejected: glyph creep in the public schema, every downstream consumer pays for a presentation choice in their type signatures, the auxiliaries would have to gain an `_aux` suffix or similar to opt out. Direction is presentation; field names are data.
- **Bake the arrow into the value** (e.g. emit `"1.0 ↑"` as a string). Rejected: turns a numeric field into a string, breaks `.model_dump()` consumers (notebooks, DataFrames), and forces every reader to parse the arrow off again.
- **A pluggable formatter Protocol** (e.g. `GradeFormatter.render(grades) -> str` with text / HTML / Rich implementations). Rejected on the CLAUDE.md abstraction rule: only one implementation exists today (text), and HTML / Rich / Markdown formatters are speculative. Adding the Protocol now creates an interface no second implementation needs to honor. Inline now; promote to a Protocol when a second renderer actually arrives.
- **Put the direction dict on the model as a `ClassVar`.** Rejected: pydantic's `model_config` is the canonical place for model metadata, and stuffing direction there couples the model to its presentation. A module-level dict next to the model is co-located without forcing the model itself to grow attributes.
- **Render `prompt_version` un-arrowed alongside the metrics.** Rejected: `prompt_version` is a string identifier (current value `"2"`), not a metric. Printing it next to the scored fields invites readers to treat it as one. Excluded entirely from the render.

## References
- ADR 0008 - narrativity metrics: the seven scored metrics and nine auxiliary primitives, plus their per-metric equation tags (the source of `NARRATIVITY_GRADE_DIRECTIONS`).
- ADR 0023 - removed `perplexity` from `ExtractionGrades`.
- ADR 0025 - removed `readability` from `ExtractionGrades`; the arrowed surface is the reduced six-field aggregate.
- CLAUDE.md - the simplicity-first and abstraction (>=2 implementations) rules drive the no-Protocol, one-renderer decision.
- `src/xains/metrics/render.py` - the renderer.
- `src/xains/metrics/grader.py` and `src/xains/metrics/narrativity.py` - the direction dicts.
