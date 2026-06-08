# 0014. Rename scoring API to `grade_*` / `*Grades`

Date: 2026-06-08
Status: Accepted

## Context

The metrics subpackage shipped two public functions and two public types
named with the prefix/suffix `score`:

- `score_extraction(...) -> ExtractionScores`
- `score_narrativity(...) -> NarrativityGrades`  (was `NarrativityScores`)

In ML/XAI contexts, the bare word "score" is ambiguous: it most commonly
denotes a prediction score, a confidence, or a probability — *outputs of
the model under explanation*, not assessments of the verbalization. This
library produces the latter (verbalization-fidelity + narrativity
assessments) and never the former. The name collision was a constant
low-grade friction in docstrings, prompts, and prose.

## Decision

Rename, as a breaking public-API change (pre-1.0, no compatibility shim):

| Old | New |
| --- | --- |
| `score_extraction` | `grade_extraction` |
| `score_narrativity` | `grade_narrativity` |
| `ExtractionScores` | `ExtractionGrades` |
| `NarrativityScores` | `NarrativityGrades` |

Also rename the source module `src/xainarratives/metrics/scorer.py` →
`grader.py` and the two test files `tests/unit/test_metrics_scorer.py` /
`tests/unit/test_narrativity_scorer.py` → `..._grader.py`, all via
`git mv` to preserve history.

Keep the functions pure (no `Grader` object). There is no state to bind:
each call takes its inputs (extraction / request / schema / narrative
text / optional perplexity provider) and returns a fresh `*Grades`
record. Wrapping them in a class would add a layer that holds nothing,
forces a two-step `Grader(...).grade(...)` call pattern on the caller,
and creates an abstraction with only one would-be implementation —
failing the project's "abstractions need ≥2 implementations" rule
(CLAUDE.md). The pure-function shape is simpler and equally testable.

## Rationale

- **"Grade" disambiguates from prediction-score / confidence.** A grade
  is what you give a piece of work after assessing it against criteria;
  that maps cleanly onto what these functions do to a generated
  narrative.
- **Types renamed for full lexical consistency.** Leaving the
  `*Scores` types while renaming the functions would make every call
  site read `grades = grade_extraction(...); assert isinstance(grades,
  ExtractionScores)` — half-renamed reads as a mistake, not a choice.
- **Variable-level scope-creep avoided.** Local variables named
  `scores` in test bodies (and notebook locals `ext_scores` /
  `narr_scores`) were renamed to `grades` for consistency. Bare prose
  uses of the words "score" / "scores" / "scorer" elsewhere
  (README verb, CHANGELOG noun, comments) were left intact — those are
  ordinary English, not references to the renamed API.

## Consequences

- Breaking: any caller importing `score_extraction`,
  `score_narrativity`, `ExtractionScores`, or `NarrativityScores`
  must update to the new names. Documented in CHANGELOG under
  `### Changed (BREAKING)`.
- Module path: `xainarratives.metrics.scorer` → `xainarratives.metrics.grader`.
  Top-level re-exports (`xainarratives.ExtractionGrades`, etc.) are
  unchanged in *path* — only the name changed.
- The notebook quickstart's narrative-grading cells reference the new
  names; the notebook needs re-execution against a live LLM to refresh
  outputs (out of scope for this commit, tracked separately).

## Rejected alternatives

- **Introduce a `Grader` object with a `.grade()` method.** Considered
  for symmetry with `Explainer`. Rejected: the functions are pure with
  no shared state worth binding, and there is no second implementation
  on the horizon (the integration is the integration). Inlining as
  pure functions follows simplicity-first + the ≥2-impl rule.
- **Rename only the functions, keep `ExtractionScores` /
  `NarrativityScores`.** Rejected: produces half-renamed call sites
  that read like a typo. Lexical consistency matters more than the
  smaller diff.
- **Keep `score_*` and add `grade_*` as aliases.** Rejected: pre-1.0,
  no external users, and the project policy is hard cutovers over
  shims (see ADR 0007's hard-cutover precedent for prompt version 1).

## References

- ADRs 0008 (narrativity metrics) and 0010 (quickstart notebook)
  introduced the now-renamed identifiers. Those ADRs are preserved as
  historical record using the names that existed when they were
  written; this ADR supersedes the naming choice without editing the
  earlier text.
- CLAUDE.md "Abstraction Rule": every Protocol/ABC needs ≥2
  implementations, or one + a committed plan for the second. Applied
  here to the rejected `Grader` class.
- The verbalization-fidelity / attribution-faithfulness distinction
  (CLAUDE.md "Vocabulary") — the renamed types live entirely on the
  verbalization side.
