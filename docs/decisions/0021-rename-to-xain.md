# 0021. Rename package xainarratives -> xain

Date: 2026-06-17
Status: Accepted

## Context

The library was developed privately under the name `xainarratives`.
Ahead of a public release from the ADMAntwerp GitHub org and a planned
JOSS submission, the package needed a final, stable public name.
`xainarratives` is descriptive but long; the bare `xain` was verified
free on PyPI and as a GitHub repo.

## Decision

Rename the Python package, import name, repo, and PyPI distribution to
`xain`, and use `xain` as the sole name everywhere -- code, prose,
documentation, citation, README, and paper. No separate display brand
is retained. So the name is `xain` uniformly: `import xain`,
`pip install xain`, and "xain" in prose and citations. Authors recorded
as Mateusz Cedro and David Martens; copyright holder University of
Antwerp (institutional research software). Repository and Issues URLs
point at `github.com/ADMAntwerp/xain`.

## Rationale

- A single short name used consistently everywhere is simplest for
  users and for the codebase: one name to learn, type, import, and
  cite.
- `xain` verified available on PyPI (the federated-learning "XAIN"
  project renamed away to `xain-fl` / `xain-sdk` / `xain-aggregators`,
  leaving the bare name free) and as the ADMAntwerp repo.
- Institutional release (ADMAntwerp / University of Antwerp) over a
  personal account: stronger affiliation for the JOSS submission, ties
  the software to the lab the reference paper comes from, and gives the
  project continuity beyond any individual.
- Done pre-publication, while the repo is still private, so the public
  history is internally consistent from its first commit (no
  post-publication rename churn).

## Consequences

- BREAKING for any code using the old name:
  `from xainarratives import X` becomes `from xain import X`;
  `pip install "xainarratives[openai]"` becomes
  `pip install "xain[openai]"`. There are no external downstream users
  yet (repo never public under the old name), so the breakage is
  internal only.
- ADRs 0001-0020 and prior CHANGELOG entries reference the former name
  `xainarratives`; these are preserved verbatim as immutable history.
  This ADR records the transition. Readers of older ADRs should read
  `xainarratives` as the prior name of the `xain` package.
- The `src/xainarratives/` directory moved to `src/xain/` via
  `git mv`, preserving per-file history.

## Validation

Full chain green after the rename: ruff, ruff format, mypy (72 source
files), and 306 passed / 2 deselected - the same test count as before
the rename, including the missing-SDK ImportError test whose hint
string and matching regex both moved to `xain[openai]` in lockstep.

## References

- The renamed package `src/xain/` (was `src/xainarratives/`).
- `pyproject.toml` - name, authors, project URLs.
- `LICENSE` - copyright holder University of Antwerp.
- ADRs 0001-0020 - recorded under the prior package name.
