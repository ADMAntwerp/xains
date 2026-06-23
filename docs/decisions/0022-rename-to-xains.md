# 0022. Rename package xain -> xains

Date: 2026-06-19
Status: Accepted

## Context

ADR 0021 renamed the package from `xainarratives` to `xain`. On
attempting to reserve the distribution name on PyPI, `xain` proved
unavailable: the name is registered to another account (the dormant
federated-learning "XAIN" project family, which published `xain-fl` /
`xain-sdk` / `xain-aggregators` and registered the bare `xain` with no
releases). PyPI rejects uploads to a project name owned by another
account, and PEP 541 name-transfer is unlikely to succeed for a new
project that has workable alternative names available.

## Decision

Rename the package, import name, repo, and PyPI distribution from `xain`
to `xains`. `xains` was verified free on PyPI (both the /simple/ index
and the JSON API return 404) and is claimable now. The name is used
uniformly everywhere: `import xains`, `pip install xains`, repo
`github.com/ADMAntwerp/xains`. ADRs 0001-0021 and prior CHANGELOG
entries are preserved verbatim as immutable history; this ADR records
the transition.

## Rationale

- `xain` was unavailable on PyPI (registered, zero releases, owned by
  another account); `xains` is genuinely free across PyPI and as a
  GitHub repo.
- A single short name used consistently (package == import == pip ==
  repo) avoids the install/import mismatch a fallback dist-name would
  create.
- `xains` is distinct enough from the established "XAIN" federated-
  learning brand to reduce the collision and discoverability problems
  that bare `xain` carried.
- Done early, while the project is days old with no external users or
  citations, so the cost of the change is minimal compared to renaming
  after adoption.

## Consequences

- BREAKING relative to the brief `xain` period: `from xain import X`
  becomes `from xains import X`; `pip install` (never published as
  `xain`) is `pip install xains`. There were no external users under
  `xain` (the repo was public only briefly and never on PyPI), so the
  breakage is effectively internal.
- The `src/xain/` directory moved to `src/xains/` via git mv (after the
  earlier src/xainarratives -> src/xain move), preserving per-file
  history.
- ADR 0021's filename and body still reference `xain`; read it as the
  prior name in the rename chain xainarratives -> xain -> xains.

## Validation

Full chain green after the rename: ruff, ruff format, mypy (72 source
files), and 306 passed / 2 deselected, plus a live re-execution of the
quickstart notebook under the new name.

## References

- ADR 0021 - the prior rename (xainarratives -> xain).
- PEP 541 - PyPI name retention / transfer policy (why claiming `xain`
  was not pursued).
- pyproject.toml - name, URLs.
- src/xains/ - the renamed package.
