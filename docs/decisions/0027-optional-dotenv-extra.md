# 0027. Optional dotenv extra for local key management

Date: 2026-06-23
Status: Accepted

## Context
xains providers read API keys from `os.environ`: `OpenAICompatibleProvider` (and its `OpenAIProvider` / `OpenRouterProvider` subclasses) resolves `api_key` from `os.environ.get(api_key_env_var)` at construction time, `OpenAICompatibleEchoProvider` (perplexity) does the same, and `AnthropicProvider` leaves resolution to the `anthropic` SDK which falls back to `ANTHROPIC_API_KEY` itself. Loading those env vars is the caller's job. In practice many users keep their keys in a project-local `.env` and load them with `python-dotenv`. The README setup docs need to describe that workflow honestly, but referencing `python-dotenv` without making the dependency available would push every user to discover and install it independently. At the same time CLAUDE.md restricts core runtime dependencies to `pydantic` only; adding `python-dotenv` to the core list is not on the table.

## Decision
Ship `python-dotenv` as an optional extra: `dotenv = ["python-dotenv>=1.0"]` in `pyproject.toml`. The library itself does not call `load_dotenv()` anywhere; runtime behaviour is unchanged. A `.env.example` lives at the repo root listing the four provider env-var names (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `TOGETHER_API_KEY`); `.env` itself is already gitignored. Users who want the workflow install with `pip install "xains[dotenv]"`, copy `.env.example` to `.env`, fill in the keys they need, and call `load_dotenv()` themselves before importing xains-using code. The README setup section documents this pattern.

## Rationale
- The dependency-discipline contract (ADR 0001, CLAUDE.md) keeps the core install at one runtime dep. `python-dotenv` is squarely an opt-in convenience, not a runtime requirement, so it belongs in an extra.
- Auto-loading `.env` inside library code is implicit and surprising. Library imports should not read the filesystem to mutate process state; that couples test isolation, container images, and CI pipelines to a file that may or may not exist. Keeping the call site explicit in the caller is honest and testable.
- Providing the dependency means README docs can recommend a concrete install command instead of hand-waving. Users who pick a different loader (`direnv`, `1Password CLI`, container secrets) are unaffected; the extra is purely additive.
- `.env.example` documents the env-var contract in one place. New providers added in the future should append their env-var names here.

## Consequences
- `pyproject.toml` gains one new optional-dependency group, `dotenv`. Core deps stay `pydantic>=2.6` alone.
- A tracked `.env.example` ships at the repo root. `.env` is already in `.gitignore`; `.env.example` is not, so it is committed.
- No `src/` code change. `grep -rn "dotenv" src/` returns nothing before and after this commit.
- Full chain unchanged: 318 tests still pass; ruff, format, mypy all clean.
- README setup docs can now reference the concrete install (`pip install "xains[dotenv]"`) and the example template.

## Rejected alternatives
- **Add `python-dotenv` to core runtime dependencies.** Rejected: violates the pydantic-only floor (ADR 0001, CLAUDE.md dependency discipline). The library does not need to load `.env` for any of its own behaviour.
- **Call `load_dotenv()` automatically inside `xains.__init__`.** Rejected: implicit filesystem read at import time, surprising in tests and containers, couples library import to caller's working directory, makes the library harder to reason about for users who deliberately do not use `.env`.
- **Ship `python-dotenv` inside an existing extra (e.g. `notebook` or `dev`).** Rejected: those extras have orthogonal purposes; a developer who wants `.env` support but does not want jupyter / shap / scikit-learn should not have to pull them in.
- **Skip the dependency, document `pip install python-dotenv` separately.** Rejected: leaves the install command outside the project's declared surface, no version floor recorded, no single command to set up a working environment.
- **Ship a tiny xains-internal helper (`xains.load_env()` wrapping `python-dotenv`).** Rejected: thin wrapper, no value over a direct `from dotenv import load_dotenv; load_dotenv()` call, and creates the impression that xains owns the loading lifecycle when it does not.

## References
- ADR 0001 - scope boundary and dependency discipline (pydantic-only core).
- CLAUDE.md - dependency discipline and simplicity rules.
- `pyproject.toml` - the `dotenv` extra.
- `.env.example` - the documented env-var template.
- `src/xains/providers/openai_compatible.py`, `src/xains/metrics/perplexity_api.py` - eager env-var reads at construction.
