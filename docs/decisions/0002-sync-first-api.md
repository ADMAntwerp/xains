# 0002. Sync as the canonical API

Date: 2026-04-23
Status: Accepted

## Context

LLM calls are I/O-bound and benefit from async (batching, concurrency,
streaming). But async-first APIs impose ergonomic cost on notebook users
and the bulk of the near-term user base is ML engineers iterating
interactively.

## Decision

The canonical `Explainer.explain()` is synchronous. Async will be
considered only when a concrete need (batch throughput in production
serving, streaming UIs) arises, and will be added as a separate thin
wrapper, not by rewriting the core.

## Consequences

- Simpler code, simpler tests, simpler stack traces.
- Notebook users don't need `await` / `asyncio.run`.
- Batch throughput for very large runs will be worse than an async-native
  design would allow. Acceptable until measured to be a problem.

## Alternatives considered

- Async-first with `asyncio.run` wrappers for sync. Rejected for v0 — the
  async-first ergonomic cost is real and we have no evidence we need the
  throughput yet.
