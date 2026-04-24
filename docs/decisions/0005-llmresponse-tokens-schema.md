# 0005. LLMResponse.tokens_used schema: {"input", "output", "total"}

Date: 2026-04-23
Status: Accepted

## Context

`LLMResponse.tokens_used` is typed `dict[str, int] | None`. The key set was
not pinned in PR 1; the first concrete provider beyond the mock
(`AnthropicProvider`) is about to land, and `OpenAIProvider`,
`LiteLLMProvider`, and `OllamaProvider` are committed follow-ups.

Different backends report usage differently:

- Anthropic Messages API: `usage.input_tokens`, `usage.output_tokens`. No
  total.
- OpenAI Chat Completions: `usage.prompt_tokens`, `usage.completion_tokens`,
  `usage.total_tokens`.
- LiteLLM: proxies the upstream provider's shape, typically OpenAI-shaped
  but not guaranteed.
- Ollama: `prompt_eval_count`, `eval_count`.

If each provider passes its native dict through, downstream code
(evaluators, cost estimators, audit logs) has to branch per provider and
silently drifts.

## Decision

`tokens_used` is one of:

- `None` — the provider did not report usage, or usage data was
  unavailable on this call. Consumers MUST handle `None`.
- A dict with **exactly** these three string keys:
  - `"input"` — prompt / input tokens, `int >= 0`.
  - `"output"` — completion / output tokens, `int >= 0`.
  - `"total"` — `input + output`, `int >= 0`. Always equal to the sum.

Providers normalize at the boundary. Providers compute `total`; callers
MUST NOT recompute or derive it.

If a provider's native response is missing either `input` or `output`,
`tokens_used` is `None`. Partial dicts are not permitted.

## Consequences

- Every new provider implementation includes the translation to this
  schema and a test asserting the three-key contract.
- `AnthropicProvider` computes `total = input_tokens + output_tokens`.
- `OpenAIProvider` (future) maps `prompt_tokens -> input`,
  `completion_tokens -> output`, `total_tokens -> total` and asserts the
  sum identity before returning.
- Evaluators and cost estimators are provider-agnostic: one key lookup
  works everywhere.
- The schema is narrow on purpose. Cache hits, reasoning tokens, tool
  tokens, and other provider-specific usage breakdowns are NOT represented
  here. If that data is needed later, it goes on a separate field (e.g.
  `tokens_used_detail: dict[str, int] | None`) so `tokens_used` stays
  stable.

## Alternatives considered

- **Pass-through native dict.** Rejected: forces every consumer to branch
  per provider; the whole point of the Protocol is that callers do not
  care which backend served the call.
- **Keys `"prompt"` / `"completion"` (OpenAI vocabulary).** Rejected:
  Anthropic's `input` / `output` names are the more neutral pair and the
  ones the Messages API uses.
- **Omit `total`, let callers sum.** Rejected: total is cheap, and pinning
  it in the schema eliminates a class of off-by-one / missing-key bugs in
  cost reporting downstream.
