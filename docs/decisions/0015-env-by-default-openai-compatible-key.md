# 0015. Env-by-default API key for `OpenAICompatibleEchoProvider`

Date: 2026-05-22
Status: Accepted

## Context

`AnthropicProvider` constructs its SDK client without an explicit key —
the `anthropic` SDK reads `ANTHROPIC_API_KEY` from the environment
automatically. Callers never need to write
`api_key=os.environ["ANTHROPIC_API_KEY"]`.

`OpenAICompatibleEchoProvider` required `api_key` as a positional
argument. Calling it forced `os.environ[...]` boilerplate at every call
site, including the quickstart notebook. The asymmetry between the two
providers was visible to anyone reading them side by side, and made the
code samples in the README harder to keep consistent.

The wrinkle: `OpenAICompatibleEchoProvider` is provider-agnostic. Its
own docstring commits the class to supporting Together.ai, vLLM, TGI's
OpenAI shim, and OpenAI's legacy `/v1/completions`. There is no single
"correct" env var to hardcode.

## Decision

Make `api_key` optional and keyword-only, with environment-variable
fallback:

```python
def __init__(
    self,
    base_url: str,
    *,
    api_key: str | None = None,
    api_key_env_var: str = "OPENAI_API_KEY",
    model: str,
    timeout: float = 30.0,
) -> None:
```

Resolution order at construction time:

1. If `api_key` is passed, use it.
2. Else read `os.environ[api_key_env_var]`.
3. If neither yields a value, raise `ValueError` naming the env var
   that was checked.

Default env var is `OPENAI_API_KEY` (matching the OpenAI SDK
convention). Together users pass `api_key_env_var="TOGETHER_API_KEY"`.
vLLM users typically pass `api_key="not-required"` (or any non-empty
string) since the local server ignores the key.

## Rationale

- **Consistency with `AnthropicProvider`.** Both providers now default
  to env-based key resolution. Notebook and README examples stop
  reaching for `os.environ[...]` boilerplate.
- **Option B (configurable env var) chosen over Option A (hardcoded
  `TOGETHER_API_KEY`).** A hardcoded Together-specific fallback would
  be wrong for vLLM / TGI / OpenAI users — the docstring already
  commits the class to provider-agnosticism. Option A would leak the
  notebook's choice of provider into the library surface.
- **Option C (rely on the OpenAI SDK's built-in `OPENAI_API_KEY`
  auto-read) rejected.** Setting `OPENAI_API_KEY=<together key>` is
  misleading, collides with the real OpenAI env var in shared shells,
  and fails for anyone running both OpenAI and a self-hosted endpoint
  in the same process. One line of explicit configuration
  (`api_key_env_var=...`) buys unambiguous semantics.
- **Keyword-only past `base_url`.** Prevents the "two adjacent string
  arguments, easy to swap" bug between `api_key` and
  `api_key_env_var`. All four existing call sites (3 test fixtures +
  the notebook cell) already pass everything as kwargs, so this is a
  no-op for current usage; future positional callers get a loud
  `TypeError` instead of a silent type-misbinding.
- **`ValueError` that names the env var.** The error message tells the
  user which env var was checked, so a misconfigured `OPENAI_API_KEY`
  (when the provider was set to read `TOGETHER_API_KEY`) is
  diagnosable at a glance.

## Consequences

- Breaking signature change. Positional callers (none in repo) now get
  `TypeError`. The three test fixtures and the notebook cell already
  use kwargs, so the only required source-code update is the notebook
  switching to demonstrate the new env-by-default pattern.
- `api_key` is no longer a required argument. Users who don't pass it
  must set the env var (`OPENAI_API_KEY` by default, or whichever name
  they pass via `api_key_env_var`); construction fails fast with a
  named `ValueError` if neither is set.
- The notebook quickstart now uses
  `OpenAICompatibleEchoProvider(base_url=..., model=...,
  api_key_env_var="TOGETHER_API_KEY")` — no `os.environ[...]` in the
  call. The cell-1 guard
  `if "TOGETHER_API_KEY" not in os.environ: raise KeyError(...)` is
  preserved; it now fails fast on the same variable the provider reads.

## Rejected alternatives

- **A: Hardcode `TOGETHER_API_KEY` as the fallback.** Rejected: leaky
  for a class the docstring commits to supporting Together, vLLM, TGI,
  and OpenAI. Non-Together users would be surprised.
- **C: Rely on the OpenAI SDK's built-in `OPENAI_API_KEY` auto-read.**
  Rejected as above — forces non-OpenAI users to set the OpenAI env
  var to a non-OpenAI key.
- **Drop `api_key_env_var`; only fall back to `OPENAI_API_KEY`.**
  Equivalent to Option C; same reasons reject it.
- **Introduce a `BaseAPIProvider` with shared env-fallback logic.**
  Rejected: only two providers, and the Anthropic SDK already handles
  its env-read internally — the abstraction would have one in-repo
  implementation. Violates the CLAUDE.md "abstractions need >=2
  implementations" rule.

## References

- `AnthropicProvider.__init__` — the comparison point. The `anthropic`
  SDK reads `ANTHROPIC_API_KEY` automatically, so the provider passes
  the key through implicitly.
- `src/xainarratives/metrics/perplexity_api.py` class docstring —
  enumerates the supported endpoints (Together, vLLM, TGI, OpenAI),
  the commitment this ADR honors.
- CLAUDE.md "Abstraction Rule" — applied to reject a shared
  `BaseAPIProvider`.
