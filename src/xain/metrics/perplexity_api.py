"""OpenAICompatibleEchoProvider — perplexity via OpenAI-compatible /v1/completions.

Computes mean negative log-probability over the input text by sending a
``POST /completions`` request with ``echo=True, logprobs=1, max_tokens=1``.
The response includes per-token logprobs for the input itself, from which
sequence perplexity is derived as ``exp(-mean(logprobs))``.

Compatible endpoints (any service speaking the OpenAI ``/v1/completions``
wire format with echo + logprobs support):

* Together.ai (``base_url="https://api.together.xyz/v1"``).
* vLLM (``--model ...`` then ``base_url="http://localhost:8000/v1"``).
* TGI's OpenAI-compatible shim.
* OpenAI's legacy ``/v1/completions`` (``gpt-3.5-turbo-instruct`` only;
  on a deprecation path).

Optional dependency: ``pip install "xain[perplexity-api]"`` (openai SDK).

Per the ``PerplexityProvider`` Protocol contract, all API errors
(``openai.OpenAIError`` and subclasses) are mapped to ``None`` rather than
raised. Callers wanting to surface specific errors should call the openai
SDK directly.
"""

import math
import os
from typing import Any

_MISSING_OPENAI_MESSAGE = (
    "The 'openai' package is required for OpenAICompatibleEchoProvider. "
    'Install with: pip install "xain[perplexity-api]"'
)


class OpenAICompatibleEchoProvider:
    """Perplexity via the OpenAI ``/v1/completions`` echo + logprobs pattern.

    API key resolution: pass ``api_key=`` explicitly, or leave it ``None``
    and the value is read from ``os.environ[api_key_env_var]``. The default
    env var is ``OPENAI_API_KEY``; for Together, pass
    ``api_key_env_var="TOGETHER_API_KEY"``. If neither path yields a key,
    ``__init__`` raises ``ValueError`` naming the env var that was checked.

    Paper-replication caveat: Cedro & Martens 2026 uses
    ``meta-llama/Llama-3.1-8B`` (the base model). That model is **not** on
    Together's serverless catalog; reproducing the paper's numbers via the
    API requires either a dedicated Together endpoint or a different
    provider entirely — typically local
    ``HuggingFacePerplexityProvider(model_name="meta-llama/Llama-3.1-8B")``
    (subject to the Hugging Face gating).

    Serverless callers should pick a model from Together's current public
    catalog. As of 2026-05, working choices include:

    * ``meta-llama/Meta-Llama-3-8B-Instruct-Lite``
    * ``Qwen/Qwen2.5-7B-Instruct-Turbo``
    * ``openai/gpt-oss-20b``
    """

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        api_key_env_var: str = "OPENAI_API_KEY",
        model: str,
        timeout: float = 30.0,
    ) -> None:
        try:
            import openai
        except ImportError as exc:
            raise ImportError(_MISSING_OPENAI_MESSAGE) from exc

        resolved_key = api_key if api_key is not None else os.environ.get(api_key_env_var)
        if resolved_key is None:
            raise ValueError(
                f"No API key for OpenAICompatibleEchoProvider: pass api_key=... "
                f"or set the {api_key_env_var} environment variable."
            )

        self._client: Any = openai.OpenAI(base_url=base_url, api_key=resolved_key, timeout=timeout)
        self._model: str = model

    def compute(self, text: str) -> float | None:
        """Return the perplexity of ``text``, or ``None`` for degenerate inputs / API errors."""
        if not text.strip():
            return None

        import openai

        try:
            response = self._client.completions.create(
                model=self._model,
                prompt=text,
                echo=True,
                logprobs=1,
                max_tokens=1,
                temperature=0,
            )
        except openai.OpenAIError:
            return None

        # Together.ai returns echo logprobs in two shapes depending on the
        # model. Try Shape B (response.prompt[...]) first — logprobs are
        # semantically attached to the echoed prompt tokens, not the
        # (1-token, temperature=0) generated continuation — and fall back to
        # Shape A (response.choices[...], OpenAI standard). The openai SDK's
        # Completion Pydantic model doesn't expose ``prompt`` as a typed
        # attribute, so we drop to model_dump() for the dual-shape probe.
        token_logprobs: list[float | None] | None = None

        raw: dict[str, Any] = response.model_dump() if hasattr(response, "model_dump") else {}
        prompt_field = raw.get("prompt")
        if isinstance(prompt_field, list) and prompt_field:
            first = prompt_field[0]
            if isinstance(first, dict):
                lp = first.get("logprobs")
                if isinstance(lp, dict):
                    candidate = lp.get("token_logprobs")
                    if candidate:
                        token_logprobs = candidate

        if token_logprobs is None:
            try:
                token_logprobs = response.choices[0].logprobs.token_logprobs
            except (AttributeError, IndexError):
                token_logprobs = None

        if not token_logprobs:
            return None

        valid = [lp for lp in token_logprobs if lp is not None]
        if not valid:
            return None

        mean_neg_logprob = -sum(valid) / len(valid)

        try:
            ppl = math.exp(mean_neg_logprob)
        except OverflowError:
            return None

        if math.isnan(ppl) or math.isinf(ppl):
            return None

        return ppl
