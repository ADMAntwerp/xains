"""Perplexity providers.

Perplexity is the one metric in this package that needs an external
language model. It is gated through the ``PerplexityProvider`` Protocol so
the metric layer never depends on a specific LLM SDK.

Concrete providers shipped:

* ``DisabledProvider`` (this module) — always returns ``None`` (default;
  no perplexity computed).
* ``HuggingFacePerplexityProvider`` (``perplexity_hf``) — local
  autoregressive model via transformers + torch.
* ``OpenAICompatibleEchoProvider`` (``perplexity_api``) — hits any
  ``/v1/completions`` endpoint with ``echo=True, logprobs=1`` (Together,
  vLLM, TGI's OpenAI shim, OpenAI legacy completions).
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class PerplexityProvider(Protocol):
    """Sync text → perplexity adapter."""

    def compute(self, text: str) -> float | None:
        """Return the sequence perplexity of ``text``, or ``None`` if unavailable."""
        ...


class DisabledProvider:
    """Always returns ``None``. Use as the default when perplexity is not wanted."""

    def compute(self, text: str) -> float | None:
        return None
