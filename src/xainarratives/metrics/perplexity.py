"""Perplexity providers.

Perplexity is the one metric in this package that needs an external
language model. It is gated through the ``PerplexityProvider`` Protocol so
the metric layer never depends on a specific LLM SDK. Two concrete
implementations ship in v0:

* ``DisabledProvider`` — always returns ``None`` (default; no perplexity).
* ``APIPerplexityProvider`` — wraps a caller-supplied callable that returns
  a perplexity value. The callable is expected to compute perplexity from
  whatever the caller's LLM exposes (token logprobs, native perplexity
  endpoint, etc.); the provider does not care.
"""

from collections.abc import Callable
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


class APIPerplexityProvider:
    """Wraps a caller-supplied perplexity callable into the Protocol."""

    def __init__(self, perplexity_fn: Callable[[str], float | None]) -> None:
        self._fn = perplexity_fn

    def compute(self, text: str) -> float | None:
        return self._fn(text)
