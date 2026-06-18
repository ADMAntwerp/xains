"""Unit tests for PerplexityProvider Protocol and concrete providers.

Note: the Protocol implementation in ``xain.metrics.perplexity`` MUST
be decorated with ``@runtime_checkable`` — without it, every ``isinstance``
assertion in this file raises ``TypeError``. The two ``_implements_protocol``
tests below catch that regression as a ``TypeError`` at test time.
"""

from xain.metrics import (
    DisabledProvider,
    PerplexityProvider,
)

# ----------------------------------------- DisabledProvider


def test_disabled_provider_always_returns_none() -> None:
    p = DisabledProvider()
    assert p.compute("any text") is None
    assert p.compute("") is None
    assert p.compute("a longer narrative with multiple sentences.") is None


def test_disabled_provider_implements_protocol() -> None:
    """Structural-typing check: DisabledProvider satisfies PerplexityProvider."""
    assert isinstance(DisabledProvider(), PerplexityProvider)


# ----------------------------------------- APIPerplexityProvider deletion


def test_api_perplexity_provider_deleted() -> None:
    """Confirm the abstract APIPerplexityProvider placeholder was removed in PR 7."""
    import xain.metrics.perplexity as ppl

    assert not hasattr(ppl, "APIPerplexityProvider")


# ----------------------------------------- Protocol structurality


def test_protocol_accepts_ad_hoc_conformer() -> None:
    """The Protocol is structural: any class with ``compute(text) -> float | None`` passes."""

    class _AdHoc:
        def compute(self, text: str) -> float | None:
            return 3.14 if text else None

    assert isinstance(_AdHoc(), PerplexityProvider)
