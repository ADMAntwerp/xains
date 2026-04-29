"""Unit tests for PerplexityProvider Protocol and concrete providers.

Note: the Protocol implementation in ``xainarratives.metrics.perplexity`` MUST
be decorated with ``@runtime_checkable`` — without it, every ``isinstance``
assertion in this file raises ``TypeError``. The two ``_implements_protocol``
tests below catch that regression as a ``TypeError`` at test time.
"""

from xainarratives.metrics import (
    APIPerplexityProvider,
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


# ----------------------------------------- APIPerplexityProvider


def test_api_perplexity_provider_calls_supplied_callable() -> None:
    """The provider forwards each compute() call to the supplied callable."""
    calls: list[str] = []

    def fake_fn(text: str) -> float | None:
        calls.append(text)
        return 42.0

    p = APIPerplexityProvider(perplexity_fn=fake_fn)
    p.compute("first")
    p.compute("second")
    assert calls == ["first", "second"]


def test_api_perplexity_provider_returns_callable_value() -> None:
    p = APIPerplexityProvider(perplexity_fn=lambda _text: 17.5)
    assert p.compute("x") == 17.5


def test_api_perplexity_provider_implements_protocol() -> None:
    p = APIPerplexityProvider(perplexity_fn=lambda _text: 1.0)
    assert isinstance(p, PerplexityProvider)


def test_api_perplexity_provider_propagates_none() -> None:
    """If the supplied callable returns None, the provider returns None."""
    p = APIPerplexityProvider(perplexity_fn=lambda _text: None)
    assert p.compute("anything") is None


# ----------------------------------------- Protocol structurality


def test_protocol_accepts_ad_hoc_conformer() -> None:
    """The Protocol is structural: any class with ``compute(text) -> float | None`` passes."""

    class _AdHoc:
        def compute(self, text: str) -> float | None:
            return 3.14 if text else None

    assert isinstance(_AdHoc(), PerplexityProvider)
