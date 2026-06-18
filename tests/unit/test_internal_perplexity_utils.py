"""Unit tests for cumulative_perplexity helper."""

from xain.metrics._internal.perplexity_utils import cumulative_perplexity


class _FakeProvider:
    """Records received texts; returns scripted perplexities in order."""

    def __init__(self, scripted: list[float | None]) -> None:
        self._scripted = scripted
        self.received: list[str] = []

    def compute(self, text: str) -> float | None:
        idx = len(self.received)
        self.received.append(text)
        return self._scripted[idx]


def test_cumulative_perplexity_returns_n_values_for_n_sentences() -> None:
    sentences = ["First.", "Second.", "Third."]
    provider = _FakeProvider(scripted=[10.0, 8.0, 7.0])
    result = cumulative_perplexity(sentences, provider)
    assert result == [10.0, 8.0, 7.0]


def test_cumulative_perplexity_calls_provider_on_growing_prefixes() -> None:
    sentences = ["First.", "Second.", "Third."]
    provider = _FakeProvider(scripted=[10.0, 8.0, 7.0])
    cumulative_perplexity(sentences, provider)
    assert provider.received == [
        "First.",
        "First. Second.",
        "First. Second. Third.",
    ]


def test_cumulative_perplexity_short_circuits_on_none() -> None:
    """When the provider returns None, the helper returns None and stops calling."""
    sentences = ["First.", "Second.", "Third."]
    provider = _FakeProvider(scripted=[10.0, None, 7.0])
    result = cumulative_perplexity(sentences, provider)
    assert result is None
    # Stopped at the first None — the third call did not happen.
    assert len(provider.received) == 2


def test_cumulative_perplexity_empty_sentences_returns_empty_list() -> None:
    """Edge case: empty input → empty list, no provider calls."""
    provider = _FakeProvider(scripted=[])
    result = cumulative_perplexity([], provider)
    assert result == []
    assert provider.received == []
