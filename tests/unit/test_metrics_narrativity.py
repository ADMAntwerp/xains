"""Unit tests for readability metric.

Readability is a Flesch reading-ease score over the original narrative text.
The ``extraction`` parameter is required by the metric signature for
symmetry with other metrics; it is not used in the PR 5 implementation.
"""

import sys

import pytest

from xainarratives import NarrativeExtraction
from xainarratives.metrics import readability


def _empty_extraction() -> NarrativeExtraction:
    return NarrativeExtraction(
        features={}, hallucinations=[], prompt_version="2", model_name="test"
    )


def test_readability_returns_float_for_nonempty_narrative() -> None:
    text = (
        "The applicant was predicted to default. The strongest driver was "
        "their high debt-to-income ratio. Their younger age slightly reduced "
        "the predicted probability."
    )
    score = readability(_empty_extraction(), text)
    assert score is not None
    assert isinstance(score, float)


def test_readability_returns_none_for_empty_narrative() -> None:
    assert readability(_empty_extraction(), "") is None
    assert readability(_empty_extraction(), "   ") is None


def test_readability_raises_clean_error_when_textstat_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forcing ``import textstat`` to fail yields a clear ImportError pointing at the extra."""
    monkeypatch.setitem(sys.modules, "textstat", None)
    with pytest.raises(ImportError, match=r'pip install "xainarratives\[textstat\]"'):
        readability(_empty_extraction(), "Some narrative text.")
