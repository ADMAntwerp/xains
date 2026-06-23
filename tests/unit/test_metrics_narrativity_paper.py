"""Unit tests for the seven paper narrativity metrics: CSR, DCPR, CCPR,
CECPR, FDR, TTCPR, VCPR.

All seven follow ``metric(text, provider) -> float | None`` and degrade to
``None`` on degenerate inputs (empty text, too few sentences, denominator
zero, provider returning None, etc.) — never raising.
"""

import pytest

from xains.metrics.narrativity import (
    ccpr,
    cecpr,
    csr,
    dcpr,
    fdr,
    ttcpr,
    vcpr,
)

SAMPLE_TEXT = (
    "The applicant was predicted to default. "
    "High debt to income drove the prediction. "
    "However, younger age slightly offset this. "
    "Therefore, the loan was denied."
)
# 4 sentences. Has connectives (however, therefore), cause-effect markers
# (therefore), verbs (was, predicted, drove, offset, denied), bigrams.

NO_LEXICAL_TEXT = "Mountain river forest. Sun moon star. Tree leaf branch. Rock stone pebble."
# 4 sentences of pure-noun terms: no connectives, no cause-effect markers,
# no verbs (NLTK tags geographic/celestial/object nouns reliably as NN).


class _FakeProvider:
    """Returns scripted perplexities in order; raises if exhausted."""

    def __init__(self, scripted: list[float | None]) -> None:
        self._scripted = scripted
        self._index = 0

    def compute(self, text: str) -> float | None:
        if self._index >= len(self._scripted):
            raise IndexError(
                f"FakeProvider called {self._index + 1} times; scripted only {len(self._scripted)}"
            )
        result = self._scripted[self._index]
        self._index += 1
        return result


# ------------------------------------------------------ CSR


def test_csr_happy_path() -> None:
    """(PPL_shuffled - PPL_ordered) / PPL_ordered = (15 - 10) / 10 = 0.5."""
    provider = _FakeProvider([10.0, 15.0])
    assert csr(SAMPLE_TEXT, provider) == pytest.approx(0.5)


def test_csr_returns_none_when_provider_returns_none() -> None:
    """If either PPL call returns None, CSR returns None."""
    p1 = _FakeProvider([None, 15.0])
    p2 = _FakeProvider([10.0, None])
    assert csr(SAMPLE_TEXT, p1) is None
    assert csr(SAMPLE_TEXT, p2) is None


def test_csr_returns_none_with_single_sentence() -> None:
    """CSR requires at least 2 sentences to have something to shuffle."""
    provider = _FakeProvider([])  # never called
    assert csr("One sentence only.", provider) is None


def test_csr_uses_seed_42_deterministically() -> None:
    """csr() is deterministic given identical inputs (random.Random(42) shuffle)."""
    p1 = _FakeProvider([10.0, 15.0])
    p2 = _FakeProvider([10.0, 15.0])
    assert csr(SAMPLE_TEXT, p1) == csr(SAMPLE_TEXT, p2)


# ------------------------------------------------------ DCPR


def test_dcpr_happy_path() -> None:
    provider = _FakeProvider([12.0, 10.0, 8.5, 7.5])  # decreasing → b > 0
    result = dcpr(SAMPLE_TEXT, provider)
    assert result is not None
    assert result > 0


def test_dcpr_returns_none_when_provider_returns_none() -> None:
    provider = _FakeProvider([12.0, None, 8.5, 7.5])
    assert dcpr(SAMPLE_TEXT, provider) is None


def test_dcpr_returns_none_below_three_sentences() -> None:
    """fit_decay_constant requires N >= 3."""
    provider = _FakeProvider([10.0, 8.0])  # enough for 2-sentence text if called
    assert dcpr("First sentence. Second sentence.", provider) is None


# ------------------------------------------------------ CCPR


def test_ccpr_happy_path() -> None:
    provider = _FakeProvider([12.0, 10.0, 8.5, 7.5])
    result = ccpr(SAMPLE_TEXT, provider)
    assert result is not None
    assert result > 0


def test_ccpr_returns_none_when_provider_returns_none() -> None:
    provider = _FakeProvider([12.0, None, 8.5, 7.5])
    assert ccpr(SAMPLE_TEXT, provider) is None


def test_ccpr_returns_none_below_three_sentences() -> None:
    provider = _FakeProvider([10.0, 8.0])
    assert ccpr("First sentence. Second sentence.", provider) is None


def test_ccpr_returns_none_when_no_connectives() -> None:
    """No connective lexicon hits → cr=0 → None."""
    provider = _FakeProvider([12.0, 10.0, 8.5, 7.5])
    assert ccpr(NO_LEXICAL_TEXT, provider) is None


# ------------------------------------------------------ CECPR


def test_cecpr_happy_path() -> None:
    """SAMPLE_TEXT contains 'therefore' → cer > 0."""
    provider = _FakeProvider([12.0, 10.0, 8.5, 7.5])
    result = cecpr(SAMPLE_TEXT, provider)
    assert result is not None
    assert result > 0


def test_cecpr_returns_none_when_provider_returns_none() -> None:
    provider = _FakeProvider([12.0, None, 8.5, 7.5])
    assert cecpr(SAMPLE_TEXT, provider) is None


def test_cecpr_returns_none_below_three_sentences() -> None:
    provider = _FakeProvider([10.0, 8.0])
    assert cecpr("First sentence. Second sentence.", provider) is None


def test_cecpr_returns_none_when_no_cause_effect_markers() -> None:
    """No cause-effect markers → cer=0 → None (per locked decision)."""
    provider = _FakeProvider([12.0, 10.0, 8.5, 7.5])
    assert cecpr(NO_LEXICAL_TEXT, provider) is None


# ------------------------------------------------------ FDR


def test_fdr_happy_path() -> None:
    """FDR = dist2² / ln(PPL); ln(10) ≈ 2.303 > 0."""
    provider = _FakeProvider([10.0])
    result = fdr(SAMPLE_TEXT, provider)
    assert result is not None
    assert result > 0


def test_fdr_returns_none_when_provider_returns_none() -> None:
    provider = _FakeProvider([None])
    assert fdr(SAMPLE_TEXT, provider) is None


def test_fdr_returns_none_when_ppl_is_one() -> None:
    """ln(1) = 0 → division undefined → None."""
    provider = _FakeProvider([1.0])
    assert fdr(SAMPLE_TEXT, provider) is None


def test_fdr_returns_none_when_no_bigrams() -> None:
    """Single-word text → 0 bigrams → None."""
    provider = _FakeProvider([10.0])
    assert fdr("hello", provider) is None


# ------------------------------------------------------ TTCPR


def test_ttcpr_happy_path() -> None:
    provider = _FakeProvider([12.0, 10.0, 8.5, 7.5])
    result = ttcpr(SAMPLE_TEXT, provider)
    assert result is not None
    assert result > 0


def test_ttcpr_returns_none_when_provider_returns_none() -> None:
    provider = _FakeProvider([12.0, None, 8.5, 7.5])
    assert ttcpr(SAMPLE_TEXT, provider) is None


def test_ttcpr_returns_none_below_three_sentences() -> None:
    provider = _FakeProvider([10.0, 8.0])
    assert ttcpr("First sentence. Second sentence.", provider) is None


# ------------------------------------------------------ VCPR


def test_vcpr_happy_path() -> None:
    provider = _FakeProvider([12.0, 10.0, 8.5, 7.5])
    result = vcpr(SAMPLE_TEXT, provider)
    assert result is not None
    assert result > 0


def test_vcpr_returns_none_when_provider_returns_none() -> None:
    provider = _FakeProvider([12.0, None, 8.5, 7.5])
    assert vcpr(SAMPLE_TEXT, provider) is None


def test_vcpr_returns_none_below_three_sentences() -> None:
    provider = _FakeProvider([10.0, 8.0])
    assert vcpr("First sentence. Second sentence.", provider) is None


def test_vcpr_returns_none_when_no_verbs() -> None:
    """Pure-noun text → vr=0 → None."""
    provider = _FakeProvider([12.0, 10.0, 8.5, 7.5])
    assert vcpr(NO_LEXICAL_TEXT, provider) is None
