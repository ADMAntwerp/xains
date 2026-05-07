"""Unit tests for score_narrativity and NarrativityScores."""

import sys

import pytest
from pydantic import ValidationError

from xainarratives import NarrativityScores, score_narrativity

SAMPLE_TEXT = (
    "The applicant was predicted to default. "
    "High debt to income drove the prediction. "
    "However, younger age slightly offset this. "
    "Therefore, the loan was denied."
)
# 4 sentences with connectives, cause-effect markers, verbs, bigrams.


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


class _ConstantProvider:
    """Always returns the same value; for tests not bounded by call count."""

    def __init__(self, value: float | None) -> None:
        self._value = value
        self.calls = 0

    def compute(self, text: str) -> float | None:
        self.calls += 1
        return self._value


# Decreasing for cumulative (first 4) → r > 0. Extras for shuffled / fdr / margin.
_RICH_SCRIPT = [12.0, 10.0, 8.5, 7.5, 14.0, 10.0, 10.0, 10.0, 10.0, 10.0]


def test_score_narrativity_populates_all_fields_for_complete_narrative() -> None:
    provider = _FakeProvider(list(_RICH_SCRIPT))
    scores = score_narrativity(SAMPLE_TEXT, provider)
    assert isinstance(scores, NarrativityScores)
    # 7 derived metrics
    assert scores.csr is not None
    assert scores.dcpr is not None
    assert scores.ccpr is not None
    assert scores.cecpr is not None
    assert scores.fdr is not None
    assert scores.ttcpr is not None
    assert scores.vcpr is not None
    # 9 auxiliaries
    assert scores.ppl_ordered is not None
    assert scores.ppl_shuffled is not None
    assert scores.decay_constant is not None
    assert scores.dist2 is not None
    assert scores.ttr is not None
    assert scores.vr is not None
    assert scores.cr is not None
    assert scores.cer is not None
    assert scores.n_sentences == 4


def test_score_narrativity_n_sentences_recorded() -> None:
    provider = _FakeProvider(list(_RICH_SCRIPT))
    scores = score_narrativity(SAMPLE_TEXT, provider)
    assert scores.n_sentences == 4


def test_score_narrativity_handles_provider_failure_gracefully() -> None:
    """Provider returns None → PPL-dependent metrics None; text-only auxiliaries still populate."""
    provider = _ConstantProvider(value=None)
    scores = score_narrativity(SAMPLE_TEXT, provider)
    # PPL-dependent metrics → None
    assert scores.csr is None
    assert scores.dcpr is None
    assert scores.ccpr is None
    assert scores.cecpr is None
    assert scores.fdr is None
    assert scores.ttcpr is None
    assert scores.vcpr is None
    assert scores.ppl_ordered is None
    assert scores.ppl_shuffled is None
    assert scores.decay_constant is None
    # Text-only auxiliaries still populate.
    assert scores.ttr is not None
    assert scores.dist2 is not None
    assert scores.vr is not None
    assert scores.cr is not None
    assert scores.cer is not None
    assert scores.n_sentences == 4


def test_score_narrativity_handles_missing_nltk_gracefully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NLTK missing: sentence-dependent metrics + POS-tagging-dependent become None;
    word/bigram/lexicon-based auxiliaries still computable.
    """
    monkeypatch.setitem(sys.modules, "nltk", None)
    provider = _FakeProvider(list(_RICH_SCRIPT))
    scores = score_narrativity(SAMPLE_TEXT, provider)
    # Sentence-dependent / POS-dependent → None
    assert scores.n_sentences is None
    assert scores.csr is None
    assert scores.dcpr is None
    assert scores.ccpr is None
    assert scores.cecpr is None
    assert scores.ttcpr is None
    assert scores.vcpr is None
    assert scores.vr is None
    # Pure-Python auxiliaries still work.
    assert scores.ttr is not None
    assert scores.dist2 is not None
    assert scores.cr is not None
    assert scores.cer is not None
    # fdr / ppl_ordered: implementation-dependent (could reuse cum_ppl[-1] or
    # call the provider directly). Intentionally not asserted.


def test_score_narrativity_handles_missing_scipy_gracefully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """scipy missing: r-based metrics become None; CSR, FDR, and auxiliaries still compute."""
    monkeypatch.setitem(sys.modules, "scipy", None)
    monkeypatch.setitem(sys.modules, "scipy.optimize", None)
    provider = _FakeProvider(list(_RICH_SCRIPT))
    scores = score_narrativity(SAMPLE_TEXT, provider)
    # r-based metrics → None (curve fit unavailable)
    assert scores.dcpr is None
    assert scores.ccpr is None
    assert scores.cecpr is None
    assert scores.ttcpr is None
    assert scores.vcpr is None
    assert scores.decay_constant is None
    # Non-r-based metrics still compute.
    assert scores.csr is not None
    assert scores.fdr is not None
    assert scores.ppl_ordered is not None
    assert scores.ppl_shuffled is not None
    # Auxiliaries.
    assert scores.n_sentences == 4
    assert scores.ttr is not None
    assert scores.dist2 is not None
    assert scores.vr is not None
    assert scores.cr is not None
    assert scores.cer is not None


def test_narrativity_scores_rejects_extra_fields() -> None:
    """NarrativityScores has ConfigDict(extra='forbid')."""
    with pytest.raises(ValidationError):
        NarrativityScores(  # type: ignore[call-arg]
            csr=None,
            dcpr=None,
            ccpr=None,
            cecpr=None,
            fdr=None,
            ttcpr=None,
            vcpr=None,
            ppl_ordered=None,
            ppl_shuffled=None,
            decay_constant=None,
            dist2=None,
            ttr=None,
            vr=None,
            cr=None,
            cer=None,
            n_sentences=None,
            unknown_extra="bogus",
        )
