"""Unit tests for grade_narrativity and NarrativityGrades."""

import sys

import pytest
from pydantic import ValidationError

from xains import NarrativityGrades, grade_narrativity

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


def test_grade_narrativity_populates_all_fields_for_complete_narrative() -> None:
    provider = _FakeProvider(list(_RICH_SCRIPT))
    grades = grade_narrativity(SAMPLE_TEXT, provider)
    assert isinstance(grades, NarrativityGrades)
    # 7 derived metrics
    assert grades.csr is not None
    assert grades.dcpr is not None
    assert grades.ccpr is not None
    assert grades.cecpr is not None
    assert grades.fdr is not None
    assert grades.ttcpr is not None
    assert grades.vcpr is not None
    # 9 auxiliaries
    assert grades.ppl_ordered is not None
    assert grades.ppl_shuffled is not None
    assert grades.decay_constant is not None
    assert grades.dist2 is not None
    assert grades.ttr is not None
    assert grades.vr is not None
    assert grades.cr is not None
    assert grades.cer is not None
    assert grades.n_sentences == 4


def test_grade_narrativity_n_sentences_recorded() -> None:
    provider = _FakeProvider(list(_RICH_SCRIPT))
    grades = grade_narrativity(SAMPLE_TEXT, provider)
    assert grades.n_sentences == 4


def test_grade_narrativity_handles_provider_failure_gracefully() -> None:
    """Provider returns None → PPL-dependent metrics None; text-only auxiliaries still populate."""
    provider = _ConstantProvider(value=None)
    grades = grade_narrativity(SAMPLE_TEXT, provider)
    # PPL-dependent metrics → None
    assert grades.csr is None
    assert grades.dcpr is None
    assert grades.ccpr is None
    assert grades.cecpr is None
    assert grades.fdr is None
    assert grades.ttcpr is None
    assert grades.vcpr is None
    assert grades.ppl_ordered is None
    assert grades.ppl_shuffled is None
    assert grades.decay_constant is None
    # Text-only auxiliaries still populate.
    assert grades.ttr is not None
    assert grades.dist2 is not None
    assert grades.vr is not None
    assert grades.cr is not None
    assert grades.cer is not None
    assert grades.n_sentences == 4


def test_grade_narrativity_handles_missing_nltk_gracefully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NLTK missing: sentence-dependent metrics + POS-tagging-dependent become None;
    word/bigram/lexicon-based auxiliaries still computable.
    """
    monkeypatch.setitem(sys.modules, "nltk", None)
    provider = _FakeProvider(list(_RICH_SCRIPT))
    grades = grade_narrativity(SAMPLE_TEXT, provider)
    # Sentence-dependent / POS-dependent → None
    assert grades.n_sentences is None
    assert grades.csr is None
    assert grades.dcpr is None
    assert grades.ccpr is None
    assert grades.cecpr is None
    assert grades.ttcpr is None
    assert grades.vcpr is None
    assert grades.vr is None
    # Pure-Python auxiliaries still work.
    assert grades.ttr is not None
    assert grades.dist2 is not None
    assert grades.cr is not None
    assert grades.cer is not None
    # fdr / ppl_ordered: implementation-dependent (could reuse cum_ppl[-1] or
    # call the provider directly). Intentionally not asserted.


def test_grade_narrativity_handles_missing_scipy_gracefully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """scipy missing: r-based metrics become None; CSR, FDR, and auxiliaries still compute."""
    monkeypatch.setitem(sys.modules, "scipy", None)
    monkeypatch.setitem(sys.modules, "scipy.optimize", None)
    provider = _FakeProvider(list(_RICH_SCRIPT))
    grades = grade_narrativity(SAMPLE_TEXT, provider)
    # r-based metrics → None (curve fit unavailable)
    assert grades.dcpr is None
    assert grades.ccpr is None
    assert grades.cecpr is None
    assert grades.ttcpr is None
    assert grades.vcpr is None
    assert grades.decay_constant is None
    # Non-r-based metrics still compute.
    assert grades.csr is not None
    assert grades.fdr is not None
    assert grades.ppl_ordered is not None
    assert grades.ppl_shuffled is not None
    # Auxiliaries.
    assert grades.n_sentences == 4
    assert grades.ttr is not None
    assert grades.dist2 is not None
    assert grades.vr is not None
    assert grades.cr is not None
    assert grades.cer is not None


def test_narrativity_grades_rejects_extra_fields() -> None:
    """NarrativityGrades has ConfigDict(extra='forbid')."""
    with pytest.raises(ValidationError):
        NarrativityGrades(  # type: ignore[call-arg]
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
