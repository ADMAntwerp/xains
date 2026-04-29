"""Unit tests for score_extraction and ExtractionScores.

The scorer is the integration point for all metrics: pure data in, scores
out. Perplexity is the only metric that touches anything provider-shaped,
and even there the provider is supplied by the caller.
"""

import sys
from typing import Any

import pytest
from pydantic import ValidationError

from xainarratives import (
    DatasetSchema,
    ExtractionScores,
    FeatureClaim,
    FeatureSchema,
    Modality,
    NarrativeExtraction,
    Prediction,
    TabularContribution,
    TabularExplanationRequest,
    TargetSchema,
)
from xainarratives.metrics import score_extraction

# ------------------------------------------------------ helpers


def _request(*contribs: tuple[str, Any, float]) -> TabularExplanationRequest:
    return TabularExplanationRequest(
        features={name: val for name, val, _ in contribs},
        prediction=Prediction(predicted_class=1),
        contributions=[
            TabularContribution(name=name, value=val, importance=imp) for name, val, imp in contribs
        ],
    )


def _schema(*names: str) -> DatasetSchema:
    return DatasetSchema(
        modality=Modality.TABULAR,
        name="test",
        description="test",
        target=TargetSchema(name="t", description="d", classes={0: "A", 1: "B"}),
        features=[FeatureSchema(name=n, dtype="numeric", description=f"feat {n}") for n in names],
    )


def _claim(name: str, *, rank: int, sign: int, value: Any = None) -> FeatureClaim:
    return FeatureClaim(
        rank=rank,
        sign=sign,
        value=value,
        narrative_name=name,
        resolved_to=name,
    )


def _extraction(
    features: dict[str, FeatureClaim],
    hallucinations: list[FeatureClaim] | None = None,
    prompt_version: str = "2",
) -> NarrativeExtraction:
    return NarrativeExtraction(
        features=features,
        hallucinations=hallucinations or [],
        prompt_version=prompt_version,
        model_name="test",
    )


class _FakeProvider:
    """Structural PerplexityProvider that records calls."""

    def __init__(self, return_value: float | None = 100.0) -> None:
        self._return_value = return_value
        self.calls: list[str] = []

    def compute(self, text: str) -> float | None:
        self.calls.append(text)
        return self._return_value


# ------------------------------------------------------ tests


def test_score_extraction_populates_all_fields_for_complete_extraction() -> None:
    request = _request(("dti", 0.41, 0.37), ("age", 29, -0.12))
    schema = _schema("dti", "age")
    extraction = _extraction(
        features={
            "dti": _claim("dti", rank=1, sign=1, value=0.41),
            "age": _claim("age", rank=2, sign=-1, value=29),
        }
    )
    narrative = "Higher dti pushed the applicant toward default. Younger age slightly offset this."
    fake = _FakeProvider(return_value=42.0)
    scores = score_extraction(
        extraction,
        request,
        schema,
        narrative_text=narrative,
        perplexity_provider=fake,
    )
    assert isinstance(scores, ExtractionScores)
    assert scores.sign_faithfulness == 1.0
    assert scores.value_faithfulness == 1.0
    assert scores.rank_correlation is not None
    assert scores.coverage == 1.0  # 2 of min(10, 2) = 2.
    assert scores.hallucination_count == 0
    assert scores.readability is not None
    assert scores.perplexity == 42.0


def test_score_extraction_uses_disabled_perplexity_by_default() -> None:
    request = _request(("dti", 0.41, 0.37))
    schema = _schema("dti")
    extraction = _extraction(features={"dti": _claim("dti", rank=1, sign=1)})
    scores = score_extraction(extraction, request, schema, narrative_text="Some narrative text.")
    assert scores.perplexity is None


def test_score_extraction_calls_supplied_perplexity_provider() -> None:
    request = _request(("dti", 0.41, 0.37))
    schema = _schema("dti")
    extraction = _extraction(features={"dti": _claim("dti", rank=1, sign=1)})
    fake = _FakeProvider(return_value=99.5)
    scores = score_extraction(
        extraction,
        request,
        schema,
        narrative_text="Some narrative text.",
        perplexity_provider=fake,
    )
    assert len(fake.calls) == 1
    assert scores.perplexity == 99.5


def test_score_extraction_with_no_resolved_features_sets_fidelity_metrics_none() -> None:
    request = _request(("dti", 0.41, 0.37))
    schema = _schema("dti")
    extraction = _extraction(features={})
    scores = score_extraction(extraction, request, schema, narrative_text="Some narrative text.")
    # Fidelity metrics are undefined when there's nothing to compare.
    assert scores.sign_faithfulness is None
    assert scores.value_faithfulness is None
    assert scores.rank_correlation is None
    # Coverage and hallucination_count are always defined.
    assert scores.coverage == 0.0
    assert scores.hallucination_count == 0


def test_score_extraction_forwards_prompt_version() -> None:
    request = _request(("dti", 0.41, 0.37))
    schema = _schema("dti")
    extraction = _extraction(
        features={"dti": _claim("dti", rank=1, sign=1)},
        prompt_version="2",
    )
    scores = score_extraction(extraction, request, schema, narrative_text="Some narrative text.")
    assert scores.prompt_version == "2"


def test_extraction_scores_rejects_extra_fields() -> None:
    """ExtractionScores has ConfigDict(extra='forbid')."""
    with pytest.raises(ValidationError):
        ExtractionScores(
            sign_faithfulness=None,
            value_faithfulness=None,
            rank_correlation=None,
            coverage=0.0,
            hallucination_count=0,
            readability=None,
            perplexity=None,
            prompt_version="2",
            unknown_extra="bogus",  # type: ignore[call-arg]
        )


def test_score_extraction_passes_narrative_text_to_perplexity_provider() -> None:
    request = _request(("dti", 0.41, 0.37))
    schema = _schema("dti")
    extraction = _extraction(features={"dti": _claim("dti", rank=1, sign=1)})
    narrative = "Exact text the provider should receive."
    fake = _FakeProvider()
    score_extraction(
        extraction,
        request,
        schema,
        narrative_text=narrative,
        perplexity_provider=fake,
    )
    assert fake.calls == [narrative]


def test_score_extraction_handles_missing_textstat_gracefully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When textstat is unavailable, readability=None but other metrics are populated."""
    monkeypatch.setitem(sys.modules, "textstat", None)
    request = _request(("dti", 0.41, 0.37))
    schema = _schema("dti")
    extraction = _extraction(features={"dti": _claim("dti", rank=1, sign=1)})
    scores = score_extraction(extraction, request, schema, narrative_text="Some narrative text.")
    assert scores.readability is None
    assert scores.coverage == 1.0
    assert scores.hallucination_count == 0
    assert scores.prompt_version == "2"
