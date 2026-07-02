"""Unit tests for grade_hybrid + HybridGrades (ADR 0041).

Hybrid grading composes ``grade_extraction`` and ``grade_counterfactual``:
``HybridGrades`` HOLDS one ``ExtractionGrades`` and one
``CounterfactualGrades`` side by side (nested, not inherited). This
resolves the abstract-grades-base question ADR 0032 deferred to the
hybrid third case — the answer is compose, not inherit.

Both sub-grades are ``Optional`` so the grader can grade only the present
half; both-``None`` returns an empty ``HybridGrades`` with no raise
(mirroring the Explainer's partial-extraction dispatch, ADR 0040).
"""

from typing import Any

import pytest
from pydantic import ValidationError

from xains import (
    CounterfactualExtraction,
    CounterfactualFeatureClaim,
    CounterfactualGrades,
    DatasetSchema,
    ExtractionGrades,
    FeatureClaim,
    FeatureSchema,
    HybridGrades,
    Modality,
    NarrativeExtraction,
    Prediction,
    TabularContribution,
    TabularCounterfactual,
    TabularExplanationRequest,
    TargetSchema,
)
from xains.metrics import grade_counterfactual, grade_extraction, grade_hybrid, render_grades

# ------------------------------------------------------ helpers


def _schema() -> DatasetSchema:
    return DatasetSchema(
        modality=Modality.TABULAR,
        name="credit_risk",
        description="Credit risk demo.",
        target=TargetSchema(
            name="default",
            description="Default outcome.",
            classes={0: "Repaid", 1: "Defaulted"},
        ),
        features=[
            FeatureSchema(name="age", dtype="numeric", unit="years", description="age"),
            FeatureSchema(name="dti", dtype="numeric", description="dti"),
        ],
    )


def _request() -> TabularExplanationRequest:
    return TabularExplanationRequest(
        features={"age": 29, "dti": 0.41},
        prediction=Prediction(predicted_class=1),
        contributions=[
            TabularContribution(name="dti", value=0.41, importance=0.37),
            TabularContribution(name="age", value=29, importance=-0.12),
        ],
        counterfactual=TabularCounterfactual(predicted_class=0, features={"age": 29, "dti": 0.20}),
    )


def _fi_claim(name: str, *, rank: int, sign: int, value: Any = None) -> FeatureClaim:
    return FeatureClaim(
        rank=rank,
        sign=sign,
        value=value,
        narrative_name=name,
        resolved_to=name,
    )


def _fi_extraction() -> NarrativeExtraction:
    return NarrativeExtraction(
        features={
            "dti": _fi_claim("dti", rank=1, sign=1, value=0.41),
            "age": _fi_claim("age", rank=2, sign=-1, value=29),
        },
        hallucinations=[],
        prompt_version="2",
        model_name="test",
    )


def _cf_claim(
    name: str, *, stated_before: Any = None, stated_after: Any = None
) -> CounterfactualFeatureClaim:
    return CounterfactualFeatureClaim(
        narrative_name=name,
        resolved_to=name,
        stated_before=stated_before,
        stated_after=stated_after,
    )


def _cf_extraction() -> CounterfactualExtraction:
    return CounterfactualExtraction(
        changes={"dti": _cf_claim("dti", stated_before=0.41, stated_after=0.20)},
        invented=[],
        prompt_version="1",
        model_name="test",
    )


# ====================================================== grade_hybrid: compose correctness


def test_grade_hybrid_with_both_present_matches_standalone_grades() -> None:
    """Both halves present -> each sub-grade equals what the standalone grader returns."""
    schema = _schema()
    req = _request()
    fi_ext = _fi_extraction()
    cf_ext = _cf_extraction()
    narrative = "Higher dti drove the default. Lowering dti to 0.20 would flip it."

    hg = grade_hybrid(fi_ext, cf_ext, req, schema, narrative_text=narrative)

    assert isinstance(hg, HybridGrades)
    assert hg.feature_importance == grade_extraction(fi_ext, req, schema, narrative_text=narrative)
    assert hg.counterfactual == grade_counterfactual(cf_ext, req, schema)


def test_grade_hybrid_forwards_k_to_extraction_grader() -> None:
    """``k`` must be threaded to grade_extraction (affects coverage)."""
    schema = _schema()
    req = _request()
    fi_ext = _fi_extraction()
    cf_ext = _cf_extraction()

    hg = grade_hybrid(fi_ext, cf_ext, req, schema, narrative_text="n", k=1)
    expected_fi = grade_extraction(fi_ext, req, schema, narrative_text="n", k=1)
    assert hg.feature_importance == expected_fi


# ====================================================== partial grading


def test_grade_hybrid_only_fi_extraction_present_leaves_cf_none() -> None:
    schema = _schema()
    req = _request()
    fi_ext = _fi_extraction()

    hg = grade_hybrid(fi_ext, None, req, schema, narrative_text="n")

    assert hg.feature_importance is not None
    assert isinstance(hg.feature_importance, ExtractionGrades)
    assert hg.counterfactual is None


def test_grade_hybrid_only_cf_extraction_present_leaves_fi_none() -> None:
    schema = _schema()
    req = _request()
    cf_ext = _cf_extraction()

    hg = grade_hybrid(None, cf_ext, req, schema, narrative_text="n")

    assert hg.feature_importance is None
    assert hg.counterfactual is not None
    assert isinstance(hg.counterfactual, CounterfactualGrades)


def test_grade_hybrid_both_none_returns_empty_hybridgrades_no_raise() -> None:
    schema = _schema()
    req = _request()

    hg = grade_hybrid(None, None, req, schema, narrative_text="n")

    assert isinstance(hg, HybridGrades)
    assert hg.feature_importance is None
    assert hg.counterfactual is None


# ====================================================== HybridGrades shape


def test_hybrid_grades_rejects_extra_fields() -> None:
    """HybridGrades has ConfigDict(extra='forbid'), matching the other two grade models."""
    with pytest.raises(ValidationError):
        HybridGrades(
            feature_importance=None,
            counterfactual=None,
            bogus="x",  # type: ignore[call-arg]
        )


def test_hybrid_grades_defaults_are_none() -> None:
    """Both sub-fields default to None so ``HybridGrades()`` is the empty container."""
    hg = HybridGrades()
    assert hg.feature_importance is None
    assert hg.counterfactual is None


# ====================================================== render round-trip


def test_hybrid_grades_render_via_unpack_produces_both_sections() -> None:
    """Documented usage: ``render_grades(extraction=hg.feature_importance,
    counterfactual=hg.counterfactual)`` renders both sections."""
    schema = _schema()
    req = _request()
    hg = grade_hybrid(_fi_extraction(), _cf_extraction(), req, schema, narrative_text="n")

    rendered = render_grades(
        extraction=hg.feature_importance,
        counterfactual=hg.counterfactual,
    )
    assert "Verbalization fidelity" in rendered
    assert "Counterfactual fidelity" in rendered
