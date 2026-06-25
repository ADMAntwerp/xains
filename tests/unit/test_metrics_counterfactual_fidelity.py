"""Unit tests for change_fidelity / cf_coverage / invented_features (ADR 0031).

Pure functions over (CounterfactualExtraction, request, schema). Ground
truth comes from build_scenarios(); each metric mirrors a fidelity.py /
coverage.py counterpart for the FI path.
"""

from typing import Any

from xains import (
    CounterfactualExtraction,
    CounterfactualFeatureClaim,
    DatasetSchema,
    FeatureSchema,
    Modality,
    Prediction,
    TabularContribution,
    TabularExplanationRequest,
    TargetSchema,
)
from xains.metrics import (
    cf_coverage,
    change_fidelity,
    invented_features,
)
from xains.types import TabularCounterfactual

# ------------------------------------------------------ schemas + fixtures


def _numeric_schema() -> DatasetSchema:
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
            FeatureSchema(name="salary", dtype="numeric", unit="EUR", description="salary"),
            FeatureSchema(name="dti", dtype="numeric", description="dti"),
        ],
    )


def _categorical_schema() -> DatasetSchema:
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
            FeatureSchema(
                name="housing",
                dtype="categorical",
                description="housing status",
                categories=["rent", "own", "free"],
            ),
            FeatureSchema(name="married", dtype="boolean", description="marital status"),
        ],
    )


def _request(
    features: dict[str, Any],
    cf_features: dict[str, Any],
    cf_predicted: int | str = 0,
) -> TabularExplanationRequest:
    return TabularExplanationRequest(
        features=features,
        prediction=Prediction(predicted_class=1),
        contributions=[
            TabularContribution(
                name=next(iter(features)), value=next(iter(features.values())), importance=0.1
            )
        ],
        counterfactual=TabularCounterfactual(predicted_class=cf_predicted, features=cf_features),
    )


def _extraction(
    changes: dict[str, CounterfactualFeatureClaim] | None = None,
    invented: list[CounterfactualFeatureClaim] | None = None,
) -> CounterfactualExtraction:
    return CounterfactualExtraction(
        changes=changes or {},
        invented=invented or [],
        prompt_version="1",
        model_name="test",
    )


def _claim(
    name: str,
    *,
    stated_before: Any = None,
    stated_after: Any = None,
    direction: str | None = None,
) -> CounterfactualFeatureClaim:
    return CounterfactualFeatureClaim(
        narrative_name=name,
        resolved_to=name,
        stated_before=stated_before,
        stated_after=stated_after,
        stated_direction=direction,
    )


# ====================================================== change_fidelity


def test_change_fidelity_both_sides_correct_numeric() -> None:
    schema = _numeric_schema()
    req = _request(
        {"age": 29, "salary": 52000, "dti": 0.41}, {"age": 29, "salary": 52000, "dti": 0.20}
    )
    extraction = _extraction(
        changes={"dti": _claim("dti", stated_before=0.41, stated_after=0.20)},
    )
    assert change_fidelity(extraction, req, schema) == 1.0


def test_change_fidelity_before_wrong_marks_incorrect() -> None:
    schema = _numeric_schema()
    req = _request({"age": 29, "dti": 0.41}, {"age": 29, "dti": 0.20})
    extraction = _extraction(
        changes={"dti": _claim("dti", stated_before=0.99, stated_after=0.20)},
    )
    assert change_fidelity(extraction, req, schema) == 0.0


def test_change_fidelity_after_wrong_marks_incorrect() -> None:
    schema = _numeric_schema()
    req = _request({"age": 29, "dti": 0.41}, {"age": 29, "dti": 0.20})
    extraction = _extraction(
        changes={"dti": _claim("dti", stated_before=0.41, stated_after=0.99)},
    )
    assert change_fidelity(extraction, req, schema) == 0.0


def test_partial_statement_before_none_marks_incorrect() -> None:
    """stated_before=None: the narrative did not commit to the BEFORE value -> incorrect."""
    schema = _numeric_schema()
    req = _request({"age": 29, "dti": 0.41}, {"age": 29, "dti": 0.20})
    extraction = _extraction(
        changes={"dti": _claim("dti", stated_before=None, stated_after=0.20)},
    )
    assert change_fidelity(extraction, req, schema) == 0.0


def test_partial_statement_after_none_marks_incorrect() -> None:
    """stated_after=None: the narrative did not commit to the AFTER value -> incorrect."""
    schema = _numeric_schema()
    req = _request({"age": 29, "dti": 0.41}, {"age": 29, "dti": 0.20})
    extraction = _extraction(
        changes={"dti": _claim("dti", stated_before=0.41, stated_after=None)},
    )
    assert change_fidelity(extraction, req, schema) == 0.0


def test_partial_statement_both_none_marks_incorrect() -> None:
    """Both sides None: the narrative resolved the feature but stated no values -> incorrect."""
    schema = _numeric_schema()
    req = _request({"age": 29, "dti": 0.41}, {"age": 29, "dti": 0.20})
    extraction = _extraction(
        changes={"dti": _claim("dti", stated_before=None, stated_after=None)},
    )
    assert change_fidelity(extraction, req, schema) == 0.0


def test_change_fidelity_numeric_with_string_stated_value_incorrect_no_crash() -> None:
    """Type-guard: a string where a number is expected -> incorrect, no exception."""
    schema = _numeric_schema()
    req = _request({"age": 29, "dti": 0.41}, {"age": 29, "dti": 0.20})
    extraction = _extraction(
        changes={"dti": _claim("dti", stated_before=0.41, stated_after="low")},
    )
    # Must not raise; must score 0.
    assert change_fidelity(extraction, req, schema) == 0.0


def test_change_fidelity_numeric_with_bool_stated_value_incorrect() -> None:
    """_is_numeric excludes bool; True/False on a numeric feature -> incorrect."""
    schema = _numeric_schema()
    req = _request({"age": 29, "dti": 0.41}, {"age": 29, "dti": 0.20})
    extraction = _extraction(
        changes={"dti": _claim("dti", stated_before=True, stated_after=False)},
    )
    assert change_fidelity(extraction, req, schema) == 0.0


def test_change_fidelity_categorical_equality_correct() -> None:
    schema = _categorical_schema()
    req = _request(
        {"housing": "rent", "married": False},
        {"housing": "own", "married": False},
    )
    extraction = _extraction(
        changes={"housing": _claim("housing", stated_before="rent", stated_after="own")},
    )
    assert change_fidelity(extraction, req, schema) == 1.0


def test_change_fidelity_categorical_equality_incorrect() -> None:
    schema = _categorical_schema()
    req = _request(
        {"housing": "rent", "married": False},
        {"housing": "own", "married": False},
    )
    extraction = _extraction(
        changes={"housing": _claim("housing", stated_before="rent", stated_after="free")},
    )
    assert change_fidelity(extraction, req, schema) == 0.0


def test_change_fidelity_boolean_equality_correct() -> None:
    schema = _categorical_schema()
    req = _request(
        {"housing": "rent", "married": False},
        {"housing": "rent", "married": True},
    )
    extraction = _extraction(
        changes={"married": _claim("married", stated_before=False, stated_after=True)},
    )
    assert change_fidelity(extraction, req, schema) == 1.0


def test_change_fidelity_categorical_with_number_stated_value_incorrect_no_crash() -> None:
    """Number stated for a categorical feature -> equality fails -> incorrect, no exception."""
    schema = _categorical_schema()
    req = _request(
        {"housing": "rent", "married": False},
        {"housing": "own", "married": False},
    )
    extraction = _extraction(
        changes={"housing": _claim("housing", stated_before=0, stated_after=1)},
    )
    assert change_fidelity(extraction, req, schema) == 0.0


def test_change_fidelity_ignores_claims_about_unchanged_features() -> None:
    """A claim about a feature that didn't actually change is not a change-fidelity question."""
    schema = _numeric_schema()
    # age is unchanged; dti is changed.
    req = _request({"age": 29, "dti": 0.41}, {"age": 29, "dti": 0.20})
    extraction = _extraction(
        changes={
            "dti": _claim("dti", stated_before=0.41, stated_after=0.20),  # correct + counts
            "age": _claim("age", stated_before=29, stated_after=99),  # ignored (unchanged)
        },
    )
    # Only dti is counted; correct -> 1/1 = 1.0
    assert change_fidelity(extraction, req, schema) == 1.0


def test_change_fidelity_returns_none_when_no_resolved_claims_about_changed_features() -> None:
    schema = _numeric_schema()
    req = _request({"age": 29, "dti": 0.41}, {"age": 29, "dti": 0.20})
    # Extraction has claims, but none of them are about the (actually-changed) dti.
    extraction = _extraction(
        changes={"age": _claim("age", stated_before=29, stated_after=99)},  # unchanged -> ignored
    )
    assert change_fidelity(extraction, req, schema) is None


def test_change_fidelity_returns_none_when_extraction_has_no_resolved_changes() -> None:
    schema = _numeric_schema()
    req = _request({"age": 29, "dti": 0.41}, {"age": 29, "dti": 0.20})
    extraction = _extraction(changes={})
    assert change_fidelity(extraction, req, schema) is None


def test_change_fidelity_invented_claims_not_counted() -> None:
    """The metric is over RESOLVED claims; invented features are scored by invented_features."""
    schema = _numeric_schema()
    req = _request({"age": 29, "dti": 0.41}, {"age": 29, "dti": 0.20})
    extraction = _extraction(
        changes={"dti": _claim("dti", stated_before=0.41, stated_after=0.20)},
        invented=[
            CounterfactualFeatureClaim(
                narrative_name="credit_history",
                resolved_to=None,
                stated_before="fair",
                stated_after="excellent",
            ),
        ],
    )
    assert change_fidelity(extraction, req, schema) == 1.0


def test_change_fidelity_mixed_correct_and_incorrect_fractional_score() -> None:
    schema = _numeric_schema()
    # CF changes age AND dti AND salary.
    req = _request(
        {"age": 29, "salary": 52000, "dti": 0.41},
        {"age": 35, "salary": 80000, "dti": 0.20},
    )
    extraction = _extraction(
        changes={
            "dti": _claim("dti", stated_before=0.41, stated_after=0.20),  # correct
            "age": _claim("age", stated_before=99, stated_after=35),  # incorrect (before wrong)
            "salary": _claim("salary", stated_before=52000, stated_after=80000),  # correct
        },
    )
    # 2 of 3 correct -> 0.666...
    result = change_fidelity(extraction, req, schema)
    assert result is not None
    assert abs(result - 2 / 3) < 1e-9


# ====================================================== cf_coverage


def test_cf_coverage_all_changed_features_resolved() -> None:
    schema = _numeric_schema()
    req = _request({"age": 29, "dti": 0.41}, {"age": 35, "dti": 0.20})
    extraction = _extraction(
        changes={
            "age": _claim("age", stated_before=29, stated_after=35),
            "dti": _claim("dti", stated_before=0.41, stated_after=0.20),
        },
    )
    assert cf_coverage(extraction, req, schema) == 1.0


def test_cf_coverage_partial_resolved() -> None:
    schema = _numeric_schema()
    # 2 features changed; only 1 resolved
    req = _request({"age": 29, "dti": 0.41}, {"age": 35, "dti": 0.20})
    extraction = _extraction(
        changes={"dti": _claim("dti", stated_before=0.41, stated_after=0.20)},
    )
    assert cf_coverage(extraction, req, schema) == 0.5


def test_cf_coverage_zero_when_no_resolved_claims() -> None:
    schema = _numeric_schema()
    req = _request({"age": 29, "dti": 0.41}, {"age": 35, "dti": 0.20})
    extraction = _extraction()
    assert cf_coverage(extraction, req, schema) == 0.0


def test_cf_coverage_unchanged_feature_claim_does_not_inflate() -> None:
    """A claim about an unchanged feature does not count toward coverage of changed features."""
    schema = _numeric_schema()
    req = _request({"age": 29, "dti": 0.41}, {"age": 29, "dti": 0.20})  # only dti changes
    extraction = _extraction(
        changes={
            "age": _claim("age", stated_before=29, stated_after=99),  # unchanged - doesn't count
        },
    )
    # 0 of 1 changed-feature resolved -> 0.0
    assert cf_coverage(extraction, req, schema) == 0.0


def test_cf_coverage_zero_when_cf_made_no_changes() -> None:
    """Degenerate: CF identical to factual -> no changed features -> 0.0 (always defined)."""
    schema = _numeric_schema()
    req = _request({"age": 29, "dti": 0.41}, {"age": 29, "dti": 0.41})
    extraction = _extraction()
    assert cf_coverage(extraction, req, schema) == 0.0


# ====================================================== invented_features


def test_invented_features_counts_invented_list() -> None:
    extraction = _extraction(
        invented=[
            CounterfactualFeatureClaim(narrative_name="X", resolved_to=None),
            CounterfactualFeatureClaim(narrative_name="Y", resolved_to=None),
        ],
    )
    assert invented_features(extraction) == 2


def test_invented_features_zero_when_none() -> None:
    extraction = _extraction()
    assert invented_features(extraction) == 0
