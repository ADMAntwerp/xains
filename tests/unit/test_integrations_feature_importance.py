"""Unit tests for xainarratives.integrations.feature_importance.

The adapter is a pure data-shape translator — no network, no numpy, no
upstream XAI libs. These tests pin the edge-case behaviour agreed in PR 3.
"""

import math

import pytest

from xainarratives import (
    Prediction,
    TabularContribution,
    TabularCounterfactual,
    TabularExplanationRequest,
)
from xainarratives.integrations import from_feature_importance

# ---------------------------------------------------------------- happy path


def test_happy_path_returns_tabular_request() -> None:
    req = from_feature_importance(
        features={"age": 29, "dti": 0.41},
        importances={"age": -0.12, "dti": 0.37},
        prediction=Prediction(predicted_class=1),
    )
    assert isinstance(req, TabularExplanationRequest)
    assert req.features == {"age": 29, "dti": 0.41}
    assert len(req.contributions) == 2
    assert all(isinstance(c, TabularContribution) for c in req.contributions)


def test_contribution_count_matches_importances_not_features() -> None:
    req = from_feature_importance(
        features={"age": 29, "dti": 0.41, "income": 50000},
        importances={"dti": 0.37},  # only one attributed
        prediction=Prediction(predicted_class=1),
    )
    assert set(req.features.keys()) == {"age", "dti", "income"}
    assert [c.name for c in req.contributions] == ["dti"]


def test_tabular_contribution_value_matches_features_dict() -> None:
    req = from_feature_importance(
        features={"age": 29, "dti": 0.41},
        importances={"age": -0.12, "dti": 0.37},
        prediction=Prediction(predicted_class=1),
    )
    by_name = {c.name: c for c in req.contributions}
    assert by_name["age"].value == 29
    assert by_name["dti"].value == 0.41


# ------------------------------------------------------------------ errors


def test_importance_for_unknown_feature_raises() -> None:
    with pytest.raises(ValueError, match="ghost"):
        from_feature_importance(
            features={"age": 29},
            importances={"age": 0.1, "ghost": 0.5},
            prediction=Prediction(predicted_class=1),
        )


def test_empty_importances_raises() -> None:
    with pytest.raises(ValueError, match="at least one"):
        from_feature_importance(
            features={"age": 29},
            importances={},
            prediction=Prediction(predicted_class=1),
        )


def test_nan_importance_raises() -> None:
    with pytest.raises(ValueError, match="age"):
        from_feature_importance(
            features={"age": 29},
            importances={"age": math.nan},
            prediction=Prediction(predicted_class=1),
        )


def test_inf_importance_raises() -> None:
    with pytest.raises(ValueError, match="age"):
        from_feature_importance(
            features={"age": 29},
            importances={"age": math.inf},
            prediction=Prediction(predicted_class=1),
        )


# ------------------------------------------------------------- preservation


def test_zero_importance_preserved() -> None:
    req = from_feature_importance(
        features={"age": 29, "dti": 0.41},
        importances={"age": 0.0, "dti": 0.37},
        prediction=Prediction(predicted_class=1),
    )
    by_name = {c.name: c for c in req.contributions}
    assert by_name["age"].importance == 0.0


def test_contribution_order_follows_importances_iteration_order() -> None:
    req = from_feature_importance(
        features={"age": 29, "dti": 0.41, "income": 50000},
        # Deliberate insertion order: income, age, dti.
        importances={"income": 0.05, "age": -0.12, "dti": 0.37},
        prediction=Prediction(predicted_class=1),
    )
    assert [c.name for c in req.contributions] == ["income", "age", "dti"]


def test_rank_is_none_on_all_contributions() -> None:
    req = from_feature_importance(
        features={"age": 29, "dti": 0.41, "income": 50000},
        importances={"age": -0.12, "dti": 0.37, "income": 0.05},
        prediction=Prediction(predicted_class=1),
    )
    assert all(c.rank is None for c in req.contributions)


# -------------------------------------------------------- pass-through args


def test_prediction_forwarded_unchanged() -> None:
    pred = Prediction(predicted_class=1, probabilities={0: 0.2, 1: 0.8})
    req = from_feature_importance(
        features={"age": 29},
        importances={"age": 0.1},
        prediction=pred,
    )
    assert req.prediction is pred


def test_counterfactuals_forwarded_unchanged() -> None:
    cf = TabularCounterfactual(
        features={"age": 45},
        predicted_class=0,
    )
    req = from_feature_importance(
        features={"age": 29},
        importances={"age": 0.1},
        prediction=Prediction(predicted_class=1),
        counterfactuals=[cf],
    )
    assert req.counterfactuals is not None
    assert len(req.counterfactuals) == 1
    assert req.counterfactuals[0] is cf


def test_contrast_class_forwarded_unchanged() -> None:
    req = from_feature_importance(
        features={"age": 29},
        importances={"age": 0.1},
        prediction=Prediction(predicted_class=1),
        contrast_class=0,
    )
    assert req.contrast_class == 0


def test_instance_id_forwarded_unchanged() -> None:
    req = from_feature_importance(
        features={"age": 29},
        importances={"age": 0.1},
        prediction=Prediction(predicted_class=1),
        instance_id="row-42",
    )
    assert req.instance_id == "row-42"


# -------------------------------------------------------------- call shape


def test_keyword_only_beyond_prediction() -> None:
    with pytest.raises(TypeError):
        from_feature_importance(  # type: ignore[misc]
            {"age": 29},
            {"age": 0.1},
            Prediction(predicted_class=1),
            None,  # positional counterfactuals — must be keyword-only
        )
