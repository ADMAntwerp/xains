"""Unit tests for CounterfactualTabularPromptTemplate (ADR 0029).

Mirrors test_prompts_feature_importance_tabular.py shape.
"""

from typing import Any

import pytest

from xains import (
    DatasetSchema,
    ExplanationConfig,
    FeatureSchema,
    Modality,
    Prediction,
    TabularContribution,
    TabularExplanationRequest,
    TargetSchema,
)
from xains.prompts.counterfactual_tabular import (
    DEFAULT_SYSTEM_TEMPLATE,
    DEFAULT_USER_TEMPLATE,
    CounterfactualTabularPromptTemplate,
)
from xains.types import TabularCounterfactual, TextCounterfactual


def _schema() -> DatasetSchema:
    return DatasetSchema(
        modality=Modality.TABULAR,
        name="credit_risk",
        description="Credit risk demo.",
        target=TargetSchema(
            name="default",
            description="Whether the applicant defaulted.",
            classes={0: "Repaid", 1: "Defaulted"},
        ),
        features=[
            FeatureSchema(name="age", dtype="numeric", unit="years", description="age"),
            FeatureSchema(name="salary", dtype="numeric", unit="EUR", description="salary"),
            FeatureSchema(name="dti", dtype="numeric", description="debt-to-income"),
        ],
    )


def _config() -> ExplanationConfig:
    return ExplanationConfig(mode="counterfactual", audience="end_user")


def _cf(
    features: dict[str, Any],
    *,
    predicted_class: int = 0,
    method: str | None = None,
    changed: list[str] | None = None,
) -> TabularCounterfactual:
    return TabularCounterfactual(
        predicted_class=predicted_class,
        features=features,
        changed_features=changed,
        method=method,
    )


def _request_with(cfs: list[TabularCounterfactual]) -> TabularExplanationRequest:
    return TabularExplanationRequest(
        features={"age": 29, "salary": 52000, "dti": 0.41},
        prediction=Prediction(predicted_class=1, probabilities={0: 0.2, 1: 0.8}),
        contributions=[TabularContribution(name="dti", value=0.41, importance=0.37)],
        counterfactuals=cfs,
    )


# ------------------------------------------------------ defaults present


def test_module_exports_default_templates() -> None:
    assert isinstance(DEFAULT_SYSTEM_TEMPLATE, str)
    assert isinstance(DEFAULT_USER_TEMPLATE, str)
    assert "{counterfactuals}" in DEFAULT_USER_TEMPLATE


# ------------------------------------------------------ single vs multi CF


def test_single_cf_renders_without_numbering_with_flip_lead() -> None:
    schema = _schema()
    req = _request_with([_cf({"age": 29, "salary": 80000, "dti": 0.41})])
    _, user = CounterfactualTabularPromptTemplate().render(req, schema, _config())
    assert "Scenario" not in user
    assert "To change the prediction from Defaulted to Repaid:" in user
    assert "  - salary: 52000 -> 80000 [EUR]" in user


def test_multiple_cfs_are_numbered_and_order_preserved() -> None:
    schema = _schema()
    cfs = [
        _cf({"age": 29, "salary": 80000, "dti": 0.41}),
        _cf({"age": 35, "salary": 52000, "dti": 0.20}),
    ]
    req = _request_with(cfs)
    _, user = CounterfactualTabularPromptTemplate().render(req, schema, _config())
    assert "Scenario 1: To change the prediction from Defaulted to Repaid:" in user
    assert "Scenario 2: To change the prediction from Defaulted to Repaid:" in user
    assert user.index("Scenario 1") < user.index("Scenario 2")
    # Scenario 1 changes salary; scenario 2 changes age and dti.
    s1 = user.index("Scenario 1")
    s2 = user.index("Scenario 2")
    assert "salary: 52000 -> 80000" in user[s1:s2]
    assert "dti: 0.41 -> 0.2" in user[s2:]


# ------------------------------------------------------ method provenance


def test_method_off_by_default_never_appears() -> None:
    schema = _schema()
    req = _request_with([_cf({"age": 29, "salary": 80000, "dti": 0.41}, method="DiCE")])
    _, user = CounterfactualTabularPromptTemplate().render(req, schema, _config())
    assert "DiCE" not in user
    assert "method" not in user


def test_method_shown_when_include_method_true_and_cf_method_set() -> None:
    schema = _schema()
    req = _request_with([_cf({"age": 29, "salary": 80000, "dti": 0.41}, method="DiCE")])
    _, user = CounterfactualTabularPromptTemplate(include_method=True).render(
        req, schema, _config()
    )
    assert "(method: DiCE)" in user


def test_method_omitted_when_include_method_true_but_cf_method_none() -> None:
    schema = _schema()
    req = _request_with([_cf({"age": 29, "salary": 80000, "dti": 0.41}, method=None)])
    _, user = CounterfactualTabularPromptTemplate(include_method=True).render(
        req, schema, _config()
    )
    assert "method" not in user


# ------------------------------------------------------ error paths


def test_non_tabular_request_raises_type_error() -> None:
    schema = _schema()
    with pytest.raises(TypeError, match="TabularExplanationRequest"):
        CounterfactualTabularPromptTemplate().render(
            object(),  # type: ignore[arg-type]
            schema,
            _config(),
        )


def test_counterfactuals_none_raises_value_error() -> None:
    schema = _schema()
    req = TabularExplanationRequest(
        features={"age": 29, "salary": 52000, "dti": 0.41},
        prediction=Prediction(predicted_class=1),
        contributions=[TabularContribution(name="dti", value=0.41, importance=0.37)],
        counterfactuals=None,
    )
    with pytest.raises(ValueError, match=r"counterfactual"):
        CounterfactualTabularPromptTemplate().render(req, schema, _config())


def test_non_tabular_counterfactual_in_list_raises_type_error() -> None:
    schema = _schema()
    tab_cf = _cf({"age": 29, "salary": 80000, "dti": 0.41})
    txt_cf = TextCounterfactual(predicted_class=0, text="counterfactual text")
    req = TabularExplanationRequest(
        features={"age": 29, "salary": 52000, "dti": 0.41},
        prediction=Prediction(predicted_class=1),
        contributions=[TabularContribution(name="dti", value=0.41, importance=0.37)],
        counterfactuals=[tab_cf, txt_cf],
    )
    with pytest.raises(TypeError, match="tabular"):
        CounterfactualTabularPromptTemplate().render(req, schema, _config())


def test_cf_changes_feature_absent_from_schema_raises_value_error() -> None:
    schema = _schema()
    # CF override flags a feature that exists in cf.features and factual,
    # but is not declared on the schema.
    req = TabularExplanationRequest(
        features={"age": 29, "salary": 52000, "dti": 0.41, "mystery": 1},
        prediction=Prediction(predicted_class=1),
        contributions=[TabularContribution(name="dti", value=0.41, importance=0.37)],
        counterfactuals=[
            _cf({"age": 29, "salary": 52000, "dti": 0.41, "mystery": 7}),
        ],
    )
    with pytest.raises(ValueError, match=r"mystery"):
        CounterfactualTabularPromptTemplate().render(req, schema, _config())


def test_factual_predicted_class_not_in_schema_raises() -> None:
    schema = _schema()
    req = TabularExplanationRequest(
        features={"age": 29, "salary": 52000, "dti": 0.41},
        prediction=Prediction(predicted_class=99),
        contributions=[TabularContribution(name="dti", value=0.41, importance=0.37)],
        counterfactuals=[_cf({"age": 29, "salary": 80000, "dti": 0.41})],
    )
    with pytest.raises(ValueError, match=r"99"):
        CounterfactualTabularPromptTemplate().render(req, schema, _config())


def test_cf_predicted_class_not_in_schema_raises() -> None:
    schema = _schema()
    req = _request_with([_cf({"age": 29, "salary": 80000, "dti": 0.41}, predicted_class=99)])
    with pytest.raises(ValueError, match=r"99"):
        CounterfactualTabularPromptTemplate().render(req, schema, _config())


# ------------------------------------------------------ editable templates (ADR 0017)


def test_extra_placeholders_conflicting_with_builtin_raises_in_init() -> None:
    with pytest.raises(ValueError, match=r"counterfactuals"):
        CounterfactualTabularPromptTemplate(extra_placeholders={"counterfactuals": "x"})


def test_custom_templates_render_with_known_placeholders() -> None:
    schema = _schema()
    req = _request_with([_cf({"age": 29, "salary": 80000, "dti": 0.41})])
    custom_sys = "Custom for {target_name}/{audience}."
    custom_user = "P:{prediction}\nC:{counterfactuals}"
    sys, user = CounterfactualTabularPromptTemplate(
        system_template=custom_sys,
        user_template=custom_user,
    ).render(req, schema, _config())
    assert sys == "Custom for default/end_user."
    assert user.startswith("P:Defaulted\nC:")


def test_unknown_token_in_template_raises_value_error() -> None:
    schema = _schema()
    req = _request_with([_cf({"age": 29, "salary": 80000, "dti": 0.41})])
    with pytest.raises(ValueError, match=r"bogus"):
        CounterfactualTabularPromptTemplate(
            user_template="P:{prediction}\nC:{counterfactuals}\nExtra:{bogus}",
        ).render(req, schema, _config())


def test_extra_placeholders_substituted_into_custom_template() -> None:
    schema = _schema()
    req = _request_with([_cf({"age": 29, "salary": 80000, "dti": 0.41})])
    _, user = CounterfactualTabularPromptTemplate(
        extra_placeholders={"region": "EU"},
        user_template="{counterfactuals}\nRegion: {region}",
    ).render(req, schema, _config())
    assert "Region: EU" in user
