"""Unit tests for build_contribution_block + build_counterfactual_block (ADR 0038).

Direct tests of the shared block-string helpers extracted from the two
tabular templates' render() methods. The template-level tests already
pin end-to-end behaviour; these tests pin the helpers' individual
contracts so the hybrid template can compose them with confidence.
"""

import pytest

from xains import (
    DatasetSchema,
    FeatureSchema,
    Modality,
    TabularContribution,
    TargetSchema,
)
from xains.counterfactuals import ChangedFeature, CounterfactualScenario
from xains.prompts._blocks import build_contribution_block, build_counterfactual_block


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
            FeatureSchema(name="salary", dtype="numeric", unit="EUR", description="salary"),
            FeatureSchema(name="dti", dtype="numeric", description="debt-to-income"),
            FeatureSchema(name="score", dtype="numeric", description="score"),
        ],
    )


# ------------------------------------------------------ build_contribution_block


def test_contribution_block_orders_by_abs_importance_descending() -> None:
    schema = _schema()
    contribs = [
        TabularContribution(name="age", value=29, importance=-0.12),
        TabularContribution(name="salary", value=52000, importance=0.21),
        TabularContribution(name="dti", value=0.41, importance=-0.37),
    ]
    out = build_contribution_block(contribs, schema, top_k=10)
    lines = out.split("\n")
    # dti (0.37) first, salary (0.21) second, age (0.12) third
    assert lines[0].startswith("- dti = 0.41")
    assert lines[1].startswith("- salary = 52000 [EUR]")
    assert lines[2].startswith("- age = 29 [years]")


def test_contribution_block_top_k_cut_widens_at_tied_boundary() -> None:
    """Ties at the k-th boundary widen the cut to include all tied contributions."""
    schema = _schema()
    # Four contributions: importances 0.5, 0.3, 0.3, 0.1. top_k=2 -> boundary 0.3, widened to 3.
    contribs = [
        TabularContribution(name="age", value=29, importance=0.5),
        TabularContribution(name="salary", value=52000, importance=0.3),
        TabularContribution(name="dti", value=0.41, importance=-0.3),
        TabularContribution(name="score", value=100, importance=0.1),
    ]
    out = build_contribution_block(contribs, schema, top_k=2)
    lines = out.split("\n")
    assert len(lines) == 3
    assert "age" in lines[0]
    assert "salary" in lines[1] or "salary" in lines[2]
    assert "dti" in lines[1] or "dti" in lines[2]
    assert "score" not in out


def test_contribution_block_top_k_cut_no_widening_when_no_tie() -> None:
    schema = _schema()
    contribs = [
        TabularContribution(name="age", value=29, importance=0.5),
        TabularContribution(name="salary", value=52000, importance=0.3),
        TabularContribution(name="dti", value=0.41, importance=0.1),
    ]
    out = build_contribution_block(contribs, schema, top_k=2)
    lines = out.split("\n")
    assert len(lines) == 2
    assert "dti" not in out


def test_contribution_block_unit_suffix_present_when_declared() -> None:
    schema = _schema()
    contribs = [TabularContribution(name="salary", value=52000, importance=0.5)]
    out = build_contribution_block(contribs, schema, top_k=10)
    assert " [EUR]" in out


def test_contribution_block_unit_suffix_absent_when_undeclared() -> None:
    """dti has no unit; its line should not carry `[...]`."""
    schema = _schema()
    contribs = [TabularContribution(name="dti", value=0.41, importance=0.5)]
    out = build_contribution_block(contribs, schema, top_k=10)
    assert " [" not in out


def test_contribution_block_sign_formatting_positive_and_negative() -> None:
    schema = _schema()
    contribs = [
        TabularContribution(name="age", value=29, importance=0.5),
        TabularContribution(name="salary", value=52000, importance=-0.5),
    ]
    out = build_contribution_block(contribs, schema, top_k=10)
    assert "importance=+0.5" in out
    assert "importance=-0.5" in out


def test_contribution_block_sign_positive_when_zero() -> None:
    """Zero uses `+` (per the >= 0 test in the original inline logic)."""
    schema = _schema()
    contribs = [TabularContribution(name="age", value=29, importance=0.0)]
    out = build_contribution_block(contribs, schema, top_k=10)
    assert "importance=+0" in out


def test_contribution_block_empty_list_yields_empty_string() -> None:
    schema = _schema()
    out = build_contribution_block([], schema, top_k=10)
    assert out == ""


def test_contribution_block_missing_feature_in_schema_raises() -> None:
    """Helper does not validate; schema.feature() KeyErrors on lookup."""
    schema = _schema()
    contribs = [TabularContribution(name="mystery", value=1, importance=0.5)]
    with pytest.raises(KeyError):
        build_contribution_block(contribs, schema, top_k=10)


# ------------------------------------------------------ build_counterfactual_block


def _cf_scenario(
    *,
    factual_label: str = "Bad credit risk",
    cf_label: str = "Good credit risk",
    changes: list[ChangedFeature] | None = None,
    method: str | None = None,
) -> CounterfactualScenario:
    return CounterfactualScenario(
        factual_label=factual_label,
        cf_label=cf_label,
        changes=changes or [],
        method=method,
    )


def test_counterfactual_block_single_change_lead_and_indented_line() -> None:
    schema = _schema()
    scenario = _cf_scenario(
        changes=[ChangedFeature(name="dti", before=0.41, after=0.20)],
    )
    out = build_counterfactual_block(scenario, schema, include_method=False)
    assert out == (
        "To change the prediction from Bad credit risk to Good credit risk:\n  - dti: 0.41 -> 0.2"
    )


def test_counterfactual_block_multiple_changes_render_multiple_lines() -> None:
    schema = _schema()
    scenario = _cf_scenario(
        changes=[
            ChangedFeature(name="age", before=29, after=35),
            ChangedFeature(name="salary", before=52000, after=80000),
        ],
    )
    out = build_counterfactual_block(scenario, schema, include_method=False)
    lines = out.split("\n")
    assert lines[0] == "To change the prediction from Bad credit risk to Good credit risk:"
    assert lines[1] == "  - age: 29 -> 35 [years]"
    assert lines[2] == "  - salary: 52000 -> 80000 [EUR]"


def test_counterfactual_block_unit_suffix_present_when_declared() -> None:
    schema = _schema()
    scenario = _cf_scenario(
        changes=[ChangedFeature(name="salary", before=52000, after=80000)],
    )
    out = build_counterfactual_block(scenario, schema, include_method=False)
    assert "  - salary: 52000 -> 80000 [EUR]" in out


def test_counterfactual_block_unit_suffix_absent_when_undeclared() -> None:
    schema = _schema()
    scenario = _cf_scenario(
        changes=[ChangedFeature(name="dti", before=0.41, after=0.20)],
    )
    out = build_counterfactual_block(scenario, schema, include_method=False)
    assert "  - dti: 0.41 -> 0.2" in out
    assert " [" not in out.split("\n")[1]


def test_counterfactual_block_include_method_true_and_method_set() -> None:
    schema = _schema()
    scenario = _cf_scenario(
        changes=[ChangedFeature(name="dti", before=0.41, after=0.20)],
        method="DiCE",
    )
    out = build_counterfactual_block(scenario, schema, include_method=True)
    assert out.split("\n")[0] == (
        "To change the prediction from Bad credit risk to Good credit risk: (method: DiCE)"
    )


def test_counterfactual_block_include_method_true_but_cf_method_none() -> None:
    schema = _schema()
    scenario = _cf_scenario(
        changes=[ChangedFeature(name="dti", before=0.41, after=0.20)],
        method=None,
    )
    out = build_counterfactual_block(scenario, schema, include_method=True)
    assert "method" not in out


def test_counterfactual_block_include_method_false_hides_method_even_when_set() -> None:
    schema = _schema()
    scenario = _cf_scenario(
        changes=[ChangedFeature(name="dti", before=0.41, after=0.20)],
        method="DiCE",
    )
    out = build_counterfactual_block(scenario, schema, include_method=False)
    assert "DiCE" not in out
    assert "method" not in out


def test_counterfactual_block_empty_changes_yields_lead_line_only() -> None:
    """Degenerate scenario: no changes -> just the flip lead."""
    schema = _schema()
    scenario = _cf_scenario(changes=[])
    out = build_counterfactual_block(scenario, schema, include_method=False)
    assert out == "To change the prediction from Bad credit risk to Good credit risk:"


def test_counterfactual_block_missing_feature_in_schema_raises() -> None:
    """Helper does not validate; schema.feature() KeyErrors on lookup."""
    schema = _schema()
    scenario = _cf_scenario(
        changes=[ChangedFeature(name="mystery", before=1, after=2)],
    )
    with pytest.raises(KeyError):
        build_counterfactual_block(scenario, schema, include_method=False)
