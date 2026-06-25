"""Unit tests for ``changed_features`` diff (ADR 0028).

Pure function over (factual dict, TabularCounterfactual) -> list[ChangedFeature].
Tabular only this PR.
"""

from typing import Any

import pytest
from pydantic import ValidationError

from xains.counterfactuals import ChangedFeature, changed_features
from xains.types import TabularCounterfactual


def _cf(features: dict[str, Any], changed: list[str] | None = None) -> TabularCounterfactual:
    return TabularCounterfactual(
        predicted_class=0,
        features=features,
        changed_features=changed,
    )


def test_full_cf_reports_only_differing_keys() -> None:
    """cf.features carries all keys, most equal; only differing ones reported."""
    factual = {"age": 29, "salary": 52000, "dti": 0.41}
    cf = _cf({"age": 29, "salary": 80000, "dti": 0.41})
    result = changed_features(factual, cf)
    assert result == [ChangedFeature(name="salary", before=52000, after=80000)]


def test_partial_cf_reports_present_keys_that_differ() -> None:
    """cf.features carries only changed keys; each differing one reported."""
    factual = {"age": 29, "salary": 52000, "dti": 0.41}
    cf = _cf({"salary": 80000, "dti": 0.20})
    result = changed_features(factual, cf)
    assert result == [
        ChangedFeature(name="salary", before=52000, after=80000),
        ChangedFeature(name="dti", before=0.41, after=0.20),
    ]


def test_explicit_changed_features_override_is_honored_without_diff() -> None:
    """When cf.changed_features is set, those keys are reported as-is even if values match."""
    factual = {"age": 29, "salary": 52000}
    # cf.features matches factual exactly, but override says salary is "changed"
    cf = _cf({"age": 29, "salary": 52000}, changed=["salary"])
    result = changed_features(factual, cf)
    assert result == [ChangedFeature(name="salary", before=52000, after=52000)]


def test_cf_key_absent_from_factual_raises_value_error() -> None:
    """Diff: a key in cf.features that the factual lacks is a user/data error."""
    factual = {"age": 29}
    cf = _cf({"age": 30, "salary": 80000})  # 'salary' missing in factual
    with pytest.raises(ValueError, match=r"salary"):
        changed_features(factual, cf)


def test_override_key_absent_from_factual_raises_value_error() -> None:
    """Override path: same rule. Naming an absent key is a user error."""
    factual = {"age": 29}
    cf = _cf({"age": 30, "salary": 80000}, changed=["salary"])
    with pytest.raises(ValueError, match=r"salary"):
        changed_features(factual, cf)


def test_identical_cf_returns_empty_list() -> None:
    """No changes, no override: empty list. No warning, no exception."""
    factual = {"age": 29, "salary": 52000}
    cf = _cf({"age": 29, "salary": 52000})
    assert changed_features(factual, cf) == []


def test_value_types_are_preserved() -> None:
    """before/after carry the original types (int stays int, str stays str)."""
    factual = {"age": 29, "status": "single", "score": 0.5}
    cf = _cf({"age": 30, "status": "married", "score": 0.7})
    result = changed_features(factual, cf)
    by_name = {c.name: c for c in result}
    assert isinstance(by_name["age"].before, int)
    assert isinstance(by_name["age"].after, int)
    assert isinstance(by_name["status"].before, str)
    assert by_name["status"].after == "married"
    assert isinstance(by_name["score"].before, float)


def test_changed_feature_rejects_extra_fields() -> None:
    """ChangedFeature has ConfigDict(extra='forbid'), matching house style."""
    with pytest.raises(ValidationError):
        ChangedFeature(name="x", before=1, after=2, extra="bogus")  # type: ignore[call-arg]
