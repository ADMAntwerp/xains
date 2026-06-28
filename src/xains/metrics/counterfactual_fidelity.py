"""Counterfactual fidelity metrics (ADR 0031).

Three pure functions over a :class:`CounterfactualExtraction`, a tabular
request, and a schema. Ground truth comes from
:func:`xains.counterfactuals.build_scenarios` - this module does not
re-derive the diff. Per ADR 0031 each request carries a single
counterfactual, so the ground-truth map has exactly one (before, after)
pair per changed feature.

- ``change_fidelity``: fraction of resolved claims about actually-changed
  features where both ``stated_before`` AND ``stated_after`` match the
  scenario's ground-truth values (strict-both-required). Returns ``None``
  when no resolved claim is about an actually-changed feature.
- ``cf_coverage``: fraction of changed feature names that the extraction
  resolved. Always defined; ``0.0`` when the CF made no changes.
- ``invented_features``: ``len(extraction.invented)``. The CF analogue of
  ``hallucination_count``. Always defined.
"""

import math
from typing import Any

from xains.counterfactuals import build_scenarios
from xains.guardrails.types import CounterfactualExtraction, CounterfactualFeatureClaim
from xains.schema import DatasetSchema, FeatureDType
from xains.types import TabularExplanationRequest

_ATOL = 1e-6


def _is_numeric(x: object) -> bool:
    """Numeric iff int or float, excluding bool (matches fidelity.py)."""
    return isinstance(x, int | float) and not isinstance(x, bool)


def _coerce_stated_number(stated: Any) -> float | None:
    """Coerce a stated value to float if it is a number or a numeric string.

    The extraction LLM may return a numeric feature's value as a JSON string
    ("3190") rather than a number (3190). Coercing the stated side lets the
    numeric comparison succeed on value, not on the extractor's typing. Bools
    are excluded; a non-numeric string ("low") yields None and so scores
    incorrect, not an exception. Only the stated side is coerced - the ground
    truth comes from build_scenarios off the feature values and is trusted to
    be correctly typed.
    """
    if isinstance(stated, bool):
        return None
    if isinstance(stated, int | float):
        return float(stated)
    if isinstance(stated, str):
        try:
            return float(stated.strip())
        except ValueError:
            return None
    return None


def _value_matches(stated: Any, ground: Any, dtype: FeatureDType) -> bool:
    """Compare a single (stated, ground) pair against a feature's dtype.

    ``numeric``: ``math.isclose`` after coercing the stated side via
    ``_coerce_stated_number`` (a numeric string like ``"3190"`` is accepted
    on value). The ground side is trusted-typed and must be a real number.
    A non-numeric stated value (``None``, ``"low"``, bool) is incorrect,
    not an exception. ``ordinal`` / ``categorical`` / ``boolean`` /
    ``text``: equality. Ordinal joins the equality branch because the
    schema (``FeatureSchema._categorical_requires_categories``) requires
    ordinal features to carry ``categories: list[str]`` - their values
    are category labels, not numbers.
    """
    if dtype == "numeric":
        if not _is_numeric(ground):
            return False
        stated_num = _coerce_stated_number(stated)
        if stated_num is None:
            return False
        return math.isclose(stated_num, float(ground), abs_tol=_ATOL, rel_tol=0.0)
    return bool(stated == ground)


def _ground_truth_changes(
    request: TabularExplanationRequest, schema: DatasetSchema
) -> dict[str, tuple[Any, Any]]:
    """Build {feature_name: (before, after)} for the single counterfactual."""
    scenario = build_scenarios(request, schema)
    return {chg.name: (chg.before, chg.after) for chg in scenario.changes}


def _claim_matches(
    claim: CounterfactualFeatureClaim,
    ground_before: Any,
    ground_after: Any,
    dtype: FeatureDType,
) -> bool:
    """Strict-both-required: both before AND after must match the ground truth."""
    return _value_matches(claim.stated_before, ground_before, dtype) and _value_matches(
        claim.stated_after, ground_after, dtype
    )


def change_fidelity(
    extraction: CounterfactualExtraction,
    request: TabularExplanationRequest,
    schema: DatasetSchema,
) -> float | None:
    """Fraction of resolved claims about actually-changed features where
    BOTH ``stated_before`` and ``stated_after`` match the ground truth.

    Claims about features that did not actually change are silently
    ignored by this metric (they are not a fidelity question; unresolved
    mentions are picked up by ``invented_features``).

    Returns ``None`` when no resolved claim is about an actually-changed
    feature (metric undefined).
    """
    ground = _ground_truth_changes(request, schema)
    correct = 0
    total = 0
    for name, claim in extraction.changes.items():
        if name not in ground:
            continue  # claim is about a feature that did not actually change
        total += 1
        dtype = schema.feature(name).dtype
        ground_before, ground_after = ground[name]
        if _claim_matches(claim, ground_before, ground_after, dtype):
            correct += 1
    if total == 0:
        return None
    return correct / total


def cf_coverage(
    extraction: CounterfactualExtraction,
    request: TabularExplanationRequest,
    schema: DatasetSchema,
) -> float:
    """Fraction of the CF's actually-changed feature names that the
    extraction resolved.

    Denominator is the number of changed features in the counterfactual;
    numerator is the count of those names that appear as keys in
    ``extraction.changes``. Always defined; ``0.0`` when no feature
    actually changed.
    """
    ground = _ground_truth_changes(request, schema)
    if not ground:
        return 0.0
    resolved = sum(1 for name in extraction.changes if name in ground)
    return resolved / len(ground)


def invented_features(extraction: CounterfactualExtraction) -> int:
    """Number of unresolved CF mentions. CF analogue of ``hallucination_count``."""
    return len(extraction.invented)
