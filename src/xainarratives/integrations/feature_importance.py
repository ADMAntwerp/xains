"""Adapter: per-feature signed importance scalars → ``TabularExplanationRequest``.

Accepts any signed-per-feature scalar attribution — SHAP values, LIME
coefficients, sklearn ``feature_importances_``, permutation importance,
integrated gradients reduced to scalars, etc. Named by the *shape* of the
input, not the upstream tool.

Scope limits:

* Tabular only. Token / region / graph attributions have their own adapters.
* One scalar per feature. Multi-class / multi-output attributions (e.g.
  SHAP values on a multi-class classifier, shape ``(n_features, n_classes)``)
  are NOT supported by this adapter. Callers must first select the
  attribution slice for the class of interest (typically
  ``prediction.predicted_class``) and pass that as ``importances``. A
  multi-class adapter is tracked as separate future work.
"""

import math
from collections.abc import Mapping
from typing import Any

from xainarratives.types import (
    CounterfactualInstance,
    Prediction,
    TabularContribution,
    TabularExplanationRequest,
)


def from_feature_importance(
    features: Mapping[str, Any],
    importances: Mapping[str, float],
    prediction: Prediction,
    *,
    counterfactuals: list[CounterfactualInstance] | None = None,
    contrast_class: int | str | None = None,
    instance_id: str | None = None,
) -> TabularExplanationRequest:
    """Build a ``TabularExplanationRequest`` from feature values and per-feature importances.

    Features without a corresponding entry in ``importances`` remain in the
    request's ``features`` payload but do not produce a contribution.
    Contributions are emitted in ``importances`` iteration order; the adapter
    does not sort or assign ``rank``. See the module docstring for scope
    limits (tabular-only, single-output).
    """
    if not importances:
        raise ValueError("from_feature_importance: `importances` must contain at least one entry.")

    unknown = [name for name in importances if name not in features]
    if unknown:
        raise ValueError(
            f"from_feature_importance: `importances` names features not present in "
            f"`features`: {sorted(unknown)}."
        )

    non_finite = [name for name, val in importances.items() if not math.isfinite(val)]
    if non_finite:
        raise ValueError(
            f"from_feature_importance: non-finite importance values for features "
            f"{sorted(non_finite)}."
        )

    contributions = [
        TabularContribution(name=name, value=features[name], importance=float(val))
        for name, val in importances.items()
    ]

    return TabularExplanationRequest(
        features=dict(features),
        prediction=prediction,
        contributions=contributions,
        counterfactuals=counterfactuals,
        contrast_class=contrast_class,
        instance_id=instance_id,
    )
