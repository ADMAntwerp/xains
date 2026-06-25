"""Build per-counterfactual scenario records from a tabular request.

Single source of truth for "what changed and what's the flip" - both the
CF prompt template (LLM path) and the templated CF generator (LLM-free
path) consume the output of :func:`build_scenarios`. Numbering and prose
are each consumer's surface choice; this module produces data only.
See ADR 0030.
"""

from pydantic import BaseModel, ConfigDict

from xains.counterfactuals.diff import ChangedFeature, changed_features
from xains.schema import DatasetSchema
from xains.types import TabularCounterfactual, TabularExplanationRequest


class CounterfactualScenario(BaseModel):
    """One counterfactual's structured view: flip labels and changed-feature list."""

    model_config = ConfigDict(extra="forbid")

    index: int  # 1-based position in request.counterfactuals
    factual_label: str
    cf_label: str
    changes: list[ChangedFeature]
    method: str | None


def build_scenarios(
    request: TabularExplanationRequest,
    schema: DatasetSchema,
) -> list[CounterfactualScenario]:
    """Build a :class:`CounterfactualScenario` per counterfactual, in request order.

    Honors ``request.counterfactuals`` ordering (ADR 0004: the library does
    not rank, filter, or reorder).
    """
    if request.counterfactuals is None:
        raise ValueError("build_scenarios requires request.counterfactuals; none were provided.")

    factual_class = request.prediction.predicted_class
    if factual_class not in schema.target.classes:
        raise ValueError(
            f"Prediction predicted_class={factual_class!r} is not in schema.target.classes."
        )
    factual_label = str(schema.target.classes[factual_class])

    feature_names = {f.name for f in (schema.features or [])}

    scenarios: list[CounterfactualScenario] = []
    for idx, cf in enumerate(request.counterfactuals, start=1):
        if not isinstance(cf, TabularCounterfactual):
            raise TypeError(
                f"build_scenarios requires every counterfactual to be tabular, "
                f"got {type(cf).__name__}."
            )
        if cf.predicted_class not in schema.target.classes:
            raise ValueError(
                f"Counterfactual predicted_class={cf.predicted_class!r} is not in "
                f"schema.target.classes."
            )
        cf_label = str(schema.target.classes[cf.predicted_class])

        changes = changed_features(request.features, cf)
        for chg in changes:
            if chg.name not in feature_names:
                raise ValueError(
                    f"Counterfactual references unknown feature {chg.name!r}; "
                    f"not in schema.features."
                )

        scenarios.append(
            CounterfactualScenario(
                index=idx,
                factual_label=factual_label,
                cf_label=cf_label,
                changes=changes,
                method=cf.method,
            )
        )
    return scenarios
