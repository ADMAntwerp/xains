"""Build a counterfactual scenario record from a tabular request.

Single source of truth for "what changed and what's the flip" - both the
CF prompt template (LLM path) and the templated CF generator (LLM-free
path) consume the output of :func:`build_scenarios`. Numbering and prose
are each consumer's surface choice; this module produces data only.
See ADR 0030 and ADR 0031 (single counterfactual per request).
"""

from pydantic import BaseModel, ConfigDict

from xains.counterfactuals.diff import ChangedFeature, changed_features
from xains.schema import DatasetSchema
from xains.types import TabularCounterfactual, TabularExplanationRequest


class CounterfactualScenario(BaseModel):
    """The single counterfactual's structured view: flip labels and changed-feature list."""

    model_config = ConfigDict(extra="forbid")

    factual_label: str
    cf_label: str
    changes: list[ChangedFeature]
    method: str | None


def build_scenarios(
    request: TabularExplanationRequest,
    schema: DatasetSchema,
) -> CounterfactualScenario:
    """Build the :class:`CounterfactualScenario` for ``request.counterfactual``.

    Per ADR 0031 a request carries exactly one counterfactual (or
    ``None``). Raises ``ValueError`` when the counterfactual is missing
    or its predicted_class is not in ``schema.target.classes``; raises
    ``TypeError`` when the counterfactual is not a
    :class:`TabularCounterfactual`.
    """
    cf = request.counterfactual
    if cf is None:
        raise ValueError("build_scenarios requires request.counterfactual; none was provided.")

    if not isinstance(cf, TabularCounterfactual):
        raise TypeError(
            f"build_scenarios requires a tabular counterfactual, got {type(cf).__name__}."
        )

    factual_class = request.prediction.predicted_class
    if factual_class not in schema.target.classes:
        raise ValueError(
            f"Prediction predicted_class={factual_class!r} is not in schema.target.classes."
        )
    factual_label = str(schema.target.classes[factual_class])

    if cf.predicted_class not in schema.target.classes:
        raise ValueError(
            f"Counterfactual predicted_class={cf.predicted_class!r} is not in "
            f"schema.target.classes."
        )
    cf_label = str(schema.target.classes[cf.predicted_class])

    changes = changed_features(request.features, cf)

    feature_names = {f.name for f in (schema.features or [])}
    for chg in changes:
        if chg.name not in feature_names:
            raise ValueError(
                f"Counterfactual references unknown feature {chg.name!r}; not in schema.features."
            )

    return CounterfactualScenario(
        factual_label=factual_label,
        cf_label=cf_label,
        changes=changes,
        method=cf.method,
    )
