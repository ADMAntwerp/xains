"""HybridTabularPromptTemplate - two-section prompt for mode='feature_importance_counterfactual'.

Composes the two shared block helpers from `xains.prompts._blocks` into a
single prompt: a feature-importance contribution block explaining why the
model made its prediction, followed by a counterfactual block describing
what would need to change to flip it. The `include_method` flag (mirroring
`CounterfactualTabularPromptTemplate`) toggles the CF-block method suffix
without affecting the FI half. Tabular only. See ADR 0039.
"""

from xains._substitution import substitute
from xains.config import ExplanationConfig
from xains.counterfactuals import build_scenarios
from xains.prompts._blocks import build_contribution_block, build_counterfactual_block
from xains.prompts.base import PromptTemplate
from xains.schema import DatasetSchema
from xains.types import ExplanationRequest, TabularExplanationRequest

DEFAULT_SYSTEM_TEMPLATE = (
    "You are explaining a model prediction for the '{target_name}' target in two parts. "
    "First, explain why the model made its prediction, based on the feature contributions. "
    "Then, describe a counterfactual scenario: what would need to change "
    "for the model to predict a different outcome. "
    "For each feature that changes in the counterfactual, state explicitly "
    "the value it changes from and the value it changes to, using those "
    "values exactly as written rather than rephrasing them. "
    "Audience: {audience}. Tone: {tone}. "
    "Keep the explanation under {max_length_words} words."
    "\n\n"
    "{narrative_rules}"
)

DEFAULT_USER_TEMPLATE = (
    "Prediction: {prediction}.\n"
    "Top contributions by magnitude:\n"
    "{contributions}\n"
    "\n"
    "Counterfactual scenario (what could change the outcome):\n"
    "{counterfactual}"
)

_BUILTIN_NAMES = frozenset(
    {
        "target_name",
        "audience",
        "tone",
        "max_length_words",
        "narrative_rules",
        "prediction",
        "contributions",
        "counterfactual",
    }
)


class HybridTabularPromptTemplate(PromptTemplate):
    """Render a tabular hybrid feature-importance + counterfactual explanation prompt.

    Composes two body blocks via `xains.prompts._blocks`:

    - `{contributions}` - the feature-importance top-k contribution list
      (same shape as `FeatureImportanceTabularPromptTemplate`).
    - `{counterfactual}` - the flip lead plus change lines
      (same shape as `CounterfactualTabularPromptTemplate`).

    `{prediction}` is the FACTUAL class label
    (`schema.target.classes[request.prediction.predicted_class]`), matching
    the FI template - not the counterfactual's target label.

    Pass `system_template=` / `user_template=` to customize the prompt
    structure; both default to `DEFAULT_SYSTEM_TEMPLATE` /
    `DEFAULT_USER_TEMPLATE` and reproduce the canonical layout byte-for-byte.

    Built-in placeholders: `{target_name}`, `{audience}`, `{tone}`,
    `{max_length_words}`, `{narrative_rules}` come from the schema and
    ExplanationConfig; `{prediction}` is the factual class label;
    `{contributions}` is the FI block; `{counterfactual}` is the CF block.

    `include_method=True` appends `" (method: <cf.method>)"` to the CF
    block's flip line when the counterfactual carries a non-`None`
    `method` field. Default is `False` to keep narratives method-agnostic.

    Pass `extra_placeholders={"name": "value", ...}` to declare additional
    substitutions; extras may not overlap built-in names. An
    identifier-shaped `{token}` in a template that is neither a built-in
    nor in `extra_placeholders` raises `ValueError` at render time.
    """

    def __init__(
        self,
        *,
        system_template: str = DEFAULT_SYSTEM_TEMPLATE,
        user_template: str = DEFAULT_USER_TEMPLATE,
        extra_placeholders: dict[str, str] | None = None,
        include_method: bool = False,
    ) -> None:
        extra = dict(extra_placeholders) if extra_placeholders else {}
        conflicts = sorted(_BUILTIN_NAMES & extra.keys())
        if conflicts:
            raise ValueError(
                f"extra_placeholders may not rebind built-in placeholder names: "
                f"{', '.join(conflicts)}."
            )
        self._system_template = system_template
        self._user_template = user_template
        self._extra = extra
        self._include_method = include_method

    def render(
        self,
        request: ExplanationRequest,
        schema: DatasetSchema,
        config: ExplanationConfig,
    ) -> tuple[str, str]:
        if not isinstance(request, TabularExplanationRequest):
            raise TypeError(
                f"HybridTabularPromptTemplate requires a TabularExplanationRequest, "
                f"got {type(request).__name__}."
            )
        if request.counterfactual is None:
            raise ValueError(
                "HybridTabularPromptTemplate requires request.counterfactual "
                "(mode='feature_importance_counterfactual' needs a counterfactual)."
            )

        # FI-half validation: guard contributions + factual predicted_class.
        feature_names = {f.name for f in (schema.features or [])}
        for c in request.contributions:
            if c.name not in feature_names:
                raise ValueError(
                    f"Contribution references unknown feature {c.name!r}; not in schema.features."
                )
        predicted_class = request.prediction.predicted_class
        if predicted_class not in schema.target.classes:
            raise ValueError(
                f"Prediction predicted_class={predicted_class!r} is not in schema.target.classes."
            )
        class_label = schema.target.classes[predicted_class]

        # CF-half validation happens inside build_scenarios (factual/CF classes,
        # changed-feature names against schema).
        scenario = build_scenarios(request, schema)

        contributions_block = build_contribution_block(
            request.contributions, schema, config.top_k_features
        )
        counterfactual_block = build_counterfactual_block(scenario, schema, self._include_method)

        values: dict[str, str] = {
            "target_name": schema.target.name,
            "audience": config.audience,
            "tone": config.tone,
            "max_length_words": str(config.max_length_words),
            "narrative_rules": config.narrative_rules,
            "prediction": str(class_label),
            "contributions": contributions_block,
            "counterfactual": counterfactual_block,
        }
        values.update(self._extra)

        system = substitute(self._system_template, values)
        user = substitute(self._user_template, values)
        return system, user
