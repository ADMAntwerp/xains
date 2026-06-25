"""CounterfactualTabularPromptTemplate - verbalize tabular counterfactual scenarios.

Mirrors :class:`FeatureImportanceTabularPromptTemplate`'s editable-template
contract (ADR 0017). Renders one block per counterfactual in the order
provided (ADR 0004: the library does not rank, filter, or reorder). Each
block leads with the flip (factual label -> counterfactual label) and lists
the changed-features beneath. Tabular only. See ADR 0029.
"""

from xains._substitution import substitute
from xains.config import ExplanationConfig
from xains.counterfactuals import build_scenarios
from xains.prompts.base import PromptTemplate
from xains.schema import DatasetSchema
from xains.types import ExplanationRequest, TabularExplanationRequest

DEFAULT_SYSTEM_TEMPLATE = (
    "You are explaining a model prediction for the '{target_name}' target "
    "by describing counterfactual scenarios: what would need to change about "
    "this instance for the model to predict a different outcome. "
    "Audience: {audience}. Tone: {tone}. "
    "Keep the explanation under {max_length_words} words."
    "\n\n"
    "{narrative_rules}"
)

DEFAULT_USER_TEMPLATE = (
    "Current prediction: {prediction}.\n"
    "Counterfactual scenarios (what could change the outcome):\n"
    "{counterfactuals}"
)

_BUILTIN_NAMES = frozenset(
    {
        "target_name",
        "audience",
        "tone",
        "max_length_words",
        "narrative_rules",
        "prediction",
        "counterfactuals",
    }
)


class CounterfactualTabularPromptTemplate(PromptTemplate):
    """Render a tabular counterfactual explanation prompt.

    Verbalizes ``request.counterfactuals`` in the order provided (no ranking
    or filtering). A single counterfactual renders without numbering;
    multiple counterfactuals are numbered ``Scenario 1``, ``Scenario 2``, ....

    Pass ``system_template=`` / ``user_template=`` to customize the prompt
    structure; both default to ``DEFAULT_SYSTEM_TEMPLATE`` /
    ``DEFAULT_USER_TEMPLATE`` and reproduce the canonical layout byte-for-byte.

    Built-in placeholders: ``{target_name}``, ``{audience}``, ``{tone}``,
    ``{max_length_words}``, ``{narrative_rules}`` come from the schema and
    ExplanationConfig; ``{prediction}`` is the factual class label
    (``schema.target.classes[request.prediction.predicted_class]``);
    ``{counterfactuals}`` is the joined scenario block.

    ``include_method=True`` appends ``" (method: <cf.method>)"`` to each
    scenario's flip line when the counterfactual carries a non-None
    ``method`` string. Default is ``False`` to keep narratives
    method-agnostic.

    Pass ``extra_placeholders={"name": "value", ...}`` to declare additional
    substitutions; extras may not overlap built-in names. An
    identifier-shaped ``{token}`` in a template that is neither a built-in
    nor in ``extra_placeholders`` raises ``ValueError`` at render time.
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
                f"CounterfactualTabularPromptTemplate requires a TabularExplanationRequest, "
                f"got {type(request).__name__}."
            )

        scenarios = build_scenarios(request, schema)
        # scenarios is non-empty: request.counterfactuals has min_length=1.
        factual_label = scenarios[0].factual_label
        numbered = len(scenarios) > 1

        scenario_blocks: list[str] = []
        for sc in scenarios:
            prefix = f"Scenario {sc.index}: " if numbered else ""
            method_suffix = ""
            if self._include_method and sc.method is not None:
                method_suffix = f" (method: {sc.method})"

            lead = (
                f"{prefix}To change the prediction from {sc.factual_label} to "
                f"{sc.cf_label}:{method_suffix}"
            )
            change_lines = []
            for chg in sc.changes:
                feat = schema.feature(chg.name)
                unit = f" [{feat.unit}]" if feat.unit else ""
                change_lines.append(f"  - {chg.name}: {chg.before} -> {chg.after}{unit}")

            scenario_blocks.append("\n".join([lead, *change_lines]))

        counterfactuals_block = "\n\n".join(scenario_blocks)

        values: dict[str, str] = {
            "target_name": schema.target.name,
            "audience": config.audience,
            "tone": config.tone,
            "max_length_words": str(config.max_length_words),
            "narrative_rules": config.narrative_rules,
            "prediction": factual_label,
            "counterfactuals": counterfactuals_block,
        }
        values.update(self._extra)

        system = substitute(self._system_template, values)
        user = substitute(self._user_template, values)
        return system, user
