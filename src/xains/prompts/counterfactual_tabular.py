"""CounterfactualTabularPromptTemplate - verbalize the tabular counterfactual scenario.

Mirrors :class:`FeatureImportanceTabularPromptTemplate`'s editable-template
contract (ADR 0017). Per ADR 0031 a request carries exactly one
counterfactual, so the template renders one flip block (no numbering,
no alternatives) consisting of the flip lead followed by the
changed-feature lines. Tabular only. See ADR 0029.
"""

from xains._substitution import substitute
from xains.config import ExplanationConfig
from xains.counterfactuals import build_scenarios
from xains.prompts._blocks import build_counterfactual_block
from xains.prompts.base import PromptTemplate
from xains.schema import DatasetSchema
from xains.types import ExplanationRequest, TabularExplanationRequest

DEFAULT_SYSTEM_TEMPLATE = (
    "You are explaining a model prediction for the '{target_name}' target "
    "by describing a counterfactual scenario: what would need to change "
    "about this instance for the model to predict a different outcome. "
    "For each feature that changes, state explicitly the value it changes from "
    "and the value it changes to, using those values exactly as written "
    "rather than rephrasing them. "
    "Audience: {audience}. Tone: {tone}. "
    "Keep the explanation under {max_length_words} words."
    "\n\n"
    "{narrative_rules}"
)

DEFAULT_USER_TEMPLATE = (
    "Current prediction: {prediction}.\n"
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
        "counterfactual",
    }
)


class CounterfactualTabularPromptTemplate(PromptTemplate):
    """Render a tabular counterfactual explanation prompt.

    Verbalizes ``request.counterfactual`` as a single flip block: a lead
    line ("To change the prediction from <factual> to <cf>:") followed by
    one indented ``  - name: before -> after [unit]`` line per changed
    feature.

    Pass ``system_template=`` / ``user_template=`` to customize the prompt
    structure; both default to ``DEFAULT_SYSTEM_TEMPLATE`` /
    ``DEFAULT_USER_TEMPLATE`` and reproduce the canonical layout
    byte-for-byte.

    Built-in placeholders: ``{target_name}``, ``{audience}``, ``{tone}``,
    ``{max_length_words}``, ``{narrative_rules}`` come from the schema and
    ExplanationConfig; ``{prediction}`` is the factual class label
    (``schema.target.classes[request.prediction.predicted_class]``);
    ``{counterfactual}`` is the rendered scenario block.

    ``include_method=True`` appends ``" (method: <cf.method>)"`` to the
    flip line when the counterfactual carries a non-``None`` ``method``
    field. Default is ``False`` to keep narratives method-agnostic.

    Pass ``extra_placeholders={"name": "value", ...}`` to declare
    additional substitutions; extras may not overlap built-in names. An
    identifier-shaped ``{token}`` in a template that is neither a
    built-in nor in ``extra_placeholders`` raises ``ValueError`` at
    render time.
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

        scenario = build_scenarios(request, schema)
        factual_label = scenario.factual_label
        counterfactual_block = build_counterfactual_block(scenario, schema, self._include_method)

        values: dict[str, str] = {
            "target_name": schema.target.name,
            "audience": config.audience,
            "tone": config.tone,
            "max_length_words": str(config.max_length_words),
            "narrative_rules": config.narrative_rules,
            "prediction": factual_label,
            "counterfactual": counterfactual_block,
        }
        values.update(self._extra)

        system = substitute(self._system_template, values)
        user = substitute(self._user_template, values)
        return system, user
