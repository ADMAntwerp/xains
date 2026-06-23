"""FeatureImportanceTabularPromptTemplate — verbalize tabular attributions as evidence prose."""

from xains._substitution import substitute
from xains.config import ExplanationConfig
from xains.prompts.base import PromptTemplate
from xains.schema import DatasetSchema
from xains.types import ExplanationRequest, TabularExplanationRequest

DEFAULT_SYSTEM_TEMPLATE = (
    "You are explaining a model prediction for the '{target_name}' target. "
    "Audience: {audience}. Tone: {tone}. "
    "Keep the explanation under {max_length_words} words."
    "\n\n"
    "{narrative_rules}"
)

DEFAULT_USER_TEMPLATE = (
    "Prediction: {prediction}.\nTop contributions by magnitude:\n{contributions}"
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
    }
)


class FeatureImportanceTabularPromptTemplate(PromptTemplate):
    """Render a feature-importance tabular explanation prompt.

    Contributions are selected top-k by ``abs(importance)`` descending; ties
    at the k-th boundary widen the cut to include all tied contributions. The
    ``rank`` field on each contribution is ignored.

    Pass ``system_template=`` / ``user_template=`` to customize the prompt
    structure; both default to ``DEFAULT_SYSTEM_TEMPLATE`` /
    ``DEFAULT_USER_TEMPLATE`` and reproduce the canonical layout byte-for-byte.

    Built-in placeholders: ``{target_name}``, ``{audience}``, ``{tone}``,
    ``{max_length_words}``, ``{narrative_rules}`` come from the schema and
    ExplanationConfig; ``{prediction}`` is
    ``schema.target.classes[predicted_class]``; ``{contributions}`` is the
    joined per-line contribution block.

    Pass ``extra_placeholders={"name": "value", ...}`` to declare additional
    substitutions; extras may not overlap built-in names. An identifier-shaped
    ``{token}`` in a template that is neither a built-in nor in
    extra_placeholders raises ``ValueError`` at render time.
    """

    def __init__(
        self,
        *,
        system_template: str = DEFAULT_SYSTEM_TEMPLATE,
        user_template: str = DEFAULT_USER_TEMPLATE,
        extra_placeholders: dict[str, str] | None = None,
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

    def render(
        self,
        request: ExplanationRequest,
        schema: DatasetSchema,
        config: ExplanationConfig,
    ) -> tuple[str, str]:
        if not isinstance(request, TabularExplanationRequest):
            raise TypeError(
                f"FeatureImportanceTabularPromptTemplate requires a TabularExplanationRequest, "
                f"got {type(request).__name__}."
            )

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

        ordered = sorted(request.contributions, key=lambda c: -abs(c.importance))
        k = config.top_k_features
        if len(ordered) > k:
            boundary = abs(ordered[k - 1].importance)
            cut = k
            while cut < len(ordered) and abs(ordered[cut].importance) == boundary:
                cut += 1
            ordered = ordered[:cut]

        contribution_lines = []
        for c in ordered:
            feat = schema.feature(c.name)
            unit = f" [{feat.unit}]" if feat.unit else ""
            sign = "+" if c.importance >= 0 else "-"
            contribution_lines.append(
                f"- {c.name} = {c.value}{unit}: importance={sign}{abs(c.importance):g}"
            )
        contributions = "\n".join(contribution_lines)

        values: dict[str, str] = {
            "target_name": schema.target.name,
            "audience": config.audience,
            "tone": config.tone,
            "max_length_words": str(config.max_length_words),
            "narrative_rules": config.narrative_rules,
            "prediction": str(class_label),
            "contributions": contributions,
        }
        values.update(self._extra)

        system = substitute(self._system_template, values)
        user = substitute(self._user_template, values)
        return system, user
