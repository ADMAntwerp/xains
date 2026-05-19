"""FactualTabularPromptTemplate — verbalize tabular attributions as evidence prose."""

from xainarratives.config import ExplanationConfig
from xainarratives.prompts.base import PromptTemplate
from xainarratives.schema import DatasetSchema
from xainarratives.types import ExplanationRequest, TabularExplanationRequest


class FactualTabularPromptTemplate(PromptTemplate):
    """Render a factual tabular explanation prompt.

    Contributions are selected top-k by ``abs(importance)`` descending; ties
    at the k-th boundary widen the cut to include all tied contributions. The
    ``rank`` field on each contribution is ignored.
    """

    def render(
        self,
        request: ExplanationRequest,
        schema: DatasetSchema,
        config: ExplanationConfig,
    ) -> tuple[str, str]:
        if not isinstance(request, TabularExplanationRequest):
            raise TypeError(
                f"FactualTabularPromptTemplate requires a TabularExplanationRequest, "
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

        system_header = (
            f"You are explaining a model prediction for the '{schema.target.name}' target. "
            f"Audience: {config.audience}. Tone: {config.tone}. "
            f"Keep the explanation under {config.max_length_words} words."
        )
        system = f"{system_header}\n\n{config.narrative_rules}"

        lines = [
            f"Prediction: {class_label}.",
            "Top contributions by magnitude:",
        ]
        for c in ordered:
            feat = schema.feature(c.name)
            unit = f" [{feat.unit}]" if feat.unit else ""
            sign = "+" if c.importance >= 0 else "-"
            lines.append(f"- {c.name} = {c.value}{unit}: importance={sign}{abs(c.importance):g}")
        user = "\n".join(lines)

        return system, user
