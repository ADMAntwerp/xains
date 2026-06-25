"""User-facing knobs for the Explainer.

Deployment-wide settings (audience, tone, length, mode) live here. Per-call
inputs (features, prediction, attributions) live in ``types.py``. Keeping
these separate means the same Explainer can run many requests without
rebuilding.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from xains.types import ExplanationMode

Audience = Literal["technical", "business", "end_user"]
Tone = Literal["neutral", "empathetic", "formal"]


# Verbatim multi-line paper-quote (Cedro & Martens 2026). The rule sentences exceed
# the line-length limit on purpose — wrapping would alter the LLM-facing text — so
# E501 is suppressed for this file via [tool.ruff.lint.per-file-ignores] in pyproject.toml.
DEFAULT_NARRATIVE_RULES = """Generate a narrative explanation (an XAI Narrative) based on the following rules:
1. An XAI Narrative should establish a continuous structure by following a clear narrative arc with a beginning, middle, and end, while using explicit linguistic connectives so that individual events can be seen in the perspective of the others.
2. An XAI Narrative should explicitly identify the underlying cause-effect mechanisms to clarify why the system made a particular prediction.
3. An XAI Narrative should explain model's prediction with linguistic fluency, avoiding repetitive, list-like structures.
4. An XAI Narrative should use a lexically diverse vocabulary with an emphasis on active verbs to express how specific features influence the final prediction."""


class ExplanationConfig(BaseModel):
    """Knobs governing how the Explainer talks to the LLM."""

    model_config = ConfigDict(extra="forbid")

    audience: Audience = "end_user"
    language: str = "en"
    max_length_words: int = Field(default=150, gt=0)
    tone: Tone = "neutral"

    # How many top-ranked contributions the prompt will reference.
    top_k_features: int = Field(default=5, gt=0)

    # Required: explain via feature-importance contributions, the counterfactual, or weave both.
    # Counterfactual and feature_importance_counterfactual require request.counterfactual.
    mode: ExplanationMode

    run_guardrails: bool = True
    extract_narrative: bool = True

    # Injected into the system prompt by every narrative-generating template.
    narrative_rules: str = DEFAULT_NARRATIVE_RULES
