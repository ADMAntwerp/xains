"""User-facing knobs for the Explainer.

Deployment-wide settings (audience, tone, length, mode) live here. Per-call
inputs (features, prediction, attributions) live in ``types.py``. Keeping
these separate means the same Explainer can run many requests without
rebuilding.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Audience = Literal["technical", "business", "end_user"]
Tone = Literal["neutral", "empathetic", "formal"]
ExplanationModeOrAuto = Literal["factual", "contrastive", "counterfactual", "auto"]


class ExplanationConfig(BaseModel):
    """Knobs governing how the Explainer talks to the LLM."""

    model_config = ConfigDict(extra="forbid")

    audience: Audience = "end_user"
    language: str = "en"
    max_length_words: int = Field(default=150, gt=0)
    tone: Tone = "neutral"

    # How many top-ranked contributions the prompt will reference.
    top_k_features: int = Field(default=5, gt=0)

    include_confidence: bool = True
    include_caveats: bool = True

    # "auto" resolves to factual / contrastive / counterfactual based on the
    # request (see Explainer._resolve_mode). Setting an explicit value forces
    # that mode and is an error if the request doesn't support it.
    mode: ExplanationModeOrAuto = "auto"

    run_guardrails: bool = True
    extract_narrative: bool = True
