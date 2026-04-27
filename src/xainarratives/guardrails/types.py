"""Pydantic models for the guardrails + narrative-extraction layer.

* ``GuardrailResult`` — result record for any single guardrail check.
* ``FeatureClaim`` — per-feature structured claim extracted from a narrative.
* ``NarrativeExtraction`` — full per-narrative extraction record.

The extraction schema mirrors Ichmoukhamedov et al. 2024 (arXiv:2412.10220).
Feature-name keys are the names *as used in the narrative* — synonyms and
hallucinations are preserved intentionally; set-membership normalization is
the job of the downstream scoring layer (PR 5).
"""

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class GuardrailResult(BaseModel):
    """Outcome of a single guardrail check."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    severity: Literal["advisory", "failure"]
    passed: bool
    details: dict[str, Any] = Field(default_factory=dict)


class FeatureClaim(BaseModel):
    """Structured claim the narrative makes about a single feature."""

    model_config = ConfigDict(extra="forbid")

    rank: int = Field(ge=1)
    sign: Literal[-1, 0, 1]
    value: Any = None
    assumption: str = ""


class NarrativeExtraction(BaseModel):
    """Full extraction record for one explanation narrative."""

    model_config = ConfigDict(extra="forbid")

    features: dict[str, FeatureClaim] = Field(default_factory=dict)
    prompt_version: str = Field(min_length=1)
    model_name: str = Field(min_length=1)

    @model_validator(mode="after")
    def _ranks_are_dense_permutation(self) -> Self:
        ranks = sorted(claim.rank for claim in self.features.values())
        expected = list(range(1, len(self.features) + 1))
        if ranks != expected:
            raise ValueError(
                f"NarrativeExtraction: ranks must be a dense permutation of "
                f"1..N (N={len(self.features)}); got {ranks}."
            )
        return self
