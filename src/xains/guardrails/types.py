"""Pydantic models for the guardrails + narrative-extraction layer.

* ``GuardrailResult`` — result record for any single guardrail check.
* ``FeatureClaim`` — per-feature structured claim from a feature-importance
  narrative.
* ``NarrativeExtraction`` — full per-narrative extraction record (FI mode).
* ``CounterfactualFeatureClaim`` — per-feature structured claim from a
  counterfactual narrative (ADR 0031).
* ``CounterfactualExtraction`` — full per-narrative extraction record (CF mode).

The FI extraction schema mirrors Ichmoukhamedov et al. 2024
(arXiv:2412.10220). Per ADR 0007, feature-name resolution happens at
extraction time: the LLM maps each narrative mention to a schema feature
name (recorded in the claim's ``resolved_to`` and used as the dict key
in the extraction record) or marks it as unresolved (``resolved_to is None``;
recorded in the ``hallucinations`` / ``invented`` channel).
"""

from itertools import chain
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
    narrative_name: str = Field(min_length=1)
    resolved_to: str | None = None


class NarrativeExtraction(BaseModel):
    """Full extraction record for one explanation narrative.

    ``features`` is keyed by schema feature name (the LLM's resolution).
    ``hallucinations`` lists feature mentions the LLM could not resolve.
    Ranks are 1-indexed narrative-order positions, dense over the union of
    both channels.
    """

    model_config = ConfigDict(extra="forbid")

    features: dict[str, FeatureClaim] = Field(default_factory=dict)
    hallucinations: list[FeatureClaim] = Field(default_factory=list)
    prompt_version: str = Field(min_length=1)
    model_name: str = Field(min_length=1)

    @model_validator(mode="after")
    def _ranks_are_dense_permutation(self) -> Self:
        ranks = sorted(claim.rank for claim in chain(self.features.values(), self.hallucinations))
        total = len(self.features) + len(self.hallucinations)
        expected = list(range(1, total + 1))
        if ranks != expected:
            raise ValueError(
                f"NarrativeExtraction: ranks must be a dense permutation of "
                f"1..N (N={total}) over features and hallucinations; got {ranks}."
            )
        return self

    @model_validator(mode="after")
    def _resolved_features_key_matches_resolved_to(self) -> Self:
        for name, claim in self.features.items():
            if claim.resolved_to != name:
                raise ValueError(
                    f"NarrativeExtraction.features[{name!r}].resolved_to must "
                    f"equal the dict key; got {claim.resolved_to!r}."
                )
        return self

    @model_validator(mode="after")
    def _hallucinations_have_no_resolution(self) -> Self:
        for i, claim in enumerate(self.hallucinations):
            if claim.resolved_to is not None:
                raise ValueError(
                    f"NarrativeExtraction.hallucinations[{i}].resolved_to "
                    f"must be None; got {claim.resolved_to!r}."
                )
        return self


# ========================================================================
# Counterfactual extraction (ADR 0031)
# ========================================================================


class CounterfactualFeatureClaim(BaseModel):
    """Structured change-claim the narrative makes about a single feature.

    ``stated_before`` / ``stated_after`` are the values the narrative
    attributes to the feature (or ``None`` when the narrative is silent).
    ``stated_direction`` is a free-form direction word the narrative uses
    (``"increased"``, ``"decreased"``, ``"changed"``, ...) captured for
    diagnostics; it is not scored in this commit (ADR 0031).
    """

    model_config = ConfigDict(extra="forbid")

    narrative_name: str = Field(min_length=1)
    resolved_to: str | None = None
    stated_before: Any = None
    stated_after: Any = None
    stated_direction: str | None = None


class CounterfactualExtraction(BaseModel):
    """Full extraction record for one counterfactual narrative.

    ``changes`` is keyed by schema feature name (the LLM's resolution).
    ``invented`` lists feature mentions the LLM could not resolve (the
    CF analogue of ``NarrativeExtraction.hallucinations``).
    """

    model_config = ConfigDict(extra="forbid")

    changes: dict[str, CounterfactualFeatureClaim] = Field(default_factory=dict)
    invented: list[CounterfactualFeatureClaim] = Field(default_factory=list)
    prompt_version: str = Field(min_length=1)
    model_name: str = Field(min_length=1)

    @model_validator(mode="after")
    def _resolved_changes_key_matches_resolved_to(self) -> Self:
        for name, claim in self.changes.items():
            if claim.resolved_to != name:
                raise ValueError(
                    f"CounterfactualExtraction.changes[{name!r}].resolved_to "
                    f"must equal the dict key; got {claim.resolved_to!r}."
                )
        return self

    @model_validator(mode="after")
    def _invented_have_no_resolution(self) -> Self:
        for i, claim in enumerate(self.invented):
            if claim.resolved_to is not None:
                raise ValueError(
                    f"CounterfactualExtraction.invented[{i}].resolved_to "
                    f"must be None; got {claim.resolved_to!r}."
                )
        return self
