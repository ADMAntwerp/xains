"""Per-instance request and response types.

All polymorphic families (contributions, counterfactuals, requests) are
discriminated unions keyed on a ``type`` / ``modality`` field so that
callers can pass any member of the family through a single parameter and
pydantic parses the right concrete model.

See docs/decisions/0003-data-model.md for the rationale.
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from xainarratives.schema import Modality

# =========================================================================
# Prediction
# =========================================================================


class Prediction(BaseModel):
    """The model's output for the instance under explanation."""

    model_config = ConfigDict(extra="forbid")

    predicted_class: int | str
    probabilities: dict[int | str, float] | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


# =========================================================================
# Contributions (polymorphic, discriminated by `type`)
# =========================================================================


class _BaseContribution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    importance: float
    rank: int | None = Field(default=None, ge=0)


class TabularContribution(_BaseContribution):
    type: Literal["tabular"] = "tabular"
    name: str = Field(min_length=1)
    value: Any


class TokenContribution(_BaseContribution):
    type: Literal["token"] = "token"
    token: str
    span: tuple[int, int]  # character offsets into the original text


class RegionContribution(_BaseContribution):
    type: Literal["region"] = "region"
    region_id: int = Field(ge=0)
    # (x_min, y_min, x_max, y_max) if applicable
    bbox: tuple[int, int, int, int] | None = None
    # A caption or semantic label for the region (text-LLM-friendly).
    description: str | None = None


class NodeContribution(_BaseContribution):
    type: Literal["node"] = "node"
    node_id: str = Field(min_length=1)
    features: dict[str, Any] = Field(default_factory=dict)
    label: str | None = None


class EdgeContribution(_BaseContribution):
    type: Literal["edge"] = "edge"
    src: str = Field(min_length=1)
    dst: str = Field(min_length=1)
    edge_type: str | None = None


Contribution = Annotated[
    TabularContribution
    | TokenContribution
    | RegionContribution
    | NodeContribution
    | EdgeContribution,
    Field(discriminator="type"),
]


# =========================================================================
# Counterfactuals (polymorphic, discriminated by `type`)
# =========================================================================


class _BaseCounterfactual(BaseModel):
    model_config = ConfigDict(extra="forbid")

    predicted_class: int | str
    probabilities: dict[int | str, float] | None = None
    distance: float | None = Field(default=None, ge=0.0)
    method: str | None = None  # e.g. "DiCE", "Wachter" — provenance only


class TabularCounterfactual(_BaseCounterfactual):
    type: Literal["tabular"] = "tabular"
    features: dict[str, Any]
    # If None, the Explainer may compute changed_features as the diff vs the factual.
    changed_features: list[str] | None = None


class TextCounterfactual(_BaseCounterfactual):
    type: Literal["text"] = "text"
    text: str


class ImageCounterfactual(_BaseCounterfactual):
    type: Literal["image"] = "image"
    # Filesystem path to the counterfactual image. We do not pass bytes around in v0.
    image_path: str
    changed_regions: list[int] | None = None


class GraphCounterfactual(_BaseCounterfactual):
    type: Literal["graph"] = "graph"
    # Changed node features, keyed by node id. Only changed nodes need be present.
    node_features: dict[str, dict[str, Any]] | None = None
    added_edges: list[tuple[str, str]] | None = None
    removed_edges: list[tuple[str, str]] | None = None


CounterfactualInstance = Annotated[
    TabularCounterfactual | TextCounterfactual | ImageCounterfactual | GraphCounterfactual,
    Field(discriminator="type"),
]


# =========================================================================
# Explanation requests (polymorphic, discriminated by `modality`)
# =========================================================================


class _BaseExplanationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction: Prediction
    counterfactuals: list[CounterfactualInstance] | None = Field(default=None, min_length=1)
    contrast_class: int | str | None = None
    instance_id: str | None = None


class TabularExplanationRequest(_BaseExplanationRequest):
    modality: Literal[Modality.TABULAR] = Modality.TABULAR
    features: dict[str, Any]
    contributions: list[TabularContribution] = Field(min_length=1)


class TextExplanationRequest(_BaseExplanationRequest):
    modality: Literal[Modality.TEXT] = Modality.TEXT
    text: str = Field(min_length=1)
    contributions: list[TokenContribution] = Field(min_length=1)


class ImageExplanationRequest(_BaseExplanationRequest):
    modality: Literal[Modality.IMAGE] = Modality.IMAGE
    image_path: str = Field(min_length=1)
    saliency_overlay_path: str | None = None
    contributions: list[RegionContribution] = Field(min_length=1)


class GraphExplanationRequest(_BaseExplanationRequest):
    modality: Literal[Modality.GRAPH] = Modality.GRAPH
    target_node_id: str = Field(min_length=1)
    subgraph_description: str | None = None
    # Graph contributions can be nodes, edges, or both.
    contributions: list[NodeContribution | EdgeContribution] = Field(min_length=1)


ExplanationRequest = Annotated[
    TabularExplanationRequest
    | TextExplanationRequest
    | ImageExplanationRequest
    | GraphExplanationRequest,
    Field(discriminator="modality"),
]


# =========================================================================
# Result
# =========================================================================


ExplanationMode = Literal[
    "feature_importance", "counterfactual", "feature_importance_counterfactual"
]


class ExplanationResult(BaseModel):
    """Output of ``Explainer.explain``. Includes audit metadata."""

    model_config = ConfigDict(extra="forbid")

    text: str
    mode: ExplanationMode
    prompt: str | None  # the rendered prompt sent to the LLM; None for non-LLM generators
    raw_llm_response: str | None
    model_name: str | None
    tokens_used: dict[str, int] | None = None
    latency_ms: float | None = Field(default=None, ge=0.0)
    guardrails: "list[GuardrailResult] | None" = None
    narrative_extraction: "NarrativeExtraction | None" = None
    guardrail_tokens_used: dict[str, int] | None = None


from xainarratives.guardrails.types import GuardrailResult, NarrativeExtraction  # noqa: E402

ExplanationResult.model_rebuild()
