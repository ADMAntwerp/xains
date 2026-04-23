"""Dataset-level schema: features, target, modality-specific metadata.

These objects describe the *model's* world, not a specific prediction. They
are set once per deployment and reused for every instance-level explanation.
"""

from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

FeatureDType = Literal["numeric", "categorical", "ordinal", "boolean", "text"]


class Modality(StrEnum):
    """Data modality the model operates on."""

    TABULAR = "tabular"
    TEXT = "text"
    IMAGE = "image"
    GRAPH = "graph"


class FeatureSchema(BaseModel):
    """Description of a single tabular feature (also used for graph node features)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    dtype: FeatureDType
    description: str = Field(min_length=1)
    unit: str | None = None
    categories: list[str] | None = None

    @model_validator(mode="after")
    def _categorical_requires_categories(self) -> Self:
        if self.dtype in ("categorical", "ordinal") and self.categories is None:
            raise ValueError(
                f"FeatureSchema {self.name!r}: dtype={self.dtype!r} requires `categories`."
            )
        return self


class TargetSchema(BaseModel):
    """Description of the prediction target."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    classes: dict[int | str, str] = Field(min_length=2)
    class_descriptions: dict[int | str, str] | None = None

    @model_validator(mode="after")
    def _class_descriptions_match_classes(self) -> Self:
        if self.class_descriptions is None:
            return self
        extra = set(self.class_descriptions) - set(self.classes)
        if extra:
            raise ValueError(
                f"TargetSchema: class_descriptions has unknown classes {sorted(map(str, extra))}."
            )
        return self


class TextSpec(BaseModel):
    """Text-modality metadata."""

    model_config = ConfigDict(extra="forbid")

    language: str = "en"


class ImageSpec(BaseModel):
    """Image-modality metadata."""

    model_config = ConfigDict(extra="forbid")

    height: int = Field(gt=0)
    width: int = Field(gt=0)
    channels: int = Field(default=3, gt=0)


class GraphSpec(BaseModel):
    """Graph-modality metadata."""

    model_config = ConfigDict(extra="forbid")

    directed: bool = False
    node_types: list[str] | None = None
    edge_types: list[str] | None = None


class DatasetSchema(BaseModel):
    """Everything the LLM needs to know about the dataset and target.

    Set once per model. Reused for every prediction.
    """

    model_config = ConfigDict(extra="forbid")

    modality: Modality
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    target: TargetSchema

    # Modality-specific payloads. Exactly one combination is valid per modality;
    # enforced in the validator below.
    features: list[FeatureSchema] | None = None
    text_spec: TextSpec | None = None
    image_spec: ImageSpec | None = None
    graph_spec: GraphSpec | None = None

    @model_validator(mode="after")
    def _require_modality_payload(self) -> Self:
        if self.modality == Modality.TABULAR and not self.features:
            raise ValueError("DatasetSchema: modality=TABULAR requires `features`.")
        if self.modality == Modality.TEXT and self.text_spec is None:
            raise ValueError("DatasetSchema: modality=TEXT requires `text_spec`.")
        if self.modality == Modality.IMAGE and self.image_spec is None:
            raise ValueError("DatasetSchema: modality=IMAGE requires `image_spec`.")
        if self.modality == Modality.GRAPH and self.graph_spec is None:
            raise ValueError("DatasetSchema: modality=GRAPH requires `graph_spec`.")
        return self

    def feature(self, name: str) -> FeatureSchema:
        """Look up a feature by name. Raises KeyError if not found."""
        if self.features is None:
            raise KeyError(f"Schema has no features; modality={self.modality}.")
        for f in self.features:
            if f.name == name:
                return f
        raise KeyError(f"Unknown feature: {name!r}")
