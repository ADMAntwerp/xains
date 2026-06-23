"""Tests for xains.schema."""

import pytest
from pydantic import ValidationError

from xains import (
    DatasetSchema,
    FeatureSchema,
    GraphSpec,
    ImageSpec,
    Modality,
    TargetSchema,
    TextSpec,
)


class TestFeatureSchema:
    def test_numeric_feature_valid(self) -> None:
        f = FeatureSchema(name="age", dtype="numeric", description="Age in years.", unit="years")
        assert f.name == "age"
        assert f.categories is None

    def test_categorical_requires_categories(self) -> None:
        with pytest.raises(ValidationError, match="requires `categories`"):
            FeatureSchema(name="color", dtype="categorical", description="Color.")

    def test_categorical_with_categories_valid(self) -> None:
        f = FeatureSchema(
            name="color",
            dtype="categorical",
            description="Color.",
            categories=["r", "g", "b"],
        )
        assert f.categories == ["r", "g", "b"]

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FeatureSchema(name="", dtype="numeric", description="x")

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FeatureSchema(
                name="x",
                dtype="numeric",
                description="x",
                unknown_field=1,  # type: ignore[call-arg]
            )


class TestTargetSchema:
    def test_minimum_two_classes(self) -> None:
        with pytest.raises(ValidationError):
            TargetSchema(name="y", description="d", classes={0: "A"})

    def test_class_descriptions_must_be_subset(self) -> None:
        with pytest.raises(ValidationError, match="unknown classes"):
            TargetSchema(
                name="y",
                description="d",
                classes={0: "A", 1: "B"},
                class_descriptions={0: "desc", 2: "desc"},
            )

    def test_class_descriptions_subset_valid(self) -> None:
        t = TargetSchema(
            name="y",
            description="d",
            classes={0: "A", 1: "B"},
            class_descriptions={0: "only zero"},
        )
        assert t.class_descriptions == {0: "only zero"}


class TestDatasetSchema:
    def test_tabular_valid(self, tabular_schema: DatasetSchema) -> None:
        assert tabular_schema.modality is Modality.TABULAR
        assert tabular_schema.feature("age").unit == "years"

    def test_tabular_requires_features(self) -> None:
        with pytest.raises(ValidationError, match="requires `features`"):
            DatasetSchema(
                modality=Modality.TABULAR,
                name="n",
                description="d",
                target=TargetSchema(name="y", description="d", classes={0: "a", 1: "b"}),
            )

    def test_text_requires_text_spec(self) -> None:
        with pytest.raises(ValidationError, match="requires `text_spec`"):
            DatasetSchema(
                modality=Modality.TEXT,
                name="n",
                description="d",
                target=TargetSchema(name="y", description="d", classes={0: "a", 1: "b"}),
            )

    def test_image_requires_image_spec(self) -> None:
        with pytest.raises(ValidationError, match="requires `image_spec`"):
            DatasetSchema(
                modality=Modality.IMAGE,
                name="n",
                description="d",
                target=TargetSchema(name="y", description="d", classes={0: "a", 1: "b"}),
            )

    def test_graph_requires_graph_spec(self) -> None:
        with pytest.raises(ValidationError, match="requires `graph_spec`"):
            DatasetSchema(
                modality=Modality.GRAPH,
                name="n",
                description="d",
                target=TargetSchema(name="y", description="d", classes={0: "a", 1: "b"}),
            )

    def test_image_schema_with_spec_valid(self) -> None:
        s = DatasetSchema(
            modality=Modality.IMAGE,
            name="n",
            description="d",
            target=TargetSchema(name="y", description="d", classes={0: "a", 1: "b"}),
            image_spec=ImageSpec(height=64, width=64),
        )
        assert s.image_spec is not None
        assert s.image_spec.channels == 3  # default

    def test_feature_lookup_unknown(self, tabular_schema: DatasetSchema) -> None:
        with pytest.raises(KeyError):
            tabular_schema.feature("nonexistent")

    def test_feature_lookup_on_non_tabular(self) -> None:
        s = DatasetSchema(
            modality=Modality.TEXT,
            name="n",
            description="d",
            target=TargetSchema(name="y", description="d", classes={0: "a", 1: "b"}),
            text_spec=TextSpec(),
        )
        with pytest.raises(KeyError, match="no features"):
            s.feature("anything")


class TestSpecs:
    def test_image_spec_positive_dims(self) -> None:
        with pytest.raises(ValidationError):
            ImageSpec(height=0, width=10)

    def test_graph_spec_defaults(self) -> None:
        g = GraphSpec()
        assert g.directed is False
        assert g.node_types is None
