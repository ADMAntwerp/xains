"""Tests for xains.types — polymorphism, discriminators, validation."""

from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from xains import (
    EdgeContribution,
    GraphCounterfactual,
    GraphExplanationRequest,
    ImageCounterfactual,
    ImageExplanationRequest,
    NodeContribution,
    Prediction,
    RegionContribution,
    TabularContribution,
    TabularCounterfactual,
    TabularExplanationRequest,
    TextCounterfactual,
    TextExplanationRequest,
    TokenContribution,
)
from xains.types import Contribution, CounterfactualInstance, ExplanationRequest


class TestPrediction:
    def test_minimal(self) -> None:
        p = Prediction(predicted_class=1)
        assert p.predicted_class == 1
        assert p.probabilities is None

    def test_confidence_bounds(self) -> None:
        with pytest.raises(ValidationError):
            Prediction(predicted_class=1, confidence=1.5)
        with pytest.raises(ValidationError):
            Prediction(predicted_class=1, confidence=-0.1)


class TestContributionDiscriminator:
    adapter: TypeAdapter[Contribution] = TypeAdapter(Contribution)

    def test_tabular_dispatch(self) -> None:
        c = self.adapter.validate_python(
            {"type": "tabular", "name": "age", "value": 29, "importance": 0.1}
        )
        assert isinstance(c, TabularContribution)

    def test_token_dispatch(self) -> None:
        c = self.adapter.validate_python(
            {"type": "token", "token": "x", "span": (0, 1), "importance": 0.5}
        )
        assert isinstance(c, TokenContribution)

    def test_region_dispatch(self) -> None:
        c = self.adapter.validate_python({"type": "region", "region_id": 0, "importance": 0.5})
        assert isinstance(c, RegionContribution)

    def test_node_dispatch(self) -> None:
        c = self.adapter.validate_python({"type": "node", "node_id": "n1", "importance": 0.5})
        assert isinstance(c, NodeContribution)

    def test_edge_dispatch(self) -> None:
        c = self.adapter.validate_python(
            {"type": "edge", "src": "a", "dst": "b", "importance": 0.5}
        )
        assert isinstance(c, EdgeContribution)

    def test_unknown_type_rejected(self) -> None:
        with pytest.raises(ValidationError):
            self.adapter.validate_python({"type": "junk", "importance": 0.1})


class TestTabularExplanationRequest:
    def test_valid(self) -> None:
        r = TabularExplanationRequest(
            features={"x": 1},
            prediction=Prediction(predicted_class=0),
            contributions=[TabularContribution(name="x", value=1, importance=0.5)],
        )
        assert r.modality.value == "tabular"

    def test_empty_contributions_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TabularExplanationRequest(
                features={"x": 1},
                prediction=Prediction(predicted_class=0),
                contributions=[],
            )

    def test_wrong_contribution_type_rejected(self) -> None:
        # A NodeContribution should not be accepted by TabularExplanationRequest
        with pytest.raises(ValidationError):
            TabularExplanationRequest(
                features={"x": 1},
                prediction=Prediction(predicted_class=0),
                contributions=[NodeContribution(node_id="n1", importance=0.5)],
            )

    def test_counterfactuals_must_be_non_empty_if_set(self) -> None:
        with pytest.raises(ValidationError):
            TabularExplanationRequest(
                features={"x": 1},
                prediction=Prediction(predicted_class=0),
                contributions=[TabularContribution(name="x", value=1, importance=0.5)],
                counterfactuals=[],
            )


class TestExplanationRequestDiscriminator:
    adapter: TypeAdapter[ExplanationRequest] = TypeAdapter(ExplanationRequest)

    @pytest.mark.parametrize(
        ("payload", "cls"),
        [
            (
                {
                    "modality": "tabular",
                    "features": {"x": 1},
                    "prediction": {"predicted_class": 0},
                    "contributions": [
                        {"type": "tabular", "name": "x", "value": 1, "importance": 0.1}
                    ],
                },
                TabularExplanationRequest,
            ),
            (
                {
                    "modality": "text",
                    "text": "hi there",
                    "prediction": {"predicted_class": "pos"},
                    "contributions": [
                        {"type": "token", "token": "hi", "span": (0, 2), "importance": 0.1}
                    ],
                },
                TextExplanationRequest,
            ),
            (
                {
                    "modality": "image",
                    "image_path": "/tmp/x.png",
                    "prediction": {"predicted_class": 0},
                    "contributions": [{"type": "region", "region_id": 0, "importance": 0.1}],
                },
                ImageExplanationRequest,
            ),
            (
                {
                    "modality": "graph",
                    "target_node_id": "n1",
                    "prediction": {"predicted_class": 0},
                    "contributions": [{"type": "node", "node_id": "n1", "importance": 0.1}],
                },
                GraphExplanationRequest,
            ),
        ],
    )
    def test_dispatch(self, payload: dict[str, Any], cls: type) -> None:
        r = self.adapter.validate_python(payload)
        assert isinstance(r, cls)


class TestCounterfactualDiscriminator:
    adapter: TypeAdapter[CounterfactualInstance] = TypeAdapter(CounterfactualInstance)

    def test_tabular(self) -> None:
        c = self.adapter.validate_python(
            {"type": "tabular", "predicted_class": 0, "features": {"x": 2}}
        )
        assert isinstance(c, TabularCounterfactual)

    def test_text(self) -> None:
        c = self.adapter.validate_python({"type": "text", "predicted_class": "neg", "text": "bad"})
        assert isinstance(c, TextCounterfactual)

    def test_image(self) -> None:
        c = self.adapter.validate_python(
            {"type": "image", "predicted_class": 0, "image_path": "/tmp/cf.png"}
        )
        assert isinstance(c, ImageCounterfactual)

    def test_graph(self) -> None:
        c = self.adapter.validate_python({"type": "graph", "predicted_class": 0})
        assert isinstance(c, GraphCounterfactual)
