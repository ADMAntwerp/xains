"""Shared fixtures: one valid schema + request pair per modality."""

import pytest

from xainarratives import (
    DatasetSchema,
    EdgeContribution,
    FeatureSchema,
    GraphExplanationRequest,
    GraphSpec,
    ImageExplanationRequest,
    ImageSpec,
    Modality,
    NodeContribution,
    Prediction,
    RegionContribution,
    TabularContribution,
    TabularExplanationRequest,
    TargetSchema,
    TextExplanationRequest,
    TextSpec,
    TokenContribution,
)

# ------------------------------ tabular ------------------------------ #


@pytest.fixture
def tabular_schema() -> DatasetSchema:
    return DatasetSchema(
        modality=Modality.TABULAR,
        name="credit_risk",
        description="Predicts 24-month default on personal loans.",
        target=TargetSchema(
            name="default",
            description="Whether the applicant defaulted within 24 months.",
            classes={0: "Repaid", 1: "Defaulted"},
        ),
        features=[
            FeatureSchema(
                name="age",
                dtype="numeric",
                unit="years",
                description="Applicant age at application.",
            ),
            FeatureSchema(
                name="dti",
                dtype="numeric",
                description="Debt-to-income ratio.",
            ),
        ],
    )


@pytest.fixture
def tabular_request() -> TabularExplanationRequest:
    return TabularExplanationRequest(
        features={"age": 29, "dti": 0.41},
        prediction=Prediction(predicted_class=1, probabilities={0: 0.2, 1: 0.8}),
        contributions=[
            TabularContribution(name="dti", value=0.41, importance=0.37, rank=0),
            TabularContribution(name="age", value=29, importance=-0.12, rank=1),
        ],
    )


# -------------------------------- text -------------------------------- #


@pytest.fixture
def text_schema() -> DatasetSchema:
    return DatasetSchema(
        modality=Modality.TEXT,
        name="sentiment",
        description="Binary sentiment of product reviews.",
        target=TargetSchema(
            name="sentiment",
            description="Overall sentiment polarity.",
            classes={"pos": "Positive", "neg": "Negative"},
        ),
        text_spec=TextSpec(language="en"),
    )


@pytest.fixture
def text_request() -> TextExplanationRequest:
    return TextExplanationRequest(
        text="the battery life is amazing",
        prediction=Prediction(predicted_class="pos"),
        contributions=[
            TokenContribution(token="amazing", span=(21, 28), importance=0.62),
            TokenContribution(token="battery", span=(4, 11), importance=0.10),
        ],
    )


# ------------------------------- image ------------------------------- #


@pytest.fixture
def image_schema() -> DatasetSchema:
    return DatasetSchema(
        modality=Modality.IMAGE,
        name="chest_xray",
        description="Pneumonia detection from chest radiographs.",
        target=TargetSchema(
            name="pneumonia",
            description="Presence of pneumonia.",
            classes={0: "Normal", 1: "Pneumonia"},
        ),
        image_spec=ImageSpec(height=224, width=224, channels=1),
    )


@pytest.fixture
def image_request() -> ImageExplanationRequest:
    return ImageExplanationRequest(
        image_path="/tmp/xray_001.png",
        prediction=Prediction(predicted_class=1, probabilities={0: 0.1, 1: 0.9}),
        contributions=[
            RegionContribution(
                region_id=0,
                bbox=(80, 120, 160, 200),
                description="Right lower lobe opacity",
                importance=0.71,
            ),
        ],
    )


# ------------------------------- graph ------------------------------- #


@pytest.fixture
def graph_schema() -> DatasetSchema:
    return DatasetSchema(
        modality=Modality.GRAPH,
        name="fraud_ring",
        description="Node-level fraud classification on a transaction graph.",
        target=TargetSchema(
            name="fraud",
            description="Whether the account is fraudulent.",
            classes={0: "Legit", 1: "Fraud"},
        ),
        graph_spec=GraphSpec(directed=True, node_types=["account"], edge_types=["transfer"]),
    )


@pytest.fixture
def graph_request() -> GraphExplanationRequest:
    return GraphExplanationRequest(
        target_node_id="acct_42",
        prediction=Prediction(predicted_class=1),
        contributions=[
            NodeContribution(node_id="acct_7", label="account", importance=0.55),
            EdgeContribution(src="acct_7", dst="acct_42", edge_type="transfer", importance=0.33),
        ],
    )
