"""Tests for xainarratives.prompts.echo."""

import json

from xainarratives import (
    DatasetSchema,
    ExplanationConfig,
    GraphExplanationRequest,
    ImageExplanationRequest,
    TabularExplanationRequest,
    TextExplanationRequest,
)
from xainarratives.prompts import EchoPromptTemplate


def _render_and_parse(
    request: TabularExplanationRequest
    | TextExplanationRequest
    | ImageExplanationRequest
    | GraphExplanationRequest,
    schema: DatasetSchema,
) -> tuple[str, dict[str, object]]:
    system, user = EchoPromptTemplate().render(request, schema, ExplanationConfig())
    payload = json.loads(user)
    return system, payload


def test_tabular_round_trip(
    tabular_schema: DatasetSchema, tabular_request: TabularExplanationRequest
) -> None:
    _, payload = _render_and_parse(tabular_request, tabular_schema)
    assert set(payload.keys()) == {"schema", "request", "config"}
    assert payload["request"]["modality"] == "tabular"  # type: ignore[index]
    assert payload["schema"]["modality"] == "tabular"  # type: ignore[index]
    assert payload["config"]["mode"] == "auto"  # type: ignore[index]


def test_text_round_trip(text_schema: DatasetSchema, text_request: TextExplanationRequest) -> None:
    _, payload = _render_and_parse(text_request, text_schema)
    assert payload["request"]["modality"] == "text"  # type: ignore[index]
    # Token contribution preserved
    contribs = payload["request"]["contributions"]  # type: ignore[index]
    assert contribs[0]["type"] == "token"


def test_image_round_trip(
    image_schema: DatasetSchema, image_request: ImageExplanationRequest
) -> None:
    _, payload = _render_and_parse(image_request, image_schema)
    assert payload["request"]["modality"] == "image"  # type: ignore[index]
    assert payload["request"]["image_path"] == "/tmp/xray_001.png"  # type: ignore[index]


def test_graph_round_trip(
    graph_schema: DatasetSchema, graph_request: GraphExplanationRequest
) -> None:
    _, payload = _render_and_parse(graph_request, graph_schema)
    assert payload["request"]["modality"] == "graph"  # type: ignore[index]
    # Both node and edge contributions survive the round-trip
    types = {c["type"] for c in payload["request"]["contributions"]}  # type: ignore[index]
    assert types == {"node", "edge"}


def test_user_prompt_is_deterministic(
    tabular_schema: DatasetSchema, tabular_request: TabularExplanationRequest
) -> None:
    """Same inputs must produce byte-identical output (sort_keys=True)."""
    template = EchoPromptTemplate()
    cfg = ExplanationConfig()
    _, u1 = template.render(tabular_request, tabular_schema, cfg)
    _, u2 = template.render(tabular_request, tabular_schema, cfg)
    assert u1 == u2
