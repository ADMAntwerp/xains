"""Tests for xainarratives.config."""

import pytest
from pydantic import ValidationError

from xainarratives import ExplanationConfig
from xainarratives.config import DEFAULT_NARRATIVE_RULES


def test_defaults() -> None:
    c = ExplanationConfig()
    assert c.audience == "end_user"
    assert c.max_length_words == 150
    assert c.top_k_features == 5
    assert c.mode == "auto"


def test_max_length_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        ExplanationConfig(max_length_words=0)


def test_top_k_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        ExplanationConfig(top_k_features=0)


def test_extra_fields_rejected() -> None:
    with pytest.raises(ValidationError):
        ExplanationConfig(unknown=True)  # type: ignore[call-arg]


def test_mode_choices() -> None:
    for m in ("factual", "contrastive", "counterfactual", "auto"):
        ExplanationConfig(mode=m)
    with pytest.raises(ValidationError):
        ExplanationConfig(mode="bogus")


def test_narrative_rules_default() -> None:
    c = ExplanationConfig()
    assert c.narrative_rules == DEFAULT_NARRATIVE_RULES
    # Spot-check paper-verbatim content.
    assert (
        "Generate a narrative explanation (an XAI Narrative) based on the following rules:"
    ) in c.narrative_rules
    for prefix in ("1. ", "2. ", "3. ", "4. "):
        assert prefix in c.narrative_rules


def test_narrative_rules_override() -> None:
    c = ExplanationConfig(narrative_rules="custom rules block")
    assert c.narrative_rules == "custom rules block"
    # Override replaces; it does not append to the default.
    assert DEFAULT_NARRATIVE_RULES not in c.narrative_rules
