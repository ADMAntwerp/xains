"""Tests for xainarratives.config."""

import pytest
from pydantic import ValidationError

from xainarratives import ExplanationConfig


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
