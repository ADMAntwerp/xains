"""Tests for xain.providers.mock + Protocol conformance."""

import pytest

from xain.providers import LLMProvider, MockLLMProvider


def test_default_response() -> None:
    p = MockLLMProvider()
    r = p.generate("sys", "user")
    assert r.text == "mock response"
    assert r.model_name == "mock-v0"


def test_list_cycles() -> None:
    p = MockLLMProvider(responses=["a", "b"])
    assert p.generate("s", "u").text == "a"
    assert p.generate("s", "u").text == "b"
    assert p.generate("s", "u").text == "a"  # cycles


def test_callable_responses() -> None:
    p = MockLLMProvider(responses=lambda sys, user: f"got:{user}")
    assert p.generate("sys", "hi").text == "got:hi"
    assert p.generate("sys", "yo").text == "got:yo"


def test_empty_list_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        MockLLMProvider(responses=[])


def test_custom_model_name() -> None:
    p = MockLLMProvider(model_name="fake-gpt")
    assert p.generate("s", "u").model_name == "fake-gpt"


def test_implements_protocol() -> None:
    """Structural-typing check: MockLLMProvider satisfies LLMProvider."""
    p = MockLLMProvider()
    assert isinstance(p, LLMProvider)
