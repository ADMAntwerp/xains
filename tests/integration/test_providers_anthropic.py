"""Integration tests for AnthropicProvider.

Two modes:

* ``test_generate_returns_llmresponse`` — cassette-replayed. Runs offline in
  CI after the cassette has been recorded once. Exercises the full SDK →
  LLMResponse mapping path against a recorded HTTP response.
* ``test_generate_live`` — hits the real Anthropic API. Marked
  ``@pytest.mark.live`` and skipped in default CI. Requires
  ``ANTHROPIC_API_KEY`` in the environment.
"""

import os

import pytest

from xains.providers import AnthropicProvider, LLMResponse


@pytest.mark.vcr
def test_generate_returns_llmresponse() -> None:
    # The api_key value is irrelevant during cassette replay (VCR intercepts
    # before any HTTP call); during recording it must be a real key. Headers
    # are scrubbed from the committed cassette via conftest.vcr_config.
    provider = AnthropicProvider(
        model="claude-haiku-4-5",
        max_tokens=64,
        api_key=os.environ.get("ANTHROPIC_API_KEY", "sk-test-placeholder"),
    )
    response = provider.generate("You are terse.", "Say hi in one word.")

    assert isinstance(response, LLMResponse)
    assert response.text
    assert response.model_name.startswith("claude-haiku-4-5")

    # ADR 0005: tokens_used is exactly {"input", "output", "total"}.
    assert response.tokens_used is not None
    assert set(response.tokens_used.keys()) == {"input", "output", "total"}
    assert response.tokens_used["input"] > 0
    assert response.tokens_used["output"] > 0
    assert (
        response.tokens_used["total"]
        == response.tokens_used["input"] + response.tokens_used["output"]
    )


@pytest.mark.live
def test_generate_live() -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    provider = AnthropicProvider(
        model="claude-haiku-4-5",
        max_tokens=64,
        api_key=os.environ["ANTHROPIC_API_KEY"],
    )
    response = provider.generate("You are terse.", "Say hi in one word.")

    assert isinstance(response, LLMResponse)
    assert response.text
    assert response.model_name.startswith("claude-haiku-4-5")
    assert response.tokens_used is not None
    assert set(response.tokens_used.keys()) == {"input", "output", "total"}
