"""Integration tests for OpenAICompatibleEchoProvider.

Two modes (mirroring tests/integration/test_providers_anthropic.py):

* Cassette-replayed tests use ``@pytest.mark.vcr`` against Together.ai's
  ``/v1/completions`` endpoint with ``echo=True, logprobs=1, max_tokens=1``.
* ``test_api_provider_live`` is ``@pytest.mark.live``, skipped by default,
  and runs only when ``TOGETHER_API_KEY`` is set.

Cassette scrubbing (``authorization`` header) is inherited from the
``vcr_config`` fixture in ``tests/integration/conftest.py``.
"""

import os
import sys

import pytest

from xainarratives.metrics import OpenAICompatibleEchoProvider, PerplexityProvider

_BASE_URL = "https://api.together.xyz/v1"
_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct-Lite"
_PLACEHOLDER_KEY = "sk-test-placeholder"


@pytest.fixture
def provider() -> OpenAICompatibleEchoProvider:
    """Provider with a real ``TOGETHER_API_KEY`` if set, else placeholder.

    Placeholder suffices for cassette replay and mock tests. Setting the
    real env var enables cassette re-recording
    (``--record-mode=once`` / ``--record-mode=rewrite``) without manual
    fixture surgery.
    """
    return OpenAICompatibleEchoProvider(
        base_url=_BASE_URL,
        api_key=os.environ.get("TOGETHER_API_KEY", _PLACEHOLDER_KEY),
        model=_MODEL,
    )


# ------------------------------------------------------ cassette


@pytest.mark.vcr
def test_api_provider_returns_float_for_normal_text(
    provider: OpenAICompatibleEchoProvider,
) -> None:
    """Cassette replay: real Together.ai response → finite positive perplexity."""
    result = provider.compute("The applicant defaulted within the window.")
    assert result is not None
    assert isinstance(result, float)
    assert result > 0


# ------------------------------------------------------ short-circuit, no HTTP


def test_api_provider_returns_none_for_empty_text(
    provider: OpenAICompatibleEchoProvider,
) -> None:
    """Empty text short-circuits before any HTTP call."""
    assert provider.compute("") is None


def test_api_provider_returns_none_for_whitespace_text(
    provider: OpenAICompatibleEchoProvider,
) -> None:
    assert provider.compute("   \n\t  ") is None


# ------------------------------------------------------ HTTP error path


def test_api_provider_returns_none_on_http_error(
    provider: OpenAICompatibleEchoProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``openai.OpenAIError`` from the SDK → returns None per Protocol contract."""
    import openai

    def _raise(**kwargs: object) -> None:
        raise openai.OpenAIError("simulated transport error")

    monkeypatch.setattr(provider._client.completions, "create", _raise)
    assert provider.compute("any narrative text here") is None


# ------------------------------------------------------ Protocol


def test_api_provider_implements_protocol(
    provider: OpenAICompatibleEchoProvider,
) -> None:
    """Structural-typing check: OpenAICompatibleEchoProvider satisfies PerplexityProvider."""
    assert isinstance(provider, PerplexityProvider)


# ------------------------------------------------------ missing-deps


def test_api_provider_raises_when_openai_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forcing ``import openai`` to fail yields ImportError with install hint."""
    monkeypatch.setitem(sys.modules, "openai", None)
    with pytest.raises(ImportError, match=r'pip install "xainarratives\[perplexity-api\]"'):
        OpenAICompatibleEchoProvider(
            base_url=_BASE_URL,
            api_key=_PLACEHOLDER_KEY,
            model=_MODEL,
        )


# ------------------------------------------------------ live


@pytest.mark.live
def test_api_provider_live() -> None:
    """Real API call against Together.ai. Requires ``TOGETHER_API_KEY`` env var."""
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        pytest.skip("TOGETHER_API_KEY not set")
    provider = OpenAICompatibleEchoProvider(
        base_url=_BASE_URL,
        api_key=api_key,
        model=_MODEL,
    )
    result = provider.compute("The applicant defaulted within the window.")
    assert result is not None
    assert isinstance(result, float)
    assert result > 0
