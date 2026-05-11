"""Unit tests for HuggingFacePerplexityProvider.

Uses sshleifer/tiny-gpt2 (~10MB) as the test model: real GPT-2 weights
(deterministic non-trivial perplexities) but tiny enough for fast tests
without GPU.
"""

import sys
import warnings

import pytest

from xainarratives.metrics import HuggingFacePerplexityProvider, PerplexityProvider

_TINY_MODEL = "sshleifer/tiny-gpt2"


@pytest.fixture(scope="module")
def hf_provider() -> HuggingFacePerplexityProvider:
    """Shared CPU provider; reused across tests that don't need fresh state."""
    return HuggingFacePerplexityProvider(model_name=_TINY_MODEL, device="cpu")


# ------------------------------------------------------ happy path


def test_hf_provider_returns_float_for_normal_text(
    hf_provider: HuggingFacePerplexityProvider,
) -> None:
    result = hf_provider.compute("The applicant defaulted within the window.")
    assert result is not None
    assert isinstance(result, float)
    assert result > 0


def test_hf_provider_deterministic(
    hf_provider: HuggingFacePerplexityProvider,
) -> None:
    """Same text → same perplexity (no_grad, eval mode, fixed weights)."""
    text = "The applicant defaulted within the window."
    a = hf_provider.compute(text)
    b = hf_provider.compute(text)
    assert a is not None and b is not None
    assert a == b


# ------------------------------------------------------ degenerate inputs


def test_hf_provider_returns_none_for_empty_text(
    hf_provider: HuggingFacePerplexityProvider,
) -> None:
    assert hf_provider.compute("") is None


def test_hf_provider_returns_none_for_whitespace_text(
    hf_provider: HuggingFacePerplexityProvider,
) -> None:
    assert hf_provider.compute("   \n\t  ") is None


def test_hf_provider_returns_none_for_single_token(
    hf_provider: HuggingFacePerplexityProvider,
) -> None:
    """'a' tokenizes to a single GPT-2 BPE token; with only one token there's
    no next-token prediction to compute loss against, so the provider returns
    None per its contract.
    """
    assert hf_provider.compute("a") is None


# ------------------------------------------------------ truncation


def test_hf_provider_truncation_emits_warning_once() -> None:
    """Long input exceeding max_length triggers a UserWarning on first call only."""
    provider = HuggingFacePerplexityProvider(model_name=_TINY_MODEL, device="cpu", max_length=10)
    long_text = "The quick brown fox jumps over the lazy dog. " * 20
    with pytest.warns(UserWarning, match="exceeds max_length"):
        provider.compute(long_text)
    # Subsequent calls do not warn.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        provider.compute(long_text)
    assert not any(issubclass(w.category, UserWarning) for w in caught)


# ------------------------------------------------------ device handling


def test_hf_provider_custom_device_cpu() -> None:
    """Explicit device='cpu' is honored."""
    provider = HuggingFacePerplexityProvider(model_name=_TINY_MODEL, device="cpu")
    assert provider._device == "cpu"
    result = provider.compute("Quick test sentence.")
    assert result is not None


def test_hf_provider_default_device_auto_detects() -> None:
    """device=None auto-detects: 'cuda' if available, else 'cpu'."""
    provider = HuggingFacePerplexityProvider(model_name=_TINY_MODEL, device=None)
    assert provider._device in {"cuda", "cpu"}


# ------------------------------------------------------ Protocol


def test_hf_provider_implements_protocol(
    hf_provider: HuggingFacePerplexityProvider,
) -> None:
    """Structural-typing check: HuggingFacePerplexityProvider satisfies PerplexityProvider."""
    assert isinstance(hf_provider, PerplexityProvider)


# ------------------------------------------------------ missing-deps


def test_hf_provider_raises_when_transformers_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forcing ``import transformers`` to fail yields ImportError with install hint."""
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(ImportError, match=r'pip install "xainarratives\[perplexity-hf\]"'):
        HuggingFacePerplexityProvider(model_name=_TINY_MODEL, device="cpu")


def test_hf_provider_raises_when_torch_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forcing ``import torch`` to fail yields ImportError with install hint."""
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ImportError, match=r'pip install "xainarratives\[perplexity-hf\]"'):
        HuggingFacePerplexityProvider(model_name=_TINY_MODEL, device="cpu")
