"""Unit tests for AnthropicProvider.

These tests MUST run without the ``anthropic`` SDK being importable by the
code under test — the ImportError path is the whole point. Protocol
conformance is checked structurally and requires only the class object.
"""

import sys

import pytest

from xain.providers import AnthropicProvider, LLMProvider


def test_implements_protocol() -> None:
    """Structural-typing check: AnthropicProvider satisfies LLMProvider."""
    p = AnthropicProvider(model="claude-haiku-4-5", max_tokens=64)
    assert isinstance(p, LLMProvider)


def test_missing_anthropic_raises_importerror(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the anthropic package is not installed, generate() raises a clear ImportError
    that tells the user the exact pip extra to install.
    """
    # Force ``import anthropic`` inside generate() to fail, regardless of
    # whether the package is installed in this environment.
    monkeypatch.setitem(sys.modules, "anthropic", None)

    p = AnthropicProvider(model="claude-haiku-4-5", max_tokens=64)

    with pytest.raises(ImportError, match=r'pip install "xain\[anthropic\]"'):
        p.generate("sys", "user")
