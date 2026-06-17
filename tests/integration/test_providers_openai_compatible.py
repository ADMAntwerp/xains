"""Integration tests for OpenAICompatibleProvider and the OpenAIProvider /
OpenRouterProvider subclasses.

All 11 tests are pure mock-spy - zero live API calls. They pin:
- the happy-path generate() return shape (LLMResponse with text, model_name,
  tokens_used in the {input, output, total} ADR 0005 contract);
- the system+user -> messages array mapping (the boundary that differs
  from Anthropic's top-level system= kwarg);
- API-key resolution (explicit kwarg, env-var fallback, ValueError on miss);
- usage edge cases (response.usage=None, message.content=None);
- the two presets (OpenAIProvider, OpenRouterProvider) including OpenRouter's
  HTTP-Referer / X-Title default_headers when set;
- the missing-SDK ImportError naming the pip extra.
"""

import sys
from types import SimpleNamespace

import pytest

from xainarratives.providers import LLMProvider, LLMResponse
from xainarratives.providers.openai_compatible import (
    OpenAICompatibleProvider,
    OpenAIProvider,
    OpenRouterProvider,
)


def _fake_response(
    *,
    text: str | None = "The applicant Defaulted.",
    model: str = "gpt-4o-mini",
    prompt_tokens: int = 12,
    completion_tokens: int = 8,
    usage_present: bool = True,
) -> SimpleNamespace:
    """Build a SimpleNamespace that quacks like an openai ChatCompletion."""
    message = SimpleNamespace(content=text)
    choice = SimpleNamespace(message=message)
    usage = (
        SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        if usage_present
        else None
    )
    return SimpleNamespace(choices=[choice], model=model, usage=usage)


def _spy_openai(
    monkeypatch: pytest.MonkeyPatch, response: SimpleNamespace
) -> tuple[dict[str, object], dict[str, object]]:
    """Replace openai.OpenAI with a spy class.

    Returns (ctor_kwargs, create_kwargs) - both dicts captured by the spy:
      - ctor_kwargs: kwargs passed to openai.OpenAI(...) at client construction
      - create_kwargs: kwargs passed to client.chat.completions.create(...) at generate()
    """
    import openai

    ctor_kwargs: dict[str, object] = {}
    create_kwargs: dict[str, object] = {}

    def _create(**kw: object) -> SimpleNamespace:
        create_kwargs.update(kw)
        return response

    class _SpyOpenAI:
        def __init__(self, **kwargs: object) -> None:
            ctor_kwargs.update(kwargs)
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=_create))

    monkeypatch.setattr(openai, "OpenAI", _SpyOpenAI)
    return ctor_kwargs, create_kwargs


# ------------------------------------------------------ happy path


def test_generate_returns_llm_response_with_populated_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Happy path: text + model_name + tokens_used per ADR 0005."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    _spy_openai(monkeypatch, _fake_response())
    provider = OpenAICompatibleProvider(base_url=None, model="gpt-4o-mini", max_tokens=256)

    result = provider.generate("S", "U")

    assert isinstance(result, LLMResponse)
    assert result.text == "The applicant Defaulted."
    assert result.model_name == "gpt-4o-mini"
    assert result.tokens_used == {"input": 12, "output": 8, "total": 20}
    # Structural Protocol check folded in: OpenAICompatibleProvider IS LLMProvider.
    assert isinstance(provider, LLMProvider)


# ------------------------------------------------------ messages array shape


def test_messages_built_with_system_then_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """system+user must be packed as [{role:system},{role:user}] in messages=."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    _, create_kwargs = _spy_openai(monkeypatch, _fake_response())
    provider = OpenAICompatibleProvider(base_url=None, model="gpt-4o-mini", max_tokens=256)

    provider.generate("you are a helper", "explain the model")

    assert create_kwargs["messages"] == [
        {"role": "system", "content": "you are a helper"},
        {"role": "user", "content": "explain the model"},
    ]
    assert create_kwargs["model"] == "gpt-4o-mini"
    assert create_kwargs["max_tokens"] == 256


# ------------------------------------------------------ API key resolution


def test_api_key_explicit_overrides_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    ctor_kwargs, _ = _spy_openai(monkeypatch, _fake_response())
    provider = OpenAICompatibleProvider(
        base_url=None, model="x", max_tokens=10, api_key="explicit-key"
    )
    provider.generate("s", "u")
    assert ctor_kwargs["api_key"] == "explicit-key"


def test_api_key_resolved_from_env_var_when_not_passed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    ctor_kwargs, _ = _spy_openai(monkeypatch, _fake_response())
    provider = OpenAICompatibleProvider(base_url=None, model="x", max_tokens=10)
    provider.generate("s", "u")
    assert ctor_kwargs["api_key"] == "from-env"


def test_missing_api_key_raises_with_env_var_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match=r"OPENAI_API_KEY"):
        OpenAICompatibleProvider(base_url=None, model="x", max_tokens=10)


# ------------------------------------------------------ usage edge case


def test_tokens_used_is_none_when_response_usage_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Some self-hosted endpoints (vLLM at low compat) return usage=None."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    _spy_openai(monkeypatch, _fake_response(usage_present=False))
    provider = OpenAICompatibleProvider(base_url=None, model="x", max_tokens=10)
    result = provider.generate("s", "u")
    assert result.tokens_used is None


# ------------------------------------------------------ presets


def test_openai_provider_preset_uses_default_endpoint_and_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAIProvider: no base_url passed to SDK; reads OPENAI_API_KEY."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    ctor_kwargs, _ = _spy_openai(monkeypatch, _fake_response())
    provider = OpenAIProvider(model="gpt-4o-mini", max_tokens=256)
    provider.generate("s", "u")
    # base_url omitted -> SDK uses its own default.
    assert "base_url" not in ctor_kwargs
    assert ctor_kwargs["api_key"] == "sk-openai"
    # No default_headers when not requested.
    assert "default_headers" not in ctor_kwargs


def test_openrouter_provider_preset_uses_openrouter_endpoint_and_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenRouterProvider: base_url=openrouter.ai/api/v1; reads OPENROUTER_API_KEY."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")
    ctor_kwargs, _ = _spy_openai(monkeypatch, _fake_response())
    provider = OpenRouterProvider(model="anthropic/claude-haiku-4-5", max_tokens=256)
    provider.generate("s", "u")
    assert ctor_kwargs["base_url"] == "https://openrouter.ai/api/v1"
    assert ctor_kwargs["api_key"] == "sk-or"
    # No referer/title -> default_headers absent from openai.OpenAI(...) kwargs.
    assert "default_headers" not in ctor_kwargs


def test_openrouter_referer_and_title_passed_as_default_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """referer + title -> openai.OpenAI(default_headers={HTTP-Referer, X-Title})."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")
    ctor_kwargs, _ = _spy_openai(monkeypatch, _fake_response())
    provider = OpenRouterProvider(
        model="anthropic/claude-haiku-4-5",
        max_tokens=256,
        referer="https://my-app.example",
        title="MyApp",
    )
    provider.generate("s", "u")
    assert ctor_kwargs["default_headers"] == {
        "HTTP-Referer": "https://my-app.example",
        "X-Title": "MyApp",
    }


# ------------------------------------------------------ missing SDK


def test_missing_openai_sdk_raises_importerror_with_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SDK missing -> ImportError naming the openai extra. Lazy import: the
    failure fires at generate() time, not at construction (matches AnthropicProvider)."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    # Construct succeeds (no openai import in __init__).
    provider = OpenAICompatibleProvider(base_url=None, model="x", max_tokens=10)
    # Mask the SDK so the lazy import inside generate() fails.
    monkeypatch.setitem(sys.modules, "openai", None)
    with pytest.raises(ImportError, match=r'pip install "xainarratives\[openai\]"'):
        provider.generate("s", "u")


# ------------------------------------------------------ content-None defensive


def test_none_content_yields_empty_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAI returns message.content=None for tool-call responses.

    Our use case is chat-only (no tools), so this is defensive: the result's
    text becomes "" rather than letting None propagate to LLMResponse.text.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    _spy_openai(monkeypatch, _fake_response(text=None))
    provider = OpenAICompatibleProvider(base_url=None, model="x", max_tokens=10)
    result = provider.generate("s", "u")
    assert result.text == ""
