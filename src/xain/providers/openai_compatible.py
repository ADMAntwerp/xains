"""OpenAICompatibleProvider + OpenAIProvider + OpenRouterProvider.

OpenAI-shaped chat-completions LLM providers. The base class is the public
extension point for any OpenAI-compatible endpoint (Together / Groq / vLLM /
LM Studio); the two subclasses are thin presets for OpenAI and OpenRouter.

The ``openai`` SDK is an **optional** dependency. It is imported lazily on the
first call to ``generate()``, never at module top. If the package is not
installed, a clear ``ImportError`` tells the caller the exact pip extra.

API-key resolution is eager (constructor-time): pass ``api_key=`` explicitly,
or leave it ``None`` and the value is read from ``os.environ[api_key_env_var]``.
If neither path yields a key, ``__init__`` raises ``ValueError`` naming the
env var that was checked. This mirrors ``OpenAICompatibleEchoProvider`` (the
perplexity sibling) and is more honest than the SDK's silent auto-read when
the endpoint is non-OpenAI (e.g. OpenRouter wants ``OPENROUTER_API_KEY``).

Conforms structurally to ``xain.providers.LLMProvider``.
"""

import os
from typing import Any

from xain.providers.base import LLMResponse

_MISSING_SDK_MESSAGE = (
    "The 'openai' package is required. Install with: pip install \"xain[openai]\""
)

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenAICompatibleProvider:
    """Sync adapter over the OpenAI Chat Completions API.

    Constructor resolves the API key eagerly (``ValueError`` if missing).
    The ``openai`` SDK is imported lazily on the first ``generate()`` call.

    Args:
        base_url: endpoint URL, or ``None`` to let the SDK use its default
            (OpenAI's own endpoint). Subclasses preset this.
        model: model id passed to ``chat.completions.create``.
        max_tokens: required - unbounded ``max_tokens`` is a cost footgun.
        api_key: explicit key, or ``None`` to resolve from
            ``os.environ[api_key_env_var]``.
        api_key_env_var: name of the env var to read when ``api_key`` is
            ``None``. Defaults to ``"OPENAI_API_KEY"``; subclasses override
            for non-OpenAI endpoints.
        default_headers: extra HTTP headers passed to ``openai.OpenAI(...)``.
            None means omit the kwarg entirely (SDK uses its defaults).
    """

    def __init__(
        self,
        *,
        base_url: str | None,
        model: str,
        max_tokens: int,
        api_key: str | None = None,
        api_key_env_var: str = "OPENAI_API_KEY",
        default_headers: dict[str, str] | None = None,
    ) -> None:
        resolved_key = api_key if api_key is not None else os.environ.get(api_key_env_var)
        if not resolved_key:
            raise ValueError(
                f"No API key for {type(self).__name__}: pass api_key=... "
                f"or set the {api_key_env_var} environment variable."
            )
        self._api_key = resolved_key
        self._base_url = base_url
        self._model = model
        self._max_tokens = max_tokens
        self._default_headers = default_headers
        self._client: Any = None

    def generate(self, system: str, user: str) -> LLMResponse:
        if self._client is None:
            try:
                import openai
            except ImportError as exc:
                raise ImportError(_MISSING_SDK_MESSAGE) from exc
            client_kwargs: dict[str, Any] = {"api_key": self._api_key}
            if self._base_url is not None:
                client_kwargs["base_url"] = self._base_url
            if self._default_headers:
                client_kwargs["default_headers"] = self._default_headers
            self._client = openai.OpenAI(**client_kwargs)

        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )

        text = response.choices[0].message.content or ""
        tokens_used: dict[str, int] | None = None
        if response.usage is not None:
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            tokens_used = {
                "input": input_tokens,
                "output": output_tokens,
                "total": input_tokens + output_tokens,
            }

        return LLMResponse(
            text=text,
            model_name=response.model,
            tokens_used=tokens_used,
        )


class OpenAIProvider(OpenAICompatibleProvider):
    """OpenAI preset: SDK default endpoint, reads ``OPENAI_API_KEY``."""

    def __init__(
        self,
        *,
        model: str,
        max_tokens: int,
        api_key: str | None = None,
    ) -> None:
        super().__init__(
            base_url=None,
            model=model,
            max_tokens=max_tokens,
            api_key=api_key,
            api_key_env_var="OPENAI_API_KEY",
        )


class OpenRouterProvider(OpenAICompatibleProvider):
    """OpenRouter preset: openrouter.ai endpoint, reads ``OPENROUTER_API_KEY``.

    The optional ``referer`` and ``title`` kwargs map to the ``HTTP-Referer``
    and ``X-Title`` headers OpenRouter uses for app attribution / leaderboards.
    Both are optional; when both are None, no extra headers are sent.
    """

    def __init__(
        self,
        *,
        model: str,
        max_tokens: int,
        api_key: str | None = None,
        referer: str | None = None,
        title: str | None = None,
    ) -> None:
        headers: dict[str, str] = {}
        if referer is not None:
            headers["HTTP-Referer"] = referer
        if title is not None:
            headers["X-Title"] = title
        super().__init__(
            base_url=_OPENROUTER_BASE_URL,
            model=model,
            max_tokens=max_tokens,
            api_key=api_key,
            api_key_env_var="OPENROUTER_API_KEY",
            default_headers=headers if headers else None,
        )
