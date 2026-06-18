"""AnthropicProvider — Anthropic Messages API adapter.

The ``anthropic`` SDK is an **optional** dependency. It is imported lazily
on the first call to :meth:`AnthropicProvider.generate`, never at module
top. If the package is not installed, a clear :class:`ImportError` tells
the caller the exact pip extra to install.

Usage::

    provider = AnthropicProvider(model="claude-haiku-4-5", max_tokens=256)
    response = provider.generate(system="...", user="...")

Error policy: SDK exceptions (``RateLimitError``, ``APIStatusError``, …)
propagate unchanged. Only the missing-package case is wrapped, because the
raw ``ModuleNotFoundError`` does not point the user at the right install
command.
"""

from typing import Any

from xain.providers.base import LLMResponse

_MISSING_SDK_MESSAGE = (
    "The 'anthropic' package is required. Install with: pip install \"xain[anthropic]\""
)


class AnthropicProvider:
    """Sync adapter over the Anthropic Messages API.

    Conforms structurally to ``xain.providers.LLMProvider``.
    """

    def __init__(self, model: str, max_tokens: int, api_key: str | None = None) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._api_key = api_key
        self._client: Any = None  # lazily constructed in generate()

    def generate(self, system: str, user: str) -> LLMResponse:
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise ImportError(_MISSING_SDK_MESSAGE) from exc
            self._client = anthropic.Anthropic(api_key=self._api_key)

        msg = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )

        text = "".join(b.text for b in msg.content if b.type == "text")
        input_tokens = msg.usage.input_tokens
        output_tokens = msg.usage.output_tokens

        return LLMResponse(
            text=text,
            model_name=msg.model,
            tokens_used={
                "input": input_tokens,
                "output": output_tokens,
                "total": input_tokens + output_tokens,
            },
        )
