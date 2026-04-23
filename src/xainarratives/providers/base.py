"""LLM provider abstraction.

The library is LLM-agnostic. Concrete providers implement the ``LLMProvider``
Protocol by forwarding ``generate(system, user)`` to whatever backend they
speak (Anthropic, OpenAI, a local model, a mock, …).

Keep this interface minimal. If a future backend needs extra inputs (images,
tools, streaming), add a new, more specific Protocol — do not bloat this one.
"""

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class LLMResponse(BaseModel):
    """Normalized response from any provider."""

    model_config = ConfigDict(extra="forbid")

    text: str
    model_name: str = Field(min_length=1)
    tokens_used: dict[str, int] | None = None


@runtime_checkable
class LLMProvider(Protocol):
    """Sync text-in / text-out LLM interface."""

    def generate(self, system: str, user: str) -> LLMResponse:
        """Send a system and user prompt; return the assistant's reply."""
        ...
