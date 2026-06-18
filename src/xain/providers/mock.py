"""MockLLMProvider — deterministic, offline, for tests and the skeleton path."""

from collections.abc import Callable

from xain.providers.base import LLMResponse


class MockLLMProvider:
    """Deterministic provider that returns pre-scripted responses.

    Accepts either:

    * a list of response strings, served in order and cycling if exhausted;
    * a callable ``(system, user) -> str`` for response logic driven by the
      actual prompt.

    This provider never touches the network.
    """

    def __init__(
        self,
        responses: list[str] | Callable[[str, str], str] | None = None,
        model_name: str = "mock-v0",
    ) -> None:
        if isinstance(responses, list) and len(responses) == 0:
            raise ValueError("MockLLMProvider: `responses` list must not be empty.")

        self._responses: list[str] | Callable[[str, str], str] = (
            responses if responses is not None else ["mock response"]
        )
        self._index = 0
        self._model_name = model_name

    def generate(self, system: str, user: str) -> LLMResponse:
        if callable(self._responses):
            text = self._responses(system, user)
        else:
            text = self._responses[self._index % len(self._responses)]
            self._index += 1
        return LLMResponse(text=text, model_name=self._model_name)
