"""Prompt-template abstraction.

A ``PromptTemplate`` turns ``(request, schema, config)`` into a concrete
``(system, user)`` pair. Different templates encode different explanation
styles (factual, contrastive, counterfactual) and different audiences
(technical, business, end-user), without the ``Explainer`` having to know.
"""

from abc import ABC, abstractmethod

from xainarratives.config import ExplanationConfig
from xainarratives.schema import DatasetSchema
from xainarratives.types import ExplanationRequest


class PromptTemplate(ABC):
    """Renders a prompt pair for a given request, schema, and config."""

    @abstractmethod
    def render(
        self,
        request: ExplanationRequest,
        schema: DatasetSchema,
        config: ExplanationConfig,
    ) -> tuple[str, str]:
        """Return ``(system_prompt, user_prompt)``."""
