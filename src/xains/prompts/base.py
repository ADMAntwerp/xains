"""Prompt-template abstraction.

A ``PromptTemplate`` turns ``(request, schema, config)`` into a concrete
``(system, user)`` pair. Different templates encode different explanation
styles (feature_importance, counterfactual,
feature_importance_counterfactual) and different audiences
(technical, business, end-user), without the ``Explainer`` having to know.
"""

from abc import ABC, abstractmethod

from xains.config import ExplanationConfig
from xains.schema import DatasetSchema
from xains.types import ExplanationRequest


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
