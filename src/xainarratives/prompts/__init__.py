"""Prompt-template implementations."""

from xainarratives.prompts.base import PromptTemplate
from xainarratives.prompts.echo import EchoPromptTemplate
from xainarratives.prompts.feature_importance_tabular import (
    DEFAULT_SYSTEM_TEMPLATE,
    DEFAULT_USER_TEMPLATE,
    FeatureImportanceTabularPromptTemplate,
)

__all__ = [
    "DEFAULT_SYSTEM_TEMPLATE",
    "DEFAULT_USER_TEMPLATE",
    "EchoPromptTemplate",
    "FeatureImportanceTabularPromptTemplate",
    "PromptTemplate",
]
