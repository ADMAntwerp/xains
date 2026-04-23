"""Prompt-template implementations."""

from xainarratives.prompts.base import PromptTemplate
from xainarratives.prompts.echo import EchoPromptTemplate
from xainarratives.prompts.factual_tabular import FactualTabularPromptTemplate

__all__ = ["EchoPromptTemplate", "FactualTabularPromptTemplate", "PromptTemplate"]
