"""Prompt-template implementations.

The default-template constants (``DEFAULT_SYSTEM_TEMPLATE`` /
``DEFAULT_USER_TEMPLATE``) for each concrete template live on their own
submodule, not at this package level - one pair per template, they would
collide if re-exported here. Import them directly, e.g.
``from xains.prompts.feature_importance_tabular import DEFAULT_SYSTEM_TEMPLATE``.
See ADR 0029.
"""

from xains.prompts.base import PromptTemplate
from xains.prompts.counterfactual_tabular import CounterfactualTabularPromptTemplate
from xains.prompts.echo import EchoPromptTemplate
from xains.prompts.feature_importance_tabular import FeatureImportanceTabularPromptTemplate
from xains.prompts.hybrid_tabular import HybridTabularPromptTemplate

__all__ = [
    "CounterfactualTabularPromptTemplate",
    "EchoPromptTemplate",
    "FeatureImportanceTabularPromptTemplate",
    "HybridTabularPromptTemplate",
    "PromptTemplate",
]
