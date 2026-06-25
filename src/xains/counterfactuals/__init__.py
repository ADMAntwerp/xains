"""Counterfactual-narrative support code.

Pure helpers consumed by the CF prompt template (LLM path) and the templated
CF generator (LLM-free path). The library never *searches* for
counterfactuals; it consumes pre-computed ``TabularCounterfactual`` instances
and verbalizes them. See ADR 0004, 0028, 0030.
"""

from xains.counterfactuals.diff import ChangedFeature, changed_features
from xains.counterfactuals.scenarios import CounterfactualScenario, build_scenarios

__all__ = [
    "ChangedFeature",
    "CounterfactualScenario",
    "build_scenarios",
    "changed_features",
]
