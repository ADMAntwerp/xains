"""Counterfactual-narrative support code.

Pure helpers consumed by the counterfactual prompt template (future commits).
The library never *searches* for counterfactuals; it consumes pre-computed
``TabularCounterfactual`` instances and verbalizes them. See ADR 0004.
"""

from xains.counterfactuals.diff import ChangedFeature, changed_features

__all__ = ["ChangedFeature", "changed_features"]
