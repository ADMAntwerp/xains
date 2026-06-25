"""Compute the changed-features diff between a factual and a tabular counterfactual.

The library never searches for counterfactuals; it consumes
:class:`TabularCounterfactual` instances and verbalizes them. This module
delivers the diff that ADR 0004 promised: ``name`` / ``before`` / ``after``
triples for every changed feature. See ADR 0028.

Tabular only.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict

from xains.types import TabularCounterfactual


class ChangedFeature(BaseModel):
    """One feature that differs between factual and counterfactual."""

    model_config = ConfigDict(extra="forbid")

    name: str
    before: Any
    after: Any


def changed_features(factual: dict[str, Any], cf: TabularCounterfactual) -> list[ChangedFeature]:
    """Diff a tabular counterfactual against the factual instance.

    If ``cf.changed_features`` is set, those keys are reported as-is (the
    user's explicit declaration is honored, no value check). Otherwise every
    key in ``cf.features`` whose value differs from ``factual[key]`` is
    reported. A key absent from ``factual`` raises ``ValueError`` either
    way - a counterfactual referencing a feature the factual lacks is a
    user / data error.
    """
    if cf.changed_features is not None:
        keys: list[str] = list(cf.changed_features)
        diff_mode = False
    else:
        keys = list(cf.features.keys())
        diff_mode = True

    result: list[ChangedFeature] = []
    for key in keys:
        if key not in factual:
            raise ValueError(f"counterfactual feature {key!r} is absent from the factual instance")
        before = factual[key]
        after = cf.features[key]
        if diff_mode and before == after:
            continue
        result.append(ChangedFeature(name=key, before=before, after=after))
    return result
