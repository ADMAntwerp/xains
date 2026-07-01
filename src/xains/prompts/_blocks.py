"""Shared block-string builders for the tabular prompt templates.

Two pure helpers, extracted from the inline block logic of
``FeatureImportanceTabularPromptTemplate.render()`` and
``CounterfactualTabularPromptTemplate.render()``:

- :func:`build_contribution_block`: contribution ordering (abs(importance)
  desc, tie-widening at the k-th boundary) + per-line formatting.
- :func:`build_counterfactual_block`: flip lead + change lines for a
  pre-built :class:`CounterfactualScenario`.

Byte-for-byte identical output to the original inline logic. See ADR 0038.
"""

from collections.abc import Sequence

from xains.counterfactuals import CounterfactualScenario
from xains.schema import DatasetSchema
from xains.types import TabularContribution


def build_contribution_block(
    contributions: Sequence[TabularContribution],
    schema: DatasetSchema,
    top_k: int,
) -> str:
    """Order contributions by ``abs(importance)`` desc and render the block.

    Ties at the k-th boundary widen the cut to include all tied contributions;
    the ``rank`` field on each contribution is ignored. Per-line format is
    ``- {name} = {value}[ [unit]]: importance={sign}{abs(importance):g}``.
    Callers must have already validated that every contribution name is in
    ``schema.features``; this helper does not re-check.
    """
    ordered = sorted(contributions, key=lambda c: -abs(c.importance))
    if len(ordered) > top_k:
        boundary = abs(ordered[top_k - 1].importance)
        cut = top_k
        while cut < len(ordered) and abs(ordered[cut].importance) == boundary:
            cut += 1
        ordered = ordered[:cut]

    lines = []
    for c in ordered:
        feat = schema.feature(c.name)
        unit = f" [{feat.unit}]" if feat.unit else ""
        sign = "+" if c.importance >= 0 else "-"
        lines.append(f"- {c.name} = {c.value}{unit}: importance={sign}{abs(c.importance):g}")
    return "\n".join(lines)


def build_counterfactual_block(
    scenario: CounterfactualScenario,
    schema: DatasetSchema,
    include_method: bool,
) -> str:
    """Render the flip lead + change lines for a pre-built scenario.

    Format: a lead line ``To change the prediction from <factual> to <cf>:``
    optionally followed by ``" (method: <cf.method>)"`` when ``include_method``
    is True and ``scenario.method`` is not None, then one indented
    ``  - name: before -> after [unit]`` line per changed feature.
    """
    method_suffix = (
        f" (method: {scenario.method})" if include_method and scenario.method is not None else ""
    )
    lead = (
        f"To change the prediction from {scenario.factual_label} to "
        f"{scenario.cf_label}:{method_suffix}"
    )
    change_lines = []
    for chg in scenario.changes:
        feat = schema.feature(chg.name)
        unit = f" [{feat.unit}]" if feat.unit else ""
        change_lines.append(f"  - {chg.name}: {chg.before} -> {chg.after}{unit}")
    return "\n".join([lead, *change_lines])
