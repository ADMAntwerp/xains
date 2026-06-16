"""Single-pass {identifier} substitution primitive.

Shared by ``prompts/feature_importance_tabular.py`` (LLM prompt rendering)
and ``generation/templated.py`` (templated narrative rendering). Private
to the library - not re-exported from the package root.

``substitute()`` is one-pass: substituted values are never re-scanned for
further placeholders, so callers can safely inject text that contains
literal ``{identifier}`` patterns. Unknown identifier-shaped tokens in the
template raise ``ValueError``. See ADR 0017 for design rationale and
brace-safety semantics.
"""

import re

PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_]\w*)\}")


def substitute(template: str, values: dict[str, str]) -> str:
    """Single-pass {identifier} substitution.

    Unknown identifier-shaped tokens raise ValueError. Substituted values are
    never re-scanned (one-pass guarantee).
    """
    for token in PLACEHOLDER_RE.findall(template):
        if token not in values:
            valid = ", ".join(sorted(values))
            raise ValueError(
                f"Unknown placeholder {{{token}}} in template. Valid placeholders: {valid}."
            )
    return PLACEHOLDER_RE.sub(lambda m: values[m.group(1)], template)
