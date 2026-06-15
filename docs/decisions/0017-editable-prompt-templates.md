# 0017. Editable prompt templates for `FeatureImportanceTabularPromptTemplate`

Date: 2026-06-14
Status: Accepted

## Context

Researchers iterating on explanation quality need to change the prompt -
wording, ordering, added context, audience phrasing - without forking or
subclassing the template. Previously the prompt was assembled inline in
``render()`` via f-strings, so any customization meant editing source.

Separately, the exact text sent to the LLM should be inspectable before
the call, so a user can verify what will actually be sent rather than
trusting the template produced what they expected.

The library already has a precedent for "expose this as a plain string
with a sensible default the user can override": ``narrative_rules``
(ADR 0011) is a config string the user can replace wholesale. The same
mental model applies here - prompts are text, and text is what gets
edited - so the natural extension is to make the whole prompt the
editable surface, not just the rules block.

## Decision

``FeatureImportanceTabularPromptTemplate`` gets two editable template
strings and a placeholder dict, all keyword-only and all defaulted to
constants that reproduce the prior output byte-for-byte. This is the
editable-strings design (named placeholders), chosen over a
composable-section-objects design.

**Constructor:**

```python
def __init__(
    self,
    *,
    system_template: str = DEFAULT_SYSTEM_TEMPLATE,
    user_template: str = DEFAULT_USER_TEMPLATE,
    extra_placeholders: dict[str, str] | None = None,
) -> None: ...
```

**Built-in placeholders** (filled from the request / schema / config):

- ``{target_name}`` - ``schema.target.name``
- ``{audience}`` - ``config.audience``
- ``{tone}`` - ``config.tone``
- ``{max_length_words}`` - ``config.max_length_words``
- ``{narrative_rules}`` - ``config.narrative_rules``
- ``{prediction}`` - ``schema.target.classes[predicted_class]``
- ``{contributions}`` - the joined per-line contribution block

**Substitution contract:**

- Recognized as a placeholder: regex ``\{([a-zA-Z_]\w*)\}`` -
  Python-identifier-shaped tokens between single braces.
- ``{name}`` in built-ins or ``extra_placeholders`` -> substituted in a
  single ``re.sub`` pass. Substituted values are not re-scanned
  (one-pass guarantee).
- ``{name}`` identifier-shaped but unknown -> ``ValueError`` at render,
  naming the token and listing valid names. Catches typos and forgotten
  ``extra_placeholders`` declarations.
- Omitted built-in -> allowed; the computed value is simply unused.
- Other braces - ``{}``, ``{ foo }``, ``{123}``, ``{"key": "val"}``,
  standalone ``{`` or ``}`` - are not matched and pass through literally.
- ``extra_placeholders`` keys overlapping built-in names -> ``ValueError``
  at the constructor (fail-fast).

**Boundary between template and code:**

- In the template: layout, ordering, narrative-rules placement,
  audience/tone wording.
- In code: top-k selection (with tie-widening), per-contribution line
  formatting, and the human-label lookup for ``{prediction}``.
  ``{contributions}`` reaches the template already joined.

**Notebook integration.** The quickstart notebook gained a preview cell
that calls ``explainer.prompt_template.render(request, schema,
explainer.config)`` on the live ``explainer`` objects and prints both
prompts before ``explain()`` - so what's printed is byte-identical to
what the LLM receives.

## Rationale

- **Editable strings over composable sections.** Researchers think
  prompts-as-text and reach for string editing, not object graphs. A
  section-object design would add unrequested vocabulary and have one
  in-repo user - violating the ">=2 implementations" rule in CLAUDE.md.
- **Matches the ``narrative_rules`` precedent.** ADR 0011 established
  "expose a plain string with a sensible default." Templates extend it
  from one knob to the whole prompt.
- **``{}`` over ``$``.** ``{...}`` is conventional in modern Python
  (f-strings, ``str.format``) and reads cleanly next to prose;
  ``string.Template``'s ``$`` was rejected on that basis.
- **Custom substitution, not ``str.format``.** ``str.format`` would
  honor ``{0}``, ``{x!r}``, ``{x:>10}`` - an unrequested mini-language -
  and raise on any literal ``{`` in user prose. The one-pass regex is
  narrower and easier to reason about.
- **Byte-for-byte additive.** The default constants reproduce the
  pre-refactor output exactly; one snapshot test pins this down.

## Consequences

- New keyword-only constructor args, all defaulted. Existing call sites
  work unchanged; output is identical. Non-breaking.
- ``DEFAULT_SYSTEM_TEMPLATE`` / ``DEFAULT_USER_TEMPLATE`` re-exported
  from ``xainarratives.prompts``.
- The ``PromptTemplate`` ABC is unchanged. When the counterfactual
  template lands as the second user, the substitution helper should move
  to ``base.py`` - it does not belong on the ABC yet under the >=2-impl
  rule.
- Substitution is one-pass: a ``narrative_rules`` value containing
  literal ``{appendix}`` is injected verbatim, not re-parsed.

## Deferred decisions

- **Literal ``{identifier}`` text in templates is not supported in v1.**
  ``{appendix}`` (neither built-in nor declared) is a hard error, guarded
  by ``test_unknown_identifier_braces_raise``. Escape syntax
  (``{{...}}``) was rejected for v1: it would partially mimic
  ``str.format``, falsely implying the rest of the mini-language works.
  Revisit if a concrete need for literal ``{word}`` text surfaces.

## Rejected alternatives

- **Composable section objects.** Heavier concept, speculative
  reordering nobody asked for, one in-repo user (>=2-impl rule).
- **``string.Template`` (``$name``).** ``{}`` chosen for familiarity.
- **Full ``str.format``.** Silently changes the meaning of literal
  braces; introduces an unrequested mini-language.
- **``{{ }}`` escape syntax.** Deferred - partial str.format mimicry.

## References

- ADR 0011 (configurable narrative-generation rules) - the
  string-override precedent.
- ``src/xainarratives/prompts/feature_importance_tabular.py`` - source.
- ``tests/unit/test_prompts_feature_importance_tabular.py`` - 14 new
  tests pin the contract.
- ``notebooks/01_quickstart.ipynb`` - the preview cell.
- CLAUDE.md "Abstraction Rule" - applied to defer moving ``_substitute``
  to the ABC.
