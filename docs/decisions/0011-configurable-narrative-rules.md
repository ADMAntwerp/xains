# 0011. Configurable narrative-generation rules

Date: 2026-05-19
Status: Accepted

## Context

xainarratives generates narrative explanations via prompt templates.
`FactualTabularPromptTemplate` is the only one that currently exists;
contrastive and counterfactual templates are planned for later PRs.

The operational definition of an XAI Narrative from Cedro & Martens 2026
- continuous structure, explicit cause-effect mechanisms, linguistic
fluency, lexical diversity - should be injected into every
narrative-generation prompt. Two requirements shape where it lives:

- It must be universal across modes (factual, contrastive, counterfactual),
  not duplicated per template.
- It must be overridable by users without subclassing a template.

## Decision

Add a `narrative_rules: str` field to `ExplanationConfig`, defaulting to a
module-level constant `DEFAULT_NARRATIVE_RULES` that carries the paper's
four-rule wording verbatim. Every prompt template that generates a
narrative reads `config.narrative_rules` and appends it to its system
message after a blank-line separator.

This PR wires the field into `ExplanationConfig` and into
`FactualTabularPromptTemplate` (the only template that currently exists).
Building contrastive and counterfactual templates is separate, later work;
those templates will read the same field when they are built.

## Rationale

- One field on the config covers every template - no template-by-template
  duplication, no shared base-class state.
- Users override by passing `ExplanationConfig(narrative_rules="...")` - no
  subclassing, no monkey-patching. The customization point is a config
  field, consistent with `audience`, `tone`, and `max_length_words`.
- The default matches the paper's wording verbatim, so paper-faithful
  narrative generation is the out-of-the-box behavior.
- The rules are domain content (an operational definition), not rendering
  logic, so they belong in config alongside the other content-shaping
  knobs rather than inside a `render()` method.

## Consequences

- `FactualTabularPromptTemplate.render()` reads `config.narrative_rules`
  and appends it to the system message.
- Future contrastive and counterfactual templates will read the same field
  by convention. Nothing at the `PromptTemplate` Protocol level enforces
  this; it is a documented expectation.
- Users wanting mode-specific rules pass a different `ExplanationConfig`
  per `Explainer.explain()` call.
- `DEFAULT_NARRATIVE_RULES` is a verbatim multi-line quote whose sentences
  exceed the line-length limit; E501 is suppressed for `config.py` via a
  scoped `per-file-ignores` entry, with a justification comment on the
  constant.

## Rejected alternatives

- **Hardcode the rules inside `FactualTabularPromptTemplate`.** Factual-only
  and not overridable without subclassing - the opposite of both
  requirements.
- **Add a base-class constant on `PromptTemplate`.** Subclassing is a
  developer escape hatch, not a user-facing knob. End users expect to set a
  config field and have the prompt change, not to subclass a template.
- **Per-template rule injection** (each template defines its own rules).
  Fragments the canonical narrativity definition across templates; the
  rules are a domain concept independent of explanation mode.

## References

- Cedro, M., & Martens, D. (2026). *On the Importance and Evaluation of
  Narrativity in Natural Language AI Explanations.* arXiv:2604.18311.
- ADR 0006: Guardrails and narrative extraction layer.
- ADR 0008: Narrativity metrics - paper-faithful composition.
- ADR 0010: Quickstart notebook.
