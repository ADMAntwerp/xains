# 0001. Scope boundary: post-hoc verbalizer only

Date: 2026-04-23
Status: Accepted

## Context

XAI tools (SHAP, LIME, Captum, GNNExplainer, DiCE, …) produce attributions
and counterfactuals. They are mature, framework-specific, and heavy
(PyTorch / TF / JAX / sklearn / etc.).

Users still need a layer that turns those numeric outputs into explanations
a human can read. That layer is currently hand-rolled per project.

## Decision

`xainarratives` owns only the **verbalization** layer. It accepts
pre-computed attributions and counterfactuals as input and produces natural
language plus verbalization-quality metrics.

It does not, and will not, include model training, inference, attribution
computation, or counterfactual search.

## Consequences

**Positive.**

- No heavy ML deps in core. Fast install, light CI, small attack surface.
- Framework-agnostic: works with any upstream tool that produces signed
  per-feature / per-token / per-region / per-node-or-edge scalars.
- Sharply testable — inputs are pydantic models, not model objects.
- Clear vocabulary: we measure *verbalization fidelity*, not *attribution
  faithfulness* (the latter is upstream).

**Negative.**

- Users must integrate the upstream XAI tool themselves. We mitigate with
  thin input-shape adapters (`from_feature_importance`, etc.).
- We cannot verify that the provided attributions actually reflect the
  model. Guardrails target the text-vs-attribution relationship only.

## Alternatives considered

- Include SHAP/LIME as optional integrations that *run* computation.
  Rejected: conflates two layers, bloats deps, and competes poorly with
  the upstream tools at their own job.
