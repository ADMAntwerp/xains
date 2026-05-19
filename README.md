# xainarratives

Natural-language verbalization of ML model predictions from **pre-computed** attributions.

> **Scope.** `xainarratives` is a post-hoc *verbalizer*, not an XAI toolkit. It takes a dataset schema, a model prediction, pre-computed attributions (SHAP, LIME, sklearn `feature_importances_`, GNNExplainer, Captum, etc.), and optional pre-computed counterfactuals, and produces natural-language explanations plus verbalization-quality metrics. It never trains models, never runs inference, never computes attributions, and never searches for counterfactuals — those are the user's responsibility, upstream.

Supports four modalities: **tabular**, **text**, **image**, **graph** (GNN).

> ⚠️ **Status: v0.0.1 alpha.** The API will break. Pin exact versions if you depend on it.

## Install

```bash
pip install xainarratives          # core only
pip install xainarratives[dev]     # dev tooling
```

## Quickstart: end-to-end notebook

For the full pipeline (load German Credit, train a RandomForest, compute SHAP, generate a narrative with Claude, extract structured claims, and score extraction fidelity plus the seven Cedro & Martens 2026 narrativity metrics) see notebooks/01_quickstart.ipynb.

Install and launch:

    pip install -e ".[notebook,anthropic,perplexity-api]"
    export ANTHROPIC_API_KEY="sk-ant-..."
    export TOGETHER_API_KEY="..."
    jupyter lab notebooks/01_quickstart.ipynb

The rendered notebook on GitHub shows committed outputs from one realization; LLM responses and perplexity numbers shift run-to-run.

## Minimal example (mock LLM, no API keys)

```python
from xainarratives import (
    DatasetSchema, FeatureSchema, TargetSchema, Modality,
    TabularExplanationRequest, TabularContribution, Prediction,
    ExplanationConfig, Explainer,
)
from xainarratives.providers import MockLLMProvider
from xainarratives.prompts import EchoPromptTemplate

schema = DatasetSchema(
    modality=Modality.TABULAR,
    name="credit_risk",
    description="Predicts 24-month default on personal loans.",
    target=TargetSchema(
        name="default",
        description="Whether the applicant defaulted.",
        classes={0: "Repaid", 1: "Defaulted"},
    ),
    features=[
        FeatureSchema(name="age", dtype="numeric", unit="years",
                      description="Applicant age at application."),
        FeatureSchema(name="dti", dtype="numeric",
                      description="Debt-to-income ratio."),
    ],
)

request = TabularExplanationRequest(
    features={"age": 29, "dti": 0.41},
    prediction=Prediction(predicted_class=1, probabilities={0: 0.2, 1: 0.8}),
    contributions=[
        TabularContribution(name="dti", value=0.41, importance=0.37),
        TabularContribution(name="age", value=29, importance=-0.12),
    ],
)

explainer = Explainer(
    schema=schema,
    llm=MockLLMProvider(responses=["High DTI is the dominant risk driver."]),
    prompt_template=EchoPromptTemplate(),
    config=ExplanationConfig(audience="end_user"),
)

result = explainer.explain(request)
print(result.text)
```

## Current capabilities

- Core data model (PR 1): pydantic schema, request, contribution, prediction, and config types across four modalities.
- AnthropicProvider (PR 2): real Anthropic Claude API integration via the official SDK.
- from_feature_importance adapter (PR 3): converts any signed per-feature attribution (SHAP, LIME, sklearn feature_importances_) into a TabularExplanationRequest.
- Guardrails and narrative extraction (PR 4): rule checks plus LLM-based extract_narrative_claims, orchestrated by Explainer.explain.
- Extraction scoring (PR 5, revised by ADR 0007): sign / value / rank faithfulness, coverage, hallucination count, readability.
- Paper narrativity metrics (PR 6): the seven Cedro & Martens 2026 metrics (CSR, DCPR, CCPR, CECPR, FDR, TTCPR, VCPR).
- Perplexity providers (PR 7): HuggingFacePerplexityProvider (local) and OpenAICompatibleEchoProvider (any OpenAI-compatible /v1/completions endpoint).

See docs/design.md for the full design and docs/decisions/ for recorded architecture decisions (ADRs 0001-0010).

## License

MIT — see [`LICENSE`](LICENSE).
