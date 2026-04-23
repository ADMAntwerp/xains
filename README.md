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

## Quickstart (skeleton / mock path)

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

## What's in v0.0.1

Skeleton + mock path only. No real LLM providers, no real prompt templates, no integrations.

See [`docs/design.md`](docs/design.md) for the full design and [`docs/decisions/`](docs/decisions/) for recorded architecture decisions.

## License

MIT — see [`LICENSE`](LICENSE).
