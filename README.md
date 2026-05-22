# xainarratives

Natural-language verbalization of tabular ML model predictions from **pre-computed** attributions.

> **Scope.** This library generates natural-language XAI narratives from technical outputs like SHAP attributions or counterfactual explanations, making the explanations more transparent and understandable.

## Install

```bash
pip install xainarratives          # core only
pip install xainarratives[dev]     # dev tooling
```

## Quickstart: end-to-end notebook

For the full pipeline - load German Credit, train a RandomForest, compute SHAP, generate a narrative with an LLM, extract structured claims, and score the narrative on faithfulness and narrativity - see `notebooks/01_quickstart.ipynb`.

Install and launch:

    pip install -e ".[notebook,anthropic,perplexity-api]"
    export ANTHROPIC_API_KEY="sk-ant-..."
    export TOGETHER_API_KEY="..."
    jupyter lab notebooks/01_quickstart.ipynb

The rendered notebook on GitHub shows committed outputs from one realization; LLM responses and perplexity numbers shift run-to-run.

## Minimal example

```python
from xainarratives import (
    DatasetSchema, FeatureSchema, TargetSchema, Modality,
    TabularExplanationRequest, TabularContribution, Prediction,
    ExplanationConfig, Explainer,
)
from xainarratives.providers import AnthropicProvider
from xainarratives.prompts import FactualTabularPromptTemplate

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
    llm=AnthropicProvider(model="claude-haiku-4-5"),
    prompt_template=FactualTabularPromptTemplate(),
    config=ExplanationConfig(mode="factual", audience="end_user"),
)

result = explainer.explain(request)
print(result.text)
```

See `docs/design.md` for the full design and `docs/decisions/` for recorded architecture decisions.

## License

MIT - see [`LICENSE`](LICENSE).
