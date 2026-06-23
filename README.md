# xains

[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org)
[![PyPI version](https://img.shields.io/pypi/v/xains.svg)](https://pypi.org/project/xains/)
[![Tests](https://github.com/ADMAntwerp/xains/actions/workflows/ci.yml/badge.svg)](https://github.com/ADMAntwerp/xains/actions/workflows/ci.yml)

<!-- TODO (add as each service comes online for ADMAntwerp/xains):
[![codecov](https://codecov.io/gh/ADMAntwerp/xains/branch/master/graph/badge.svg)](https://codecov.io/gh/ADMAntwerp/xains)
[![Read the Docs](https://readthedocs.org/projects/xains/badge/?version=latest)](https://xains.readthedocs.io/en/latest/)
-->

xains generates explainable AI (XAI) narratives - hence the name. It turns technical XAI outputs such as SHAP attributions and counterfactuals into clear natural-language explanations that make model decisions understandable to a broad audience.

> **Scope.** This library generates natural-language XAI narratives from technical outputs like SHAP attributions or counterfactual explanations, making the explanations more transparent and understandable.

## Install

```bash
git clone https://github.com/ADMAntwerp/xains.git
cd xains
pip install -e .
```

## Minimal example

```python
import xains
import xains.prompts

schema = xains.DatasetSchema(
    modality=xains.Modality.TABULAR,
    name="credit_risk",
    description="Predicts 24-month default on personal loans.",
    target=xains.TargetSchema(
        name="default",
        description="Whether the applicant defaulted.",
        classes={0: "Repaid", 1: "Defaulted"},
    ),
    features=[
        xains.FeatureSchema(name="age", dtype="numeric", unit="years",
                           description="Applicant age at application."),
        xains.FeatureSchema(name="salary", dtype="numeric", unit="EUR",
                           description="Annual gross salary."),
        xains.FeatureSchema(name="debt_to_income", dtype="numeric",
                           description="Debt-to-income ratio."),
    ],
)

request = xains.TabularExplanationRequest(
    features={"age": 29, "salary": 52000, "debt_to_income": 0.41},
    prediction=xains.Prediction(predicted_class=1, probabilities={0: 0.2, 1: 0.8}),
    contributions=[
        xains.TabularContribution(name="debt_to_income", value=0.41, importance=0.37),
        xains.TabularContribution(name="salary", value=52000, importance=-0.21),
        xains.TabularContribution(name="age", value=29, importance=-0.12),
    ],
)

llm = xains.AnthropicProvider(model="claude-haiku-4-5", max_tokens=512)
explainer = xains.Explainer(
    schema=schema,
    generator=xains.LLMNarrativeGenerator(
        prompt_template=xains.prompts.FeatureImportanceTabularPromptTemplate(),
        llm=llm,
    ),
    config=xains.ExplanationConfig(
        mode="feature_importance", audience="end_user",
        max_length_words=40, extract_narrative=True,
    ),
    judge_llm=llm,  # required when extract_narrative=True
)

result = explainer.explain(request)
print(result.text)
```

Output is illustrative; LLM responses vary run-to-run:

```text
Your profile indicates elevated default risk. A debt-to-income ratio of 0.41
substantially increases this concern, signaling that your debt obligations
consume a meaningful portion of earnings. Although your salary of EUR 52,000
and relatively young age of 29 provide some protective factors that work
against default, they ultimately prove insufficient to offset the debt burden
weighing on your financial stability.
```

### Scoring the narrative

A narrative is only useful if it is faithful to the attributions and reads well. xains scores both. `grade_extraction` checks the claims the narrative makes against the input attributions - sign, value, and rank fidelity, coverage, hallucination count, and readability (perplexity is added when a perplexity provider is supplied):

```python
grades = xains.grade_extraction(
    extraction=result.narrative_extraction,
    request=request,
    schema=schema,
    narrative_text=result.text,
    k=5,
)
print(grades)
```

```text
sign_faithfulness=1.0 value_faithfulness=1.0 rank_correlation=1.0 coverage=1.0
hallucination_count=0 readability=30.09 perplexity=None prompt_version='2'
```

`grade_narrativity` scores how well the text reads as a narrative, using the metrics from Cedro & Martens 2026. It needs a perplexity provider (any OpenAI-compatible endpoint that returns logprobs):

```python
from xains.metrics import OpenAICompatibleEchoProvider

ppl = OpenAICompatibleEchoProvider(
    base_url="https://api.together.xyz/v1",
    model="meta-llama/Meta-Llama-3-8B-Instruct-Lite",
    api_key_env_var="TOGETHER_API_KEY",
)
narrativity = xains.grade_narrativity(result.text, ppl)
print(narrativity.fdr, narrativity.csr)
```

```text
0.29 0.11
```

The two values are Fluency-Diversity Rate (FDR) and Continuous Structure Rate (CSR), both higher-is-better - two of the seven Cedro & Martens 2026 narrativity metrics; the notebook computes all seven plus auxiliary primitives.

### End-to-end notebook

For the full pipeline (load German Credit, train a RandomForest, compute SHAP, generate the narrative, extract structured claims, and score on faithfulness and narrativity), see the tutorial in [`notebooks/01_quickstart.ipynb`](notebooks/01_quickstart.ipynb).

See `docs/design.md` for the full design and `docs/decisions/` for recorded architecture decisions.

## Choosing a model

Any `LLMProvider` drops into `xains.LLMNarrativeGenerator(llm=...)` - pick the provider for the model you want:

```python
import xains

# Anthropic (reads ANTHROPIC_API_KEY)
llm = xains.AnthropicProvider(model="claude-haiku-4-5", max_tokens=512)

# OpenAI (reads OPENAI_API_KEY)
llm = xains.OpenAIProvider(model="gpt-4o-mini", max_tokens=512)

# OpenRouter - Llama, and many others (reads OPENROUTER_API_KEY)
llm = xains.OpenRouterProvider(model="meta-llama/llama-3.3-70b-instruct", max_tokens=512)

# Any OpenAI-compatible endpoint (Together, Groq, vLLM, ...) - set base_url + the env var to read
llm = xains.OpenAICompatibleProvider(
    base_url="https://api.together.xyz/v1",
    api_key_env_var="TOGETHER_API_KEY",
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    max_tokens=512,
)
```

Each reads its API key from the named env var (or pass `api_key=` explicitly); drop any of them into `xains.LLMNarrativeGenerator(llm=...)` exactly as the Minimal example does.

## License

MIT - see [`LICENSE`](LICENSE).
