# `xains`

[![PyPI version](https://badge.fury.io/py/xains.svg)](https://badge.fury.io/py/xains)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/ADMAntwerp/xains/actions/workflows/ci.yml/badge.svg)](https://github.com/ADMAntwerp/xains/actions/workflows/ci.yml)

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org)
[![Downloads](https://static.pepy.tech/badge/xains)](https://pepy.tech/project/xains)

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Last commit](https://img.shields.io/github/last-commit/ADMAntwerp/xains)](https://github.com/ADMAntwerp/xains/commits/master)

<!-- TODO (add as each service comes online for ADMAntwerp/xains):
[![codecov](https://codecov.io/gh/ADMAntwerp/xains/branch/master/graph/badge.svg)](https://codecov.io/gh/ADMAntwerp/xains)
[![Read the Docs](https://readthedocs.org/projects/xains/badge/?version=latest)](https://xains.readthedocs.io/en/latest/)
-->

`xains` generates explainable AI (XAI) Narratives - hence the name. It turns technical XAI outputs, e.g. SHAP or LIME attributions and counterfactuals, into clear natural-language narrative that makes the explanation of the model's decision understandable to a broad audience. 

## Installation
`xains` is intended to work with **Python 3.11 and above**.
Installation can be done via `pip` :
```sh
pip install xains
```

Or via `uv` :

```sh
uv add xains
```

## Quickstart

Imaging a classifier flagged this applicant as a likely default. The raw feature importances (for example from SHAP) are: `{debt_to_income: +0.37, salary: -0.21, age: -0.12}`. Fine, is this explanation understandable to broad audience? `xains` turns that explanations into a narrative. 

`xains` needs three things: a `schema` (what the features and target mean), a `request` (this instance plus its importances), and an `explainer` (which model verbalizes it).

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

The resulting XAIN (XAI narrative):

```text
Your profile indicates elevated default risk. A debt-to-income ratio of 0.41
substantially increases this concern, signaling that your debt obligations
consume a meaningful portion of earnings. Although your salary of EUR 52,000
and relatively young age of 29 provide some protective factors that work
against default, they ultimately prove insufficient to offset the debt burden
weighing on your financial stability.
```

### Scoring the narrative

A narrative is only useful if it is faithful to the attributions and reads well. `xains` scores both. `grade_extraction` checks the claims the narrative makes against the input attributions - sign, value, and rank fidelity, coverage, and hallucination count:

```python
grades = xains.grade_extraction(
    extraction=result.narrative_extraction,
    request=request,
    schema=schema,
    narrative_text=result.text,
    k=5,
)
print(xains.render_grades(extraction=grades))
```

```text
Verbalization fidelity
  sign_faithfulness ↑: 1.0
  value_faithfulness ↑: 1.0
  rank_correlation ↑: 1.0
  coverage ↑: 1.0
  hallucination_count ↓: 0
```

Arrows mark the desired direction for each metric (`↑` higher is better, `↓` lower is better). `render_grades` also accepts a `narrativity=` argument and emits a second `Narrativity` section.

`grade_narrativity` scores how well the text reads as a narrative, using the metrics from Cedro & Martens 2026. It needs a perplexity provider (any OpenAI-compatible endpoint that returns logprobs):

```python
from xains.metrics import OpenAICompatibleEchoProvider

ppl = OpenAICompatibleEchoProvider(
    base_url="https://api.together.xyz/v1",
    model="meta-llama/Meta-Llama-3-8B-Instruct-Lite",
    api_key_env_var="TOGETHER_API_KEY",
)
narrativity = xains.grade_narrativity(result.text, ppl)
print(xains.render_grades(narrativity=narrativity, scored_only=True))
```

```text
Narrativity
  csr ↑: 0.27
  dcpr ↓: 1.3
  ccpr ↓: 203.14
  cecpr ↓: 29252.68
  fdr ↑: 0.29
  ttcpr ↓: 2.29
  vcpr ↓: 50.79
```

These are the seven Cedro & Martens 2026 narrativity metrics. `scored_only=True` hides the nine auxiliary primitives (`ppl_ordered`, `ttr`, `n_sentences`, ...) which `grade_narrativity` also captures for paper replication; omit the flag to see them.

### End-to-end notebook

For the full pipeline (load German Credit, train a RandomForest, compute SHAP, generate the narrative, extract structured claims, and score on faithfulness and narrativity), see the tutorial in [`notebooks/01_quickstart.ipynb`](notebooks/01_quickstart.ipynb).

### API keys

The remote LLM providers need API keys (set them in your environment or a `.env` file; see `.env.example`).

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

## License

MIT - see [`LICENSE`](LICENSE).
