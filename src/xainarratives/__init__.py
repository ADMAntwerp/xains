"""xainarratives — Natural-language verbalization of ML predictions from pre-computed attributions.

See :mod:`xainarratives.schema`, :mod:`xainarratives.types`, and
:mod:`xainarratives.explainer` for the core public API. See :doc:`docs/design.md`
for the architectural overview.
"""

from xainarratives.config import ExplanationConfig
from xainarratives.explainer import Explainer
from xainarratives.schema import (
    DatasetSchema,
    FeatureSchema,
    GraphSpec,
    ImageSpec,
    Modality,
    TargetSchema,
    TextSpec,
)
from xainarratives.types import (
    EdgeContribution,
    ExplanationMode,
    ExplanationRequest,
    ExplanationResult,
    GraphCounterfactual,
    GraphExplanationRequest,
    ImageCounterfactual,
    ImageExplanationRequest,
    NodeContribution,
    Prediction,
    RegionContribution,
    TabularContribution,
    TabularCounterfactual,
    TabularExplanationRequest,
    TextCounterfactual,
    TextExplanationRequest,
    TokenContribution,
)

__version__ = "0.0.1"

__all__ = [
    "DatasetSchema",
    "EdgeContribution",
    "Explainer",
    "ExplanationConfig",
    "ExplanationMode",
    "ExplanationRequest",
    "ExplanationResult",
    "FeatureSchema",
    "GraphCounterfactual",
    "GraphExplanationRequest",
    "GraphSpec",
    "ImageCounterfactual",
    "ImageExplanationRequest",
    "ImageSpec",
    "Modality",
    "NodeContribution",
    "Prediction",
    "RegionContribution",
    "TabularContribution",
    "TabularCounterfactual",
    "TabularExplanationRequest",
    "TargetSchema",
    "TextCounterfactual",
    "TextExplanationRequest",
    "TextSpec",
    "TokenContribution",
    "__version__",
]
