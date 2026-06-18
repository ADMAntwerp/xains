"""Rule-based guardrails.

One plain function: ``class_name_mentioned``. Strict failure-severity. The
schema target label for ``prediction.predicted_class`` must appear in the
explanation text as a case-insensitive substring. Modality-agnostic.

A rule-based feature-invention check was considered and rejected — see
ADR 0006.
"""

from xain.guardrails.types import GuardrailResult
from xain.schema import DatasetSchema
from xain.types import Prediction


def class_name_mentioned(
    text: str, schema: DatasetSchema, prediction: Prediction
) -> GuardrailResult:
    """Strict: the predicted class's human label appears in ``text`` (case-insensitive)."""
    label = schema.target.classes[prediction.predicted_class]
    return GuardrailResult(
        name="class_name_mentioned",
        severity="failure",
        passed=label.lower() in text.lower(),
        details={
            "expected_label": label,
            "predicted_class": prediction.predicted_class,
        },
    )
