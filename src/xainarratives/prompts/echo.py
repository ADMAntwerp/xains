"""EchoPromptTemplate — renders the full request as JSON.

Used for skeleton-phase testing: the output is fully deterministic and
mechanically checkable. Not meant for production explanations.
"""

import json

from xainarratives.config import ExplanationConfig
from xainarratives.prompts.base import PromptTemplate
from xainarratives.schema import DatasetSchema
from xainarratives.types import ExplanationRequest


class EchoPromptTemplate(PromptTemplate):
    """Render the request, schema, and config as a single JSON blob.

    The resulting prompt is useless for real LLMs but perfect for verifying
    that the Explainer plumbing routes every field correctly across all four
    modalities.
    """

    _SYSTEM = "You are a mock explainer used for testing. Echo back a short acknowledgement only."

    def render(
        self,
        request: ExplanationRequest,
        schema: DatasetSchema,
        config: ExplanationConfig,
    ) -> tuple[str, str]:
        payload = {
            "schema": schema.model_dump(mode="json"),
            "request": request.model_dump(mode="json"),
            "config": config.model_dump(mode="json"),
        }
        user = json.dumps(payload, indent=2, sort_keys=True, default=str)
        return self._SYSTEM, user
