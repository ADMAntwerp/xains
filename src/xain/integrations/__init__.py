"""Integration adapters. Named by the shape of the input, not the upstream tool."""

from xain.integrations.feature_importance import from_feature_importance

__all__ = ["from_feature_importance"]
