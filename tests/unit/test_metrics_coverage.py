"""Unit tests for coverage and hallucination_count metrics."""

import pytest

from xains import (
    DatasetSchema,
    FeatureClaim,
    FeatureSchema,
    Modality,
    NarrativeExtraction,
    TargetSchema,
)
from xains.metrics import coverage, hallucination_count

# ------------------------------------------------------ helpers


def _schema(*names: str) -> DatasetSchema:
    return DatasetSchema(
        modality=Modality.TABULAR,
        name="test",
        description="test",
        target=TargetSchema(name="t", description="d", classes={0: "A", 1: "B"}),
        features=[FeatureSchema(name=n, dtype="numeric", description=f"feat {n}") for n in names],
    )


def _claim(name: str, *, rank: int) -> FeatureClaim:
    return FeatureClaim(rank=rank, sign=1, narrative_name=name, resolved_to=name)


def _hallucination(name: str, *, rank: int) -> FeatureClaim:
    return FeatureClaim(rank=rank, sign=1, narrative_name=name, resolved_to=None)


def _extraction(
    features: dict[str, FeatureClaim],
    hallucinations: list[FeatureClaim] | None = None,
) -> NarrativeExtraction:
    return NarrativeExtraction(
        features=features,
        hallucinations=hallucinations or [],
        prompt_version="2",
        model_name="test",
    )


# ------------------------------------------------------ coverage


def test_coverage_full_when_all_top_k_mentioned() -> None:
    schema = _schema("a", "b", "c")
    ext = _extraction(
        features={
            "a": _claim("a", rank=1),
            "b": _claim("b", rank=2),
            "c": _claim("c", rank=3),
        }
    )
    # 3 of min(3, 3) = 3.
    assert coverage(ext, schema, k=3) == 1.0


def test_coverage_partial() -> None:
    schema = _schema("a", "b", "c", "d")
    ext = _extraction(
        features={
            "a": _claim("a", rank=1),
            "b": _claim("b", rank=2),
        }
    )
    # 2 of min(4, 4) = 4 → 0.5.
    assert coverage(ext, schema, k=4) == 0.5


def test_coverage_zero_when_no_resolved_features() -> None:
    schema = _schema("a", "b", "c")
    ext = _extraction(features={})
    assert coverage(ext, schema, k=3) == 0.0


def test_coverage_k_larger_than_features_clamps() -> None:
    schema = _schema("a", "b", "c")
    ext = _extraction(
        features={
            "a": _claim("a", rank=1),
            "b": _claim("b", rank=2),
            "c": _claim("c", rank=3),
        }
    )
    # k=10, schema has 3 features → denominator = min(10, 3) = 3 → 1.0.
    assert coverage(ext, schema, k=10) == 1.0


def test_coverage_invalid_k_raises() -> None:
    schema = _schema("a")
    ext = _extraction(features={})
    with pytest.raises(ValueError):
        coverage(ext, schema, k=0)
    with pytest.raises(ValueError):
        coverage(ext, schema, k=-1)


def test_coverage_ignores_hallucinations() -> None:
    schema = _schema("a", "b", "c")
    ext = _extraction(
        features={"a": _claim("a", rank=1)},
        hallucinations=[
            _hallucination("ghost1", rank=2),
            _hallucination("ghost2", rank=3),
        ],
    )
    # 1 of min(3, 3) = 3 → 1/3.
    result = coverage(ext, schema, k=3)
    assert result == pytest.approx(1.0 / 3.0)


# ------------------------------------------------------ hallucination_count


def test_hallucination_count_zero_when_empty() -> None:
    ext = _extraction(features={})
    assert hallucination_count(ext) == 0


def test_hallucination_count_matches_list_length() -> None:
    ext = _extraction(
        features={"a": _claim("a", rank=1)},
        hallucinations=[
            _hallucination("g1", rank=2),
            _hallucination("g2", rank=3),
            _hallucination("g3", rank=4),
        ],
    )
    assert hallucination_count(ext) == 3
