"""Render grade aggregates as text with desired-direction arrows.

Arrows mark scored metrics only. Auxiliary primitives on ``NarrativityGrades``
render without arrows (they are diagnostics). ``prompt_version`` is metadata
and is omitted from the render. See ADR 0024 and ADR 0026.
"""

from pydantic import BaseModel

from xains.metrics.grader import EXTRACTION_GRADE_DIRECTIONS, ExtractionGrades
from xains.metrics.narrativity import NARRATIVITY_GRADE_DIRECTIONS, NarrativityGrades


def render_grades(
    extraction: ExtractionGrades | None = None,
    narrativity: NarrativityGrades | None = None,
    scored_only: bool = False,
) -> str:
    """Render extraction and narrativity grades as a grouped, arrow-annotated block.

    Each scored metric renders as ``name <arrow>: value``; auxiliaries render as
    ``name: value``. When ``scored_only=True``, fields absent from the direction
    dict are omitted entirely (drops the 9 NarrativityGrades auxiliaries; no
    visible effect on ExtractionGrades, whose fields are all scored). Returns
    ``""`` when neither aggregate is supplied.
    """
    sections: list[str] = []
    if extraction is not None:
        sections.append(
            _render_section(
                "Verbalization fidelity",
                extraction,
                EXTRACTION_GRADE_DIRECTIONS,
                scored_only=scored_only,
            )
        )
    if narrativity is not None:
        sections.append(
            _render_section(
                "Narrativity",
                narrativity,
                NARRATIVITY_GRADE_DIRECTIONS,
                scored_only=scored_only,
            )
        )
    return "\n\n".join(sections)


def _render_section(
    header: str,
    grades: BaseModel,
    directions: dict[str, str],
    *,
    scored_only: bool,
) -> str:
    lines = [header]
    for field, value in grades.model_dump().items():
        if field == "prompt_version":
            continue
        if scored_only and field not in directions:
            continue
        arrow = directions.get(field, "")
        marker = f" {arrow}" if arrow else ""
        rendered = f"{value:.2f}" if isinstance(value, float) else str(value)
        lines.append(f"  {field}{marker}: {rendered}")
    return "\n".join(lines)
