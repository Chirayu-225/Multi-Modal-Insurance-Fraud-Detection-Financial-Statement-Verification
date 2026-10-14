"""
Output validator: nothing a generative model says is trusted until it
passes here.

Three checks, applied in order:

  1. Schema   the response must parse as JSON matching a Pydantic model.
              Failure raises SchemaViolationError -- the same exception
              vision.py already raises -- so pipeline code keeps one
              fail-closed path: a max-severity flag and manual review,
              never a crash and never a silent pass.
  2. Citation every cited chunk ID must be one that was actually
              retrieved and sent to the model.
  3. Grounding every number must appear (within tolerance) in the cited
              chunks, or be the output of the calculator tool. This is
              what stops a model from "helpfully" producing a plausible
              revenue figure that is in no source at all.

Status: real and unit-tested. Unit conversion is deliberately minimal
(Indian digit grouping and plain decimals); figures printed in a
different unit from the one the model reports will fail grounding and
go to review, which is the safe direction to fail in.
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from pydantic import BaseModel, Field, ValidationError

from app.services.vision import SchemaViolationError

_NUMBER = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")


def numbers_in(text: str) -> list[float]:
    out = []
    for m in _NUMBER.finditer(text):
        try:
            out.append(float(m.group().replace(",", "")))
        except ValueError:
            continue
    return out


def is_grounded(value: float, candidates: Iterable[float], rel_tol: float = 0.005) -> bool:
    for c in candidates:
        if c == value or (c != 0 and abs(c - value) / abs(c) <= rel_tol):
            return True
    return False


class GroundedFigure(BaseModel):
    metric: str
    value: float
    unit: str = "INR_crore"
    fiscal_year: str
    source_chunk_ids: list[str] = Field(min_length=1)


class FigureExtraction(BaseModel):
    figures: list[GroundedFigure]


@dataclass
class ValidationOutcome:
    accepted: list[GroundedFigure] = field(default_factory=list)
    rejected: list[tuple[GroundedFigure, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.rejected


def parse_json_model(text: str, model: type[BaseModel]) -> BaseModel:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned)
    try:
        return model.model_validate(json.loads(cleaned))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise SchemaViolationError(f"model output did not match {model.__name__}: {exc}") from exc


def validate_figure_extraction(
    text: str,
    retrieved_chunks: dict[str, str],
    computed_values: Iterable[float] = (),
) -> ValidationOutcome:
    extraction: FigureExtraction = parse_json_model(text, FigureExtraction)  # type: ignore[assignment]
    computed = list(computed_values)
    outcome = ValidationOutcome()
    for fig in extraction.figures:
        unknown = [cid for cid in fig.source_chunk_ids if cid not in retrieved_chunks]
        if unknown:
            outcome.rejected.append((fig, f"cites chunk(s) that were never retrieved: {unknown}"))
            continue
        candidates = [n for cid in fig.source_chunk_ids for n in numbers_in(retrieved_chunks[cid])] + computed
        if not is_grounded(fig.value, candidates):
            outcome.rejected.append((fig, f"value {fig.value} does not appear in the cited chunks or calculator output"))
            continue
        outcome.accepted.append(fig)
    return outcome
