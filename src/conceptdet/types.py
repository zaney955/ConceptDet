from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from conceptdet.errors import InputError


@dataclass(frozen=True)
class Box:
    """An axis-aligned XYXY box with an exclusive right/bottom edge."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        values = (self.x1, self.y1, self.x2, self.y2)
        if not all(math.isfinite(value) for value in values):
            raise InputError(f"Bounding box contains a non-finite value: {values}")
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise InputError(f"Bounding box must satisfy x2>x1 and y2>y1: {values}")

    @classmethod
    def from_sequence(cls, values: Sequence[float]) -> Box:
        if len(values) != 4:
            raise InputError(f"Expected four bbox values, got {len(values)}: {values}")
        try:
            return cls(*(float(value) for value in values))
        except (TypeError, ValueError) as exc:
            raise InputError(f"Bounding box values must be numeric: {values}") from exc

    def shifted(self, dx: float, dy: float) -> Box:
        return Box(self.x1 + dx, self.y1 + dy, self.x2 + dx, self.y2 + dy)

    def scaled(self, scale_x: float, scale_y: float) -> Box:
        return Box(
            self.x1 * scale_x,
            self.y1 * scale_y,
            self.x2 * scale_x,
            self.y2 * scale_y,
        )

    def clamp(self, width: int, height: int) -> Box:
        if width <= 0 or height <= 0:
            raise InputError(f"Invalid image size: {(width, height)}")
        x1 = min(max(self.x1, 0.0), float(width))
        y1 = min(max(self.y1, 0.0), float(height))
        x2 = min(max(self.x2, 0.0), float(width))
        y2 = min(max(self.y2, 0.0), float(height))
        if x2 <= x1 or y2 <= y1:
            raise InputError(
                f"Bounding box {self.to_list()} lies outside image size {(width, height)}"
            )
        return Box(x1, y1, x2, y2)

    def to_list(self, *, rounded: bool = False) -> list[float] | list[int]:
        values = [self.x1, self.y1, self.x2, self.y2]
        return [int(round(value)) for value in values] if rounded else values


def parse_boxes(value: str | Iterable[Sequence[float]] | Sequence[float]) -> tuple[Box, ...]:
    """Parse ``x1,y1,x2,y2;x1,y1,x2,y2`` or an equivalent sequence."""

    if isinstance(value, str):
        groups = [part.strip() for part in value.split(";") if part.strip()]
        if not groups:
            raise InputError("At least one reference box is required")
        boxes = []
        for group in groups:
            tokens = [token for token in re.split(r"[\s,]+", group) if token]
            boxes.append(Box.from_sequence(tokens))
        return tuple(boxes)

    materialized = list(value)
    if not materialized:
        raise InputError("At least one reference box is required")
    if len(materialized) == 4 and all(isinstance(item, (int, float)) for item in materialized):
        return (Box.from_sequence(materialized),)
    return tuple(Box.from_sequence(item) for item in materialized)
