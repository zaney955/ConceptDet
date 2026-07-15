from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

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
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise InputError(f"Bounding box must be a four-value sequence: {values}")
        if len(values) != 4:
            raise InputError(f"Expected four bbox values, got {len(values)}: {values}")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
            raise InputError(f"Bounding box values must be numeric: {values}")
        return cls(*(float(value) for value in values))

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

    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)
