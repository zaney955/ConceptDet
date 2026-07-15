"""PROTOTYPE — pure Qwen3-VL visual-budget calculations.

Question: which shared per-image visual-token cap preserves useful dynamic
resolution for a reference/target pair while leaving enough memory headroom for
single-GPU LoRA smoke training?
"""

from __future__ import annotations

from dataclasses import dataclass

PATCH_SIZE = 16
MERGE_SIZE = 2
PIXELS_PER_TOKEN = (PATCH_SIZE * MERGE_SIZE) ** 2


@dataclass(frozen=True)
class Budget:
    tokens_per_image: int

    @property
    def max_pixels(self) -> int:
        return self.tokens_per_image * PIXELS_PER_TOKEN


@dataclass(frozen=True)
class ProcessedImage:
    temporal_grid: int
    height_grid: int
    width_grid: int

    @property
    def visual_tokens(self) -> int:
        return self.temporal_grid * self.height_grid * self.width_grid // (MERGE_SIZE**2)

    @property
    def processed_height(self) -> int:
        return self.height_grid * PATCH_SIZE

    @property
    def processed_width(self) -> int:
        return self.width_grid * PATCH_SIZE


def processed_image(grid_thw: list[int]) -> ProcessedImage:
    if len(grid_thw) != 3:
        raise ValueError("image_grid_thw must contain temporal, height, and width grids")
    return ProcessedImage(*map(int, grid_thw))
