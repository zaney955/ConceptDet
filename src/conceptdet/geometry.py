from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from PIL import Image, ImageDraw

from conceptdet.errors import InputError
from conceptdet.types import Box


@dataclass(frozen=True)
class ImageTransform:
    """Bidirectional coordinate transform between source and model images."""

    source_size: tuple[int, int]
    model_size: tuple[int, int]

    def __post_init__(self) -> None:
        if min(*self.source_size, *self.model_size) <= 0:
            raise InputError(
                f"Image dimensions must be positive: {self.source_size} -> {self.model_size}"
            )

    @property
    def scale_x(self) -> float:
        return self.model_size[0] / self.source_size[0]

    @property
    def scale_y(self) -> float:
        return self.model_size[1] / self.source_size[1]

    def to_model(self, box: Box) -> Box:
        return box.scaled(self.scale_x, self.scale_y)

    def to_source(self, box: Box) -> Box:
        return box.scaled(1.0 / self.scale_x, 1.0 / self.scale_y)


@dataclass(frozen=True)
class PreparedReference:
    image: Image.Image
    model_boxes: tuple[Box, ...]
    crop_box: tuple[int, int, int, int]


def _expanded_crop(
    boxes: tuple[Box, ...], width: int, height: int, context_scale: float
) -> tuple[int, int, int, int]:
    if context_scale < 1.0:
        raise InputError("reference crop context scale must be >= 1")
    x1 = min(box.x1 for box in boxes)
    y1 = min(box.y1 for box in boxes)
    x2 = max(box.x2 for box in boxes)
    y2 = max(box.y2 for box in boxes)
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    crop_width = min(width, max(1.0, (x2 - x1) * context_scale))
    crop_height = min(height, max(1.0, (y2 - y1) * context_scale))

    left = min(max(0.0, center_x - crop_width / 2), width - crop_width)
    top = min(max(0.0, center_y - crop_height / 2), height - crop_height)
    right = left + crop_width
    bottom = top + crop_height
    return (
        max(0, math.floor(left)),
        max(0, math.floor(top)),
        min(width, math.ceil(right)),
        min(height, math.ceil(bottom)),
    )


def prepare_reference(
    image: Image.Image,
    boxes: tuple[Box, ...],
    *,
    model_size: tuple[int, int] = (600, 600),
    crop_mode: Literal["full", "crop"] = "full",
    context_scale: float = 4.0,
    box_color: str = "red",
    box_width: int = 2,
) -> PreparedReference:
    if not boxes:
        raise InputError("At least one reference box is required")
    if crop_mode not in {"full", "crop"}:
        raise InputError("reference crop mode must be 'full' or 'crop'")
    if box_width < 1:
        raise InputError("box width must be >= 1")

    width, height = image.size
    clamped = tuple(box.clamp(width, height) for box in boxes)
    crop_box = (0, 0, width, height)
    if crop_mode == "crop":
        crop_box = _expanded_crop(clamped, width, height, context_scale)

    cropped = image.crop(crop_box)
    transform = ImageTransform(cropped.size, model_size)
    offset_x, offset_y = crop_box[:2]
    model_boxes = tuple(transform.to_model(box.shifted(-offset_x, -offset_y)) for box in clamped)
    prompt_image = cropped.resize(model_size, Image.Resampling.BICUBIC)
    draw = ImageDraw.Draw(prompt_image)
    for box in model_boxes:
        draw.rectangle(box.to_list(rounded=True), outline=box_color, width=box_width)
    return PreparedReference(prompt_image, model_boxes, crop_box)


def prepare_target(
    image: Image.Image, model_size: tuple[int, int] = (600, 600)
) -> tuple[Image.Image, ImageTransform]:
    transform = ImageTransform(image.size, model_size)
    return image.resize(model_size, Image.Resampling.BICUBIC), transform
