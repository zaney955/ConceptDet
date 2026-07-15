from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from PIL import Image, ImageDraw


@dataclass(frozen=True)
class RenderStyle:
    """A Reference Box style expressed in post-resize image pixels."""

    inner_color: str = "#ff2020"
    halo_color: str = "#ffffff"
    inner_width: int = 3
    halo_width: int = 2

def adaptive_style(processed_size: tuple[int, int]) -> RenderStyle:
    """Keep strokes visible without letting them dominate small instances."""
    shortest_side = min(processed_size)
    inner_width = min(4, max(2, round(shortest_side / 256)))
    return RenderStyle(inner_width=inner_width, halo_width=2)


def _scaled_box(
    box: tuple[float, float, float, float],
    source_size: tuple[int, int],
    processed_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    source_width, source_height = source_size
    processed_width, processed_height = processed_size
    x1, y1, x2, y2 = box
    return (
        round(x1 * processed_width / source_width),
        round(y1 * processed_height / source_height),
        round(x2 * processed_width / source_width),
        round(y2 * processed_height / source_height),
    )


def render_after_resize(
    image: Image.Image,
    boxes: Iterable[tuple[float, float, float, float]],
    processed_size: tuple[int, int],
    style: RenderStyle,
) -> Image.Image:
    """Reference implementation: resize once, then render in processed pixels."""
    source_size = image.size
    rendered = image.convert("RGB").resize(processed_size, Image.Resampling.BICUBIC)
    draw = ImageDraw.Draw(rendered)
    for box in boxes:
        processed_box = _scaled_box(box, source_size, processed_size)
        x1, y1, x2, y2 = processed_box
        if style.halo_width:
            draw.rectangle(
                (
                    x1 - style.halo_width,
                    y1 - style.halo_width,
                    x2 + style.halo_width,
                    y2 + style.halo_width,
                ),
                outline=style.halo_color,
                width=style.halo_width,
            )
        draw.rectangle(
            processed_box,
            outline=style.inner_color,
            width=style.inner_width,
        )
    return rendered


def render_legacy_before_resize(
    image: Image.Image,
    boxes: Iterable[tuple[float, float, float, float]],
    processed_size: tuple[int, int],
    *,
    width: int = 2,
) -> Image.Image:
    """Reproduce the old fixed-width-original-pixel behavior for comparison."""
    rendered = image.convert("RGB")
    draw = ImageDraw.Draw(rendered)
    for box in boxes:
        draw.rectangle(tuple(round(value) for value in box), outline="#ff2020", width=width)
    return rendered.resize(processed_size, Image.Resampling.BICUBIC)


def crop_around_box(
    image: Image.Image,
    box: tuple[float, float, float, float],
    source_size: tuple[int, int],
    *,
    output_size: tuple[int, int] = (320, 240),
    context_scale: float = 4.0,
) -> Image.Image:
    """Make an inspection crop without changing the model input."""
    x1, y1, x2, y2 = _scaled_box(box, source_size, image.size)
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    crop_width = max(1, (x2 - x1) * context_scale)
    crop_height = max(1, (y2 - y1) * context_scale)
    target_ratio = output_size[0] / output_size[1]
    if crop_width / crop_height < target_ratio:
        crop_width = crop_height * target_ratio
    else:
        crop_height = crop_width / target_ratio
    left = min(max(0, math.floor(center_x - crop_width / 2)), image.width - crop_width)
    top = min(max(0, math.floor(center_y - crop_height / 2)), image.height - crop_height)
    crop = image.crop(
        (
            round(left),
            round(top),
            round(left + crop_width),
            round(top + crop_height),
        )
    )
    return crop.resize(output_size, Image.Resampling.NEAREST)
