#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from conceptdet.prototypes.reference_box_rendering import (
    RenderStyle,
    adaptive_style,
    crop_around_box,
    render_after_resize,
    render_legacy_before_resize,
)


DEFAULT_BOXES = (
    (1165.0, 2911.0, 1354.0, 3230.0),
    (4064.0, 3087.0, 4208.0, 3375.0),
)
PROCESSED_SIZE = (832, 736)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    try:
        return ImageFont.truetype(path, size=size)
    except OSError:
        return ImageFont.load_default()


def _variant_panel(
    title: str,
    image: Image.Image,
    boxes: tuple[tuple[float, float, float, float], ...],
    source_size: tuple[int, int],
) -> Image.Image:
    margin = 18
    title_height = 48
    crop_gap = 12
    crop_size = (320, 240)
    panel_width = max(image.width, crop_size[0] * 2 + crop_gap) + margin * 2
    panel_height = title_height + image.height + crop_size[1] + margin * 3
    panel = Image.new("RGB", (panel_width, panel_height), "#181818")
    draw = ImageDraw.Draw(panel)
    draw.text((margin, 12), title, font=_font(22), fill="white")
    image_x = (panel_width - image.width) // 2
    image_y = title_height + margin
    panel.paste(image, (image_x, image_y))
    crop_y = image_y + image.height + margin
    total_crop_width = crop_size[0] * 2 + crop_gap
    crop_x = (panel_width - total_crop_width) // 2
    for index, box in enumerate(boxes):
        crop = crop_around_box(image, box, source_size, output_size=crop_size)
        panel.paste(crop, (crop_x + index * (crop_size[0] + crop_gap), crop_y))
    return panel


def build_contact_sheet(image: Image.Image) -> Image.Image:
    source_size = image.size
    style = adaptive_style(PROCESSED_SIZE)
    variants = (
        (
            "A · legacy: 2 px in original image",
            render_legacy_before_resize(image, DEFAULT_BOXES, PROCESSED_SIZE),
        ),
        (
            f"B · adaptive red: {style.inner_width} px after resize",
            render_after_resize(
                image,
                DEFAULT_BOXES,
                PROCESSED_SIZE,
                RenderStyle(inner_width=style.inner_width, halo_width=0),
            ),
        ),
        (
            f"C · selected: {style.inner_width} px red + {style.halo_width} px outer halo",
            render_after_resize(image, DEFAULT_BOXES, PROCESSED_SIZE, style),
        ),
    )
    panels = [
        _variant_panel(title, rendered, DEFAULT_BOXES, source_size)
        for title, rendered in variants
    ]
    gap = 18
    sheet_width = sum(panel.width for panel in panels) + gap * (len(panels) - 1)
    sheet = Image.new(
        "RGB",
        (sheet_width, max(panel.height for panel in panels)),
        "#0d0d0d",
    )
    x = 0
    for panel in panels:
        sheet.paste(panel, (x, 0))
        x += panel.width + gap
    return sheet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Qwen3-VL Reference Box styles")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reference_box_rendering.png"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Image.open(args.image) as opened:
        image = ImageOps.exif_transpose(opened).copy()
    sheet = build_contact_sheet(image)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    print(f"source={image.size[0]}x{image.size[1]}")
    print(f"processed={PROCESSED_SIZE[0]}x{PROCESSED_SIZE[1]}")
    print(f"style={adaptive_style(PROCESSED_SIZE)}")
    print(f"wrote={args.output}")


if __name__ == "__main__":
    main()
