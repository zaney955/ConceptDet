from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from conceptdet.errors import InputError
from conceptdet.types import Box


def annotate(
    image: Image.Image,
    boxes: tuple[Box, ...],
    *,
    label: str | None = None,
    color: str = "red",
    width: int = 4,
) -> Image.Image:
    if width < 1:
        raise InputError("annotation width must be >= 1")
    result = image.copy()
    draw = ImageDraw.Draw(result)
    font = ImageFont.load_default()
    for box in boxes:
        coordinates = box.to_list()
        draw.rectangle(coordinates, outline=color, width=width)
        if label:
            left, top = int(box.x1), int(box.y1)
            text_box = draw.textbbox((left, top), label, font=font, stroke_width=1)
            text_width = text_box[2] - text_box[0]
            text_height = text_box[3] - text_box[1]
            label_top = max(0, top - text_height - 4)
            draw.rectangle(
                (left, label_top, left + text_width + 6, label_top + text_height + 4),
                fill=color,
            )
            draw.text((left + 3, label_top + 2), label, fill="white", font=font)
    return result


def compose_triptych(
    reference_image: Image.Image,
    target_image: Image.Image,
    result_image: Image.Image,
) -> Image.Image:
    """Compose Reference | Target | Detection using a shared panel size."""

    panel_size = target_image.size
    panels = []
    for image in (reference_image, target_image, result_image):
        panel = image.convert("RGB")
        if panel.size != panel_size:
            panel = panel.resize(panel_size, Image.Resampling.BICUBIC)
        panels.append(panel)

    combined = Image.new("RGB", (panel_size[0] * 3, panel_size[1]), "black")
    for index, panel in enumerate(panels):
        combined.paste(panel, (index * panel_size[0], 0))
    return combined
