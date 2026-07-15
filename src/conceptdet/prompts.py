from __future__ import annotations

from typing import Any

from PIL import Image

STRICT_DETECTION_PROMPT = (
    "Picture 1: Reference Image; red boxes mark the Visual Concept. "
    "Picture 2: Target Image. Find every matching instance ({query!r}). "
    'Output only a JSON array of {{"bbox_2d":[x1,y1,x2,y2]}} with integer XYXY '
    "coordinates on 0-1000. Use [] for no match. No Markdown, prose, reasoning, "
    "or extra keys."
)


def build_messages(
    reference: Image.Image, target: Image.Image, query: str
) -> list[dict[str, Any]]:
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": reference},
                {"type": "image", "image": target},
                {"type": "text", "text": STRICT_DETECTION_PROMPT.format(query=query)},
            ],
        }
    ]
