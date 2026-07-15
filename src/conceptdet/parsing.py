from __future__ import annotations

import re
from dataclasses import dataclass

from conceptdet.errors import OutputFormatError
from conceptdet.types import Box

_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"


@dataclass(frozen=True)
class ParsedCompletion:
    boxes: tuple[Box, ...]
    answer: str | None
    rule: str | None
    reasoning: str | None


def _tag(text: str, name: str) -> str | None:
    match = re.search(rf"<{name}>\s*(.*?)\s*</{name}>", text, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else None


def parse_completion(text: str) -> ParsedCompletion:
    raw_boxes = re.findall(
        r"<bbox>\s*\[?\s*(.*?)\s*\]?\s*</bbox>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    boxes: list[Box] = []
    for raw in raw_boxes:
        values = re.findall(_NUMBER, raw)
        if len(values) != 4:
            continue
        try:
            boxes.append(Box.from_sequence(values))
        except Exception:
            continue
    if not boxes:
        excerpt = " ".join(text.strip().split())[:240]
        raise OutputFormatError(f"Model output contains no valid <bbox> tag: {excerpt!r}")
    return ParsedCompletion(
        boxes=tuple(boxes),
        answer=_tag(text, "answer"),
        rule=_tag(text, "rule"),
        reasoning=_tag(text, "think"),
    )
