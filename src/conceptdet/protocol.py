from __future__ import annotations

import functools
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

from conceptdet.errors import InputError, OutputFormatError
from conceptdet.types import Box


@dataclass(frozen=True)
class ProtocolDetection:
    box: Box
    label: str | None = None

    def to_model_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {"bbox_2d": self.box.to_list(rounded=True)}
        if self.label is not None:
            payload["label"] = self.label
        return payload


def parse_detection_set(raw: str) -> tuple[ProtocolDetection, ...]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise OutputFormatError(f"Model output is not strict JSON: {exc.msg}") from exc
    if not isinstance(value, list):
        raise OutputFormatError("Detection Set must be a top-level JSON array")

    detections: list[ProtocolDetection] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise OutputFormatError(f"Detection Set item {index} must be an object")
        extra = set(item) - {"bbox_2d", "label"}
        if extra:
            raise OutputFormatError(
                f"Detection Set item {index} has unsupported keys: {', '.join(sorted(extra))}"
            )
        coordinates = item.get("bbox_2d")
        if not isinstance(coordinates, list) or len(coordinates) != 4:
            raise OutputFormatError(f"Detection Set item {index}.bbox_2d needs four integers")
        if any(isinstance(number, bool) or not isinstance(number, int) for number in coordinates):
            raise OutputFormatError(f"Detection Set item {index}.bbox_2d needs four integers")
        x1, y1, x2, y2 = coordinates
        if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
            raise OutputFormatError(
                f"Detection Set item {index}.bbox_2d must be valid XYXY on 0-1000"
            )
        label = item.get("label")
        if label is not None and not isinstance(label, str):
            raise OutputFormatError(f"Detection Set item {index}.label must be a string")
        detections.append(ProtocolDetection(Box(x1, y1, x2, y2), label))
    return tuple(detections)


def serialize_detection_set(detections: Iterable[ProtocolDetection]) -> str:
    return json.dumps(
        [detection.to_model_dict() for detection in detections],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _half_up(value: float) -> int:
    return int(Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def encode_pixel_box(box: Box, image_size: tuple[int, int]) -> Box:
    width, height = image_size
    if width <= 0 or height <= 0:
        raise OutputFormatError(f"Invalid image size for encoding: {image_size}")
    clamped = box.clamp(width, height)
    try:
        return Box(
            _half_up(clamped.x1 / width * 1000),
            _half_up(clamped.y1 / height * 1000),
            _half_up(clamped.x2 / width * 1000),
            _half_up(clamped.y2 / height * 1000),
        )
    except InputError as exc:
        raise OutputFormatError(
            f"Pixel bbox collapses on the 0-1000 grid: {box.to_list()}"
        ) from exc


def decode_model_box(box: Box, image_size: tuple[int, int]) -> Box:
    width, height = image_size
    if width <= 0 or height <= 0:
        raise OutputFormatError(f"Invalid image size for decoding: {image_size}")
    try:
        return Box(
            box.x1 / 1000 * width,
            box.y1 / 1000 * height,
            box.x2 / 1000 * width,
            box.y2 / 1000 * height,
        ).clamp(width, height)
    except Exception as exc:
        raise OutputFormatError(f"Model produced unusable bbox: {box.to_list()}") from exc


def box_iou(left: Box, right: Box) -> float:
    x1, y1 = max(left.x1, right.x1), max(left.y1, right.y1)
    x2, y2 = min(left.x2, right.x2), min(left.y2, right.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = left.area + right.area - intersection
    return intersection / union if union else 0.0


def hard_set_counts(
    predictions: Sequence[Box], targets: Sequence[Box], threshold: float
) -> tuple[int, int, int]:
    """Exact maximum-cardinality matching with maximum-IoU tie breaking."""
    if not 0 <= threshold <= 1:
        raise ValueError("IoU threshold must be between zero and one")
    if len(predictions) > 20 or len(targets) > 20:
        raise ValueError("exact matcher supports at most 20 predictions and targets")

    @functools.cache
    def solve(target_index: int, used: int) -> tuple[int, float]:
        if target_index == len(targets):
            return 0, 0.0
        best = solve(target_index + 1, used)
        for prediction_index, prediction in enumerate(predictions):
            if used & (1 << prediction_index):
                continue
            iou = box_iou(prediction, targets[target_index])
            if iou < threshold:
                continue
            count, total_iou = solve(target_index + 1, used | (1 << prediction_index))
            best = max(best, (count + 1, total_iou + iou))
        return best

    true_positives = solve(0, 0)[0]
    return true_positives, len(predictions) - true_positives, len(targets) - true_positives
