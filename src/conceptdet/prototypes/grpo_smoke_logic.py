"""PROTOTYPE — strict Detection Set reward used by the GRPO smoke test."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class RewardState:
    valid_json: bool
    error: str | None
    predictions: tuple[Box, ...]
    targets: tuple[Box, ...]
    soft_f1: float
    format_reward: float
    quality_reward: float
    total_reward: float


def _parse(raw: str) -> tuple[tuple[Box, ...], str | None]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        return (), f"invalid JSON: {exc.msg}"
    if not isinstance(value, list):
        return (), "top level must be a JSON array"

    boxes: list[Box] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            return (), f"item {index} must be an object"
        if set(item) - {"bbox_2d", "label"}:
            return (), f"item {index} contains unsupported keys"
        coords = item.get("bbox_2d")
        if not isinstance(coords, list) or len(coords) != 4:
            return (), f"item {index}.bbox_2d must contain four integers"
        if any(isinstance(v, bool) or not isinstance(v, int) for v in coords):
            return (), f"item {index}.bbox_2d must contain four integers"
        x1, y1, x2, y2 = coords
        if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
            return (), f"item {index}.bbox_2d is outside the 0-1000 XYXY range"
        if "label" in item and not isinstance(item["label"], str):
            return (), f"item {index}.label must be a string"
        boxes.append((x1, y1, x2, y2))
    return tuple(boxes), None


def _iou(left: Box, right: Box) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = (
        (left[2] - left[0]) * (left[3] - left[1])
        + (right[2] - right[0]) * (right[3] - right[1])
        - intersection
    )
    return intersection / union if union else 0.0


def _maximum_iou_sum(predictions: tuple[Box, ...], targets: tuple[Box, ...]) -> float:
    if len(predictions) > 12 or len(targets) > 12:
        raise ValueError("prototype supports at most 12 predictions and targets")

    @lru_cache(maxsize=None)
    def solve(target_index: int, used: int) -> float:
        if target_index == len(targets):
            return 0.0
        best = solve(target_index + 1, used)
        for prediction_index, prediction in enumerate(predictions):
            if used & (1 << prediction_index):
                continue
            best = max(
                best,
                _iou(prediction, targets[target_index])
                + solve(target_index + 1, used | (1 << prediction_index)),
            )
        return best

    return solve(0, 0)


def evaluate_reward(raw: str, targets: Sequence[Box]) -> RewardState:
    predictions, error = _parse(raw)
    target_tuple = tuple(targets)
    if error is not None:
        return RewardState(False, error, (), target_tuple, 0.0, 0.0, 0.0, 0.0)

    if not predictions and not target_tuple:
        soft_f1 = 1.0
    else:
        matched_iou = _maximum_iou_sum(predictions, target_tuple)
        precision = matched_iou / len(predictions) if predictions else 0.0
        recall = matched_iou / len(target_tuple) if target_tuple else 0.0
        soft_f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    return RewardState(
        True,
        None,
        predictions,
        target_tuple,
        soft_f1,
        0.10,
        0.90 * soft_f1,
        0.10 + 0.90 * soft_f1,
    )
