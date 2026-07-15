"""PROTOTYPE — pure Detection Set parsing, matching, and reward logic.

Question: does a strict JSON array plus optimal, order-independent IoU matching
produce intuitive rewards for exact detections, localization errors, misses,
duplicates, false positives, and correct/incorrect empty predictions?

This module is portable on purpose, but remains throwaway until the decision in
the Wayfinder ticket is validated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class Match:
    prediction_index: int
    target_index: int
    iou: float


@dataclass(frozen=True)
class RewardState:
    valid_json: bool
    error: str | None
    predictions: tuple[Box, ...]
    targets: tuple[Box, ...]
    iou_matrix: tuple[tuple[float, ...], ...]
    soft_matches: tuple[Match, ...]
    hard_matches: tuple[Match, ...]
    soft_precision: float
    soft_recall: float
    soft_f1: float
    true_positives_50: int
    false_positives_50: int
    false_negatives_50: int
    f1_50: float
    format_reward: float
    quality_reward: float
    total_reward: float


def parse_detection_set(raw: str) -> tuple[tuple[Box, ...], str | None]:
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
        if any(isinstance(value, bool) or not isinstance(value, int) for value in coords):
            return (), f"item {index}.bbox_2d must contain four integers"
        x1, y1, x2, y2 = coords
        if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
            return (), f"item {index}.bbox_2d is outside the valid 0–1000 XYXY range"
        if "label" in item and not isinstance(item["label"], str):
            return (), f"item {index}.label must be a string when present"
        boxes.append((x1, y1, x2, y2))
    return tuple(boxes), None


def box_iou(left: Box, right: Box) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def _soft_assignment(matrix: tuple[tuple[float, ...], ...]) -> tuple[Match, ...]:
    prediction_count = len(matrix)
    target_count = len(matrix[0]) if prediction_count else 0
    if prediction_count > 12 or target_count > 12:
        raise ValueError("prototype supports at most 12 predictions and 12 targets")

    @lru_cache(maxsize=None)
    def solve(target_index: int, used_predictions: int) -> tuple[float, tuple[Match, ...]]:
        if target_index == target_count:
            return 0.0, ()
        best_score, best_matches = solve(target_index + 1, used_predictions)
        for prediction_index in range(prediction_count):
            if used_predictions & (1 << prediction_index):
                continue
            tail_score, tail_matches = solve(
                target_index + 1, used_predictions | (1 << prediction_index)
            )
            candidate_iou = matrix[prediction_index][target_index]
            candidate_score = candidate_iou + tail_score
            if candidate_score > best_score:
                best_score = candidate_score
                best_matches = (
                    Match(prediction_index, target_index, candidate_iou),
                    *tail_matches,
                )
        return best_score, best_matches

    return tuple(match for match in solve(0, 0)[1] if match.iou > 0)


def _hard_assignment(
    matrix: tuple[tuple[float, ...], ...], threshold: float = 0.5
) -> tuple[Match, ...]:
    prediction_count = len(matrix)
    target_count = len(matrix[0]) if prediction_count else 0

    @lru_cache(maxsize=None)
    def solve(
        target_index: int, used_predictions: int
    ) -> tuple[int, float, tuple[Match, ...]]:
        if target_index == target_count:
            return 0, 0.0, ()
        best = solve(target_index + 1, used_predictions)
        for prediction_index in range(prediction_count):
            if used_predictions & (1 << prediction_index):
                continue
            candidate_iou = matrix[prediction_index][target_index]
            if candidate_iou < threshold:
                continue
            count, score, matches = solve(
                target_index + 1, used_predictions | (1 << prediction_index)
            )
            candidate = (
                count + 1,
                score + candidate_iou,
                (Match(prediction_index, target_index, candidate_iou), *matches),
            )
            if candidate[:2] > best[:2]:
                best = candidate
        return best

    return solve(0, 0)[2]


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def evaluate_reward(raw_prediction: str, targets: Sequence[Box]) -> RewardState:
    predictions, error = parse_detection_set(raw_prediction)
    target_tuple = tuple(targets)
    if error is not None:
        return RewardState(
            valid_json=False,
            error=error,
            predictions=(),
            targets=target_tuple,
            iou_matrix=(),
            soft_matches=(),
            hard_matches=(),
            soft_precision=0.0,
            soft_recall=0.0,
            soft_f1=0.0,
            true_positives_50=0,
            false_positives_50=0,
            false_negatives_50=len(target_tuple),
            f1_50=0.0,
            format_reward=0.0,
            quality_reward=0.0,
            total_reward=0.0,
        )

    matrix = tuple(
        tuple(box_iou(prediction, target) for target in target_tuple)
        for prediction in predictions
    )
    soft_matches = _soft_assignment(matrix)
    hard_matches = _hard_assignment(matrix)
    matched_iou = sum(match.iou for match in soft_matches)

    if not predictions and not target_tuple:
        soft_precision = soft_recall = soft_f1 = 1.0
    else:
        soft_precision = matched_iou / len(predictions) if predictions else 0.0
        soft_recall = matched_iou / len(target_tuple) if target_tuple else 0.0
        soft_f1 = _f1(soft_precision, soft_recall)

    true_positives = len(hard_matches)
    false_positives = len(predictions) - true_positives
    false_negatives = len(target_tuple) - true_positives
    hard_precision = true_positives / len(predictions) if predictions else float(not target_tuple)
    hard_recall = true_positives / len(target_tuple) if target_tuple else float(not predictions)
    f1_50 = _f1(hard_precision, hard_recall)

    format_reward = 0.10
    quality_reward = 0.90 * soft_f1
    return RewardState(
        valid_json=True,
        error=None,
        predictions=predictions,
        targets=target_tuple,
        iou_matrix=matrix,
        soft_matches=soft_matches,
        hard_matches=hard_matches,
        soft_precision=soft_precision,
        soft_recall=soft_recall,
        soft_f1=soft_f1,
        true_positives_50=true_positives,
        false_positives_50=false_positives,
        false_negatives_50=false_negatives,
        f1_50=f1_50,
        format_reward=format_reward,
        quality_reward=quality_reward,
        total_reward=format_reward + quality_reward,
    )
