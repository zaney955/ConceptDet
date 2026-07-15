from __future__ import annotations

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


def maximum_iou_assignment(
    predictions: Sequence[Box],
    targets: Sequence[Box],
    *,
    threshold: float | None = None,
) -> tuple[tuple[tuple[int, int, float], ...], float]:
    """Return an exact maximum-weight one-to-one IoU assignment.

    With a threshold, cardinality is maximized before total IoU. Without one,
    total IoU alone is maximized. A small residual-network implementation keeps
    this exact for Detection Sets larger than the old bit-mask matcher limit.
    """
    if threshold is not None and not 0 <= threshold <= 1:
        raise ValueError("IoU threshold must be between zero and one")
    if not predictions or not targets:
        return (), 0.0

    prediction_count = len(predictions)
    target_count = len(targets)
    source = prediction_count + target_count
    sink = source + 1
    graph: list[list[list[float | int]]] = [[] for _ in range(sink + 1)]

    def add_edge(left: int, right: int, capacity: int, cost: float) -> None:
        forward: list[float | int] = [right, len(graph[right]), capacity, cost]
        reverse: list[float | int] = [left, len(graph[left]), 0, -cost]
        graph[left].append(forward)
        graph[right].append(reverse)

    for prediction_index in range(prediction_count):
        add_edge(source, prediction_index, 1, 0.0)
    for target_index in range(target_count):
        add_edge(prediction_count + target_index, sink, 1, 0.0)

    cardinality_bonus = float(min(prediction_count, target_count) + 1)
    for prediction_index, prediction in enumerate(predictions):
        for target_index, target in enumerate(targets):
            iou = box_iou(prediction, target)
            if threshold is not None and iou < threshold:
                continue
            if threshold is None and iou <= 0:
                continue
            reward = iou + (cardinality_bonus if threshold is not None else 0.0)
            add_edge(prediction_index, prediction_count + target_index, 1, -reward)

    # Successive shortest augmenting paths. Bellman-Ford is deliberate: residual
    # reverse edges have negative/positive costs and Detection Sets are small.
    while True:
        distances = [float("inf")] * len(graph)
        previous: list[tuple[int, int] | None] = [None] * len(graph)
        distances[source] = 0.0
        for _ in range(len(graph) - 1):
            changed = False
            for node, edges in enumerate(graph):
                if distances[node] == float("inf"):
                    continue
                for edge_index, edge in enumerate(edges):
                    destination, _, capacity, cost = edge
                    if int(capacity) <= 0:
                        continue
                    candidate = distances[node] + float(cost)
                    if candidate < distances[int(destination)] - 1e-12:
                        distances[int(destination)] = candidate
                        previous[int(destination)] = (node, edge_index)
                        changed = True
            if not changed:
                break
        if previous[sink] is None or distances[sink] >= -1e-12:
            break
        node = sink
        while node != source:
            prior = previous[node]
            if prior is None:  # pragma: no cover - guarded by the sink path
                raise RuntimeError("Incomplete assignment augmenting path")
            parent, edge_index = prior
            edge = graph[parent][edge_index]
            reverse_index = int(edge[1])
            edge[2] = int(edge[2]) - 1
            graph[node][reverse_index][2] = int(graph[node][reverse_index][2]) + 1
            node = parent

    matches: list[tuple[int, int, float]] = []
    for prediction_index in range(prediction_count):
        for edge in graph[prediction_index]:
            destination, _, capacity, _ = edge
            target_index = int(destination) - prediction_count
            if not 0 <= target_index < target_count or int(capacity) != 0:
                continue
            iou = box_iou(predictions[prediction_index], targets[target_index])
            if iou > 0 or threshold == 0:
                matches.append((prediction_index, target_index, iou))
    matches.sort()
    return tuple(matches), sum(match[2] for match in matches)


def hard_set_counts(
    predictions: Sequence[Box], targets: Sequence[Box], threshold: float
) -> tuple[int, int, int]:
    """Exact maximum-cardinality matching with maximum-IoU tie breaking."""
    matches, _ = maximum_iou_assignment(predictions, targets, threshold=threshold)
    true_positives = len(matches)
    return true_positives, len(predictions) - true_positives, len(targets) - true_positives
