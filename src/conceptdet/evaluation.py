from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

from conceptdet.artifact import AdapterArtifact
from conceptdet.config import EvaluationConfig
from conceptdet.dataset import DatasetArtifact
from conceptdet.errors import EvaluationError, OutputFormatError
from conceptdet.protocol import (
    ProtocolDetection,
    hard_set_counts,
    maximum_iou_assignment,
    parse_detection_set,
    serialize_detection_set,
)
from conceptdet.types import Box

REPORT_FILE = "report.json"
RECORDS_FILE = "records.jsonl"
THRESHOLDS = tuple(round(0.5 + index * 0.05, 2) for index in range(10))
METRIC_CONTRACT = "conceptdet.confidence-free-detection-set-evaluation.v1"


def _box_key(box: Box) -> tuple[float, float, float, float]:
    return box.x1, box.y1, box.x2, box.y2


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _set_f1(true_positives: int, false_positives: int, false_negatives: int) -> float:
    denominator = 2 * true_positives + false_positives + false_negatives
    return 2 * true_positives / denominator if denominator else 1.0


def _soft_set_f1(predictions: tuple[Box, ...], targets: tuple[Box, ...]) -> float:
    if not predictions and not targets:
        return 1.0
    if not predictions or not targets:
        return 0.0
    _, matched_iou = maximum_iou_assignment(predictions, targets)
    return 2 * matched_iou / (len(predictions) + len(targets))


def _strict_truth(value: object, record_id: str) -> tuple[Box, ...]:
    try:
        raw = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        return tuple(sorted((item.box for item in parse_detection_set(raw)), key=_box_key))
    except (OutputFormatError, TypeError, ValueError) as exc:
        raise EvaluationError(f"Dataset record {record_id} has invalid truth: {exc}") from exc


def _pixel_boxes(value: object, record_id: str) -> tuple[Box, ...]:
    if not isinstance(value, list):
        raise EvaluationError(f"Dataset record {record_id} target.boxes_xyxy must be a list")
    try:
        return tuple(Box.from_sequence(item) for item in value)
    except Exception as exc:
        raise EvaluationError(
            f"Dataset record {record_id} has invalid target pixel boxes: {exc}"
        ) from exc


@dataclass(frozen=True)
class _Example:
    record_id: str
    visual_concept: str
    query: str
    target_key: str
    reference_key: str
    targets: tuple[Box, ...]
    target_pixel_boxes: tuple[Box, ...]
    target_width: int
    target_height: int
    raw_completion: str


@dataclass(frozen=True)
class _ScoredExample:
    example: _Example
    format_valid: bool
    format_error: str | None
    predictions: tuple[Box, ...]
    counts: tuple[tuple[int, int, int], ...]
    set_f1: tuple[float, ...]
    soft_set_f1: float
    area_slice: str | None

    @property
    def positive(self) -> bool:
        return bool(self.example.targets)

    @property
    def count_slice(self) -> str:
        count = len(self.example.targets)
        return "empty" if count == 0 else "one" if count == 1 else "multi"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.example.record_id,
            "visual_concept": self.example.visual_concept,
            "positive": self.positive,
            "target_count": len(self.example.targets),
            "prediction_count": len(self.predictions),
            "format_valid": self.format_valid,
            "format_error": self.format_error,
            "prediction": (
                json.loads(
                    serialize_detection_set(
                        ProtocolDetection(box) for box in self.predictions
                    )
                )
                if self.format_valid
                else None
            ),
            "set_f1": {
                f"{threshold:.2f}": value
                for threshold, value in zip(THRESHOLDS, self.set_f1, strict=True)
            },
            "mean_set_f1_50_95": _mean(list(self.set_f1)),
            "soft_set_f1": self.soft_set_f1,
            "count_slice": self.count_slice,
            "relative_area_slice": self.area_slice,
        }


def _relative_area_slice(example: _Example) -> str | None:
    if not example.target_pixel_boxes:
        return None
    image_area = example.target_width * example.target_height
    if image_area <= 0:
        raise EvaluationError(f"Dataset record {example.record_id} has invalid target size")
    mean_relative_area = sum(box.area / image_area for box in example.target_pixel_boxes) / len(
        example.target_pixel_boxes
    )
    if mean_relative_area < 0.01:
        return "small"
    if mean_relative_area < 0.10:
        return "medium"
    return "large"


def _score(example: _Example) -> _ScoredExample:
    try:
        predictions = tuple(
            sorted(
                (item.box for item in parse_detection_set(example.raw_completion)),
                key=_box_key,
            )
        )
        valid = True
        error = None
    except OutputFormatError as exc:
        predictions = ()
        valid = False
        error = str(exc)
    counts = tuple(
        hard_set_counts(predictions, example.targets, threshold) for threshold in THRESHOLDS
    )
    set_f1 = (
        tuple(_set_f1(*item) for item in counts)
        if valid
        else tuple(0.0 for _ in THRESHOLDS)
    )
    soft = _soft_set_f1(predictions, example.targets) if valid else 0.0
    return _ScoredExample(
        example,
        valid,
        error,
        predictions,
        counts,
        set_f1,
        soft,
        _relative_area_slice(example),
    )


def _micro_metrics(entries: list[_ScoredExample]) -> dict[str, float | int]:
    true_positives = sum(item.counts[0][0] for item in entries)
    false_positives = sum(item.counts[0][1] for item in entries)
    false_negatives = sum(item.counts[0][2] for item in entries)
    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives
    precision = (
        true_positives / precision_denominator
        if precision_denominator
        else (1.0 if false_negatives == 0 else 0.0)
    )
    recall = (
        true_positives / recall_denominator
        if recall_denominator
        else (1.0 if false_positives == 0 else 0.0)
    )
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": _set_f1(true_positives, false_positives, false_negatives),
    }


def _summarize(entries: list[_ScoredExample]) -> dict[str, Any]:
    positives = [item for item in entries if item.positive]
    negatives = [item for item in entries if not item.positive]
    all_means = [_mean(list(item.set_f1)) for item in entries]
    positive_means = [_mean(list(item.set_f1)) for item in positives]
    return {
        "examples": len(entries),
        "positive_examples": len(positives),
        "negative_examples": len(negatives),
        "positive_macro_mean_set_f1_50_95": _mean(
            [value for value in positive_means if value is not None]
        ),
        "all_macro_mean_set_f1_50_95": _mean(
            [value for value in all_means if value is not None]
        ),
        "micro_at_0_5": _micro_metrics(entries),
        "positive_macro_soft_set_f1": _mean([item.soft_set_f1 for item in positives]),
        "strict_valid_rate": _mean([1.0 if item.format_valid else 0.0 for item in entries]),
        "correct_empty_rate": _mean(
            [
                1.0 if item.format_valid and not item.predictions else 0.0
                for item in negatives
            ]
        ),
        "negative_false_positive_boxes_per_image": _mean(
            [float(len(item.predictions)) for item in negatives]
        ),
    }


def _reference_swap(entries: list[_ScoredExample]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[_ScoredExample]] = {}
    for entry in entries:
        key = (
            entry.example.target_key,
            entry.example.query,
            entry.example.visual_concept,
        )
        grouped.setdefault(key, []).append(entry)

    pairs: list[tuple[_ScoredExample, _ScoredExample]] = []
    group_count = 0
    for members in grouped.values():
        references = {member.example.reference_key for member in members}
        if len(references) < 2:
            continue
        truth_sets = {
            tuple(sorted(tuple(box.to_list()) for box in member.example.targets))
            for member in members
        }
        if len(truth_sets) != 1:
            raise EvaluationError("Reference-swap group has inconsistent target truth")
        group_count += 1
        ordered_members = sorted(members, key=lambda item: item.example.record_id)
        pairs.extend(
            (left, right)
            for left, right in combinations(ordered_members, 2)
            if left.example.reference_key != right.example.reference_key
        )

    if not pairs:
        return {
            "applicable": False,
            "groups": 0,
            "pairs": 0,
            "strict_valid_pair_rate": None,
            "exact_detection_set_rate": None,
            "pairwise_macro_mean_set_f1_50_95": None,
        }

    pairwise_scores: list[float] = []
    exact_scores: list[float] = []
    strict_scores: list[float] = []
    for left, right in pairs:
        both_valid = left.format_valid and right.format_valid
        strict_scores.append(1.0 if both_valid else 0.0)
        left_boxes = tuple(sorted(tuple(box.to_list()) for box in left.predictions))
        right_boxes = tuple(sorted(tuple(box.to_list()) for box in right.predictions))
        exact_scores.append(1.0 if both_valid and left_boxes == right_boxes else 0.0)
        if not both_valid:
            pairwise_scores.append(0.0)
            continue
        scores = [
            _set_f1(*hard_set_counts(left.predictions, right.predictions, threshold))
            for threshold in THRESHOLDS
        ]
        pairwise_scores.append(_mean(scores) or 0.0)
    return {
        "applicable": True,
        "groups": group_count,
        "pairs": len(pairs),
        "strict_valid_pair_rate": _mean(strict_scores),
        "exact_detection_set_rate": _mean(exact_scores),
        "pairwise_macro_mean_set_f1_50_95": _mean(pairwise_scores),
    }


def _load_predictions(path: Path) -> tuple[dict[str, str], str]:
    if not path.is_file():
        raise EvaluationError(f"Prediction JSONL does not exist: {path}")
    predictions: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvaluationError(f"Invalid prediction JSON at {path}:{line_number}") from exc
        if not isinstance(row, dict) or set(row) != {"id", "raw_completion"}:
            raise EvaluationError(
                f"Prediction row {line_number} must contain exactly id and raw_completion"
            )
        record_id = row["id"]
        raw_completion = row["raw_completion"]
        if not isinstance(record_id, str) or not record_id:
            raise EvaluationError(f"Prediction row {line_number} id must be nonempty text")
        if not isinstance(raw_completion, str):
            raise EvaluationError(
                f"Prediction row {line_number} raw_completion must be text"
            )
        if record_id in predictions:
            raise EvaluationError(f"Duplicate prediction id: {record_id}")
        predictions[record_id] = raw_completion
    canonical: list[dict[str, Any]] = []
    for record_id in sorted(predictions):
        raw_completion = predictions[record_id]
        try:
            boxes = sorted(
                (item.box for item in parse_detection_set(raw_completion)), key=_box_key
            )
            semantic_prediction: dict[str, Any] = {
                "format_valid": True,
                "boxes": [box.to_list(rounded=True) for box in boxes],
            }
        except OutputFormatError:
            semantic_prediction = {
                "format_valid": False,
                "raw_completion": raw_completion,
            }
        canonical.append({"id": record_id, "prediction": semantic_prediction})
    return predictions, hashlib.sha256(_canonical_json(canonical)).hexdigest()


def _examples(
    dataset: DatasetArtifact, split: str, predictions: dict[str, str]
) -> list[_Example]:
    records = sorted(dataset.iter_records(split), key=lambda row: str(row.get("id", "")))
    record_ids: set[str] = set()
    examples: list[_Example] = []
    for row in records:
        record_id = row.get("id")
        concept = row.get("visual_concept")
        query = row.get("query")
        reference = row.get("reference")
        target = row.get("target")
        if (
            not isinstance(record_id, str)
            or not record_id
            or record_id in record_ids
            or not isinstance(concept, str)
            or not concept
            or not isinstance(query, str)
            or not isinstance(reference, dict)
            or not isinstance(target, dict)
        ):
            raise EvaluationError(f"Dataset split contains an invalid record: {record_id!r}")
        record_ids.add(record_id)
        width, height = target.get("width"), target.get("height")
        if (
            isinstance(width, bool)
            or not isinstance(width, int)
            or width <= 0
            or isinstance(height, bool)
            or not isinstance(height, int)
            or height <= 0
        ):
            raise EvaluationError(f"Dataset record {record_id} has invalid target size")
        target_key = f"{target.get('source')}:{target.get('path')}"
        reference_key = f"{reference.get('source')}:{reference.get('path')}"
        targets = _strict_truth(row.get("detection_set"), record_id)
        target_pixel_boxes = _pixel_boxes(target.get("boxes_xyxy"), record_id)
        if len(targets) != len(target_pixel_boxes):
            raise EvaluationError(
                f"Dataset record {record_id} normalized/pixel truth count mismatch"
            )
        if row.get("positive") is not bool(targets):
            raise EvaluationError(f"Dataset record {record_id} positive flag disagrees with truth")
        for box in target_pixel_boxes:
            if box != box.clamp(width, height):
                raise EvaluationError(
                    f"Dataset record {record_id} has an out-of-bounds target pixel box"
                )
        examples.append(
            _Example(
                record_id,
                concept,
                query,
                target_key,
                reference_key,
                targets,
                target_pixel_boxes,
                width,
                height,
                predictions.get(record_id, ""),
            )
        )
    missing = sorted(record_ids - set(predictions))
    extra = sorted(set(predictions) - record_ids)
    if missing or extra:
        raise EvaluationError(
            f"Prediction coverage mismatch: missing={missing[:5]} extra={extra[:5]}"
        )
    return examples


@dataclass(frozen=True)
class EvaluationArtifact:
    path: Path
    report: dict[str, Any]

    @property
    def fingerprint(self) -> str:
        return str(self.report["evaluation_fingerprint"])

    @classmethod
    def load(cls, path: str | Path) -> EvaluationArtifact:
        artifact_path = Path(path).expanduser().resolve()
        report_path = artifact_path / REPORT_FILE
        records_path = artifact_path / RECORDS_FILE
        if not report_path.is_file() or not records_path.is_file():
            raise EvaluationError(f"Frozen evaluation report is incomplete: {artifact_path}")
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise EvaluationError(f"Cannot read frozen evaluation report: {report_path}") from exc
        if not isinstance(report, dict) or report.get("metric_contract") != METRIC_CONTRACT:
            raise EvaluationError("Frozen evaluation report contract is incompatible")
        files = report.get("files")
        if not isinstance(files, dict) or files.get(RECORDS_FILE, {}).get("sha256") != _sha256_file(
            records_path
        ):
            raise EvaluationError(f"Frozen evaluation record hash mismatch: {records_path}")
        fingerprint = report.get("evaluation_fingerprint")
        payload = dict(report)
        payload.pop("evaluation_fingerprint", None)
        if not isinstance(fingerprint, str) or hashlib.sha256(
            _canonical_json(payload)
        ).hexdigest() != fingerprint:
            raise EvaluationError("Frozen evaluation fingerprint mismatch")
        return cls(artifact_path, report)


def evaluate(config: EvaluationConfig, *, workers: int = 1) -> EvaluationArtifact:
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise EvaluationError("workers must be an integer >= 1")
    if config.output_dir.exists():
        raise EvaluationError(f"Evaluation output already exists: {config.output_dir}")
    dataset = DatasetArtifact.load(config.dataset_dir)
    adapter = AdapterArtifact.load(config.artifact)
    predictions, prediction_fingerprint = _load_predictions(config.predictions)
    examples = _examples(dataset, config.split, predictions)
    if workers == 1:
        scored = [_score(example) for example in examples]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            scored = list(executor.map(_score, examples))
    scored.sort(key=lambda item: item.example.record_id)

    records_bytes = b"".join(
        _canonical_json(item.to_dict()) + b"\n" for item in scored
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "metric_contract": METRIC_CONTRACT,
        "thresholds": list(THRESHOLDS),
        "relative_area_definition": (
            "mean Target Instance bbox area / original Target Image area; "
            "small <0.01, medium [0.01,0.10), large >=0.10"
        ),
        "config_hash": config.config_hash,
        "dataset_fingerprint": dataset.fingerprint,
        "adapter_artifact_fingerprint": adapter.fingerprint,
        "adapter_contract_fingerprint": adapter.contract["contract_fingerprint"],
        "prediction_content_fingerprint": prediction_fingerprint,
        "split": config.split,
        "metrics": _summarize(scored),
        "slices": {
            "count": {
                name: _summarize([item for item in scored if item.count_slice == name])
                for name in ("empty", "one", "multi")
            },
            "relative_area": {
                name: _summarize([item for item in scored if item.area_slice == name])
                for name in ("small", "medium", "large")
            },
            "visual_concept": {
                concept: _summarize(
                    [item for item in scored if item.example.visual_concept == concept]
                )
                for concept in sorted({item.example.visual_concept for item in scored})
            },
            "reference_swap": _reference_swap(scored),
        },
        "files": {
            RECORDS_FILE: {
                "sha256": hashlib.sha256(records_bytes).hexdigest(),
                "records": len(scored),
            }
        },
    }
    report["evaluation_fingerprint"] = hashlib.sha256(_canonical_json(report)).hexdigest()

    config.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{config.output_dir.name}.", dir=config.output_dir.parent)
    )
    try:
        (temporary / RECORDS_FILE).write_bytes(records_bytes)
        (temporary / REPORT_FILE).write_text(
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, config.output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return EvaluationArtifact.load(config.output_dir)
