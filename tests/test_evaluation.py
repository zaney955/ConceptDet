import hashlib
import json
import shutil
from pathlib import Path

import pytest

from conceptdet.artifact import (
    ADAPTER_CONFIG_FILE,
    TARGET_MODULE_PATTERNS,
    WEIGHTS_FILE,
    initialize_artifact,
)
from conceptdet.config import ArtifactInitConfig, EvaluationConfig
from conceptdet.errors import EvaluationError
from conceptdet.evaluation import EvaluationArtifact, evaluate


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _adapter(tmp_path: Path) -> Path:
    source = tmp_path / "source-adapter"
    source.mkdir()
    (source / WEIGHTS_FILE).write_bytes(b"evaluation-fixture")
    (source / ADAPTER_CONFIG_FILE).write_text(
        json.dumps(
            {
                "base_model_name_or_path": "Qwen/Qwen3-VL-8B-Instruct",
                "peft_type": "LORA",
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "bias": "none",
                "target_modules": sorted(TARGET_MODULE_PATTERNS),
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "adapter"
    initialize_artifact(
        ArtifactInitConfig(1, "artifact.init", source, output, "sft", None, tmp_path, "a")
    )
    return output


def _record(
    record_id: str,
    *,
    reference: str,
    target: str,
    concept: str = "bolt",
    boxes: list[list[int]],
    pixel_boxes: list[list[int]],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "id": record_id,
        "split": "test",
        "visual_concept": concept,
        "query": "same Visual Concept",
        "positive": bool(boxes),
        "reference": {
            "source": "fixture",
            "path": reference,
            "width": 100,
            "height": 100,
            "boxes_xyxy": [[1, 1, 10, 10]],
        },
        "target": {
            "source": "fixture",
            "path": target,
            "width": 100,
            "height": 100,
            "boxes_xyxy": pixel_boxes,
        },
        "detection_set": [{"bbox_2d": box} for box in boxes],
    }


def _dataset(tmp_path: Path) -> Path:
    records = [
        _record(
            "small-a",
            reference="ref-a.jpg",
            target="shared.jpg",
            boxes=[[10, 10, 50, 50]],
            pixel_boxes=[[1, 1, 5, 5]],
        ),
        _record(
            "small-b",
            reference="ref-b.jpg",
            target="shared.jpg",
            boxes=[[10, 10, 50, 50]],
            pixel_boxes=[[1, 1, 5, 5]],
        ),
        _record(
            "multi",
            reference="ref-c.jpg",
            target="multi.jpg",
            boxes=[[100, 100, 300, 300], [500, 500, 700, 700]],
            pixel_boxes=[[10, 10, 30, 30], [50, 50, 70, 70]],
        ),
        _record(
            "empty-correct",
            reference="ref-a.jpg",
            target="empty-a.jpg",
            boxes=[],
            pixel_boxes=[],
        ),
        _record(
            "empty-fp",
            reference="ref-a.jpg",
            target="empty-b.jpg",
            boxes=[],
            pixel_boxes=[],
        ),
        _record(
            "empty-invalid",
            reference="ref-a.jpg",
            target="empty-c.jpg",
            boxes=[],
            pixel_boxes=[],
        ),
        _record(
            "large-missed",
            reference="ref-d.jpg",
            target="large.jpg",
            concept="nut",
            boxes=[[100, 100, 600, 900]],
            pixel_boxes=[[10, 10, 60, 90]],
        ),
    ]
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    manifest = dataset / "test.jsonl"
    manifest.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in records
        ),
        encoding="utf-8",
    )
    metadata = {
        "dataset_schema_version": 1,
        "contract_id": "conceptdet.reference-detection-dataset",
        "contract_version": 1,
        "files": {
            "test.jsonl": {
                "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "records": len(records),
            }
        },
        "sources": [],
    }
    metadata["dataset_fingerprint"] = hashlib.sha256(_canonical(metadata)).hexdigest()
    (dataset / "dataset.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return dataset


def _prediction_rows() -> list[dict[str, str]]:
    exact = '[{"bbox_2d":[10,10,50,50]}]'
    return [
        {"id": "small-a", "raw_completion": exact},
        {"id": "small-b", "raw_completion": exact},
        {
            "id": "multi",
            "raw_completion": (
                '[{"bbox_2d":[100,100,300,300]},'
                '{"bbox_2d":[500,500,700,700]},'
                '{"bbox_2d":[100,100,300,300]}]'
            ),
        },
        {"id": "empty-correct", "raw_completion": "[]"},
        {"id": "empty-fp", "raw_completion": exact},
        {"id": "empty-invalid", "raw_completion": "```json\n[]\n```"},
        {"id": "large-missed", "raw_completion": "[]"},
    ]


def test_frozen_evaluation_is_order_and_worker_invariant(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    rows = _prediction_rows()
    predictions.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    config = EvaluationConfig(
        1,
        "evaluate",
        _dataset(tmp_path),
        _adapter(tmp_path),
        predictions,
        "test",
        tmp_path / "evaluation",
        tmp_path / "evaluate.yaml",
        "stable-config-hash",
    )

    first = evaluate(config, workers=1)
    first_report = (first.path / "report.json").read_bytes()
    first_records = (first.path / "records.jsonl").read_bytes()
    metrics = first.report["metrics"]
    assert metrics["positive_macro_mean_set_f1_50_95"] == pytest.approx(0.7)
    assert metrics["micro_at_0_5"] == {
        "true_positives": 4,
        "false_positives": 2,
        "false_negatives": 1,
        "precision": 4 / 6,
        "recall": 4 / 5,
        "f1": 8 / 11,
    }
    assert metrics["positive_macro_soft_set_f1"] == pytest.approx(0.7)
    assert metrics["strict_valid_rate"] == pytest.approx(6 / 7)
    assert metrics["correct_empty_rate"] == pytest.approx(1 / 3)
    assert metrics["negative_false_positive_boxes_per_image"] == pytest.approx(1 / 3)
    assert first.report["slices"]["count"]["multi"][
        "positive_macro_mean_set_f1_50_95"
    ] == pytest.approx(0.8)
    assert first.report["slices"]["relative_area"]["small"]["examples"] == 2
    assert first.report["slices"]["relative_area"]["medium"]["examples"] == 1
    assert first.report["slices"]["relative_area"]["large"]["examples"] == 1
    assert first.report["slices"]["reference_swap"] == {
        "applicable": True,
        "groups": 1,
        "pairs": 1,
        "strict_valid_pair_rate": 1.0,
        "exact_detection_set_rate": 1.0,
        "pairwise_macro_mean_set_f1_50_95": 1.0,
    }

    shutil.rmtree(first.path)
    reordered_rows = list(reversed(_prediction_rows()))
    next(row for row in reordered_rows if row["id"] == "multi")["raw_completion"] = (
        '[{"bbox_2d":[100,100,300,300]},'
        '{"bbox_2d":[100,100,300,300]},'
        '{"bbox_2d":[500,500,700,700]}]'
    )
    next(row for row in reordered_rows if row["id"] == "small-a")["raw_completion"] = (
        '[{"bbox_2d":[10,10,50,50],"label":"ignored"}]'
    )
    predictions.write_text(
        "".join(json.dumps(row) + "\n" for row in reordered_rows),
        encoding="utf-8",
    )
    second = evaluate(config, workers=4)
    assert (second.path / "report.json").read_bytes() == first_report
    assert (second.path / "records.jsonl").read_bytes() == first_records


def test_evaluation_rejects_prediction_coverage_drift_and_report_tampering(
    tmp_path: Path,
) -> None:
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        json.dumps({"id": "unknown", "raw_completion": "[]"}) + "\n",
        encoding="utf-8",
    )
    config = EvaluationConfig(
        1,
        "evaluate",
        _dataset(tmp_path),
        _adapter(tmp_path),
        predictions,
        "test",
        tmp_path / "evaluation",
        tmp_path / "evaluate.yaml",
        "stable-config-hash",
    )
    with pytest.raises(EvaluationError, match="coverage mismatch"):
        evaluate(config)

    predictions.write_text(
        "".join(json.dumps(row) + "\n" for row in _prediction_rows()),
        encoding="utf-8",
    )
    result = evaluate(config)
    with (result.path / "records.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(EvaluationError, match="hash mismatch"):
        EvaluationArtifact.load(result.path)
