from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from conceptdet.adapter import AdapterInput, DetectionAdapter
from conceptdet.artifact import AdapterArtifact
from conceptdet.config import DatasetPredictionConfig
from conceptdet.dataset import DatasetArtifact
from conceptdet.errors import EvaluationError, InputError
from conceptdet.model import Qwen3VLAdapter
from conceptdet.run_state import (
    ProcessContext,
    assert_distributed_consensus,
    distributed_barrier,
    distributed_objects,
)

_SEQUENCE_LIMIT = re.compile(
    r"Prompt \((?P<prompt>\d+)\) \+ completion \((?P<completion>\d+)\) exceeds (?P<limit>\d+)"
)


@dataclass(frozen=True)
class PredictionResult:
    path: Path
    records: int
    content_sha256: str


def _load_rgb(path: Path) -> Image.Image:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(path) as opened:
                return ImageOps.exif_transpose(opened).convert("RGB")
    except OSError as exc:
        raise EvaluationError(f"Cannot read prediction image: {path}") from exc


def _generate(adapter: DetectionAdapter, request: Any, maximum: int) -> str:
    model_input = AdapterInput(
        _load_rgb(request.reference_image),
        request.reference_boxes,
        _load_rgb(request.target_image),
        request.query,
    )
    try:
        return adapter.generate(model_input, max_new_tokens=maximum).completion
    except InputError as exc:
        match = _SEQUENCE_LIMIT.search(str(exc))
        if match is None:
            raise
        available = int(match.group("limit")) - int(match.group("prompt"))
        if not 1 <= available < maximum:
            raise
        return adapter.generate(model_input, max_new_tokens=available).completion


def _prediction_bytes(rows: list[dict[str, str]]) -> bytes:
    return b"".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
        for row in rows
    )


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def generate_dataset_predictions(
    config: DatasetPredictionConfig,
    *,
    adapter: DetectionAdapter | None = None,
) -> PredictionResult:
    """Generate one raw completion per split record and publish one ordered JSONL."""
    context = ProcessContext.current()
    torch: Any | None = None
    accelerator: Any | None = None
    if context.distributed:
        try:
            import torch as imported_torch
            from accelerate import Accelerator
        except ImportError as exc:
            raise EvaluationError("Distributed prediction dependencies are unavailable") from exc
        torch = imported_torch
        accelerator = Accelerator()

    dataset = DatasetArtifact.load(config.dataset_dir)
    artifact = AdapterArtifact.load(config.artifact)
    if config.predictions.exists():
        raise EvaluationError(f"Prediction output already exists: {config.predictions}")
    if context.distributed:
        assert torch is not None
        assert_distributed_consensus(
            torch,
            (config.config_hash, dataset.fingerprint, artifact.fingerprint),
            context,
            name="prediction identity",
        )

    records = sorted(dataset.iter_records(config.split), key=lambda row: str(row["id"]))
    local_records = records[context.rank :: context.world_size]
    if adapter is None:
        runtime = replace(config.runtime, device=context.cuda_device(config.runtime.device))
        adapter = Qwen3VLAdapter.load(config.artifact, runtime)

    local_rows: list[dict[str, str]] = []
    for record in local_records:
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id:
            raise EvaluationError(f"Dataset prediction record has invalid id: {record_id!r}")
        completion = _generate(
            adapter,
            dataset.detection_request(record),
            config.runtime.max_new_tokens,
        )
        local_rows.append({"id": record_id, "raw_completion": completion})

    if context.distributed:
        assert torch is not None
        gathered = distributed_objects(torch, local_rows, context)
        rows = [row for rank_rows in gathered for row in rank_rows]  # type: ignore[union-attr]
    else:
        rows = local_rows
    rows.sort(key=lambda row: row["id"])
    expected_ids = [str(record["id"]) for record in records]
    if [row["id"] for row in rows] != expected_ids:
        raise EvaluationError("Distributed prediction coverage or ordering mismatch")
    payload = _prediction_bytes(rows)
    digest = hashlib.sha256(payload).hexdigest()
    if context.is_main:
        _atomic_write(config.predictions, payload)
    if context.distributed:
        assert torch is not None
        distributed_barrier(torch, context)
    del accelerator
    return PredictionResult(config.predictions, len(rows), digest)
