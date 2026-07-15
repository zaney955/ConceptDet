from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

from conceptdet.application import DetectionApplication, run_detect_config
from conceptdet.artifact import (
    AdapterArtifact,
    initialize_artifact,
    validate_source_adapter,
)
from conceptdet.config import (
    ArtifactInitConfig,
    BatchConfig,
    DatasetPredictionConfig,
    DataVocConfig,
    DetectConfig,
    EvaluationConfig,
    GRPOStageConfig,
    OutputConfig,
    RequestConfig,
    SFTStageConfig,
    config_to_dict,
    load_config,
)
from conceptdet.dataset import DatasetArtifact, compile_voc_dataset
from conceptdet.errors import (
    ArtifactError,
    ConceptDetError,
    ConfigurationError,
    DatasetError,
    EvaluationError,
    InputError,
)
from conceptdet.model import Qwen3VLAdapter
from conceptdet.protocol import serialize_detection_set
from conceptdet.types import Box


def _config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, type=Path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="conceptdet", description="Qwen3-VL reference-guided Detection Sets"
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.7.0")
    domains = parser.add_subparsers(dest="domain", required=True)

    infer = domains.add_parser("infer", help="Run reference-guided inference")
    infer_commands = infer.add_subparsers(dest="operation", required=True)
    _config_argument(infer_commands.add_parser("detect", help="Run one request"))
    _config_argument(infer_commands.add_parser("batch", help="Run a JSONL manifest"))

    config = domains.add_parser("config", help="Validate and render typed YAML")
    config_commands = config.add_subparsers(dest="operation", required=True)
    _config_argument(config_commands.add_parser("validate"))
    render = config_commands.add_parser("render")
    _config_argument(render)
    render.add_argument("--output", type=Path)

    artifact = domains.add_parser("artifact", help="Manage immutable Adapter Artifacts")
    artifact_commands = artifact.add_subparsers(dest="operation", required=True)
    _config_argument(artifact_commands.add_parser("init"))
    inspect = artifact_commands.add_parser("inspect")
    inspect.add_argument("artifact", type=Path)
    inspect.add_argument("--json", action="store_true")

    data = domains.add_parser("data", help="Compile bbox-native training datasets")
    data_commands = data.add_subparsers(dest="operation", required=True)
    _config_argument(data_commands.add_parser("voc", help="Compile VOC XML into JSONL"))

    train = domains.add_parser("train", help="Run bbox-native training stages")
    train_commands = train.add_subparsers(dest="operation", required=True)
    sft = train_commands.add_parser("sft", help="Run Qwen3-VL LoRA SFT")
    _config_argument(sft)
    sft.add_argument(
        "--resume",
        default="none",
        help="none, auto, or an explicit checkpoint directory",
    )
    grpo = train_commands.add_parser(
        "grpo", help="Run native Qwen3-VL LoRA GRPO from an SFT Artifact"
    )
    _config_argument(grpo)
    grpo.add_argument(
        "--resume",
        default="none",
        help="none, auto, or an explicit complete checkpoint directory",
    )
    predict = domains.add_parser(
        "predict", help="Generate raw predictions for a compiled dataset split"
    )
    predict_commands = predict.add_subparsers(dest="operation", required=True)
    _config_argument(
        predict_commands.add_parser("dataset", help="Predict one complete dataset split")
    )
    evaluate = domains.add_parser(
        "evaluate", help="Evaluate saved strict Detection Set predictions"
    )
    _config_argument(evaluate)
    evaluate.add_argument("--workers", type=int, default=1)

    accept = domains.add_parser("accept", help="Run or assemble acceptance gates")
    accept_commands = accept.add_subparsers(dest="operation", required=True)
    cpu = accept_commands.add_parser("cpu", help="Run C0 and C1 CPU gates")
    cpu.add_argument("--root", type=Path, default=Path.cwd())
    cpu.add_argument("--output", type=Path, required=True)
    assemble = accept_commands.add_parser(
        "assemble", help="Assemble PR, release, or distributed gate evidence"
    )
    assemble.add_argument(
        "--profile", choices=("pr", "release", "distributed"), required=True
    )
    assemble.add_argument("--root", type=Path, default=Path.cwd())
    assemble.add_argument("--evidence-dir", type=Path, required=True)
    assemble.add_argument("--output", type=Path, required=True)
    return parser


def _load_adapter(config: DetectConfig | BatchConfig) -> Qwen3VLAdapter:
    return Qwen3VLAdapter.load(config.artifact, config.runtime)


def _validate_resources(
    config: (
        DetectConfig
        | BatchConfig
        | ArtifactInitConfig
        | DataVocConfig
        | SFTStageConfig
        | GRPOStageConfig
        | DatasetPredictionConfig
        | EvaluationConfig
    ),
) -> None:
    if isinstance(config, DetectConfig):
        AdapterArtifact.load(config.artifact)
        for description, path in (
            ("reference image", config.request.reference_image),
            ("target image", config.request.target_image),
        ):
            if not path.is_file():
                raise InputError(f"{description.capitalize()} does not exist: {path}")
    elif isinstance(config, BatchConfig):
        AdapterArtifact.load(config.artifact)
        for _, row in _manifest_rows(config.manifest):
            request = _request_from_row(row, config.manifest.parent)
            for description, path in (
                ("reference image", request.reference_image),
                ("target image", request.target_image),
            ):
                if not path.is_file():
                    raise InputError(f"{description.capitalize()} does not exist: {path}")
    elif isinstance(config, ArtifactInitConfig):
        validate_source_adapter(config.source_adapter)
        if config.output_dir.exists():
            raise ArtifactError(f"Artifact output already exists: {config.output_dir}")
        if config.parent_artifact is not None:
            AdapterArtifact.load(config.parent_artifact)
    elif isinstance(config, DataVocConfig):
        for source in config.sources:
            if not source.image_dir.is_dir():
                raise DatasetError(f"VOC image directory does not exist: {source.image_dir}")
            if not source.annotation_dir.is_dir():
                raise DatasetError(
                    f"VOC annotation directory does not exist: {source.annotation_dir}"
                )
        if config.output_dir.exists():
            raise DatasetError(
                f"Compiled dataset output already exists: {config.output_dir}"
            )
    elif isinstance(config, SFTStageConfig):
        dataset = DatasetArtifact.load(config.dataset_dir)
        from conceptdet.dataset import validate_training_dataset

        validate_training_dataset(dataset)
        if config.artifact_dir.exists():
            raise ArtifactError(f"SFT Artifact output already exists: {config.artifact_dir}")
    elif isinstance(config, GRPOStageConfig):
        from conceptdet.grpo import validate_grpo_inputs

        validate_grpo_inputs(config)
    elif isinstance(config, DatasetPredictionConfig):
        DatasetArtifact.load(config.dataset_dir)
        AdapterArtifact.load(config.artifact)
        if config.predictions.exists():
            raise EvaluationError(
                f"Prediction output already exists: {config.predictions}"
            )
    else:
        from conceptdet.evaluation import EvaluationArtifact

        DatasetArtifact.load(config.dataset_dir)
        AdapterArtifact.load(config.artifact)
        if not config.predictions.is_file():
            raise EvaluationError(
                f"Prediction JSONL does not exist: {config.predictions}"
            )
        if config.output_dir.exists():
            # A valid frozen report is still immutable and cannot be overwritten.
            EvaluationArtifact.load(config.output_dir)
            raise EvaluationError(
                f"Evaluation output already exists: {config.output_dir}"
            )


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "task"


def _manifest_rows(path: Path) -> list[tuple[int, dict[str, Any]]]:
    if not path.is_file():
        raise InputError(f"Manifest does not exist: {path}")
    rows: list[tuple[int, dict[str, Any]]] = []
    output_names: dict[str, int] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise InputError(f"Invalid JSON at {path}:{line_number}: {exc.msg}") from exc
        if not isinstance(row, dict):
            raise InputError(f"Manifest row {line_number} must be an object")
        allowed = {"id", "reference_image", "reference_boxes", "target_image", "query"}
        missing = allowed - {"id"} - set(row)
        unknown = set(row) - allowed
        if missing or unknown:
            raise InputError(
                f"Manifest row {line_number} missing={sorted(missing)} unknown={sorted(unknown)}"
            )
        task_id = row.get("id", f"task-{len(rows):06d}")
        if (
            isinstance(task_id, bool)
            or not isinstance(task_id, (str, int))
            or not str(task_id).strip()
        ):
            raise InputError(f"Manifest row {line_number} id must be a nonempty string or integer")
        output_name = _safe_name(str(task_id))
        if output_name in output_names:
            raise InputError(
                f"Manifest rows {output_names[output_name]} and {line_number} "
                f"map to the same output name: {output_name}"
            )
        output_names[output_name] = line_number
        rows.append((line_number, row))
    if not rows:
        raise InputError(f"Manifest has no records: {path}")
    return rows


def _manifest_path(value: object, base: Path, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise InputError(f"Manifest {field} must be a path string")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _request_from_row(row: dict[str, Any], base: Path) -> RequestConfig:
    raw_boxes = row["reference_boxes"]
    if not isinstance(raw_boxes, list) or not raw_boxes:
        raise InputError("reference_boxes must be a nonempty list")
    query = row["query"]
    if not isinstance(query, str) or not query.strip():
        raise InputError("query must be a nonempty string")
    return RequestConfig(
        _manifest_path(row["reference_image"], base, "reference_image"),
        tuple(Box.from_sequence(box) for box in raw_boxes),
        _manifest_path(row["target_image"], base, "target_image"),
        query.strip(),
    )


def _run_batch(config: BatchConfig, adapter: Any) -> int:
    application = DetectionApplication(adapter)
    rows = _manifest_rows(config.manifest)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    failures = 0
    for index, (line_number, row) in enumerate(rows):
        task_id = str(row.get("id", f"task-{index:06d}"))
        image_output = config.output_dir / f"{_safe_name(task_id)}.png"
        json_output = image_output.with_suffix(".json")
        if image_output.exists() and not config.overwrite:
            summaries.append({"id": task_id, "status": "skipped", "output": str(image_output)})
            continue
        try:
            request = _request_from_row(row, config.manifest.parent)
            result = application.run(
                request,
                OutputConfig(image_output, json_output, config.layout),
                max_new_tokens=config.runtime.max_new_tokens,
                config_hash=config.config_hash,
            )
            summaries.append(
                {
                    "id": task_id,
                    "status": "ok",
                    "result": str(json_output),
                    "detection_set": [
                        item.to_model_dict() for item in result.protocol_detections
                    ],
                }
            )
        except (ConceptDetError, OSError, ValueError) as exc:
            failures += 1
            summaries.append(
                {"id": task_id, "status": "failed", "line": line_number, "error": str(exc)}
            )
    summary_path = config.output_dir / "results.jsonl"
    summary_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in summaries),
        encoding="utf-8",
    )
    print(f"Batch summary: {summary_path}", file=sys.stderr)
    return 1 if failures else 0


def _execute(args: argparse.Namespace) -> int:
    if args.domain == "accept":
        from conceptdet.acceptance import assemble_acceptance_report, run_cpu_gates

        if args.operation == "cpu":
            report = run_cpu_gates(args.root.resolve(), args.output.resolve())
        else:
            report = assemble_acceptance_report(
                root=args.root.resolve(),
                profile=args.profile,
                evidence_dir=args.evidence_dir.resolve(),
                output=args.output.resolve(),
            )
        payload = json.loads(report.read_text(encoding="utf-8"))
        print(json.dumps({"report": str(report), "accepted": payload["accepted"]}))
        return 0 if payload["accepted"] else 1

    if args.domain == "artifact" and args.operation == "inspect":
        artifact = AdapterArtifact.load(args.artifact)
        payload = {
            "path": str(artifact.path),
            "artifact_fingerprint": artifact.fingerprint,
            "contract_fingerprint": artifact.contract["contract_fingerprint"],
            "contract": artifact.contract,
            "summary": artifact.summary,
        }
        if args.json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            print(f"Artifact: {artifact.path}")
            print(f"Artifact fingerprint: {artifact.fingerprint}")
            print(f"Contract fingerprint: {artifact.contract['contract_fingerprint']}")
            print(f"Stage: {artifact.summary.get('stage')}")
        return 0

    config = load_config(args.config)
    if args.domain == "config":
        payload = config_to_dict(config)
        if args.operation == "validate":
            _validate_resources(config)
            print(json.dumps({"kind": config.kind, "config_hash": config.config_hash}))
            return 0
        payload.pop("config_hash")
        rendered = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        else:
            print(rendered, end="")
        return 0

    if args.domain == "artifact":
        if not isinstance(config, ArtifactInitConfig):
            raise ConfigurationError("artifact init requires kind: artifact.init")
        artifact = initialize_artifact(config)
        print(
            json.dumps(
                {"artifact": str(artifact.path), "artifact_fingerprint": artifact.fingerprint}
            )
        )
        return 0

    if args.domain == "data":
        if not isinstance(config, DataVocConfig):
            raise ConfigurationError("data voc requires kind: data.voc")
        dataset = compile_voc_dataset(config)
        print(
            json.dumps(
                {
                    "dataset": str(dataset.path),
                    "dataset_fingerprint": dataset.fingerprint,
                    "files": dataset.metadata["files"],
                },
                ensure_ascii=False,
            )
        )
        return 0

    if args.domain == "train":
        resume: str | Path = args.resume
        if resume not in {"none", "auto"}:
            resume = Path(resume)
        if args.operation == "sft":
            if not isinstance(config, SFTStageConfig):
                raise ConfigurationError("train sft requires kind: train.sft")
            from conceptdet.training import run_sft

            result = run_sft(config, resume=resume)  # type: ignore[arg-type]
            payload = {
                "artifact": str(result.artifact.path),
                "artifact_fingerprint": result.artifact.fingerprint,
                "optimizer_steps": result.optimizer_steps,
                "micro_steps": result.micro_steps,
                "final_loss": result.final_loss,
                "peak_reserved_gib": result.peak_reserved_gib,
                "lifecycle_report": str(result.lifecycle_report),
            }
        else:
            if not isinstance(config, GRPOStageConfig):
                raise ConfigurationError("train grpo requires kind: train.grpo")
            from conceptdet.grpo import run_grpo

            result = run_grpo(config, resume=resume)  # type: ignore[arg-type]
            payload = {
                "artifact": str(result.artifact.path),
                "artifact_fingerprint": result.artifact.fingerprint,
                "optimizer_steps": result.optimizer_steps,
                "reward_events": result.reward_events,
                "nonzero_advantage_groups": result.nonzero_advantage_groups,
                "peak_reserved_gib": result.peak_reserved_gib,
                "lifecycle_report": str(result.lifecycle_report),
            }
        from conceptdet.run_state import ProcessContext

        if ProcessContext.current().is_main:
            print(json.dumps(payload))
        return 0

    if args.domain == "predict":
        if not isinstance(config, DatasetPredictionConfig):
            raise ConfigurationError("predict dataset requires kind: predict.dataset")
        from conceptdet.prediction import generate_dataset_predictions

        result = generate_dataset_predictions(config)
        from conceptdet.run_state import ProcessContext

        if ProcessContext.current().is_main:
            print(
                json.dumps(
                    {
                        "predictions": str(result.path),
                        "records": result.records,
                        "content_sha256": result.content_sha256,
                    }
                )
            )
        return 0

    if args.domain == "evaluate":
        if not isinstance(config, EvaluationConfig):
            raise ConfigurationError("evaluate requires kind: evaluate")
        from conceptdet.evaluation import evaluate

        result = evaluate(config, workers=args.workers)
        print(
            json.dumps(
                {
                    "evaluation": str(result.path),
                    "evaluation_fingerprint": result.fingerprint,
                    "metrics": result.report["metrics"],
                },
                ensure_ascii=False,
            )
        )
        return 0

    if args.operation == "detect":
        if not isinstance(config, DetectConfig):
            raise ConfigurationError("infer detect requires kind: infer.detect")
        _validate_resources(config)
        result = run_detect_config(config, _load_adapter(config))
        print(serialize_detection_set(result.protocol_detections))
        print(f"Output image: {config.output.image}", file=sys.stderr)
        print(f"JSON result: {config.output.json}", file=sys.stderr)
        return 0
    if not isinstance(config, BatchConfig):
        raise ConfigurationError("infer batch requires kind: infer.batch")
    _validate_resources(config)
    return _run_batch(config, _load_adapter(config))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _execute(args)
    except (
        ConfigurationError,
        ArtifactError,
        DatasetError,
        EvaluationError,
        InputError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (ConceptDetError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
