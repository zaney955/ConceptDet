from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from conceptdet.errors import ConceptDetError, InputError
from conceptdet.model import TransformersBackend
from conceptdet.pipeline import DetectionPipeline, DetectionRequest
from conceptdet.types import Box, parse_boxes


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="ConceptSeg-R1 checkpoint directory")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument(
        "--attention",
        choices=("auto", "eager", "sdpa", "flash_attention_2"),
        default="flash_attention_2",
    )
    parser.add_argument("--input-size", type=int, default=600)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--box-color", default="red")
    parser.add_argument("--box-width", type=int, default=2)
    parser.add_argument(
        "--reference-box-width",
        type=int,
        default=2,
        help="Reference prompt bbox width; keep at 2 for ConceptSeg compatibility",
    )
    parser.add_argument(
        "--output-layout",
        choices=("triptych", "annotated"),
        default="triptych",
        help="Save Reference|Target|Detection or only the original-size detection",
    )


def _add_reference_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--reference", required=True, help="Reference image")
    parser.add_argument(
        "--reference-box",
        required=True,
        action="append",
        help="Original-image XYXY box. Repeat the option or separate boxes with ';'.",
    )
    parser.add_argument(
        "--reference-crop",
        choices=("full", "crop"),
        default="full",
        help="Use the full reference image or a bbox-centered crop",
    )
    parser.add_argument("--reference-context-scale", type=float, default=4.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="conceptdet",
        description="Reference-guided bbox detection without SAM3 segmentation",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    detect = subparsers.add_parser("detect", help="Run one detection request")
    _add_model_arguments(detect)
    _add_reference_arguments(detect)
    detect.add_argument("--target", required=True, help="Target image")
    detect.add_argument("--query", required=True, help="Concept description")
    detect.add_argument("--output", required=True, help="Annotated target image")
    detect.add_argument("--json-output", help="Structured result; defaults beside --output")
    detect.add_argument("--print-raw", action="store_true", help="Print raw model completion")

    batch = subparsers.add_parser("batch", help="Run tasks from a JSONL manifest")
    _add_model_arguments(batch)
    batch.add_argument("--manifest", required=True, help="One JSON object per line")
    batch.add_argument("--output-dir", required=True)
    batch.add_argument("--overwrite", action="store_true")
    return parser


def _pipeline(args: argparse.Namespace) -> DetectionPipeline:
    backend = TransformersBackend.load(
        args.model,
        device=args.device,
        dtype=args.dtype,
        attention=args.attention,
    )
    return DetectionPipeline(
        backend,
        input_size=args.input_size,
        max_new_tokens=args.max_new_tokens,
        annotation_color=args.box_color,
        annotation_width=args.box_width,
        reference_box_width=args.reference_box_width,
        output_layout=args.output_layout,
    )


def _boxes_from_cli(values: list[str]) -> tuple[Box, ...]:
    boxes: list[Box] = []
    for value in values:
        boxes.extend(parse_boxes(value))
    return tuple(boxes)


def _run_detect(args: argparse.Namespace) -> int:
    pipeline = _pipeline(args)
    request = DetectionRequest(
        reference_path=Path(args.reference),
        reference_boxes=_boxes_from_cli(args.reference_box),
        target_path=Path(args.target),
        query=args.query,
        reference_crop_mode=args.reference_crop,
        reference_crop_context_scale=args.reference_context_scale,
    )
    output_path = Path(args.output)
    json_path = Path(args.json_output) if args.json_output else None
    result = pipeline.run(request, output_path=output_path, json_path=json_path)
    print(json.dumps(result.to_dict()["detections"], ensure_ascii=False))
    print(f"Output image: {output_path.expanduser().resolve()}")
    print(f"JSON result: {(json_path or output_path.with_suffix('.json')).expanduser().resolve()}")
    if args.print_raw:
        print(result.raw_completion)
    return 0


def _read_manifest(path: Path) -> list[tuple[int, dict[str, Any]]]:
    if not path.is_file():
        raise InputError(f"Manifest does not exist: {path}")
    records: list[tuple[int, dict[str, Any]]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise InputError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
        if not isinstance(record, dict):
            raise InputError(f"Manifest record at {path}:{line_number} must be an object")
        records.append((line_number, record))
    if not records:
        raise InputError(f"Manifest contains no tasks: {path}")
    return records


def _resolve_manifest_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else base_dir / path


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned or "task"


def _manifest_request(record: dict[str, Any], base_dir: Path) -> DetectionRequest:
    required = {"reference", "reference_boxes", "target", "query"}
    missing = sorted(required - record.keys())
    if missing:
        raise InputError(f"Manifest task is missing fields: {', '.join(missing)}")
    return DetectionRequest(
        reference_path=_resolve_manifest_path(str(record["reference"]), base_dir),
        reference_boxes=parse_boxes(record["reference_boxes"]),
        target_path=_resolve_manifest_path(str(record["target"]), base_dir),
        query=str(record["query"]),
        reference_crop_mode=str(record.get("reference_crop", "full")),
        reference_crop_context_scale=float(record.get("reference_context_scale", 4.0)),
    )


def _run_batch(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).expanduser().resolve()
    records = _read_manifest(manifest_path)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = _pipeline(args)
    summary_path = output_dir / "results.jsonl"
    failures = 0
    summaries: list[dict[str, object]] = []

    for index, (line_number, record) in enumerate(records):
        task_id = str(record.get("id", f"task-{index:06d}"))
        filename = str(record.get("output", f"{_safe_name(task_id)}.png"))
        output_path = Path(filename)
        if not output_path.is_absolute():
            output_path = output_dir / output_path
        json_path = output_path.with_suffix(".json")
        if output_path.exists() and not args.overwrite:
            item = {"id": task_id, "status": "skipped", "output": str(output_path)}
            summaries.append(item)
            print(f"SKIP {task_id}: {output_path}")
            continue
        try:
            request = _manifest_request(record, manifest_path.parent)
            result = pipeline.run(request, output_path=output_path, json_path=json_path)
            item = {
                "id": task_id,
                "status": "ok",
                "output": str(output_path),
                "result": str(json_path),
                "detections": result.to_dict()["detections"],
            }
            print(f"OK   {task_id}: {output_path}")
        except (ConceptDetError, OSError, ValueError) as exc:
            failures += 1
            item = {
                "id": task_id,
                "status": "failed",
                "manifest_line": line_number,
                "error": str(exc),
            }
            print(f"FAIL {task_id}: {exc}", file=sys.stderr)
        summaries.append(item)

    summary_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in summaries),
        encoding="utf-8",
    )
    print(f"Batch summary: {summary_path}")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _run_detect(args) if args.command == "detect" else _run_batch(args)
    except (ConceptDetError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
