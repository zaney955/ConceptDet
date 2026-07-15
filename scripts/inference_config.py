#!/usr/bin/env python3
"""Edit the configuration below, then run: bash scripts/run_inference.sh"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from conceptdet.batch import discover_images, plan_output_paths  # noqa: E402
from conceptdet.multi_gpu import (  # noqa: E402
    RuntimeConfig,
    TaskOutcome,
    TaskSpec,
    run_on_gpus,
)
from conceptdet.pipeline import DetectionRequest  # noqa: E402
from conceptdet.types import parse_boxes  # noqa: E402


# ========================
# 只需要修改以下配置
# ========================
CONFIG: dict[str, Any] = {
    # "tasks": 手工逐项填写 TASKS。
    # "batch": 扫描 BATCH_CONFIG.input_paths 中的图片。
    "mode": "batch",  # tasks / batch

    # 模型与设备。相对路径统一相对于本仓库根目录。
    "model_path": "../ConceptSeg-R1/ConceptSeg-R1-7B",
    # 一张 GPU 写 [0]；多张 GPU 写 [0, 1, 2, 3]。
    "gpu_ids": [3, 4, 5, 6, 7],
    # 启动前跳过空闲显存低于该值的 GPU，并把任务重新分配给其余 GPU。
    "min_free_memory_gb": 24.0,
    # 显存状态在预检后变化、导致模型加载失败时，在存活 GPU 上重试一次。
    "retry_model_load_failures": True,
    "dtype": "bfloat16",  # auto / float32 / float16 / bfloat16
    # 为对齐 ConceptSeg bbox_only，请保持 flash_attention_2。
    "attention": "flash_attention_2",  # auto / eager / sdpa / flash_attention_2

    # checkpoint 使用的提示图边长，一般不要修改。
    "input_size": 600,
    "max_new_tokens": 768,

    # 输出框样式。
    "box_color": "red",
    "box_width": 2,
    # 送入模型的参考图红框宽度；该 checkpoint 对此敏感，兼容值为 2。
    "reference_box_width": 2,
    # triptych: 参考图 | 目标图 | bbox 结果图；annotated: 仅保存原图结果。
    "output_layout": "triptych",  # triptych / annotated
}


# mode="tasks" 时使用。可填写一个或多个任务。
TASKS: list[dict[str, Any]] = [
    {
        "reference_path": "../ConceptSeg-R1/example_images/cod_ref.png",
        "reference_boxes": "325,64,570,270;146,420,184,491;313,434,516,561",
        "target_path": "../ConceptSeg-R1/example_images/infer.jpg",
        "query": "camouflaged object",
        "output_path": "outputs/config_demo.png",
        # None 表示自动保存为与 output_path 同名的 .json。
        "json_output_path": None,
        "reference_crop_mode": "full",  # full / crop
        "reference_crop_context_scale": 4.0,
    },
]


# mode="batch" 时使用。所有目标图片共享参考图、参考框和 query。
BATCH_CONFIG: dict[str, Any] = {
    # 每一项可以是图片或目录；目录会自动发现支持的图片。
    "input_paths": [
        "inputs/non_ref/GX2",
        # "/data/target_images",
    ],
    "recursive": False,

    # 所有目标图片共用的参考信息。
    "reference_path": (
        "./inputs/ref/GX/17286d22__3e852fa4-5dcc-40c6-a00e-addb89753b63.jpg"
    ),
    "reference_boxes": "1165,2911,1354,3230;4064,3087,4208,3375",
    "query": "the same bolt as the red-boxed example in the reference image",
    "reference_crop_mode": "full",  # full / crop
    "reference_crop_context_scale": 4.0,

    # 批量输出。递归输入会保留相对目录结构。
    "output_dir": "outputs/batch",
    "skip_existing": True,
    "log_path": "outputs/batch/results.jsonl",
}
# ========================
# 配置结束，下面通常无需修改
# ========================


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _validate_shared_config(config: dict[str, Any]) -> None:
    if int(config["input_size"]) < 1:
        raise ValueError("CONFIG['input_size'] 必须大于 0")
    if config.get("output_layout", "triptych") not in {"triptych", "annotated"}:
        raise ValueError("CONFIG['output_layout'] 必须是 'triptych' 或 'annotated'")
    if int(config.get("box_width", 2)) < 1:
        raise ValueError("CONFIG['box_width'] 必须大于 0")
    if int(config.get("reference_box_width", 2)) < 1:
        raise ValueError("CONFIG['reference_box_width'] 必须大于 0")
    gpu_ids = config.get("gpu_ids")
    if not isinstance(gpu_ids, list) or not gpu_ids:
        raise ValueError("CONFIG['gpu_ids'] 必须是非空列表，例如 [0] 或 [0, 1, 2, 3]")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("CONFIG['gpu_ids'] 不能包含重复编号")
    if float(config.get("min_free_memory_gb", 0)) < 0:
        raise ValueError("CONFIG['min_free_memory_gb'] 不能小于 0")


def _validate_explicit_tasks(tasks: list[dict[str, Any]]) -> None:
    if not tasks:
        raise ValueError("mode='tasks' 时 TASKS 不能为空")
    required = {"reference_path", "reference_boxes", "target_path", "query", "output_path"}
    for index, task in enumerate(tasks):
        missing = sorted(required - task.keys())
        if missing:
            raise ValueError(f"TASKS[{index}] 缺少字段: {', '.join(missing)}")
        if not str(task["query"]).strip():
            raise ValueError(f"TASKS[{index}].query 不能为空")


def _validate_batch_config(batch_config: dict[str, Any]) -> None:
    required = {
        "input_paths",
        "reference_path",
        "reference_boxes",
        "query",
        "output_dir",
    }
    missing = sorted(required - batch_config.keys())
    if missing:
        raise ValueError(f"BATCH_CONFIG 缺少字段: {', '.join(missing)}")
    if not batch_config["input_paths"]:
        raise ValueError("BATCH_CONFIG['input_paths'] 不能为空")
    if not str(batch_config["query"]).strip():
        raise ValueError("BATCH_CONFIG['query'] 不能为空")


def validate_config(
    config: dict[str, Any],
    tasks: list[dict[str, Any]],
    batch_config: dict[str, Any] | None = None,
) -> None:
    _validate_shared_config(config)
    mode = str(config.get("mode", "tasks"))
    if mode == "tasks":
        _validate_explicit_tasks(tasks)
    elif mode == "batch":
        _validate_batch_config(batch_config or {})
    else:
        raise ValueError("CONFIG['mode'] 必须是 'tasks' 或 'batch'")


def build_request(task: dict[str, Any]) -> DetectionRequest:
    return DetectionRequest(
        reference_path=resolve_path(task["reference_path"]),
        reference_boxes=parse_boxes(task["reference_boxes"]),
        target_path=resolve_path(task["target_path"]),
        query=str(task["query"]),
        reference_crop_mode=str(task.get("reference_crop_mode", "full")),
        reference_crop_context_scale=float(task.get("reference_crop_context_scale", 4.0)),
    )


def build_task_specs(tasks: list[dict[str, Any]]) -> list[TaskSpec]:
    specs: list[TaskSpec] = []
    output_paths: set[Path] = set()
    for index, task in enumerate(tasks):
        output_path = resolve_path(task["output_path"])
        if output_path in output_paths:
            raise ValueError(f"多个任务不能写入同一个输出文件: {output_path}")
        output_paths.add(output_path)
        raw_json_path = task.get("json_output_path")
        specs.append(
            TaskSpec(
                index=index,
                request=build_request(task),
                output_path=output_path,
                json_path=resolve_path(raw_json_path) if raw_json_path else None,
            )
        )
    return specs


def build_batch_task_specs(batch_config: dict[str, Any]) -> list[TaskSpec]:
    reference_path = resolve_path(batch_config["reference_path"])
    output_directory = resolve_path(batch_config["output_dir"])
    input_paths = [resolve_path(path) for path in batch_config["input_paths"]]
    images = discover_images(
        input_paths,
        recursive=bool(batch_config.get("recursive", False)),
        exclude_paths=(reference_path,),
        exclude_directories=(output_directory,),
    )
    if not images:
        raise ValueError("BATCH_CONFIG.input_paths 中没有发现可推理图片")
    output_paths = plan_output_paths(images, output_directory)
    reference_boxes = parse_boxes(batch_config["reference_boxes"])

    return [
        TaskSpec(
            index=index,
            request=DetectionRequest(
                reference_path=reference_path,
                reference_boxes=reference_boxes,
                target_path=image.path,
                query=str(batch_config["query"]),
                reference_crop_mode=str(batch_config.get("reference_crop_mode", "full")),
                reference_crop_context_scale=float(
                    batch_config.get("reference_crop_context_scale", 4.0)
                ),
            ),
            output_path=output_path,
        )
        for index, (image, output_path) in enumerate(zip(images, output_paths))
    ]


def _write_batch_log(
    log_path: Path,
    specs: list[TaskSpec],
    outcomes: list[TaskOutcome],
    skipped_indices: set[int],
) -> None:
    outcomes_by_index = {outcome.index: outcome for outcome in outcomes}
    rows: list[dict[str, object]] = []
    for spec in specs:
        if spec.index in skipped_indices:
            rows.append(
                {
                    "index": spec.index,
                    "status": "skipped",
                    "gpu_id": None,
                    "target_path": str(spec.request.target_path),
                    "output_path": str(spec.output_path),
                }
            )
            continue
        outcome = outcomes_by_index[spec.index]
        rows.append(
            {
                "index": spec.index,
                "status": outcome.status,
                "gpu_id": outcome.gpu_id,
                "target_path": str(spec.request.target_path),
                "output_path": outcome.output_path,
                "detections": outcome.detections,
                "error": outcome.error,
                "failure_stage": outcome.failure_stage,
            }
        )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    validate_config(CONFIG, TASKS, BATCH_CONFIG)
    mode = str(CONFIG.get("mode", "tasks"))
    model_path = resolve_path(CONFIG["model_path"])
    gpu_ids = [int(gpu_id) for gpu_id in CONFIG["gpu_ids"]]
    runtime = RuntimeConfig(
        model_path=model_path,
        dtype=str(CONFIG.get("dtype", "auto")),
        attention=str(CONFIG.get("attention", "auto")),
        input_size=int(CONFIG.get("input_size", 600)),
        max_new_tokens=int(CONFIG.get("max_new_tokens", 768)),
        box_color=str(CONFIG.get("box_color", "red")),
        box_width=int(CONFIG.get("box_width", 2)),
        reference_box_width=int(CONFIG.get("reference_box_width", 2)),
        output_layout=str(CONFIG.get("output_layout", "triptych")),
    )
    task_specs = (
        build_task_specs(TASKS)
        if mode == "tasks"
        else build_batch_task_specs(BATCH_CONFIG)
    )

    skipped_indices: set[int] = set()
    active_specs = task_specs
    if mode == "batch" and bool(BATCH_CONFIG.get("skip_existing", True)):
        skipped_indices = {
            spec.index for spec in task_specs if spec.output_path.is_file()
        }
        active_specs = [spec for spec in task_specs if spec.index not in skipped_indices]

    print(f"Mode: {mode}")
    print(f"Model: {model_path}")
    print(f"GPU IDs: {gpu_ids}")
    print(
        f"Tasks: total={len(task_specs)}, active={len(active_specs)}, "
        f"skipped={len(skipped_indices)}"
    )
    outcomes = (
        run_on_gpus(
            active_specs,
            runtime,
            gpu_ids,
            min_free_memory_gb=float(CONFIG.get("min_free_memory_gb", 0)),
            retry_model_load_failures=bool(
                CONFIG.get("retry_model_load_failures", True)
            ),
        )
        if active_specs
        else []
    )

    if mode == "batch":
        log_path = resolve_path(BATCH_CONFIG.get("log_path", "outputs/batch/results.jsonl"))
        _write_batch_log(log_path, task_specs, outcomes, skipped_indices)
        print(f"Batch log: {log_path}")

    failures = [outcome for outcome in outcomes if outcome.status != "ok"]
    succeeded = sum(outcome.status == "ok" for outcome in outcomes)
    print(
        f"Completed: ok={succeeded}, skipped={len(skipped_indices)}, failed={len(failures)}"
    )
    for outcome in failures:
        print(
            f"Task {outcome.index} failed on cuda:{outcome.gpu_id}: {outcome.error}",
            file=sys.stderr,
        )
    if failures:
        raise SystemExit(f"{len(failures)} task(s) failed")


if __name__ == "__main__":
    main()
