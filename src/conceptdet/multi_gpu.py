from __future__ import annotations

import concurrent.futures
import multiprocessing
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from conceptdet.errors import InputError
from conceptdet.model import TransformersBackend
from conceptdet.pipeline import DetectionPipeline, DetectionRequest


@dataclass(frozen=True)
class RuntimeConfig:
    model_path: Path
    dtype: str = "bfloat16"
    attention: str = "flash_attention_2"
    input_size: int = 600
    max_new_tokens: int = 768
    box_color: str = "red"
    box_width: int = 2
    reference_box_width: int = 2
    output_layout: str = "triptych"


@dataclass(frozen=True)
class TaskSpec:
    index: int
    request: DetectionRequest
    output_path: Path
    json_path: Path | None = None


@dataclass(frozen=True)
class TaskOutcome:
    index: int
    gpu_id: int
    status: str
    output_path: str
    detections: list[dict[str, object]] | None = None
    error: str | None = None
    failure_stage: str | None = None


def validate_gpu_ids(gpu_ids: Sequence[int], device_count: int) -> tuple[int, ...]:
    if not gpu_ids:
        raise InputError("gpu_ids 不能为空，例如 [0] 或 [0, 1, 2, 3]")
    normalized = tuple(int(gpu_id) for gpu_id in gpu_ids)
    if len(set(normalized)) != len(normalized):
        raise InputError(f"gpu_ids 不能包含重复编号: {list(normalized)}")
    invalid = [gpu_id for gpu_id in normalized if gpu_id < 0 or gpu_id >= device_count]
    if invalid:
        raise InputError(
            f"无效 GPU 编号 {invalid}；当前环境检测到 {device_count} 张 CUDA GPU"
        )
    return normalized


def select_gpus_by_free_memory(
    gpu_ids: Sequence[int],
    memory_gb: Mapping[int, tuple[float, float]],
    min_free_memory_gb: float,
) -> tuple[int, ...]:
    if min_free_memory_gb < 0:
        raise InputError("min_free_memory_gb 不能小于 0")
    return tuple(
        gpu_id
        for gpu_id in gpu_ids
        if memory_gb[gpu_id][0] >= min_free_memory_gb
    )


def split_tasks(
    tasks: Sequence[TaskSpec], gpu_ids: Sequence[int]
) -> list[tuple[int, list[TaskSpec]]]:
    if not tasks:
        return []
    active_gpu_ids = tuple(gpu_ids[: min(len(gpu_ids), len(tasks))])
    chunks = [[] for _ in active_gpu_ids]
    for index, task in enumerate(tasks):
        chunks[index % len(chunks)].append(task)
    return list(zip(active_gpu_ids, chunks))


def _failed_outcomes(
    tasks: Sequence[TaskSpec],
    gpu_id: int,
    error: str,
    failure_stage: str,
) -> list[TaskOutcome]:
    return [
        TaskOutcome(
            index=task.index,
            gpu_id=gpu_id,
            status="failed",
            output_path=str(task.output_path),
            error=error,
            failure_stage=failure_stage,
        )
        for task in tasks
    ]


def _run_worker(
    gpu_id: int, tasks: list[TaskSpec], runtime: RuntimeConfig
) -> list[TaskOutcome]:
    import torch

    device = f"cuda:{gpu_id}"
    try:
        torch.cuda.set_device(gpu_id)
        print(f"[GPU {gpu_id}] loading model for {len(tasks)} task(s)", flush=True)
        backend = TransformersBackend.load(
            runtime.model_path,
            device=device,
            dtype=runtime.dtype,
            attention=runtime.attention,
        )
        pipeline = DetectionPipeline(
            backend,
            input_size=runtime.input_size,
            max_new_tokens=runtime.max_new_tokens,
            annotation_color=runtime.box_color,
            annotation_width=runtime.box_width,
            reference_box_width=runtime.reference_box_width,
            output_layout=runtime.output_layout,
        )
    except Exception as exc:
        message = f"model load failed on {device}: {type(exc).__name__}: {exc}"
        print(f"[GPU {gpu_id}] {message}", flush=True)
        return _failed_outcomes(tasks, gpu_id, message, "model_load")

    outcomes: list[TaskOutcome] = []
    for position, task in enumerate(tasks, start=1):
        try:
            result = pipeline.run(
                task.request,
                output_path=task.output_path,
                json_path=task.json_path,
            )
            outcomes.append(
                TaskOutcome(
                    index=task.index,
                    gpu_id=gpu_id,
                    status="ok",
                    output_path=str(task.output_path),
                    detections=result.to_dict()["detections"],
                )
            )
            print(
                f"[GPU {gpu_id}] [{position}/{len(tasks)}] OK: {task.output_path}",
                flush=True,
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            outcomes.append(
                TaskOutcome(
                    index=task.index,
                    gpu_id=gpu_id,
                    status="failed",
                    output_path=str(task.output_path),
                    error=message,
                    failure_stage="inference",
                )
            )
            print(
                f"[GPU {gpu_id}] [{position}/{len(tasks)}] FAILED: {message}",
                flush=True,
            )
    return outcomes


def _execute_assignments(
    assignments: list[tuple[int, list[TaskSpec]]], runtime: RuntimeConfig
) -> list[TaskOutcome]:
    assignment_text = ", ".join(
        f"cuda:{gpu_id}={len(chunk)} task(s)" for gpu_id, chunk in assignments
    )
    print(f"Multi-GPU assignment: {assignment_text}", flush=True)

    if len(assignments) == 1:
        gpu_id, chunk = assignments[0]
        return sorted(_run_worker(gpu_id, chunk, runtime), key=lambda item: item.index)

    outcomes: list[TaskOutcome] = []
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=len(assignments), mp_context=context
    ) as executor:
        future_assignments = {
            executor.submit(_run_worker, gpu_id, chunk, runtime): (gpu_id, chunk)
            for gpu_id, chunk in assignments
        }
        for future in concurrent.futures.as_completed(future_assignments):
            gpu_id, chunk = future_assignments[future]
            try:
                outcomes.extend(future.result())
            except Exception as exc:
                message = f"worker crashed on cuda:{gpu_id}: {type(exc).__name__}: {exc}"
                outcomes.extend(_failed_outcomes(chunk, gpu_id, message, "worker"))

    return sorted(outcomes, key=lambda item: item.index)


def plan_model_load_retries(
    tasks: Sequence[TaskSpec],
    outcomes: Sequence[TaskOutcome],
    assigned_gpu_ids: Sequence[int],
) -> list[tuple[int, list[TaskSpec]]]:
    failed_indices = {
        outcome.index for outcome in outcomes if outcome.failure_stage == "model_load"
    }
    failed_gpu_ids = {
        outcome.gpu_id for outcome in outcomes if outcome.failure_stage == "model_load"
    }
    survivor_gpu_ids = [
        gpu_id for gpu_id in assigned_gpu_ids if gpu_id not in failed_gpu_ids
    ]
    retry_tasks = [task for task in tasks if task.index in failed_indices]
    return split_tasks(retry_tasks, survivor_gpu_ids) if survivor_gpu_ids else []


def run_on_gpus(
    tasks: Sequence[TaskSpec],
    runtime: RuntimeConfig,
    gpu_ids: Sequence[int],
    *,
    min_free_memory_gb: float = 0.0,
    retry_model_load_failures: bool = True,
) -> list[TaskOutcome]:
    if not tasks:
        return []

    import torch

    if not torch.cuda.is_available():
        raise InputError("CUDA 不可用，多卡推理需要 CUDA 环境")
    validated_gpu_ids = validate_gpu_ids(gpu_ids, torch.cuda.device_count())
    gib = 1024**3
    memory_gb = {
        gpu_id: tuple(value / gib for value in torch.cuda.mem_get_info(gpu_id))
        for gpu_id in validated_gpu_ids
    }
    usable_gpu_ids = select_gpus_by_free_memory(
        validated_gpu_ids, memory_gb, min_free_memory_gb
    )
    for gpu_id in validated_gpu_ids:
        free_gb, total_gb = memory_gb[gpu_id]
        decision = "USE" if gpu_id in usable_gpu_ids else "SKIP busy"
        print(
            f"GPU preflight: cuda:{gpu_id} free={free_gb:.2f}/{total_gb:.2f} GiB "
            f"required>={min_free_memory_gb:.2f} GiB -> {decision}",
            flush=True,
        )
    if not usable_gpu_ids:
        raise InputError(
            "配置的 GPU 均没有足够空闲显存；"
            f"当前阈值 min_free_memory_gb={min_free_memory_gb:.2f}"
        )

    assignments = split_tasks(tasks, usable_gpu_ids)
    outcomes = _execute_assignments(assignments, runtime)
    if retry_model_load_failures:
        assigned_gpu_ids = [gpu_id for gpu_id, _ in assignments]
        retry_assignments = plan_model_load_retries(tasks, outcomes, assigned_gpu_ids)
        if retry_assignments:
            retry_count = sum(len(chunk) for _, chunk in retry_assignments)
            retry_gpu_ids = [gpu_id for gpu_id, _ in retry_assignments]
            print(
                f"Retrying {retry_count} task(s) after model-load failure on "
                f"surviving GPUs {retry_gpu_ids}",
                flush=True,
            )
            retried_outcomes = _execute_assignments(retry_assignments, runtime)
            retried_by_index = {outcome.index: outcome for outcome in retried_outcomes}
            outcomes = [retried_by_index.get(outcome.index, outcome) for outcome in outcomes]

    return sorted(outcomes, key=lambda item: item.index)
