import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from conceptdet.errors import InputError
import conceptdet.multi_gpu as multi_gpu
from conceptdet.multi_gpu import (
    RuntimeConfig,
    TaskOutcome,
    TaskSpec,
    plan_model_load_retries,
    select_gpus_by_free_memory,
    split_tasks,
    validate_gpu_ids,
)
from conceptdet.pipeline import DetectionRequest
from conceptdet.types import Box


def _task(index: int) -> TaskSpec:
    return TaskSpec(
        index=index,
        request=DetectionRequest(
            reference_path=Path("reference.jpg"),
            reference_boxes=(Box(1, 1, 10, 10),),
            target_path=Path(f"target-{index}.jpg"),
            query="bolt",
        ),
        output_path=Path(f"output-{index}.png"),
    )


def test_validate_gpu_ids() -> None:
    assert validate_gpu_ids([0, 2, 3], 4) == (0, 2, 3)
    with pytest.raises(InputError, match="重复"):
        validate_gpu_ids([0, 0], 4)
    with pytest.raises(InputError, match="无效"):
        validate_gpu_ids([0, 4], 4)
    with pytest.raises(InputError, match="不能为空"):
        validate_gpu_ids([], 4)


def test_split_tasks_round_robins_and_does_not_start_idle_gpu() -> None:
    assignments = split_tasks([_task(index) for index in range(5)], [0, 1, 2, 3])
    assert [(gpu_id, [task.index for task in tasks]) for gpu_id, tasks in assignments] == [
        (0, [0, 4]),
        (1, [1]),
        (2, [2]),
        (3, [3]),
    ]

    assignments = split_tasks([_task(0), _task(1)], [3, 4, 5])
    assert [gpu_id for gpu_id, _ in assignments] == [3, 4]


def test_select_gpus_by_free_memory_filters_busy_devices() -> None:
    memory = {
        3: (31.0, 48.0),
        4: (0.06, 48.0),
        5: (0.02, 48.0),
        6: (28.0, 48.0),
        7: (47.0, 48.0),
    }
    assert select_gpus_by_free_memory([3, 4, 5, 6, 7], memory, 24.0) == (3, 6, 7)
    with pytest.raises(InputError, match="不能小于"):
        select_gpus_by_free_memory([3], memory, -1)


def test_model_load_failures_are_redistributed_to_surviving_gpus() -> None:
    tasks = [_task(index) for index in range(4)]
    outcomes = [
        TaskOutcome(0, 3, "ok", "output-0.png"),
        TaskOutcome(
            1,
            4,
            "failed",
            "output-1.png",
            error="OOM",
            failure_stage="model_load",
        ),
        TaskOutcome(
            2,
            5,
            "failed",
            "output-2.png",
            error="OOM",
            failure_stage="model_load",
        ),
        TaskOutcome(3, 6, "ok", "output-3.png"),
    ]
    retry_assignments = plan_model_load_retries(tasks, outcomes, [3, 4, 5, 6])
    assert [
        (gpu_id, [task.index for task in assigned])
        for gpu_id, assigned in retry_assignments
    ] == [(3, [1]), (6, [2])]


def test_run_on_gpus_filters_busy_gpu_and_retries_load_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gib = 1024**3
    memory = {
        3: (30 * gib, 48 * gib),
        4: (1 * gib, 48 * gib),
        6: (30 * gib, 48 * gib),
    }
    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 7,
        mem_get_info=lambda gpu_id: memory[gpu_id],
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=fake_cuda))

    calls: list[list[tuple[int, list[int]]]] = []

    def fake_execute(assignments, runtime):  # noqa: ANN001, ANN202, ARG001
        calls.append(
            [
                (gpu_id, [task.index for task in assigned])
                for gpu_id, assigned in assignments
            ]
        )
        if len(calls) == 1:
            return [
                TaskOutcome(0, 3, "ok", "output-0.png"),
                TaskOutcome(
                    1,
                    6,
                    "failed",
                    "output-1.png",
                    error="race OOM",
                    failure_stage="model_load",
                ),
            ]
        return [TaskOutcome(1, 3, "ok", "output-1.png")]

    monkeypatch.setattr(multi_gpu, "_execute_assignments", fake_execute)
    outcomes = multi_gpu.run_on_gpus(
        [_task(0), _task(1)],
        RuntimeConfig(model_path=Path("model")),
        [3, 4, 6],
        min_free_memory_gb=24,
        retry_model_load_failures=True,
    )

    assert calls == [[(3, [0]), (6, [1])], [(3, [1])]]
    assert [(outcome.index, outcome.gpu_id, outcome.status) for outcome in outcomes] == [
        (0, 3, "ok"),
        (1, 3, "ok"),
    ]
