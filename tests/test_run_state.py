import random
from pathlib import Path

import pytest
import torch

from conceptdet.errors import TrainingError
from conceptdet.run_state import (
    ProcessContext,
    RunIdentity,
    capture_rng_state,
    load_checkpoint_metadata,
    resolve_resume,
    restore_rng_state,
    write_checkpoint_metadata,
)


def _identity(**changes: object) -> RunIdentity:
    values = {
        "stage": "sft",
        "config_hash": "config",
        "dataset_fingerprint": "dataset",
        "contract_fingerprint": "contract",
        "parent_artifact_fingerprint": None,
        "semantics_fingerprint": "semantics",
        "world_size": 2,
        **changes,
    }
    return RunIdentity(**values)  # type: ignore[arg-type]


def test_checkpoint_identity_rejects_every_semantic_and_topology_drift(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint-00000003"
    checkpoint.mkdir()
    write_checkpoint_metadata(checkpoint, _identity(), {"optimizer_step": 3})
    (checkpoint / "complete").write_text("complete\n", encoding="utf-8")
    assert load_checkpoint_metadata(checkpoint, _identity()) == {"optimizer_step": 3}
    for field, value in (
        ("config_hash", "changed"),
        ("dataset_fingerprint", "changed"),
        ("contract_fingerprint", "changed"),
        ("parent_artifact_fingerprint", "parent"),
        ("semantics_fingerprint", "changed"),
        ("world_size", 1),
    ):
        with pytest.raises(TrainingError, match=field):
            load_checkpoint_metadata(checkpoint, _identity(**{field: value}))


def test_resume_ignores_partial_checkpoint_and_explicit_partial_fails(
    tmp_path: Path,
) -> None:
    partial = tmp_path / "checkpoint-00000002"
    complete = tmp_path / "checkpoint-00000001"
    partial.mkdir()
    complete.mkdir()
    (complete / "complete").write_text("complete\n", encoding="utf-8")
    assert resolve_resume(tmp_path, "auto", stage="sft") == complete
    with pytest.raises(TrainingError, match="incomplete"):
        resolve_resume(tmp_path, partial, stage="sft")


def test_rng_state_round_trip_restores_python_and_torch() -> None:
    random.seed(7)
    torch.manual_seed(7)
    state = capture_rng_state(torch)
    expected = (random.random(), torch.rand(3))
    random.seed(99)
    torch.manual_seed(99)
    restore_rng_state(torch, state)
    actual = (random.random(), torch.rand(3))
    assert actual[0] == expected[0]
    assert torch.equal(actual[1], expected[1])


def test_process_context_maps_distributed_rank_to_local_cuda() -> None:
    context = ProcessContext(rank=1, local_rank=3, world_size=4)
    assert context.cuda_device("cuda:0") == "cuda:3"
    assert context.is_main is False
