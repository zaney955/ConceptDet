from pathlib import Path

import pytest
import torch
from PIL import Image

from conceptdet.errors import DatasetError, TrainingError
from conceptdet.training import (
    SFTBatchBuilder,
    _compatible_sft_schedule,
    _ordered_records,
    _resolve_resume,
)


class _Dataset:
    def resolve_image(self, payload: dict[str, object]) -> Path:
        return Path(str(payload["path"]))


class _Processor:
    def apply_chat_template(
        self, messages: list[dict[str, object]], *, add_generation_prompt: bool, **_: object
    ) -> dict[str, torch.Tensor]:
        if add_generation_prompt:
            return {"input_ids": torch.tensor([[1, 2, 3]])}
        return {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "image_grid_thw": torch.tensor([[1, 16, 16], [1, 16, 16]]),
        }


def test_sft_batch_masks_prompt_and_keeps_assistant_labels(tmp_path: Path) -> None:
    reference = tmp_path / "reference.png"
    target = tmp_path / "target.png"
    Image.new("RGB", (256, 256), "white").save(reference)
    Image.new("RGB", (256, 256), "gray").save(target)
    record = {
        "id": "record-1",
        "positive": True,
        "query": "the same Visual Concept as the boxed Reference Instances",
        "reference": {"path": str(reference), "boxes_xyxy": [[10, 10, 40, 40]]},
        "target": {"path": str(target)},
        "detection_set": [{"bbox_2d": [100, 100, 300, 300]}],
    }
    batch, provenance = SFTBatchBuilder(_Dataset(), _Processor()).build(record)  # type: ignore[arg-type]
    assert batch["labels"].tolist() == [[-100, -100, -100, 4, 5]]
    assert provenance.prompt_tokens == 3
    assert provenance.completion_tokens == 2
    assert provenance.image_grids == ((1, 16, 16), (1, 16, 16))


def test_sft_order_reaches_negative_records_in_first_cycle() -> None:
    records = [
        *({"id": f"p-{index}", "positive": True} for index in range(11)),
        *({"id": f"n-{index}", "positive": False} for index in range(2)),
    ]
    ordered = _ordered_records(records, 17, 0)
    assert sum(not bool(row["positive"]) for row in ordered[:6]) == 1
    assert {str(row["id"]) for row in ordered} == {str(row["id"]) for row in records}


def test_sft_schedule_excludes_only_contract_overlength_records() -> None:
    class _Builder:
        def build(self, record: dict[str, object]) -> None:
            if record["id"] == "long":
                raise DatasetError("Record long has 1537 tokens; truncation is forbidden")

    records = [
        {"id": "ok-1", "positive": True},
        {"id": "long", "positive": True},
        {"id": "ok-2", "positive": False},
    ]
    compatible, excluded = _compatible_sft_schedule(  # type: ignore[arg-type]
        records, _Builder(), required_micro_steps=None
    )
    assert [record["id"] for record in compatible] == ["ok-1", "ok-2"]
    assert excluded == ["long"]


def test_sft_smoke_schedule_frontloads_positive_and_negative() -> None:
    class _Builder:
        def build(self, record: dict[str, object]) -> None:
            return None

    records = [
        {"id": "p-1", "positive": True},
        {"id": "p-2", "positive": True},
        {"id": "p-3", "positive": True},
        {"id": "n-1", "positive": False},
    ]
    compatible, excluded = _compatible_sft_schedule(  # type: ignore[arg-type]
        records, _Builder(), required_micro_steps=3
    )
    assert [record["id"] for record in compatible] == ["p-1", "n-1", "p-2"]
    assert excluded == []


def test_resume_is_explicit_and_auto_selects_latest_complete_checkpoint(
    tmp_path: Path,
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    assert _resolve_resume(work, "none") is None
    for step in (1, 2):
        checkpoint = work / f"checkpoint-{step:08d}"
        checkpoint.mkdir()
        (checkpoint / "complete").write_text("complete\n", encoding="utf-8")
    assert _resolve_resume(work, "auto") == work / "checkpoint-00000002"
    assert _resolve_resume(work, work / "checkpoint-00000001") == (
        work / "checkpoint-00000001"
    )
    with pytest.raises(TrainingError, match="not empty"):
        _resolve_resume(work, "none")
