from pathlib import Path

import pytest
import torch
from PIL import Image

from conceptdet.errors import DatasetError
from conceptdet.evaluation import score_detection_reward
from conceptdet.grpo import (
    GRPOBatchBuilder,
    _frontload_compatible_smoke_records,
    _ordered_records,
)
from conceptdet.types import Box


class _Dataset:
    def resolve_image(self, payload: dict[str, object]) -> Path:
        return Path(str(payload["path"]))


class _Processor:
    def __init__(self, prompt_tokens: int = 100) -> None:
        self.prompt_tokens = prompt_tokens

    def apply_chat_template(self, *_: object, **__: object) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.ones((1, self.prompt_tokens), dtype=torch.long),
            "image_grid_thw": torch.tensor([[1, 16, 16], [1, 16, 16]]),
        }


def _record(tmp_path: Path, *, positive: bool = True) -> dict[str, object]:
    reference = tmp_path / "reference.png"
    target = tmp_path / "target.png"
    Image.new("RGB", (256, 256), "white").save(reference)
    Image.new("RGB", (256, 256), "gray").save(target)
    return {
        "id": "positive" if positive else "negative",
        "positive": positive,
        "query": "the same Visual Concept",
        "reference": {"path": str(reference), "boxes_xyxy": [[10, 10, 40, 40]]},
        "target": {"path": str(target)},
        "detection_set": (
            [{"bbox_2d": [100, 100, 300, 300]}] if positive else []
        ),
    }


def test_grpo_batch_uses_raw_prompt_ordered_images_and_normalized_truth(
    tmp_path: Path,
) -> None:
    prepared, provenance = GRPOBatchBuilder(  # type: ignore[arg-type]
        _Dataset(), _Processor()
    ).build(_record(tmp_path))
    assert prepared["prompt"][0]["role"] == "user"
    assert isinstance(prepared["prompt"][0]["content"], str)
    assert len(prepared["images"]) == 2
    assert prepared["ground_truth"] == [[100, 100, 300, 300]]
    assert prepared["positive"] is True
    assert provenance.prompt_tokens == 100
    assert provenance.image_grids == ((1, 16, 16), (1, 16, 16))

    with pytest.raises(DatasetError, match=r"prompt \+ 192"):
        GRPOBatchBuilder(  # type: ignore[arg-type]
            _Dataset(), _Processor(prompt_tokens=1345)
        ).build(_record(tmp_path))


def test_grpo_schedule_frontloads_positive_and_negative_groups() -> None:
    records = [
        *({"id": f"p-{index}", "positive": True} for index in range(5)),
        *({"id": f"n-{index}", "positive": False} for index in range(2)),
    ]
    ordered = _ordered_records(records, 23)
    assert [bool(record["positive"]) for record in ordered[:4]] == [
        True,
        False,
        True,
        False,
    ]
    assert {record["id"] for record in ordered} == {record["id"] for record in records}


def test_grpo_full_preflight_excludes_overlength_without_truncation(
    tmp_path: Path,
) -> None:
    positive = _record(tmp_path, positive=True)
    negative = _record(tmp_path, positive=False)
    overlength = {**positive, "id": "overlength", "query": "overlength"}

    class _VariableProcessor(_Processor):
        def apply_chat_template(
            self, messages: object, *_: object, **__: object
        ) -> dict[str, torch.Tensor]:
            tokens = 1345 if "overlength" in str(messages) else 100
            return {
                "input_ids": torch.ones((1, tokens), dtype=torch.long),
                "image_grid_thw": torch.tensor([[1, 16, 16], [1, 16, 16]]),
                "pixel_values": torch.zeros((2, 3, 4, 4)),
            }

    compatible, _, excluded = _frontload_compatible_smoke_records(
        [overlength, positive, negative],
        GRPOBatchBuilder(_Dataset(), _VariableProcessor()),  # type: ignore[arg-type]
        required_records=None,
    )
    assert [record["id"] for record in compatible] == ["positive", "negative"]
    assert excluded == ["overlength"]


def test_grpo_reward_reuses_strict_soft_set_f1_contract() -> None:
    target = (Box(100, 100, 300, 300),)
    perfect = score_detection_reward('[{"bbox_2d":[100,100,300,300]}]', target)
    assert perfect.format_valid is True
    assert perfect.soft_set_f1 == 1.0
    assert perfect.total_reward == 1.0

    duplicate = score_detection_reward(
        '[{"bbox_2d":[100,100,300,300]},{"bbox_2d":[100,100,300,300]}]',
        target,
    )
    assert duplicate.soft_set_f1 == pytest.approx(2 / 3)
    assert duplicate.total_reward == pytest.approx(0.7)

    assert score_detection_reward("[]", ()).total_reward == 1.0
    invalid = score_detection_reward("```json\n[]\n```", ())
    assert invalid.format_valid is False
    assert invalid.total_reward == 0.0
