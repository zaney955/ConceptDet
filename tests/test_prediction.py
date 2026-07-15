import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from conceptdet.adapter import FakeAdapter
from conceptdet.config import DatasetPredictionConfig, RuntimeConfig
from conceptdet.prediction import generate_dataset_predictions
from conceptdet.types import Box


class _Dataset:
    fingerprint = "dataset-fingerprint"

    def __init__(self, image: Path) -> None:
        self.image = image

    def iter_records(self, split: str):
        assert split == "test"
        yield {"id": "record-b"}
        yield {"id": "record-a"}

    def detection_request(self, record: dict[str, str]) -> SimpleNamespace:
        return SimpleNamespace(
            reference_image=self.image,
            reference_boxes=(Box(1, 1, 8, 8),),
            target_image=self.image,
            query=f"query {record['id']}",
        )


def test_prediction_publishes_complete_ordered_jsonl(
    tmp_path: Path, monkeypatch
) -> None:
    image = tmp_path / "image.png"
    Image.new("RGB", (16, 16), "white").save(image)
    dataset = _Dataset(image)
    monkeypatch.setattr(
        "conceptdet.prediction.DatasetArtifact.load", lambda _: dataset
    )
    monkeypatch.setattr(
        "conceptdet.prediction.AdapterArtifact.load",
        lambda _: SimpleNamespace(fingerprint="adapter-fingerprint"),
    )
    output = tmp_path / "predictions.jsonl"
    config = DatasetPredictionConfig(
        1,
        "predict.dataset",
        tmp_path / "dataset",
        tmp_path / "artifact",
        output,
        "test",
        RuntimeConfig(device="cpu", dtype="float32", max_new_tokens=64),
        tmp_path / "predict.yaml",
        "config-hash",
    )

    result = generate_dataset_predictions(config, adapter=FakeAdapter("[]"))

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows == [
        {"id": "record-a", "raw_completion": "[]"},
        {"id": "record-b", "raw_completion": "[]"},
    ]
    assert result.records == 2
    assert len(result.content_sha256) == 64
