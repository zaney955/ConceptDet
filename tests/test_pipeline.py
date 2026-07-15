import json
from pathlib import Path

import pytest
from PIL import Image

from conceptdet.errors import InputError, OutputFormatError
from conceptdet.pipeline import DetectionPipeline, DetectionRequest
from conceptdet.types import Box


class FakeBackend:
    def __init__(self, completion: str) -> None:
        self.completion = completion
        self.calls: list[tuple[object, object, int]] = []

    def generate(self, messages, images, *, max_new_tokens):  # noqa: ANN001, ANN201
        self.calls.append((messages, images, max_new_tokens))
        return self.completion


def _request(tmp_path: Path) -> DetectionRequest:
    reference_path = tmp_path / "reference.jpg"
    target_path = tmp_path / "target.jpg"
    Image.new("RGB", (200, 100), "white").save(reference_path)
    Image.new("RGB", (1000, 500), "white").save(target_path)
    return DetectionRequest(
        reference_path=reference_path,
        reference_boxes=(Box(20, 10, 80, 50),),
        target_path=target_path,
        query="matching bolt",
    )


def test_pipeline_maps_bbox_to_original_target_and_writes_outputs(tmp_path: Path) -> None:
    backend = FakeBackend(
        "<think>match</think><rule>hex head</rule>"
        "<bbox>[60,120,300,480]</bbox><answer>bolt</answer>"
    )
    pipeline = DetectionPipeline(backend, output_layout="annotated")
    output_path = tmp_path / "nested" / "result.png"
    result = pipeline.run(_request(tmp_path), output_path=output_path)

    detection = result.detections[0]
    assert detection.model_box == Box(60, 120, 300, 480)
    assert detection.target_box == Box(100, 100, 500, 400)
    assert output_path.is_file()
    assert output_path.with_suffix(".json").is_file()
    payload = json.loads(output_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["detections"][0]["bbox_xyxy"] == [100, 100, 500, 400]
    assert payload["output_layout"] == "annotated"
    assert Image.open(output_path).size == (1000, 500)
    assert Image.open(output_path).getpixel((100, 100))[0] > 200

    messages, images, max_new_tokens = backend.calls[0]
    assert len(messages[1]["content"]) == 3
    assert images[0].size == (600, 600)
    assert images[1].size == (600, 600)
    assert images[0].getpixel((60, 60))[0] > 200
    assert images[0].getpixel((61, 61))[0] > 200
    assert images[0].getpixel((62, 62)) == (255, 255, 255)
    assert max_new_tokens == 768


def test_pipeline_saves_reference_target_detection_triptych(tmp_path: Path) -> None:
    backend = FakeBackend(
        "<rule>hex head</rule><bbox>[60,120,300,480]</bbox><answer>bolt</answer>"
    )
    pipeline = DetectionPipeline(backend, output_layout="triptych")
    output_path = tmp_path / "triptych.png"
    result = pipeline.run(_request(tmp_path), output_path=output_path)

    saved = Image.open(output_path).convert("RGB")
    assert saved.size == (1800, 600)
    assert result.output_layout == "triptych"
    assert result.annotated_image.size == (1000, 500)
    assert result.output_image.size == (1800, 600)
    # Reference red box, unannotated target, annotated target.
    assert saved.getpixel((60, 60))[0] > 200
    assert saved.getpixel((660, 120)) == (255, 255, 255)
    assert saved.getpixel((1260, 120))[0] > 200


def test_pipeline_rejects_unknown_output_layout() -> None:
    with pytest.raises(InputError, match="output_layout"):
        DetectionPipeline(FakeBackend("unused"), output_layout="grid")


def test_pipeline_rejects_completion_without_bbox(tmp_path: Path) -> None:
    pipeline = DetectionPipeline(FakeBackend("<answer>bolt</answer>"))
    with pytest.raises(OutputFormatError):
        pipeline.detect(_request(tmp_path))
