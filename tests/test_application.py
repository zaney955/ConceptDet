import json
from pathlib import Path

from PIL import Image

from conceptdet.adapter import FakeAdapter
from conceptdet.application import DetectionApplication
from conceptdet.config import OutputConfig, RequestConfig
from conceptdet.types import Box


def _request(tmp_path: Path) -> RequestConfig:
    reference = tmp_path / "reference.png"
    target = tmp_path / "target.png"
    Image.new("RGB", (200, 100), "white").save(reference)
    Image.new("RGB", (1000, 500), "white").save(target)
    return RequestConfig(reference, (Box(20, 10, 80, 50),), target, "matching bolt")


def test_fake_adapter_tracer_bullet_writes_detection_set_and_image(tmp_path: Path) -> None:
    adapter = FakeAdapter('[{"bbox_2d":[100,200,500,600]}]')
    application = DetectionApplication(adapter)
    output = OutputConfig(tmp_path / "result.png", tmp_path / "result.json", "annotated")
    result = application.run(
        _request(tmp_path), output, max_new_tokens=64, config_hash="resolved-hash"
    )
    assert result.detections[0].target_box == Box(100, 100, 500, 300)
    assert result.protocol_detections[0].box == Box(100, 200, 500, 600)
    assert output.image.is_file()
    payload = json.loads(output.json.read_text(encoding="utf-8"))
    assert payload["detection_set"] == [{"bbox_2d": [100, 200, 500, 600]}]
    assert payload["config_hash"] == "resolved-hash"
    assert len(adapter.requests) == 1


def test_empty_detection_set_is_a_success(tmp_path: Path) -> None:
    result = DetectionApplication(FakeAdapter("[]")).detect(
        _request(tmp_path), max_new_tokens=64, config_hash="h"
    )
    assert result.detections == ()
    assert result.protocol_detections == ()
