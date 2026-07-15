from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageOps

from conceptdet.adapter import AdapterInput, DetectionAdapter
from conceptdet.config import DetectConfig, OutputConfig, RequestConfig
from conceptdet.errors import InputError
from conceptdet.protocol import ProtocolDetection, decode_model_box
from conceptdet.protocol import parse_detection_set as parse_protocol_detection_set
from conceptdet.types import Box
from conceptdet.visualization import annotate, compose_triptych


@dataclass(frozen=True)
class Detection:
    model_box: Box
    target_box: Box
    label: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "bbox_2d": self.model_box.to_list(rounded=True),
            "bbox_xyxy": self.target_box.to_list(rounded=True),
            "label": self.label,
        }


@dataclass(frozen=True)
class DetectionResult:
    request: RequestConfig
    protocol_detections: tuple[ProtocolDetection, ...]
    detections: tuple[Detection, ...]
    raw_completion: str
    image_grids: tuple[tuple[int, int, int], ...]
    prompt_tokens: int | None
    config_hash: str
    annotated_image: Image.Image = field(repr=False, compare=False)
    output_image: Image.Image = field(repr=False, compare=False)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "reference_image": str(self.request.reference_image),
            "reference_boxes_xyxy": [
                box.to_list(rounded=True) for box in self.request.reference_boxes
            ],
            "target_image": str(self.request.target_image),
            "query": self.request.query,
            "detection_set": [
                detection.to_model_dict() for detection in self.protocol_detections
            ],
            "detections": [detection.to_dict() for detection in self.detections],
            "image_grids": [list(grid) for grid in self.image_grids],
            "prompt_tokens": self.prompt_tokens,
            "config_hash": self.config_hash,
            "raw_completion": self.raw_completion,
        }


class DetectionApplication:
    def __init__(self, adapter: DetectionAdapter) -> None:
        self.adapter = adapter

    @staticmethod
    def _load_image(path: Path) -> Image.Image:
        if not path.is_file():
            raise InputError(f"Image does not exist: {path}")
        try:
            with Image.open(path) as opened:
                return ImageOps.exif_transpose(opened).convert("RGB")
        except OSError as exc:
            raise InputError(f"Cannot read image: {path}") from exc

    def detect(
        self, request: RequestConfig, *, max_new_tokens: int, config_hash: str
    ) -> DetectionResult:
        reference = self._load_image(request.reference_image)
        target = self._load_image(request.target_image)
        for box in request.reference_boxes:
            box.clamp(*reference.size)
        generated = self.adapter.generate(
            AdapterInput(reference, request.reference_boxes, target, request.query),
            max_new_tokens=max_new_tokens,
        )
        protocol_detections = parse_protocol_detection_set(generated.completion)
        detections = tuple(
            Detection(
                item.box,
                decode_model_box(item.box, target.size),
                item.label,
            )
            for item in protocol_detections
        )
        annotated = annotate(
            target,
            tuple(detection.target_box for detection in detections),
            width=4,
        )
        prepared_result = annotate(
            generated.prepared_target,
            tuple(
                decode_model_box(detection.model_box, generated.prepared_target.size)
                for detection in detections
            ),
            width=4,
        )
        triptych = compose_triptych(
            generated.prepared_reference,
            generated.prepared_target,
            prepared_result,
        )
        return DetectionResult(
            request,
            protocol_detections,
            detections,
            generated.completion,
            generated.image_grids,
            generated.prompt_tokens,
            config_hash,
            annotated,
            triptych,
        )

    def run(
        self,
        request: RequestConfig,
        output: OutputConfig,
        *,
        max_new_tokens: int,
        config_hash: str,
    ) -> DetectionResult:
        result = self.detect(
            request, max_new_tokens=max_new_tokens, config_hash=config_hash
        )
        output.image.parent.mkdir(parents=True, exist_ok=True)
        output.json.parent.mkdir(parents=True, exist_ok=True)
        image = result.output_image if output.layout == "triptych" else result.annotated_image
        image.save(output.image)
        output.json.write_text(
            json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return result


def run_detect_config(config: DetectConfig, adapter: DetectionAdapter) -> DetectionResult:
    return DetectionApplication(adapter).run(
        config.request,
        config.output,
        max_new_tokens=config.runtime.max_new_tokens,
        config_hash=config.config_hash,
    )
