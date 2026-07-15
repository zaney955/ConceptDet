from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageOps

from conceptdet.errors import InputError, OutputFormatError
from conceptdet.geometry import prepare_reference, prepare_target
from conceptdet.model import GenerationBackend
from conceptdet.parsing import parse_completion
from conceptdet.prompts import build_messages, build_problem
from conceptdet.types import Box
from conceptdet.visualization import annotate, compose_triptych


@dataclass(frozen=True)
class DetectionRequest:
    reference_path: Path
    reference_boxes: tuple[Box, ...]
    target_path: Path
    query: str
    reference_crop_mode: str = "full"
    reference_crop_context_scale: float = 4.0


@dataclass(frozen=True)
class Detection:
    model_box: Box
    target_box: Box
    label: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "bbox_xyxy": self.target_box.to_list(rounded=True),
            "model_bbox_xyxy": self.model_box.to_list(rounded=True),
            "label": self.label,
        }


@dataclass(frozen=True)
class DetectionResult:
    request: DetectionRequest
    detections: tuple[Detection, ...]
    rule: str | None
    reasoning: str | None
    raw_completion: str
    model_input_size: tuple[int, int]
    reference_crop_box: tuple[int, int, int, int]
    output_layout: str
    annotated_image: Image.Image = field(repr=False, compare=False)
    output_image: Image.Image = field(repr=False, compare=False)

    def to_dict(self) -> dict[str, object]:
        return {
            "reference_image": str(self.request.reference_path),
            "reference_boxes_xyxy": [
                box.to_list(rounded=True) for box in self.request.reference_boxes
            ],
            "reference_crop_box_xyxy": list(self.reference_crop_box),
            "target_image": str(self.request.target_path),
            "query": self.request.query,
            "model_input_size": list(self.model_input_size),
            "output_layout": self.output_layout,
            "detections": [detection.to_dict() for detection in self.detections],
            "rule": self.rule,
            "reasoning": self.reasoning,
            "raw_completion": self.raw_completion,
        }


class DetectionPipeline:
    def __init__(
        self,
        backend: GenerationBackend,
        *,
        input_size: int = 600,
        max_new_tokens: int = 768,
        annotation_color: str = "red",
        annotation_width: int = 2,
        reference_box_width: int = 2,
        output_layout: str = "triptych",
    ) -> None:
        if input_size < 1:
            raise InputError("input_size must be >= 1")
        if annotation_width < 1:
            raise InputError("annotation_width must be >= 1")
        if reference_box_width < 1:
            raise InputError("reference_box_width must be >= 1")
        if output_layout not in {"triptych", "annotated"}:
            raise InputError("output_layout must be 'triptych' or 'annotated'")
        self.backend = backend
        self.model_size = (input_size, input_size)
        self.max_new_tokens = max_new_tokens
        self.annotation_color = annotation_color
        self.annotation_width = annotation_width
        self.reference_box_width = reference_box_width
        self.output_layout = output_layout

    @staticmethod
    def _load_image(path: Path) -> Image.Image:
        path = path.expanduser().resolve()
        if not path.is_file():
            raise InputError(f"Image does not exist: {path}")
        try:
            with Image.open(path) as opened:
                return ImageOps.exif_transpose(opened).convert("RGB")
        except OSError as exc:
            raise InputError(f"Cannot read image: {path}") from exc

    def detect(self, request: DetectionRequest) -> DetectionResult:
        reference = self._load_image(request.reference_path)
        target = self._load_image(request.target_path)
        prepared_reference = prepare_reference(
            reference,
            request.reference_boxes,
            model_size=self.model_size,
            crop_mode=request.reference_crop_mode,
            context_scale=request.reference_crop_context_scale,
            box_width=self.reference_box_width,
        )
        prepared_target, target_transform = prepare_target(target, self.model_size)
        problem = build_problem(
            request.query, len(request.reference_boxes), input_size=self.model_size[0]
        )
        messages = build_messages(problem)
        completion = self.backend.generate(
            messages,
            (prepared_reference.image, prepared_target),
            max_new_tokens=self.max_new_tokens,
        )
        parsed = parse_completion(completion)

        detections: list[Detection] = []
        for model_box in parsed.boxes:
            try:
                clamped_model_box = model_box.clamp(*self.model_size)
                target_box = target_transform.to_source(clamped_model_box).clamp(*target.size)
            except InputError as exc:
                raise OutputFormatError(f"Model produced an unusable bbox: {model_box}") from exc
            detections.append(Detection(clamped_model_box, target_box, parsed.answer))

        annotated_image = annotate(
            target,
            tuple(detection.target_box for detection in detections),
            label=parsed.answer,
            color=self.annotation_color,
            width=self.annotation_width,
        )
        annotated_prompt_image = annotate(
            prepared_target,
            tuple(detection.model_box for detection in detections),
            label=parsed.answer,
            color=self.annotation_color,
            width=self.annotation_width,
        )
        output_image = (
            compose_triptych(
                prepared_reference.image,
                prepared_target,
                annotated_prompt_image,
            )
            if self.output_layout == "triptych"
            else annotated_image
        )
        return DetectionResult(
            request=request,
            detections=tuple(detections),
            rule=parsed.rule,
            reasoning=parsed.reasoning,
            raw_completion=completion,
            model_input_size=self.model_size,
            reference_crop_box=prepared_reference.crop_box,
            output_layout=self.output_layout,
            annotated_image=annotated_image,
            output_image=output_image,
        )

    def run(
        self,
        request: DetectionRequest,
        *,
        output_path: Path,
        json_path: Path | None = None,
    ) -> DetectionResult:
        result = self.detect(request)
        output_path = output_path.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.output_image.save(output_path)

        json_path = json_path or output_path.with_suffix(".json")
        json_path = json_path.expanduser().resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return result
