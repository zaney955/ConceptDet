from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from conceptdet.adapter import AdapterGeneration, AdapterInput
from conceptdet.artifact import AdapterArtifact
from conceptdet.config import RuntimeConfig
from conceptdet.errors import InputError, ModelLoadError
from conceptdet.peft_weights import load_exact_adapter_weights
from conceptdet.prompts import build_messages

MIN_PIXELS = 65_536
MAX_PIXELS = 655_360
MIN_VISUAL_TOKENS = 64
MAX_VISUAL_TOKENS = 640
MAX_TOTAL_SEQUENCE_TOKENS = 1536


def smart_image_size(image_size: tuple[int, int]) -> tuple[int, int]:
    try:
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    except ImportError as exc:
        raise ModelLoadError("Transformers Qwen-VL image processing is unavailable") from exc
    height, width = smart_resize(
        image_size[1],
        image_size[0],
        factor=32,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    return width, height


def prepare_images(request: AdapterInput) -> tuple[Image.Image, Image.Image]:
    reference_size = smart_image_size(request.reference_image.size)
    target_size = smart_image_size(request.target_image.size)
    reference = request.reference_image.resize(reference_size, Image.Resampling.BICUBIC)
    target = request.target_image.resize(target_size, Image.Resampling.BICUBIC)
    scale_x = reference.width / request.reference_image.width
    scale_y = reference.height / request.reference_image.height
    inner_width = min(4, max(2, round(min(reference.size) / 256)))
    draw = ImageDraw.Draw(reference)
    for original in request.reference_boxes:
        box = original.clamp(*request.reference_image.size)
        coordinates = (
            round(box.x1 * scale_x),
            round(box.y1 * scale_y),
            round(box.x2 * scale_x),
            round(box.y2 * scale_y),
        )
        left, top, right, bottom = coordinates
        draw.rectangle(
            (left - 2, top - 2, right + 2, bottom + 2),
            outline="#ffffff",
            width=2,
        )
        draw.rectangle(coordinates, outline="#ff2020", width=inner_width)
    return reference, target


class Qwen3VLAdapter:
    def __init__(
        self,
        artifact: AdapterArtifact,
        model: Any,
        processor: Any,
        device: str,
        dtype: Any,
    ) -> None:
        self.artifact = artifact
        self.model = model
        self.processor = processor
        self.device = device
        self.dtype = dtype

    @staticmethod
    def _resolve_device(device: str, torch: Any) -> str:
        if device == "auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise ModelLoadError(f"CUDA device requested but unavailable: {device}")
        return device

    @staticmethod
    def _resolve_dtype(dtype: str, device: str, torch: Any) -> Any:
        if dtype == "auto":
            return torch.bfloat16 if device.startswith("cuda") else torch.float32
        mapping = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if dtype not in mapping:
            raise InputError(f"Unsupported dtype: {dtype}")
        if device == "cpu" and dtype != "float32":
            return torch.float32
        return mapping[dtype]

    @staticmethod
    def _resolve_attention(attention: str, device: str) -> str:
        if attention != "auto":
            return attention
        has_flash = importlib.util.find_spec("flash_attn") is not None
        return "flash_attention_2" if device.startswith("cuda") and has_flash else "sdpa"

    @classmethod
    def load(cls, artifact_path: str | Path, runtime: RuntimeConfig) -> Qwen3VLAdapter:
        # Contract and PEFT files are validated before importing/allocating the 8B model.
        artifact = AdapterArtifact.load(artifact_path)
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise ModelLoadError(
                "Qwen3-VL inference dependencies are unavailable; install the default runtime "
                "and optional flash-attn profile"
            ) from exc

        device = cls._resolve_device(runtime.device, torch)
        dtype = cls._resolve_dtype(runtime.dtype, device, torch)
        attention = cls._resolve_attention(runtime.attention, device)
        base = artifact.contract["base"]
        processor_contract = artifact.contract["processor"]
        try:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                base["model_id"],
                revision=base["model_revision"],
                dtype=dtype,
                attn_implementation=attention,
                device_map={"": device},
                low_cpu_mem_usage=True,
                local_files_only=runtime.local_files_only,
            )
            model = PeftModel.from_pretrained(model, artifact.path, is_trainable=False)
            load_exact_adapter_weights(model, artifact.path)
            model.eval()
            model.config.use_cache = True
            model.generation_config.do_sample = False
            model.generation_config.temperature = None
            model.generation_config.top_p = None
            model.generation_config.top_k = None
            processor = AutoProcessor.from_pretrained(
                processor_contract["processor_id"],
                revision=processor_contract["processor_revision"],
                min_pixels=MIN_PIXELS,
                max_pixels=MAX_PIXELS,
                local_files_only=runtime.local_files_only,
            )
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load Qwen3-VL Artifact {artifact.path}: {exc}"
            ) from exc
        return cls(artifact, model, processor, device, dtype)

    def generate(self, request: AdapterInput, *, max_new_tokens: int) -> AdapterGeneration:
        if not 1 <= max_new_tokens <= 192:
            raise InputError("max_new_tokens must be between 1 and 192")
        import torch

        reference, target = prepare_images(request)
        messages = build_messages(reference, target, request.query)
        encoded = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            add_vision_id=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"do_resize": False},
        )
        prompt_tokens = int(encoded["input_ids"].shape[1])
        if prompt_tokens + max_new_tokens > MAX_TOTAL_SEQUENCE_TOKENS:
            raise InputError(
                f"Prompt ({prompt_tokens}) + completion ({max_new_tokens}) exceeds "
                f"{MAX_TOTAL_SEQUENCE_TOKENS}; truncation is forbidden"
            )
        grids = tuple(tuple(int(value) for value in row) for row in encoded["image_grid_thw"])
        if len(grids) != 2:
            raise ModelLoadError(f"Expected two image grids, found {grids}")
        for grid in grids:
            visual_tokens = grid[0] * grid[1] * grid[2] // 4
            if not MIN_VISUAL_TOKENS <= visual_tokens <= MAX_VISUAL_TOKENS:
                raise ModelLoadError(
                    f"Image grid {grid} produces {visual_tokens} visual tokens outside contract"
                )
        encoded = encoded.to(device=self.device, dtype=self.dtype)
        with torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        completion_ids = generated[:, prompt_tokens:]
        completion = self.processor.batch_decode(
            completion_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        return AdapterGeneration(completion, reference, target, grids, prompt_tokens)
