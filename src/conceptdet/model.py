from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Protocol

from PIL import Image

from conceptdet.errors import InputError, ModelLoadError


class GenerationBackend(Protocol):
    def generate(
        self,
        messages: list[dict[str, object]],
        images: tuple[Image.Image, Image.Image],
        *,
        max_new_tokens: int,
    ) -> str: ...


class TransformersBackend:
    """Minimal Qwen2.5-VL generation backend; no SAM modules are constructed."""

    _IGNORED_CHECKPOINT_PREFIXES = (
        "learnable_query.",
        "connector.",
        "proj_to_sam.",
        "conv_1d.",
    )

    def __init__(self, model: object, processor: object, device: str, dtype: object):
        self.model = model
        self.processor = processor
        self.device = device
        self.dtype = dtype

    @classmethod
    def load(
        cls,
        model_path: str | Path,
        *,
        device: str = "auto",
        dtype: str = "auto",
        attention: str = "auto",
    ) -> TransformersBackend:
        model_path = Path(model_path).expanduser().resolve()
        cls._validate_checkpoint(model_path)

        try:
            import torch
            import torchvision  # noqa: F401
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise ModelLoadError(
                "Inference dependencies (including torchvision) are unavailable. "
                "Create the standalone environment with `bash scripts/create_env.sh`."
            ) from exc

        resolved_device = cls._resolve_device(device, torch)
        resolved_dtype = cls._resolve_dtype(dtype, resolved_device, torch)
        resolved_attention = cls._resolve_attention(attention, resolved_device)

        try:
            model, loading_info = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                str(model_path),
                dtype=resolved_dtype,
                attn_implementation=resolved_attention,
                device_map={"": resolved_device},
                low_cpu_mem_usage=True,
                output_loading_info=True,
            )
            cls._validate_loading_info(loading_info)
            model.eval()
            model.generation_config.do_sample = False
            model.generation_config.temperature = None
            # Transformers 5.13.1 forwards `backend` to Qwen2VLVideoProcessor,
            # whose backend property is read-only. Keep the still-supported
            # compatibility argument until that upstream processor bug is fixed.
            processor = AutoProcessor.from_pretrained(str(model_path), use_fast=False)
        except ModelLoadError:
            raise
        except Exception as exc:
            raise ModelLoadError(f"Failed to load checkpoint {model_path}: {exc}") from exc

        return cls(model, processor, resolved_device, resolved_dtype)

    @staticmethod
    def _validate_checkpoint(model_path: Path) -> None:
        if not model_path.is_dir():
            raise ModelLoadError(f"Checkpoint directory does not exist: {model_path}")
        config_path = model_path / "config.json"
        if not config_path.is_file():
            raise ModelLoadError(f"Checkpoint has no config.json: {model_path}")
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelLoadError(f"Cannot read checkpoint config: {config_path}") from exc
        if config.get("model_type") not in {"qwen2_vl", "qwen2_5_vl"}:
            raise ModelLoadError(
                f"Unsupported model_type={config.get('model_type')!r}; "
                "expected a ConceptSeg-R1/Qwen VL checkpoint"
            )

    @staticmethod
    def _resolve_device(device: str, torch: object) -> str:
        if device == "auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise ModelLoadError(f"CUDA device requested but CUDA is unavailable: {device}")
        return device

    @staticmethod
    def _resolve_dtype(dtype: str, device: str, torch: object) -> object:
        if dtype == "auto":
            if device.startswith("cuda"):
                return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            return torch.float32
        mapping = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if dtype not in mapping:
            raise InputError(f"Unsupported dtype: {dtype}")
        return mapping[dtype]

    @staticmethod
    def _resolve_attention(attention: str, device: str) -> str:
        allowed = {"auto", "eager", "sdpa", "flash_attention_2"}
        if attention not in allowed:
            raise InputError(f"Unsupported attention backend: {attention}")
        if attention != "auto":
            return attention
        has_flash_attn = importlib.util.find_spec("flash_attn") is not None
        return "flash_attention_2" if device.startswith("cuda") and has_flash_attn else "sdpa"

    @classmethod
    def _validate_loading_info(cls, loading_info: dict[str, object]) -> None:
        missing = list(loading_info.get("missing_keys", []))
        if missing:
            preview = ", ".join(missing[:8])
            raise ModelLoadError(f"Checkpoint is missing base Qwen weights: {preview}")
        unexpected = list(loading_info.get("unexpected_keys", []))
        unsupported = [
            key for key in unexpected if not key.startswith(cls._IGNORED_CHECKPOINT_PREFIXES)
        ]
        if unsupported:
            preview = ", ".join(unsupported[:8])
            raise ModelLoadError(f"Checkpoint contains unsupported unexpected weights: {preview}")

    def generate(
        self,
        messages: list[dict[str, object]],
        images: tuple[Image.Image, Image.Image],
        *,
        max_new_tokens: int,
    ) -> str:
        import torch

        if max_new_tokens < 1:
            raise InputError("max_new_tokens must be >= 1")
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=[[images[0], images[1]]],
            padding=True,
            return_tensors="pt",
        ).to(device=self.device, dtype=self.dtype)
        prompt_length = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        completion_ids = generated[:, prompt_length:]
        return self.processor.batch_decode(
            completion_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
