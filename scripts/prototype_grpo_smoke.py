#!/usr/bin/env python3
"""PROTOTYPE — native two-image Qwen3-VL + TRL GRPO smoke test."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import PeftModel
from PIL import Image, ImageDraw, ImageOps
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from trl import GRPOConfig, GRPOTrainer

from conceptdet.prototypes.grpo_smoke_logic import Box, evaluate_reward

MIN_PIXELS = 65_536
MAX_PIXELS = 655_360
MEMORY_GATE_GIB = 44.0
EXPECTED_TRAINABLE = 44_793_856
REFERENCE_BOXES = (
    (1165, 2911, 1354, 3230),
    (4064, 3087, 4208, 3375),
)
PROMPT = (
    "Picture 1 is the reference image; its matching concept is marked by red boxes. "
    "Picture 2 is the target image. Return every Picture 2 instance matching the "
    "reference concept as a JSON array only. Each item must be "
    '{"bbox_2d":[x1,y1,x2,y2]}. Use integer coordinates on the 0-1000 grid. '
    "Return [] when no instance matches. Do not use Markdown or code fences: the "
    "first output character must be [ and the last must be ]."
)


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as opened:
        return ImageOps.exif_transpose(opened).convert("RGB")


def _prepare_images(
    path: Path,
) -> tuple[Image.Image, Image.Image, Image.Image, tuple[int, int]]:
    source = _load_rgb(path)
    height, width = smart_resize(
        source.height,
        source.width,
        factor=32,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    size = (width, height)
    target = source.resize(size, Image.Resampling.BICUBIC)
    reference = target.copy()
    draw = ImageDraw.Draw(reference)
    scale_x, scale_y = width / source.width, height / source.height
    inner_width = min(4, max(2, round(min(size) / 256)))
    for x1, y1, x2, y2 in REFERENCE_BOXES:
        box = (
            round(x1 * scale_x),
            round(y1 * scale_y),
            round(x2 * scale_x),
            round(y2 * scale_y),
        )
        left, top, right, bottom = box
        draw.rectangle(
            (left - 2, top - 2, right + 2, bottom + 2),
            outline="#ffffff",
            width=2,
        )
        draw.rectangle(box, outline="#ff2020", width=inner_width)
    return reference, target, Image.new("RGB", size, "#808080"), source.size


def _half_up(value: float) -> int:
    return int(value + 0.5)


def _ground_truth(source_size: tuple[int, int]) -> list[list[int]]:
    width, height = source_size
    return [
        [
            _half_up(x1 / width * 1000),
            _half_up(y1 / height * 1000),
            _half_up(x2 / width * 1000),
            _half_up(y2 / height * 1000),
        ]
        for x1, y1, x2, y2 in REFERENCE_BOXES
    ]


def _messages(reference: Image.Image, target: Image.Image) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": reference},
                {"type": "image", "image": target},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]


def _preflight_processor(
    processor: Any, reference: Image.Image, target: Image.Image
) -> dict[str, Any]:
    kwargs = {
        "conversation": _messages(reference, target),
        "tokenize": True,
        "add_generation_prompt": True,
        "add_vision_id": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    default = processor.apply_chat_template(**kwargs)
    fixed = processor.apply_chat_template(
        **kwargs, processor_kwargs={"do_resize": False}
    )
    default_pixels = default["pixel_values"].float()
    fixed_pixels = fixed["pixel_values"].float()
    same_shape = default_pixels.shape == fixed_pixels.shape
    max_difference = (
        float((default_pixels - fixed_pixels).abs().max()) if same_shape else None
    )
    rendered = processor.apply_chat_template(
        _messages(reference, target),
        tokenize=False,
        add_generation_prompt=True,
        add_vision_id=True,
    )
    grids = default["image_grid_thw"].tolist()
    return {
        "prepared_size": [reference.width, reference.height],
        "default_grids": grids,
        "no_resize_grids": fixed["image_grid_thw"].tolist(),
        "pixel_shapes_equal": same_shape,
        "pixel_max_abs_difference": max_difference,
        "pixel_tensors_equal": bool(same_shape and max_difference == 0.0),
        "picture_order_rendered": "Picture 1:" in rendered and "Picture 2:" in rendered,
        "rendered_prompt_excerpt": rendered[:400],
    }


def _completion_text(completion: Any) -> str:
    content = completion[-1]["content"] if isinstance(completion, list) else completion
    if isinstance(content, list):
        return "".join(
            block.get("text", "") for block in content if block.get("type") == "text"
        )
    return str(content)


def _trainable_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        if not parameter.requires_grad:
            continue
        digest.update(name.encode())
        digest.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _memory(device: torch.device, label: str) -> dict[str, float | str]:
    torch.cuda.synchronize(device)
    free, total = torch.cuda.mem_get_info(device)
    return {
        "checkpoint": label,
        "allocated_gib": torch.cuda.memory_allocated(device) / 2**30,
        "reserved_gib": torch.cuda.memory_reserved(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        "device_used_gib": (total - free) / 2**30,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.visible_gpu):
        raise RuntimeError("set CUDA_VISIBLE_DEVICES to --visible-gpu before Python starts")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    checkpoints = [_memory(device, "process_start")]

    processor = AutoProcessor.from_pretrained(
        args.model,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
        local_files_only=True,
    )
    reference, positive, negative, source_size = _prepare_images(args.reference)
    preflight = _preflight_processor(processor, reference, positive)
    if preflight["default_grids"] != [[1, 46, 52], [1, 46, 52]]:
        raise RuntimeError(f"unexpected visual grid: {preflight['default_grids']}")
    if not preflight["pixel_tensors_equal"]:
        raise RuntimeError("processor default path rewrites adapter-prepared pixels")
    if not preflight["picture_order_rendered"]:
        raise RuntimeError("processor did not render ordered Picture 1/Picture 2 markers")

    base = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": device},
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    base.config.use_cache = False
    model = PeftModel.from_pretrained(base, args.init_adapter, is_trainable=True)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable != EXPECTED_TRAINABLE:
        raise RuntimeError(
            f"expected {EXPECTED_TRAINABLE} trainable params, found {trainable}"
        )
    before_digest = _trainable_digest(model)
    checkpoints.append(_memory(device, "adapter_loaded"))

    dataset = Dataset.from_list(
        [
            {
                "prompt": [{"role": "user", "content": PROMPT}],
                "images": [reference, positive],
                "ground_truth": _ground_truth(source_size),
                "sample_kind": "positive",
            },
            {
                "prompt": [{"role": "user", "content": PROMPT}],
                "images": [reference, negative],
                "ground_truth": [],
                "sample_kind": "negative",
            },
        ]
    )
    reward_events: list[dict[str, Any]] = []

    def detection_set_reward(
        prompts: list[Any],
        completions: list[Any],
        ground_truth: list[list[list[int]]],
        sample_kind: list[str],
        **_: Any,
    ) -> list[float]:
        del prompts
        values = []
        for completion, raw_targets, kind in zip(
            completions, ground_truth, sample_kind, strict=True
        ):
            text = _completion_text(completion)
            targets: list[Box] = [tuple(box) for box in raw_targets]  # type: ignore[misc]
            state = evaluate_reward(text, targets)
            reward_events.append(
                {"sample_kind": kind, "completion": text, **asdict(state)}
            )
            values.append(state.total_reward)
        return values

    trainer_args = GRPOConfig(
        output_dir=str(args.output / "trainer"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        generation_batch_size=2,
        num_generations=2,
        max_steps=2,
        max_completion_length=192,
        learning_rate=1e-5,
        beta=0.0,
        use_vllm=False,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_cache=False,
        shuffle_dataset=False,
        chat_template_kwargs={"add_vision_id": True},
        logging_steps=1,
        logging_first_step=True,
        log_completions=True,
        num_completions_to_print=2,
        report_to="none",
        save_strategy="no",
        disable_tqdm=True,
        dataloader_pin_memory=False,
        seed=args.seed,
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=detection_set_reward,
        args=trainer_args,
        train_dataset=dataset,
        processing_class=processor,
    )
    if trainer.ref_model is not None:
        raise RuntimeError("beta=0 unexpectedly created a reference model")
    checkpoints.append(_memory(device, "trainer_ready"))
    train_result = trainer.train()
    checkpoints.append(_memory(device, "training_complete"))
    after_digest = _trainable_digest(trainer.model)
    adapter_output = args.output / "adapter"
    trainer.save_model(str(adapter_output))
    checkpoints.append(_memory(device, "adapter_saved"))

    kinds = {event["sample_kind"] for event in reward_events}
    peak = max(float(point["peak_reserved_gib"]) for point in checkpoints)
    acceptance = {
        "native_grpo_trainer": trainer.__class__ is GRPOTrainer,
        "no_reference_model": trainer.ref_model is None,
        "ordered_two_image_grid": preflight["default_grids"]
        == [[1, 46, 52], [1, 46, 52]],
        "no_processor_contract_drift": preflight["pixel_tensors_equal"],
        "reward_saw_positive_and_negative": kinds == {"positive", "negative"},
        "reward_completion_count_at_least_four": len(reward_events) >= 4,
        "adapter_saved": (adapter_output / "adapter_model.safetensors").is_file(),
        "adapter_parameters_changed": before_digest != after_digest,
        "peak_reserved_within_44_gib": peak <= MEMORY_GATE_GIB,
    }
    report = {
        "question": (
            "Can native TRL GRPO train Qwen3-VL-8B LoRA on ordered two-image "
            "Detection Set prompts?"
        ),
        "versions": {
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "trl": __import__("trl").__version__,
            "peft": __import__("peft").__version__,
        },
        "model": args.model,
        "initial_adapter": str(args.init_adapter),
        "device": torch.cuda.get_device_name(device),
        "device_uuid": str(torch.cuda.get_device_properties(device).uuid),
        "trainable_parameters": trainable,
        "preflight": preflight,
        "trainer": {
            "class": f"{trainer.__class__.__module__}.{trainer.__class__.__name__}",
            "beta": trainer_args.beta,
            "num_generations": trainer_args.num_generations,
            "generation_batch_size": trainer_args.generation_batch_size,
            "steps_per_generation": trainer_args.steps_per_generation,
            "max_steps": trainer_args.max_steps,
            "metrics": train_result.metrics,
        },
        "reward_events": reward_events,
        "adapter_parameters_changed": before_digest != after_digest,
        "before_digest": before_digest,
        "after_digest": after_digest,
        "memory": checkpoints,
        "peak_reserved_gib": peak,
        "acceptance": acceptance,
        "passed": all(acceptance.values()),
        "required_seam": (
            "dataset emits raw conversational prompt plus ordered images=[reference,target]; "
            "adapter pre-resizes both images to the 640-token contract; stock ProcessorMixin and "
            "stock GRPOTrainer handle tokenization, generation, and forward recomputation"
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise RuntimeError(f"acceptance failed; see {report_path}")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--init-adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/grpo-smoke"))
    parser.add_argument("--visible-gpu", type=int, default=6)
    parser.add_argument("--seed", type=int, default=23)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.run:
        run(args)
        return
    while True:
        choice = input("[r] run native GRPO smoke  [q] quit > ").strip().lower()
        if choice == "r":
            run(args)
            return
        if choice == "q":
            return


if __name__ == "__main__":
    main()
