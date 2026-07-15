#!/usr/bin/env python3
"""PROTOTYPE — interactive/one-shot Qwen3-VL LoRA lifecycle profiler."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image, ImageDraw, ImageOps
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from conceptdet.prototypes.lora_smoke_logic import (
    SPECS,
    Topology,
    discover_targets,
    strict_detection_set,
    summarize_reports,
)

MIN_PIXELS = 65_536
MAX_PIXELS = 655_360
MEMORY_GATE_GIB = 44.0
REFERENCE_BOXES = (
    (1165, 2911, 1354, 3230),
    (4064, 3087, 4208, 3375),
)
PROMPT = (
    "Picture 1 is the reference image; its matching concept is marked by red boxes. "
    "Picture 2 is the target image. Return every Picture 2 instance matching the "
    "reference concept as a JSON array only. Each item must be "
    '{"bbox_2d":[x1,y1,x2,y2]}. Use integer coordinates on the 0-1000 grid. '
    "Return [] when no instance matches."
)


def _half_up(value: float) -> int:
    return int(value + 0.5)


def _normalize_box(
    box: tuple[int, int, int, int], source_size: tuple[int, int]
) -> list[int]:
    width, height = source_size
    x1, y1, x2, y2 = box
    return [
        _half_up(x1 / width * 1000),
        _half_up(y1 / height * 1000),
        _half_up(x2 / width * 1000),
        _half_up(y2 / height * 1000),
    ]


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as opened:
        return ImageOps.exif_transpose(opened).convert("RGB")


def _resize_plan(image: Image.Image) -> tuple[int, int]:
    height, width = smart_resize(
        image.height,
        image.width,
        factor=32,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    return width, height


def _prepare_images(
    reference_path: Path,
) -> tuple[Image.Image, Image.Image, Image.Image, tuple[int, int]]:
    source = _load_rgb(reference_path)
    processed_size = _resize_plan(source)
    reference = source.resize(processed_size, Image.Resampling.BICUBIC)
    target = reference.copy()
    scale_x = reference.width / source.width
    scale_y = reference.height / source.height
    inner_width = min(4, max(2, round(min(processed_size) / 256)))
    halo_width = 2
    draw = ImageDraw.Draw(reference)
    for x1, y1, x2, y2 in REFERENCE_BOXES:
        processed_box = (
            round(x1 * scale_x),
            round(y1 * scale_y),
            round(x2 * scale_x),
            round(y2 * scale_y),
        )
        left, top, right, bottom = processed_box
        draw.rectangle(
            (
                left - halo_width,
                top - halo_width,
                right + halo_width,
                bottom + halo_width,
            ),
            outline="#ffffff",
            width=halo_width,
        )
        draw.rectangle(processed_box, outline="#ff2020", width=inner_width)
    negative = Image.new("RGB", processed_size, "#808080")
    return reference, target, negative, source.size


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


def _processor_call(processor: Any, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
    return processor.apply_chat_template(
        messages,
        tokenize=True,
        add_vision_id=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={"do_resize": False},
        **kwargs,
    )


def _training_batch(
    processor: Any,
    reference: Image.Image,
    target: Image.Image,
    answer: str,
    device: torch.device,
) -> tuple[Any, int, list[list[int]]]:
    prompt_messages = _messages(reference, target)
    full_messages = [*prompt_messages, {"role": "assistant", "content": answer}]
    prompt = _processor_call(
        processor,
        prompt_messages,
        add_generation_prompt=True,
    )
    batch = _processor_call(
        processor,
        full_messages,
        add_generation_prompt=False,
    )
    prompt_length = prompt["input_ids"].shape[1]
    if not torch.equal(prompt["input_ids"], batch["input_ids"][:, :prompt_length]):
        raise RuntimeError("prompt tokens are not a prefix of the SFT conversation")
    labels = batch["input_ids"].clone()
    labels[:, :prompt_length] = -100
    batch["labels"] = labels
    grids = batch["image_grid_thw"].tolist()
    batch = batch.to(device=device, dtype=torch.bfloat16)
    return batch, int(batch["input_ids"].shape[1]), grids


def _memory(label: str, device: torch.device) -> dict[str, float | str]:
    torch.cuda.synchronize(device)
    free, total = torch.cuda.mem_get_info(device)
    return {
        "checkpoint": label,
        "allocated_gib": torch.cuda.memory_allocated(device) / 2**30,
        "reserved_gib": torch.cuda.memory_reserved(device) / 2**30,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        "device_used_gib": (total - free) / 2**30,
    }


def _release(*objects: Any) -> None:
    del objects
    gc.collect()
    torch.cuda.empty_cache()


def _mean_loss(model: torch.nn.Module, batches: list[Any]) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in batches:
            losses.append(float(model(**batch).loss.detach()))
    model.train()
    return sum(losses) / len(losses)


def _load_base(model_id: str, attention: str, device: torch.device) -> torch.nn.Module:
    return Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        attn_implementation=attention,
        device_map={"": device},
        low_cpu_mem_usage=True,
        local_files_only=True,
    )


def run_profile(args: argparse.Namespace, topology: Topology) -> dict[str, Any]:
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    checkpoints = [_memory("process_start", device)]
    spec = SPECS[topology]
    processor = AutoProcessor.from_pretrained(
        args.model,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
        local_files_only=True,
    )
    model = _load_base(args.model, args.attention, device)
    model.config.use_cache = False
    checkpoints.append(_memory("base_loaded", device))

    targets = discover_targets(model, topology)
    target_hash = hashlib.sha256("\n".join(targets).encode()).hexdigest()
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            target_modules=targets,
        ),
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if trainable != spec.expected_trainable_parameters:
        raise RuntimeError(f"expected {spec.expected_trainable_parameters}, found {trainable}")
    checkpoints.append(_memory("adapter_installed", device))

    reference, positive_target, negative_target, source_size = _prepare_images(args.reference)
    positive_answer = json.dumps(
        [{"bbox_2d": _normalize_box(box, source_size)} for box in REFERENCE_BOXES],
        separators=(",", ":"),
    )
    positive, positive_tokens, positive_grids = _training_batch(
        processor, reference, positive_target, positive_answer, device
    )
    negative, negative_tokens, negative_grids = _training_batch(
        processor, reference, negative_target, "[]", device
    )
    batches = [positive, negative]
    checkpoints.append(_memory("batches_on_device", device))

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )
    loss_before = _mean_loss(model, batches)
    torch.cuda.reset_peak_memory_stats(device)
    model.train()
    for batch in batches:
        output = model(**batch)
        (output.loss / len(batches)).backward()
        checkpoint = "backward_positive" if batch is positive else "backward_negative"
        checkpoints.append(_memory(checkpoint, device))
        del output
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    checkpoints.append(_memory("optimizer_complete", device))
    loss_after = _mean_loss(model, batches)
    checkpoints.append(_memory("post_step_loss", device))

    artifact_directory = args.artifact_directory / topology
    artifact_directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(artifact_directory, safe_serialization=True)
    checkpoints.append(_memory("adapter_saved", device))

    del optimizer, positive, negative, batches, model
    gc.collect()
    torch.cuda.empty_cache()
    checkpoints.append(_memory("training_model_released", device))

    reloaded_base = _load_base(args.model, args.attention, device)
    reloaded = PeftModel.from_pretrained(reloaded_base, artifact_directory)
    reloaded.eval()
    reloaded.config.use_cache = True
    checkpoints.append(_memory("adapter_reloaded", device))
    generation_messages = _messages(reference, positive_target)
    generation_batch = _processor_call(
        processor,
        generation_messages,
        add_generation_prompt=True,
    ).to(device=device, dtype=torch.bfloat16)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        generated = reloaded.generate(
            **generation_batch,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    completion_ids = generated[:, generation_batch["input_ids"].shape[1] :]
    completion = processor.batch_decode(completion_ids, skip_special_tokens=True)[0].strip()
    parse_error = None
    parsed: list[dict[str, Any]] | None = None
    try:
        parsed = strict_detection_set(completion)
    except (ValueError, json.JSONDecodeError) as error:
        parse_error = str(error)
    checkpoints.append(_memory("generation_complete", device))

    peak_reserved = max(float(point["peak_reserved_gib"]) for point in checkpoints)
    peak_device_used = max(float(point["device_used_gib"]) for point in checkpoints)
    result = {
        "topology": asdict(spec),
        "host_cuda_ordinal": args.visible_gpu,
        "gpu": torch.cuda.get_device_name(device),
        "gpu_uuid": str(torch.cuda.get_device_properties(device).uuid),
        "capacity_gib": torch.cuda.get_device_properties(device).total_memory / 2**30,
        "attention": args.attention,
        "dtype": "bfloat16",
        "visual_tokens_per_image_cap": 640,
        "max_pixels_per_image": MAX_PIXELS,
        "processed_size": [reference.width, reference.height],
        "image_grid_thw": {
            "positive": positive_grids,
            "negative": negative_grids,
        },
        "sequence_tokens": {
            "positive": positive_tokens,
            "negative": negative_tokens,
        },
        "target_module_count": len(targets),
        "target_modules_sha256": target_hash,
        "trainable_parameters": trainable,
        "mean_loss_before": loss_before,
        "mean_loss_after": loss_after,
        "mean_loss_improvement": loss_before - loss_after,
        "completion": completion,
        "parsed_detection_set": parsed,
        "parse_error": parse_error,
        "peak_reserved_gib": peak_reserved,
        "peak_device_used_gib": peak_device_used,
        "memory_gate_gib": MEMORY_GATE_GIB,
        "accepted": peak_reserved <= MEMORY_GATE_GIB and parse_error is None,
        "checkpoints": checkpoints,
    }
    args.report_directory.mkdir(parents=True, exist_ok=True)
    report_path = args.report_directory / f"{topology}.json"
    report_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


def _render_state(args: argparse.Namespace, message: str = "ready") -> None:
    comparison = summarize_reports(args.report_directory)
    print("\033[2J\033[H", end="")
    print("\033[1mQwen3-VL LoRA smoke prototype\033[0m")
    print(f"\033[1mstate\033[0m: {message}")
    print(f"\033[1mgpu\033[0m: CUDA_VISIBLE_DEVICES={args.visible_gpu}")
    print(f"\033[1mmemory gate\033[0m: {MEMORY_GATE_GIB:.1f} GiB peak reserved")
    for topology, report in comparison["reports"].items():
        if report is None:
            print(f"\033[1m{topology}\033[0m: not run")
        else:
            print(
                f"\033[1m{topology}\033[0m: accepted={report['accepted']} "
                f"peak={report['peak_reserved_gib']:.2f} GiB "
                f"loss_delta={report['mean_loss_improvement']:.4f}"
            )
    print("\n\033[1m[f]\033[0m full  \033[1m[a]\033[0m attention  ", end="")
    print("\033[1m[c]\033[0m compare  \033[1m[q]\033[0m quit")


def tui(args: argparse.Namespace) -> None:
    message = "ready"
    while True:
        _render_state(args, message)
        action = input("> ").strip().lower()
        if action == "q":
            return
        if action == "f":
            result = run_profile(args, "full")
            message = f"full finished: accepted={result['accepted']}"
        elif action == "a":
            result = run_profile(args, "attention")
            message = f"attention finished: accepted={result['accepted']}"
        elif action == "c":
            message = json.dumps(summarize_reports(args.report_directory), indent=2)
        else:
            message = f"unknown action: {action!r}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--visible-gpu", type=int, default=4)
    parser.add_argument("--attention", default="flash_attention_2")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--artifact-directory", type=Path, default=Path("outputs/lora-smoke"))
    parser.add_argument(
        "--report-directory",
        type=Path,
        default=Path("outputs/lora-smoke/reports"),
    )
    parser.add_argument("--run", choices=tuple(SPECS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    visible = str(args.visible_gpu)
    configured = os.environ.get("CUDA_VISIBLE_DEVICES")
    if configured != visible:
        raise RuntimeError(
            "set CUDA_VISIBLE_DEVICES before Python starts; "
            f"expected {visible!r}, found {configured!r}"
        )
    if args.run:
        result = run_profile(args, args.run)
        if not result["accepted"]:
            sys.exit(1)
    else:
        tui(args)


if __name__ == "__main__":
    main()
