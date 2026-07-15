#!/usr/bin/env python3
"""PROTOTYPE — one-process Qwen3-VL rank-16 LoRA memory profile."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from PIL import Image, ImageDraw, ImageOps
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as opened:
        return ImageOps.exif_transpose(opened).convert("RGB")


def prepare_reference(path: Path, boxes: list[tuple[int, int, int, int]]) -> Image.Image:
    image = load_rgb(path)
    draw = ImageDraw.Draw(image)
    for box in boxes:
        draw.rectangle(box, outline="red", width=8)
    return image


def parse_box(value: str) -> tuple[int, int, int, int]:
    coords = tuple(int(part) for part in value.split(","))
    if len(coords) != 4:
        raise argparse.ArgumentTypeError("box must be x1,y1,x2,y2")
    return coords  # type: ignore[return-value]


def discover_targets(model: torch.nn.Module) -> list[str]:
    text_pattern = re.compile(
        r"model\.language_model\.layers\.\d+\."
        r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))$"
    )
    merger_pattern = re.compile(
        r"model\.visual\.(?:merger|deepstack_merger_list\.\d+)\.linear_fc[12]$"
    )
    targets = [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and (text_pattern.fullmatch(name) or merger_pattern.fullmatch(name))
    ]
    if len(targets) != 260:
        preview = "\n".join(targets[:20])
        raise RuntimeError(f"expected 260 LoRA targets, found {len(targets)}\n{preview}")
    return targets


def memory(label: str, device: torch.device) -> dict[str, float | str]:
    torch.cuda.synchronize(device)
    return {
        "checkpoint": label,
        "allocated_gib": torch.cuda.memory_allocated(device) / 2**30,
        "reserved_gib": torch.cuda.memory_reserved(device) / 2**30,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--tokens-per-image", type=int, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--attention", default="flash_attention_2")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--reference-box", action="append", type=parse_box, default=[])
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    checkpoints: list[dict[str, float | str]] = [memory("process_start", device)]
    max_pixels = args.tokens_per_image * 32 * 32
    processor = AutoProcessor.from_pretrained(
        args.model,
        min_pixels=64 * 32 * 32,
        max_pixels=max_pixels,
        local_files_only=True,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation=args.attention,
        device_map={"": device},
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.config.use_cache = False
    checkpoints.append(memory("base_loaded", device))

    targets = discover_targets(model)
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
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable != 44_793_856:
        raise RuntimeError(f"unexpected trainable parameter count: {trainable}")
    checkpoints.append(memory("adapter_installed", device))

    reference = prepare_reference(args.reference, args.reference_box)
    target = load_rgb(args.target)
    user_content = [
        {"type": "image", "image": reference},
        {"type": "image", "image": target},
        {"type": "text", "text": "Return every target matching the red-boxed reference as JSON."},
    ]
    prompt_messages = [{"role": "user", "content": user_content}]
    full_messages = [
        *prompt_messages,
        {"role": "assistant", "content": '[{"bbox_2d":[220,180,310,300]}]'},
    ]
    prompt = processor.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
        add_vision_id=True,
        return_dict=True,
        return_tensors="pt",
    )
    batch = processor.apply_chat_template(
        full_messages,
        tokenize=True,
        add_generation_prompt=False,
        add_vision_id=True,
        return_dict=True,
        return_tensors="pt",
    )
    labels = batch["input_ids"].clone()
    labels[:, : prompt["input_ids"].shape[1]] = -100
    batch["labels"] = labels
    grids = batch["image_grid_thw"].tolist()
    batch = batch.to(device=device, dtype=torch.bfloat16)
    checkpoints.append(memory("batch_on_device", device))

    torch.cuda.reset_peak_memory_stats(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad), lr=1e-4
    )
    output = model(**batch)
    checkpoints.append(memory("forward_complete", device))
    output.loss.backward()
    checkpoints.append(memory("backward_complete", device))
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    checkpoints.append(memory("optimizer_complete", device))

    result = {
        "gpu": torch.cuda.get_device_name(device),
        "capacity_gib": torch.cuda.get_device_properties(device).total_memory / 2**30,
        "tokens_per_image_cap": args.tokens_per_image,
        "max_pixels_per_image": max_pixels,
        "image_grid_thw": grids,
        "sequence_tokens": int(batch["input_ids"].shape[1]),
        "trainable_parameters": trainable,
        "attention": args.attention,
        "loss": float(output.loss.detach()),
        "checkpoints": checkpoints,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
