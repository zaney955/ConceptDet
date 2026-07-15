#!/usr/bin/env python3
"""PROTOTYPE TUI for Qwen3-VL two-image visual-token budgets."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps
from transformers import AutoProcessor

from conceptdet.prototypes.token_budget_logic import Budget, processed_image

BOLD = "\x1b[1m"
DIM = "\x1b[2m"
RESET = "\x1b[0m"
CLEAR = "\033[2J\033[H"
CANDIDATES = (256, 512, 640, 768, 1024, 1280)


def parse_box(value: str) -> tuple[int, int, int, int]:
    coords = tuple(int(part) for part in value.split(","))
    if len(coords) != 4:
        raise argparse.ArgumentTypeError("box must be x1,y1,x2,y2")
    return coords  # type: ignore[return-value]


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as opened:
        return ImageOps.exif_transpose(opened).convert("RGB")


def prepare_reference(path: Path, boxes: list[tuple[int, int, int, int]]) -> Image.Image:
    image = load_rgb(path)
    draw = ImageDraw.Draw(image)
    for box in boxes:
        draw.rectangle(box, outline="red", width=8)
    return image


def scan(model: str, reference: Image.Image, target: Image.Image) -> list[dict[str, object]]:
    rows = []
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": reference},
            {"type": "image", "image": target},
            {"type": "text", "text": "Return every target matching the red-boxed reference."},
        ],
    }]
    for token_cap in CANDIDATES:
        budget = Budget(token_cap)
        processor = AutoProcessor.from_pretrained(
            model,
            min_pixels=64 * 32 * 32,
            max_pixels=budget.max_pixels,
            local_files_only=True,
        )
        batch = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            add_vision_id=True,
            return_dict=True,
            return_tensors="pt",
        )
        images = [processed_image(grid.tolist()) for grid in batch["image_grid_thw"]]
        rows.append({
            "cap": token_cap,
            "max_pixels": budget.max_pixels,
            "reference": images[0],
            "target": images[1],
            "total_visual_tokens": sum(image.visual_tokens for image in images),
            "sequence_tokens": int(batch["input_ids"].shape[1]),
        })
    return rows


def render(rows: list[dict[str, object]], selected: int, reference: Image.Image, target: Image.Image) -> None:
    print(CLEAR, end="")
    print(f"{BOLD}PROTOTYPE — Qwen3-VL two-image token budget{RESET}")
    print(f"{DIM}Question: which shared cap balances detail and a 44 GiB training gate?{RESET}\n")
    print(f"{BOLD}Original reference{RESET}: {reference.width}×{reference.height}")
    print(f"{BOLD}Original target{RESET}:    {target.width}×{target.height}\n")
    print("    cap   reference resize/tokens      target resize/tokens         total visual   sequence")
    for index, row in enumerate(rows):
        ref = row["reference"]
        tgt = row["target"]
        marker = ">" if index == selected else " "
        print(
            f"{marker} [{index + 1}] {row['cap']:4d}   "
            f"{ref.processed_width:4d}×{ref.processed_height:<4d}/{ref.visual_tokens:<4d}       "
            f"{tgt.processed_width:4d}×{tgt.processed_height:<4d}/{tgt.visual_tokens:<4d}       "
            f"{row['total_visual_tokens']:<12d}   {row['sequence_tokens']}"
        )
    chosen = rows[selected]
    print(f"\n{BOLD}Selected shared cap{RESET}: {chosen['cap']} tokens/image")
    print(f"{BOLD}Selected max_pixels{RESET}: {chosen['max_pixels']:,} pixels/image")
    print(f"{BOLD}Observed pair tokens{RESET}: {chosen['total_visual_tokens']}")
    print(f"\n{BOLD}[1–{len(CANDIDATES)}]{RESET} select candidate   {BOLD}[q]{RESET} quit")
    print(f"{DIM}GPU peak measurements are recorded by the companion profile pass.{RESET}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--reference-box", action="append", type=parse_box, default=[])
    args = parser.parse_args()
    reference = prepare_reference(args.reference, args.reference_box)
    target = load_rgb(args.target)
    rows = scan(args.model, reference, target)
    selected = 2
    while True:
        render(rows, selected, reference, target)
        choice = input("> ").strip().lower()
        if choice == "q":
            return
        if choice in {str(index) for index in range(1, len(CANDIDATES) + 1)}:
            selected = int(choice) - 1


if __name__ == "__main__":
    main()
