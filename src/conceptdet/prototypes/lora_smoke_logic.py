"""PROTOTYPE — pure topology discovery and smoke-verdict logic.

Question: can the selected Qwen3-VL rank-16 LoRA topology complete a positive
and negative training step, save/reload, and strict-JSON generation below the
44 GiB peak-reserved gate; and how does it compare with attention-only LoRA?
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

Topology = Literal["full", "attention"]

_TEXT_ATTENTION = re.compile(
    r"model\.language_model\.layers\.\d+\.self_attn\."
    r"(?:q_proj|k_proj|v_proj|o_proj)$"
)
_TEXT_MLP = re.compile(
    r"model\.language_model\.layers\.\d+\.mlp\."
    r"(?:gate_proj|up_proj|down_proj)$"
)
_MERGER = re.compile(
    r"model\.visual\.(?:merger|deepstack_merger_list\.\d+)\.linear_fc[12]$"
)


@dataclass(frozen=True)
class TopologySpec:
    name: Topology
    expected_modules: int
    expected_trainable_parameters: int


SPECS = {
    "full": TopologySpec("full", 260, 44_793_856),
    "attention": TopologySpec("attention", 152, 16_482_304),
}


def discover_targets(model: torch.nn.Module, topology: Topology) -> list[str]:
    spec = SPECS[topology]
    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if _MERGER.fullmatch(name) or _TEXT_ATTENTION.fullmatch(name):
            targets.append(name)
        elif topology == "full" and _TEXT_MLP.fullmatch(name):
            targets.append(name)
    targets.sort()
    if len(targets) != spec.expected_modules:
        raise RuntimeError(
            f"{topology}: expected {spec.expected_modules} target modules, found {len(targets)}"
        )
    return targets


def strict_detection_set(text: str) -> list[dict[str, Any]]:
    value = json.loads(text)
    if not isinstance(value, list):
        raise ValueError("completion is not a top-level JSON array")
    for item in value:
        if not isinstance(item, dict) or set(item) - {"bbox_2d", "label"}:
            raise ValueError("detection item has invalid fields")
        box = item.get("bbox_2d")
        if (
            not isinstance(box, list)
            or len(box) != 4
            or any(not isinstance(coordinate, int) for coordinate in box)
            or not (0 <= box[0] < box[2] <= 1000)
            or not (0 <= box[1] < box[3] <= 1000)
        ):
            raise ValueError("detection item has an invalid bbox_2d")
        if "label" in item and not isinstance(item["label"], str):
            raise ValueError("label must be a string")
    return value


def load_report(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def summarize_reports(report_directory: Path) -> dict[str, Any]:
    reports = {
        topology: load_report(report_directory / f"{topology}.json")
        for topology in SPECS
    }
    complete = all(report is not None for report in reports.values())
    comparison: dict[str, Any] = {"complete": complete, "reports": reports}
    if complete:
        full = reports["full"]
        attention = reports["attention"]
        assert full is not None and attention is not None
        comparison["delta"] = {
            "trainable_parameters": (
                full["trainable_parameters"] - attention["trainable_parameters"]
            ),
            "peak_reserved_gib": full["peak_reserved_gib"] - attention["peak_reserved_gib"],
            "mean_loss_improvement": (
                full["mean_loss_improvement"] - attention["mean_loss_improvement"]
            ),
        }
    return comparison
