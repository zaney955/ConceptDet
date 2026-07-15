#!/usr/bin/env python3
"""PROTOTYPE TUI for the Detection Set reward decision."""

from __future__ import annotations

import json

from conceptdet.prototypes.detection_set_reward_logic import Box, RewardState, evaluate_reward

BOLD = "\x1b[1m"
DIM = "\x1b[2m"
RESET = "\x1b[0m"
CLEAR = "\033[2J\033[H"

TARGETS: tuple[Box, ...] = ((100, 100, 300, 300), (600, 500, 850, 800))

CASES = {
    "1": (
        "exact, reversed order",
        '[{"bbox_2d":[600,500,850,800]},{"bbox_2d":[100,100,300,300]}]',
        TARGETS,
    ),
    "2": (
        "localized but shifted",
        '[{"bbox_2d":[120,120,320,320]},{"bbox_2d":[620,520,870,820]}]',
        TARGETS,
    ),
    "3": ("one missed target", '[{"bbox_2d":[100,100,300,300]}]', TARGETS),
    "4": (
        "duplicate prediction",
        '[{"bbox_2d":[100,100,300,300]},{"bbox_2d":[102,102,298,298]},'
        '{"bbox_2d":[600,500,850,800]}]',
        TARGETS,
    ),
    "5": (
        "one extra false positive",
        '[{"bbox_2d":[100,100,300,300]},{"bbox_2d":[600,500,850,800]},'
        '{"bbox_2d":[400,100,500,200]}]',
        TARGETS,
    ),
    "6": ("correct empty prediction", "[]", ()),
    "7": ("false positive on a negative sample", '[{"bbox_2d":[100,100,300,300]}]', ()),
    "8": ("malformed/prose-wrapped output", 'Result: [{"bbox_2d":[100,100,300,300]}]', TARGETS),
}


def _format_matches(state: RewardState, hard: bool = False) -> str:
    matches = state.hard_matches if hard else state.soft_matches
    if not matches:
        return "[]"
    return "[" + ", ".join(
        f"P{match.prediction_index}→T{match.target_index}@{match.iou:.3f}" for match in matches
    ) + "]"


def _render(name: str, raw: str, targets: tuple[Box, ...], state: RewardState) -> None:
    print(CLEAR, end="")
    print(f"{BOLD}PROTOTYPE — Detection Set reward{RESET}")
    print(f"{DIM}Question: does strict JSON + optimal set matching reward errors intuitively?{RESET}\n")
    print(f"{BOLD}Case{RESET}: {name}")
    print(f"{BOLD}Prediction JSON{RESET}: {raw}")
    print(f"{BOLD}Target Detection Set{RESET}: {json.dumps(targets)}")
    print(f"{BOLD}Valid schema{RESET}: {state.valid_json}")
    print(f"{BOLD}Parse error{RESET}: {state.error or 'none'}")
    print(f"{BOLD}Parsed predictions{RESET}: {state.predictions}")
    print(f"{BOLD}IoU matrix (prediction rows){RESET}: {state.iou_matrix}")
    print(f"{BOLD}Optimal soft matches{RESET}: {_format_matches(state)}")
    print(f"{BOLD}IoU≥0.5 matches{RESET}: {_format_matches(state, hard=True)}")
    print(f"{BOLD}Soft precision / recall / F1{RESET}: "
          f"{state.soft_precision:.3f} / {state.soft_recall:.3f} / {state.soft_f1:.3f}")
    print(f"{BOLD}TP / FP / FN @0.5{RESET}: "
          f"{state.true_positives_50} / {state.false_positives_50} / {state.false_negatives_50}")
    print(f"{BOLD}F1 @0.5{RESET}: {state.f1_50:.3f}")
    print(f"{BOLD}Format reward (10%){RESET}: {state.format_reward:.3f}")
    print(f"{BOLD}Soft-set quality (90%){RESET}: {state.quality_reward:.3f}")
    print(f"{BOLD}Suggested total reward{RESET}: {state.total_reward:.3f}\n")
    print(f"{BOLD}Cases{RESET}: " + "  ".join(f"[{key}] {value[0]}" for key, value in CASES.items()))
    print(f"{BOLD}[c]{RESET} custom JSON/targets   {BOLD}[q]{RESET} quit")


def _custom_case() -> tuple[str, str, tuple[Box, ...]]:
    print(CLEAR, end="")
    raw = input("Prediction JSON array: ")
    target_raw = input("Target boxes JSON, e.g. [[100,100,300,300]]: ")
    try:
        values = json.loads(target_raw)
        targets = tuple(tuple(value) for value in values)
    except (json.JSONDecodeError, TypeError):
        targets = ()
    return "custom", raw, targets  # type: ignore[return-value]


def main() -> None:
    name, raw, targets = CASES["1"]
    while True:
        state = evaluate_reward(raw, targets)
        _render(name, raw, targets, state)
        choice = input("> ").strip().lower()
        if choice == "q":
            return
        if choice == "c":
            name, raw, targets = _custom_case()
        elif choice in CASES:
            name, raw, targets = CASES[choice]


if __name__ == "__main__":
    main()
