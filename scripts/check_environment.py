#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import os
import sys
from pathlib import Path


EXPECTED_VERSIONS = {
    "accelerate": "1.14.0",
    "flash-attn": "2.8.3.post1",
    "numpy": "2.5.1",
    "pillow": "12.3.0",
    "safetensors": "0.8.0",
    "sentencepiece": "0.2.2",
    "torch": "2.13.0",
    "torchvision": "0.28.0",
    "transformers": "5.13.1",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the independent ConceptDet runtime")
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail unless CUDA and at least one GPU are available",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    expected_prefix = Path(
        os.environ.get("CONCEPTDET_ENV_DIR", repo_root / ".venv")
    ).resolve()
    active_prefix = Path(sys.prefix).resolve()
    if active_prefix != expected_prefix:
        raise RuntimeError(
            f"Wrong Python environment: {active_prefix}; expected {expected_prefix}"
        )

    mismatches: list[str] = []
    for package, expected in EXPECTED_VERSIONS.items():
        try:
            installed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            installed = "not installed"
        print(f"{package}={installed}")
        if installed.split("+")[0] != expected:
            mismatches.append(f"{package}: expected {expected}, found {installed}")
    if mismatches:
        raise RuntimeError("Version mismatch:\n  " + "\n  ".join(mismatches))

    import flash_attn  # noqa: F401
    import torch
    import torchvision  # noqa: F401
    from transformers import Qwen2_5_VLForConditionalGeneration  # noqa: F401

    print(f"python={sys.version.split()[0]}")
    print(f"environment={active_prefix}")
    print(f"cuda_runtime={torch.version.cuda}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but no CUDA GPU is available")
    print("ConceptDet environment check: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
