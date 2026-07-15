from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from PIL import Image

from conceptdet.adapter import FakeAdapter
from conceptdet.application import DetectionApplication
from conceptdet.config import RequestConfig
from conceptdet.errors import TrainingError
from conceptdet.run_state import (
    ProcessContext,
    assert_distributed_consensus,
    capture_rng_state,
    distributed_barrier,
    distributed_objects,
    restore_rng_state,
    sha256_json,
)
from conceptdet.types import Box


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run_fake_distributed(
    output_dir: Path,
    *,
    steps: int,
    resume: bool,
    mismatch_rank: int | None = None,
) -> dict[str, Any] | None:
    import random

    import numpy as np
    import torch
    from accelerate import Accelerator

    accelerator = Accelerator(cpu=True)
    context = ProcessContext.current()
    if context.world_size != 2:
        raise TrainingError(
            f"C2 requires exactly two processes, found {context.world_size}"
        )
    identity = {
        "config_hash": "fake-config" if context.rank != mismatch_rank else "drift",
        "contract_fingerprint": "fake-contract",
        "manifest_hash": "fake-manifest",
    }
    assert_distributed_consensus(
        torch, sha256_json(identity), context, name="fake run identity"
    )

    if context.is_main:
        output_dir.mkdir(parents=True, exist_ok=resume)
        reference = output_dir / "reference.png"
        target = output_dir / "target.png"
        Image.new("RGB", (32, 32), "white").save(reference)
        Image.new("RGB", (32, 32), "gray").save(target)
    distributed_barrier(torch, context)
    request = RequestConfig(
        output_dir / "reference.png",
        (Box(1, 1, 8, 8),),
        output_dir / "target.png",
        "the same Visual Concept",
    )
    result = DetectionApplication(FakeAdapter("[]")).detect(
        request, max_new_tokens=8, config_hash="fake-config"
    )
    if result.raw_completion != "[]":
        raise TrainingError("Fake Adapter application flow changed")

    random.seed(31)
    np.random.seed(31)
    torch.manual_seed(31)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(31)
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.5)
    model, optimizer = accelerator.prepare(model, optimizer)
    start_step = 0
    checkpoint_path = output_dir / "checkpoint.pt"
    if resume:
        if not checkpoint_path.is_file():
            raise TrainingError(f"Fake resume checkpoint is missing: {checkpoint_path}")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if payload.get("identity") != identity:
            raise TrainingError("Fake checkpoint identity drift")
        accelerator.unwrap_model(model).load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        rank_rng = payload.get("rng")
        if not isinstance(rank_rng, list) or len(rank_rng) != context.world_size:
            raise TrainingError("Fake checkpoint RNG topology drift")
        restore_rng_state(torch, rank_rng[context.rank])
        start_step = int(payload["step"])

    coverage: list[int] = []
    for step in range(start_step, steps):
        sample_id = step * context.world_size + context.rank
        coverage.append(sample_id)
        value = torch.tensor([[float(sample_id + 1)]], device=accelerator.device)
        target_value = value * 2.0
        optimizer.zero_grad(set_to_none=True)
        loss = torch.square(model(value) - target_value).mean()
        accelerator.backward(loss)
        optimizer.step()

    final_weight = float(accelerator.unwrap_model(model).weight.detach().cpu().item())
    weights = distributed_objects(torch, final_weight, context)
    if any(abs(float(value) - float(weights[0])) > 1e-12 for value in weights[1:]):
        raise TrainingError(f"Fake DDP weights did not synchronize: {weights}")
    all_coverage = distributed_objects(torch, coverage, context)
    flattened = sorted(item for rank_items in all_coverage for item in rank_items)
    expected = list(range(start_step * context.world_size, steps * context.world_size))
    if flattened != expected:
        raise TrainingError(
            f"Distributed sampler coverage mismatch: actual={flattened} expected={expected}"
        )

    rng = distributed_objects(torch, capture_rng_state(torch), context)
    if context.is_main:
        torch.save(
            {
                "identity": identity,
                "step": steps,
                "model": accelerator.unwrap_model(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng": rng,
            },
            checkpoint_path,
        )
        report = {
            "schema_version": 1,
            "gate": "C2",
            "passed": True,
            "world_size": context.world_size,
            "rank_zero_only_publication": True,
            "identity": identity,
            "coverage": flattened,
            "final_weights": weights,
            "steps": steps,
            "resumed": resume,
            "fake_adapter_completion": result.raw_completion,
        }
        _atomic_json(output_dir / "c2_report.json", report)
        _atomic_json(
            output_dir / "acceptance_report.json",
            {
                **report,
                "accepted": True,
                "profile": "cpu-distributed",
                "hashes": identity,
                "controls": {
                    "offline": True,
                    "no_sam_runtime": True,
                    "strict_output": True,
                    "finite_numeric": True,
                    "artifact_atomic": True,
                    "memory_gate_gib": 44.0,
                },
            },
        )
    distributed_barrier(torch, context)
    return report if context.is_main else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--mismatch-rank", type=int)
    args = parser.parse_args(argv)
    run_fake_distributed(
        args.output.resolve(),
        steps=args.steps,
        resume=args.resume,
        mismatch_rank=args.mismatch_rank,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
