from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from conceptdet.errors import TrainingError

CHECKPOINT_SCHEMA_VERSION = 1
_CHECKPOINT = re.compile(r"checkpoint-(\d+)$")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def code_fingerprint(*paths: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((item.resolve() for item in paths), key=str):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class RunIdentity:
    stage: Literal["sft", "grpo", "fake"]
    config_hash: str
    dataset_fingerprint: str
    contract_fingerprint: str
    parent_artifact_fingerprint: str | None
    semantics_fingerprint: str
    world_size: int

    def __post_init__(self) -> None:
        if self.world_size < 1:
            raise TrainingError("Checkpoint world_size must be at least one")

    @property
    def fingerprint(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True)
class ProcessContext:
    rank: int
    local_rank: int
    world_size: int

    @classmethod
    def current(cls) -> ProcessContext:
        return cls(
            int(os.environ.get("RANK", "0")),
            int(os.environ.get("LOCAL_RANK", "0")),
            int(os.environ.get("WORLD_SIZE", "1")),
        )

    def __post_init__(self) -> None:
        if self.world_size < 1 or not 0 <= self.rank < self.world_size:
            raise TrainingError(
                f"Invalid distributed topology rank={self.rank} world_size={self.world_size}"
            )
        if self.local_rank < 0:
            raise TrainingError(f"Invalid distributed local rank: {self.local_rank}")

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    def cuda_device(self, configured: str) -> str:
        if self.distributed:
            return f"cuda:{self.local_rank}"
        return "cuda:0" if configured == "auto" else configured


def _checkpoint_step(path: Path) -> int | None:
    match = _CHECKPOINT.fullmatch(path.name)
    return int(match.group(1)) if match else None


def complete_checkpoints(work_dir: Path) -> list[Path]:
    if not work_dir.is_dir():
        return []
    candidates: list[tuple[int, Path]] = []
    for path in work_dir.glob("checkpoint-*"):
        step = _checkpoint_step(path)
        if step is not None and path.is_dir() and (path / "complete").is_file():
            candidates.append((step, path.resolve()))
    candidates.sort()
    if len({step for step, _ in candidates}) != len(candidates):
        raise TrainingError(f"Ambiguous checkpoint steps in {work_dir}")
    return [path for _, path in candidates]


def resolve_resume(
    work_dir: Path,
    resume: Literal["none", "auto"] | Path,
    *,
    stage: str,
) -> Path | None:
    work_dir = work_dir.resolve()
    if resume == "none":
        if work_dir.exists() and any(work_dir.iterdir()):
            raise TrainingError(
                f"{stage.upper()} work directory is not empty; use --resume auto or a "
                f"checkpoint: {work_dir}"
            )
        return None
    if resume == "auto":
        candidates = complete_checkpoints(work_dir)
        if not candidates:
            raise TrainingError(f"No complete resumable checkpoint in {work_dir}")
        return candidates[-1]
    checkpoint = Path(resume).expanduser().resolve()
    if _checkpoint_step(checkpoint) is None or not (checkpoint / "complete").is_file():
        raise TrainingError(f"Resume checkpoint is incomplete or invalid: {checkpoint}")
    return checkpoint


def write_checkpoint_metadata(
    directory: Path,
    identity: RunIdentity,
    state: dict[str, Any],
) -> None:
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "identity": asdict(identity),
        "identity_fingerprint": identity.fingerprint,
        "state": state,
    }
    (directory / "run_state.json").write_bytes(_canonical_json(payload) + b"\n")


def load_checkpoint_metadata(
    checkpoint: Path,
    identity: RunIdentity,
) -> dict[str, Any]:
    path = checkpoint / "run_state.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingError(f"Cannot read checkpoint metadata: {checkpoint}") from exc
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise TrainingError(f"Unsupported checkpoint schema in {checkpoint}")
    stored_identity = payload.get("identity")
    stored_fingerprint = payload.get("identity_fingerprint")
    if not isinstance(stored_identity, dict) or stored_fingerprint != sha256_json(
        stored_identity
    ):
        raise TrainingError(f"Checkpoint identity fingerprint is invalid: {checkpoint}")
    expected = asdict(identity)
    for key, expected_value in expected.items():
        if stored_identity.get(key) != expected_value:
            raise TrainingError(
                f"Checkpoint {key}={stored_identity.get(key)!r} does not match "
                f"{expected_value!r}"
            )
    state = payload.get("state")
    if not isinstance(state, dict):
        raise TrainingError(f"Checkpoint state is invalid: {checkpoint}")
    return state


def capture_rng_state(torch: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state()
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:
        pass
    return state


def restore_rng_state(torch: Any, state: dict[str, Any]) -> None:
    try:
        random.setstate(state["python"])
        torch.set_rng_state(state["torch_cpu"])
        if torch.cuda.is_available() and "torch_cuda" in state:
            torch.cuda.set_rng_state(state["torch_cuda"])
        if "numpy" in state:
            import numpy as np

            np.random.set_state(state["numpy"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TrainingError("Checkpoint RNG state is invalid") from exc


def atomic_checkpoint_directory(work_dir: Path, step: int) -> tuple[Path, Path]:
    target = work_dir / f"checkpoint-{step:08d}"
    if target.exists():
        raise TrainingError(f"Checkpoint already exists: {target}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=work_dir))
    return temporary, target


def publish_checkpoint(temporary: Path, target: Path) -> None:
    try:
        (temporary / "complete").write_text("complete\n", encoding="utf-8")
        os.replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def distributed_objects(torch: Any, value: object, context: ProcessContext) -> list[object]:
    if not context.distributed:
        return [value]
    if not torch.distributed.is_initialized():
        raise TrainingError("Distributed process group is not initialized")
    values: list[object] = [None] * context.world_size
    torch.distributed.all_gather_object(values, value)
    return values


def assert_distributed_consensus(
    torch: Any,
    value: object,
    context: ProcessContext,
    *,
    name: str,
) -> None:
    values = distributed_objects(torch, value, context)
    if any(item != values[0] for item in values[1:]):
        raise TrainingError(f"Cross-rank {name} mismatch: {values}")


def distributed_barrier(torch: Any, context: ProcessContext) -> None:
    if context.distributed:
        if not torch.distributed.is_initialized():
            raise TrainingError("Distributed process group is not initialized")
        torch.distributed.barrier()
