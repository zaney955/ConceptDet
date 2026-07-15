from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import re
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from PIL import Image, ImageOps

from conceptdet.adapter import AdapterInput
from conceptdet.application import DetectionApplication
from conceptdet.artifact import (
    MODEL_ID,
    MODEL_REVISION,
    TARGET_MODULES_SHA256,
    AdapterArtifact,
    default_contract,
    initialize_artifact,
)
from conceptdet.config import ArtifactInitConfig, RequestConfig, SFTStageConfig
from conceptdet.dataset import (
    DatasetArtifact,
    detection_request,
    validate_training_dataset,
)
from conceptdet.errors import DatasetError, TrainingError
from conceptdet.model import (
    MAX_TOTAL_SEQUENCE_TOKENS,
    MAX_VISUAL_TOKENS,
    MIN_VISUAL_TOKENS,
    Qwen3VLAdapter,
    prepare_images,
    smart_image_size,
)
from conceptdet.peft_weights import load_exact_adapter_weights
from conceptdet.prompts import build_messages
from conceptdet.protocol import parse_detection_set, serialize_detection_set
from conceptdet.run_state import (
    ProcessContext,
    RunIdentity,
    atomic_checkpoint_directory,
    capture_rng_state,
    code_fingerprint,
    complete_checkpoints,
    distributed_barrier,
    distributed_objects,
    load_checkpoint_metadata,
    publish_checkpoint,
    resolve_resume,
    restore_rng_state,
    write_checkpoint_metadata,
)

EXPECTED_TARGET_MODULES = 260
EXPECTED_TRAINABLE_PARAMETERS = 44_793_856
MEMORY_GATE_GIB = 44.0
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
class BatchProvenance:
    record_id: str
    positive: bool
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    image_grids: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True)
class SFTResult:
    artifact: AdapterArtifact
    optimizer_steps: int
    micro_steps: int
    final_loss: float
    peak_reserved_gib: float
    positive_completion: str
    negative_completion: str
    lifecycle_report: Path


def discover_lora_targets(model: Any) -> list[str]:
    try:
        import torch
    except ImportError as exc:
        raise TrainingError("PyTorch is required for SFT") from exc
    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if _MERGER.fullmatch(name) or _TEXT_ATTENTION.fullmatch(name) or _TEXT_MLP.fullmatch(name):
            targets.append(name)
    targets.sort()
    if len(targets) != EXPECTED_TARGET_MODULES:
        raise TrainingError(
            f"Expected {EXPECTED_TARGET_MODULES} LoRA targets, found {len(targets)}"
        )
    digest = hashlib.sha256("\n".join(targets).encode()).hexdigest()
    if digest != TARGET_MODULES_SHA256:
        raise TrainingError(f"LoRA target topology hash mismatch: {digest}")
    return targets


def _load_rgb(path: Path) -> Image.Image:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(path) as opened:
                return ImageOps.exif_transpose(opened).convert("RGB")
    except OSError as exc:
        raise DatasetError(f"Cannot read training image: {path}") from exc


class SFTBatchBuilder:
    """Deep internal module for model-visible SFT preprocessing and label masking."""

    def __init__(self, dataset: DatasetArtifact, processor: Any) -> None:
        self.dataset = dataset
        self.processor = processor
        self._prepared_size_cache: dict[Path, tuple[int, int]] = {}

    def _prepared_size(self, path: Path) -> tuple[int, int]:
        cached = self._prepared_size_cache.get(path)
        if cached is not None:
            return cached
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", Image.DecompressionBombWarning)
                with Image.open(path) as opened:
                    width, height = opened.size
                    orientation = int(opened.getexif().get(274, 1))
        except (OSError, TypeError, ValueError) as exc:
            raise DatasetError(f"Cannot read training image metadata: {path}") from exc
        if orientation in {5, 6, 7, 8}:
            width, height = height, width
        prepared = smart_image_size((width, height))
        self._prepared_size_cache[path] = prepared
        return prepared

    def build(self, record: dict[str, Any]) -> tuple[Any, BatchProvenance]:
        try:
            import torch
        except ImportError as exc:
            raise TrainingError("PyTorch is required for SFT batches") from exc
        request = detection_request(self.dataset, record)
        reference, target = prepare_images(
            AdapterInput(
                _load_rgb(request.reference_image),
                request.reference_boxes,
                _load_rgb(request.target_image),
                request.query,
            )
        )
        raw_detection_set = record.get("detection_set")
        if not isinstance(raw_detection_set, list):
            raise DatasetError(f"Training record has no Detection Set: {record.get('id')}")
        answer = json.dumps(
            raw_detection_set, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        detections = parse_detection_set(answer)
        if serialize_detection_set(detections) != answer:
            raise DatasetError(f"Training record is not canonically serialized: {record.get('id')}")
        prompt_messages = build_messages(reference, target, request.query)
        full_messages = [*prompt_messages, {"role": "assistant", "content": answer}]
        common = {
            "tokenize": True,
            "add_vision_id": True,
            "return_dict": True,
            "return_tensors": "pt",
            "processor_kwargs": {"do_resize": False},
        }
        prompt = self.processor.apply_chat_template(
            prompt_messages, add_generation_prompt=True, **common
        )
        batch = self.processor.apply_chat_template(
            full_messages, add_generation_prompt=False, **common
        )
        prompt_tokens = int(prompt["input_ids"].shape[1])
        total_tokens = int(batch["input_ids"].shape[1])
        if not torch.equal(prompt["input_ids"], batch["input_ids"][:, :prompt_tokens]):
            raise TrainingError("Prompt tokens are not a prefix of the SFT conversation")
        completion_tokens = total_tokens - prompt_tokens
        if total_tokens > MAX_TOTAL_SEQUENCE_TOKENS:
            raise DatasetError(
                f"Record {record.get('id')} has {total_tokens} tokens; "
                f"maximum is {MAX_TOTAL_SEQUENCE_TOKENS} and truncation is forbidden"
            )
        if not 1 <= completion_tokens <= 192:
            raise DatasetError(
                f"Record {record.get('id')} completion has {completion_tokens} tokens; "
                "expected 1-192"
            )
        labels = batch["input_ids"].clone()
        labels[:, :prompt_tokens] = -100
        batch["labels"] = labels
        grids = tuple(
            tuple(int(value) for value in row) for row in batch["image_grid_thw"]
        )
        if len(grids) != 2:
            raise TrainingError(f"Expected two image grids, found {grids}")
        for grid in grids:
            visual_tokens = grid[0] * grid[1] * grid[2] // 4
            if not MIN_VISUAL_TOKENS <= visual_tokens <= MAX_VISUAL_TOKENS:
                raise TrainingError(
                    f"Image grid {grid} produces {visual_tokens} visual tokens outside contract"
                )
        return batch, BatchProvenance(
            str(record["id"]),
            bool(record["positive"]),
            total_tokens,
            prompt_tokens,
            completion_tokens,
            grids,
        )

    def preflight(self, record: dict[str, Any]) -> BatchProvenance:
        """Compute exact token/grid limits without allocating model pixel tensors."""

        request = detection_request(self.dataset, record)
        raw_detection_set = record.get("detection_set")
        if not isinstance(raw_detection_set, list):
            raise DatasetError(f"Training record has no Detection Set: {record.get('id')}")
        answer = json.dumps(
            raw_detection_set, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        detections = parse_detection_set(answer)
        if serialize_detection_set(detections) != answer:
            raise DatasetError(
                f"Training record is not canonically serialized: {record.get('id')}"
            )

        sizes = [
            self._prepared_size(path)
            for path in (request.reference_image, request.target_image)
        ]
        patch_size = int(self.processor.image_processor.patch_size)
        merge_size = int(self.processor.image_processor.merge_size)
        grids = tuple((1, height // patch_size, width // patch_size) for width, height in sizes)
        visual_tokens = [grid[1] * grid[2] // (merge_size**2) for grid in grids]
        if any(
            not MIN_VISUAL_TOKENS <= count <= MAX_VISUAL_TOKENS
            for count in visual_tokens
        ):
            raise TrainingError(f"Preflight image grids produce invalid tokens: {grids}")

        placeholder = Image.new("RGB", (1, 1))
        prompt_messages = build_messages(placeholder, placeholder, request.query)
        full_messages = [*prompt_messages, {"role": "assistant", "content": answer}]

        def expanded_tokens(messages: list[dict[str, Any]], generation: bool) -> int:
            rendered = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=generation,
                add_vision_id=True,
            )
            token_ids = self.processor.tokenizer(
                rendered, add_special_tokens=False
            )["input_ids"]
            image_token_id = int(self.processor.image_token_id)
            if token_ids.count(image_token_id) != 2:
                raise TrainingError("SFT preflight expected exactly two image placeholders")
            return len(token_ids) - 2 + sum(visual_tokens)

        prompt_tokens = expanded_tokens(prompt_messages, True)
        total_tokens = expanded_tokens(full_messages, False)
        completion_tokens = total_tokens - prompt_tokens
        if total_tokens > MAX_TOTAL_SEQUENCE_TOKENS:
            raise DatasetError(
                f"Record {record.get('id')} has {total_tokens} tokens; "
                f"maximum is {MAX_TOTAL_SEQUENCE_TOKENS} and truncation is forbidden"
            )
        if not 1 <= completion_tokens <= 192:
            raise DatasetError(
                f"Record {record.get('id')} completion has {completion_tokens} tokens; "
                "expected 1-192"
            )
        return BatchProvenance(
            str(record["id"]),
            bool(record["positive"]),
            total_tokens,
            prompt_tokens,
            completion_tokens,
            grids,
        )


def _ordered_records(records: list[dict[str, Any]], seed: int, epoch: int) -> list[dict[str, Any]]:
    positives = sorted(
        (row for row in records if bool(row["positive"])),
        key=lambda row: hashlib.sha256(f"{seed}:{epoch}:p:{row['id']}".encode()).hexdigest(),
    )
    negatives = sorted(
        (row for row in records if not bool(row["positive"])),
        key=lambda row: hashlib.sha256(f"{seed}:{epoch}:n:{row['id']}".encode()).hexdigest(),
    )
    if not positives or not negatives:
        raise DatasetError("SFT training split must contain positive and negative records")
    ratio = max(1, len(positives) // len(negatives))
    ordered: list[dict[str, Any]] = []
    positive_index = 0
    for negative in negatives:
        ordered.extend(positives[positive_index : positive_index + ratio])
        positive_index += ratio
        ordered.append(negative)
    ordered.extend(positives[positive_index:])
    return ordered


def _schedule(records: list[dict[str, Any]], epochs: float, seed: int) -> list[dict[str, Any]]:
    whole_epochs = int(epochs)
    fraction = epochs - whole_epochs
    scheduled: list[dict[str, Any]] = []
    for epoch in range(whole_epochs):
        scheduled.extend(_ordered_records(records, seed, epoch))
    if fraction:
        partial = _ordered_records(records, seed, whole_epochs)
        scheduled.extend(partial[: math.ceil(len(partial) * fraction)])
    return scheduled


def _memory(torch: Any, device: Any, checkpoint: str) -> dict[str, float | str]:
    torch.cuda.synchronize(device)
    free, total = torch.cuda.mem_get_info(device)
    return {
        "checkpoint": checkpoint,
        "allocated_gib": torch.cuda.memory_allocated(device) / 2**30,
        "reserved_gib": torch.cuda.memory_reserved(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        "device_used_gib": (total - free) / 2**30,
    }


def _checkpoint_path(work_dir: Path, optimizer_step: int) -> Path:
    return work_dir / f"checkpoint-{optimizer_step:08d}"


def _latest_checkpoint(work_dir: Path) -> Path | None:
    candidates = complete_checkpoints(work_dir)
    return candidates[-1] if candidates else None


def _resolve_resume(work_dir: Path, resume: Literal["none", "auto"] | Path) -> Path | None:
    return resolve_resume(work_dir, resume, stage="sft")


def _run_identity(
    config: SFTStageConfig, dataset: DatasetArtifact, context: ProcessContext
) -> RunIdentity:
    return RunIdentity(
        "sft",
        config.config_hash,
        dataset.fingerprint,
        str(default_contract()["contract_fingerprint"]),
        None,
        code_fingerprint(
            Path(__file__),
            Path(__file__).with_name("model.py"),
            Path(__file__).with_name("peft_weights.py"),
            Path(__file__).with_name("prompts.py"),
            Path(__file__).with_name("protocol.py"),
        ),
        context.world_size,
    )


def _compatible_sft_schedule(
    schedule: list[dict[str, Any]],
    builder: SFTBatchBuilder,
    *,
    required_micro_steps: int | None,
    require_both_kinds: bool = True,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Validate the exact model-visible sequence without ever truncating a record."""

    compatible: list[dict[str, Any]] = []
    excluded: list[str] = []
    selected: dict[bool, dict[str, Any]] = {}
    for record in schedule:
        try:
            builder.preflight(record)
        except DatasetError as exc:
            message = str(exc)
            if "truncation is forbidden" not in message and "expected 1-192" not in message:
                raise
            excluded.append(str(record["id"]))
            continue
        compatible.append(record)
        selected.setdefault(bool(record["positive"]), record)
        enough_kinds = not require_both_kinds or set(selected) == {False, True}
        if (
            required_micro_steps is not None
            and len(compatible) >= required_micro_steps
            and enough_kinds
        ):
            break
    if required_micro_steps is not None and len(compatible) < required_micro_steps:
        raise TrainingError(
            f"Only {len(compatible)} compatible SFT records remain; "
            f"{required_micro_steps} are required"
        )
    if not compatible:
        raise TrainingError("SFT has no records within the 1,536-token contract")
    if required_micro_steps is not None:
        if require_both_kinds and set(selected) != {False, True}:
            raise TrainingError("SFT smoke schedule has no compatible positive/negative pair")
        if require_both_kinds:
            front = [selected[True], selected[False]]
            compatible = [
                *front,
                *(record for record in compatible if record not in front),
            ]
        compatible = compatible[:required_micro_steps]
    return compatible, excluded


def _save_checkpoint(
    model: Any,
    optimizer: Any,
    scheduler: Any,
    work_dir: Path,
    state: dict[str, Any],
    identity: RunIdentity,
    rng_states: list[object] | None = None,
) -> Path:
    target = _checkpoint_path(work_dir, int(state["optimizer_step"]))
    temporary, target = atomic_checkpoint_directory(
        work_dir, int(state["optimizer_step"])
    )
    try:
        model.save_pretrained(temporary, safe_serialization=True)
        import torch

        torch.save(
            {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
            temporary / "optimizer.pt",
        )
        torch.save(
            {"ranks": rng_states or [capture_rng_state(torch)]},
            temporary / "rng.pt",
        )
        write_checkpoint_metadata(temporary, identity, state)
        publish_checkpoint(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target


def _load_state(
    checkpoint: Path, identity: RunIdentity
) -> dict[str, Any]:
    return load_checkpoint_metadata(checkpoint, identity)


def _request_from_training_record(
    dataset: DatasetArtifact, record: dict[str, Any]
) -> RequestConfig:
    return dataset.detection_request(record)


def run_sft(
    config: SFTStageConfig, *, resume: Literal["none", "auto"] | Path = "none"
) -> SFTResult:
    try:
        import torch
        from accelerate import Accelerator
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    except ImportError as exc:
        raise TrainingError(
            "SFT dependencies are unavailable; install the default runtime and flash-attn"
        ) from exc
    if not torch.cuda.is_available():
        raise TrainingError("Qwen3-VL SFT requires CUDA")
    context = ProcessContext.current()
    accelerator = Accelerator() if context.distributed else None
    device_name = context.cuda_device(config.runtime.device)
    if not device_name.startswith("cuda"):
        raise TrainingError("Qwen3-VL SFT requires a CUDA device")
    if config.runtime.dtype not in {"auto", "bfloat16"}:
        raise TrainingError("Qwen3-VL SFT supports bfloat16 only")
    device = accelerator.device if accelerator is not None else torch.device(device_name)
    torch.cuda.set_device(device)
    torch.manual_seed(config.optimization.seed)
    torch.cuda.manual_seed_all(config.optimization.seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    dataset = DatasetArtifact.load(config.dataset_dir)
    validation = validate_training_dataset(dataset)
    identity = _run_identity(config, dataset, context)
    records = list(dataset.iter_records("train"))
    schedule = _schedule(records, config.optimization.epochs, config.optimization.seed)
    if not schedule:
        raise TrainingError("SFT schedule is empty")
    if context.distributed:
        schedule = schedule[context.rank :: context.world_size]
        if not schedule:
            raise TrainingError(
                f"SFT rank {context.rank} received no records for world_size={context.world_size}"
            )
    resume_path = _resolve_resume(config.work_dir, resume)
    if config.artifact_dir.exists():
        raise TrainingError(f"SFT Artifact output already exists: {config.artifact_dir}")
    config.work_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = config.work_dir / "metrics.jsonl"
    memory_points = [_memory(torch, device, "process_start")]
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        min_pixels=65_536,
        max_pixels=655_360,
        local_files_only=config.runtime.local_files_only,
    )
    builder = SFTBatchBuilder(dataset, processor)
    required_micro_steps = (
        config.optimization.max_steps
        * config.optimization.gradient_accumulation_steps
        if config.optimization.max_steps is not None
        else None
    )
    schedule, excluded_overlength_records = _compatible_sft_schedule(
        schedule,
        builder,
        required_micro_steps=required_micro_steps,
        require_both_kinds=not context.distributed,
    )
    schedule_lengths = distributed_objects(torch, len(schedule), context)
    padded_record_ids: list[str] = []
    maximum_schedule_length = max(int(value) for value in schedule_lengths)
    while len(schedule) < maximum_schedule_length:
        schedule.append(schedule[len(padded_record_ids) % len(schedule)])
        padded_record_ids.append(str(schedule[-1]["id"]))
    attention = config.runtime.attention
    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            dtype=torch.bfloat16,
            attn_implementation=attention,
            device_map={"": device_name},
            low_cpu_mem_usage=True,
            local_files_only=config.runtime.local_files_only,
        )
    except torch.OutOfMemoryError as exc:
        raise TrainingError(
            f"CUDA out of memory while loading Qwen3-VL on {device_name}; "
            "select a GPU with at least 44 GiB free"
        ) from exc
    model.config.use_cache = False
    memory_points.append(_memory(torch, device, "base_loaded"))
    if resume_path is None:
        targets = discover_lora_targets(model)
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
        state = {
            "schema_version": 1,
            "config_hash": config.config_hash,
            "dataset_fingerprint": dataset.fingerprint,
            "base_model_revision": MODEL_REVISION,
            "schedule_index": 0,
            "micro_step": 0,
            "optimizer_step": 0,
            "seen_positive": False,
            "seen_negative": False,
        }
    else:
        state = _load_state(resume_path, identity)
        model = PeftModel.from_pretrained(model, resume_path, is_trainable=True)
        try:
            load_exact_adapter_weights(model, resume_path)
        except ValueError as exc:
            raise TrainingError(f"Cannot restore exact SFT adapter weights: {exc}") from exc
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    model.train()
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if trainable != EXPECTED_TRAINABLE_PARAMETERS:
        raise TrainingError(
            f"Expected {EXPECTED_TRAINABLE_PARAMETERS} trainable parameters, found {trainable}"
        )
    unexpected = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" not in name
    ]
    if unexpected:
        raise TrainingError(f"Non-LoRA parameters are trainable: {unexpected[:10]}")
    memory_points.append(_memory(torch, device, "adapter_installed"))

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.optimization.learning_rate,
        weight_decay=config.optimization.weight_decay,
    )
    warmup = config.optimization.warmup_steps
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, (step + 1) / warmup) if warmup else 1.0,
    )
    if resume_path is not None:
        payload = torch.load(resume_path / "optimizer.pt", map_location=device, weights_only=True)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        rng_payload = torch.load(
            resume_path / "rng.pt", map_location="cpu", weights_only=False
        )
        rank_states = rng_payload.get("ranks") if isinstance(rng_payload, dict) else None
        if not isinstance(rank_states, list) or len(rank_states) != context.world_size:
            raise TrainingError("Checkpoint RNG topology does not match this run")
        restore_rng_state(torch, rank_states[context.rank])
    if accelerator is not None:
        model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    accumulation = config.optimization.gradient_accumulation_steps
    maximum_steps = config.optimization.max_steps
    final_loss = float("nan")
    seen_positive = bool(state.get("seen_positive", False))
    seen_negative = bool(state.get("seen_negative", False))
    optimizer.zero_grad(set_to_none=True)

    for schedule_index in range(int(state["schedule_index"]), len(schedule)):
        if maximum_steps is not None and int(state["optimizer_step"]) >= maximum_steps:
            break
        record = schedule[schedule_index]
        batch, provenance = builder.build(record)
        batch = batch.to(device=device, dtype=torch.bfloat16)
        try:
            output = model(**batch)
            loss = output.loss / accumulation
            if accelerator is None:
                loss.backward()
            else:
                accelerator.backward(loss)
        except torch.OutOfMemoryError as exc:
            raise TrainingError(
                f"CUDA out of memory on record {provenance.record_id}; "
                "the SFT batch contract requires one sample per device"
            ) from exc
        final_loss = float(output.loss.detach())
        seen_positive |= provenance.positive
        seen_negative |= not provenance.positive
        state["seen_positive"] = seen_positive
        state["seen_negative"] = seen_negative
        state["schedule_index"] = schedule_index + 1
        state["micro_step"] = int(state["micro_step"]) + 1
        should_step = int(state["micro_step"]) % accumulation == 0
        is_final_record = schedule_index + 1 == len(schedule)
        if should_step or is_final_record:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            state["optimizer_step"] = int(state["optimizer_step"]) + 1
            metric = {
                "optimizer_step": state["optimizer_step"],
                "micro_step": state["micro_step"],
                "record_id": provenance.record_id,
                "positive": provenance.positive,
                "loss": final_loss,
                "learning_rate": scheduler.get_last_lr()[0],
                "total_tokens": provenance.total_tokens,
                "completion_tokens": provenance.completion_tokens,
                "image_grids": [list(grid) for grid in provenance.image_grids],
            }
            if context.is_main:
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(metric, sort_keys=True) + "\n")
            memory_points.append(
                _memory(torch, device, f"optimizer_step_{state['optimizer_step']}")
            )
            if int(state["optimizer_step"]) % config.optimization.checkpoint_steps == 0:
                seen_by_rank = distributed_objects(
                    torch, (seen_positive, seen_negative), context
                )
                seen_positive = any(bool(value[0]) for value in seen_by_rank)
                seen_negative = any(bool(value[1]) for value in seen_by_rank)
                state["seen_positive"] = seen_positive
                state["seen_negative"] = seen_negative
                rng_states = distributed_objects(
                    torch, capture_rng_state(torch), context
                )
                if context.is_main:
                    checkpoint_model = (
                        accelerator.unwrap_model(model)
                        if accelerator is not None
                        else model
                    )
                    _save_checkpoint(
                        checkpoint_model,
                        optimizer,
                        scheduler,
                        config.work_dir,
                        dict(state),
                        identity,
                        rng_states,
                    )
                distributed_barrier(torch, context)
        del batch, output, loss

    if maximum_steps is not None and int(state["optimizer_step"]) < maximum_steps:
        raise TrainingError(
            f"SFT schedule ended at {state['optimizer_step']} before max_steps={maximum_steps}"
        )
    seen_by_rank = distributed_objects(torch, (seen_positive, seen_negative), context)
    seen_positive = any(bool(value[0]) for value in seen_by_rank)
    seen_negative = any(bool(value[1]) for value in seen_by_rank)
    if not seen_positive or not seen_negative:
        raise TrainingError("This SFT run did not consume both positive and negative examples")
    final_peft = config.work_dir / "final-peft"
    if final_peft.exists():
        raise TrainingError(f"Final PEFT output already exists: {final_peft}")
    final_model = accelerator.unwrap_model(model) if accelerator is not None else model
    if context.is_main:
        final_model.save_pretrained(final_peft, safe_serialization=True)
    distributed_barrier(torch, context)
    memory_points.append(_memory(torch, device, "adapter_saved"))
    local_peak_before_reload = max(
        float(point["peak_reserved_gib"]) for point in memory_points
    )
    rank_peaks = distributed_objects(torch, local_peak_before_reload, context)
    peak_before_reload = max(float(value) for value in rank_peaks)
    if peak_before_reload > MEMORY_GATE_GIB:
        raise TrainingError(
            f"SFT peak reserved {peak_before_reload:.3f} GiB exceeds {MEMORY_GATE_GIB:.1f} GiB"
        )

    del optimizer, scheduler, model, final_model
    gc.collect()
    torch.cuda.empty_cache()
    memory_points.append(_memory(torch, device, "training_model_released"))
    if not context.is_main:
        distributed_barrier(torch, context)
        artifact = AdapterArtifact.load(config.artifact_dir)
        lifecycle_path = config.work_dir / "lifecycle.json"
        lifecycle = json.loads(lifecycle_path.read_text(encoding="utf-8"))
        return SFTResult(
            artifact,
            int(lifecycle["optimizer_steps"]),
            int(lifecycle["micro_steps"]),
            float(lifecycle["final_loss"]),
            float(lifecycle["peak_reserved_gib"]),
            str(lifecycle["positive_completion"]),
            str(lifecycle["negative_completion"]),
            lifecycle_path,
        )
    candidate = config.artifact_dir.parent / f".{config.artifact_dir.name}.candidate"
    artifact_config = ArtifactInitConfig(
        1,
        "artifact.init",
        final_peft,
        candidate,
        "sft",
        None,
        config.config_path,
        config.config_hash,
    )
    artifact = initialize_artifact(
        artifact_config,
        provenance={
            **validation,
            "optimizer_steps": state["optimizer_step"],
            "micro_steps": state["micro_step"],
            "final_loss": final_loss,
            "peak_reserved_gib_before_reload": peak_before_reload,
            "trainable_parameters": trainable,
            "excluded_overlength_records": excluded_overlength_records,
            "padded_record_ids": padded_record_ids,
        },
    )

    validation_records = list(dataset.iter_records("validation"))
    positive_record = next(row for row in validation_records if bool(row["positive"]))
    negative_record = next(row for row in validation_records if not bool(row["positive"]))
    inference = Qwen3VLAdapter.load(candidate, config.runtime)
    application = DetectionApplication(inference)
    smoke_max_new_tokens = min(config.runtime.max_new_tokens, 128)
    positive_result = application.detect(
        _request_from_training_record(dataset, positive_record),
        max_new_tokens=smoke_max_new_tokens,
        config_hash=config.config_hash,
    )
    negative_result = application.detect(
        _request_from_training_record(dataset, negative_record),
        max_new_tokens=smoke_max_new_tokens,
        config_hash=config.config_hash,
    )
    memory_points.append(_memory(torch, device, "strict_generation_complete"))
    peak_reserved = max(float(point["peak_reserved_gib"]) for point in memory_points)
    if peak_reserved > MEMORY_GATE_GIB:
        raise TrainingError(
            f"SFT lifecycle peak reserved {peak_reserved:.3f} GiB exceeds {MEMORY_GATE_GIB:.1f} GiB"
        )
    if config.artifact_dir.exists():
        raise TrainingError(f"SFT Artifact output already exists: {config.artifact_dir}")
    os.replace(candidate, config.artifact_dir)
    artifact = AdapterArtifact.load(config.artifact_dir)
    lifecycle = {
        "schema_version": 1,
        "accepted": True,
        "dataset_fingerprint": dataset.fingerprint,
        "artifact_fingerprint": artifact.fingerprint,
        "optimizer_steps": state["optimizer_step"],
        "micro_steps": state["micro_step"],
        "final_loss": final_loss,
        "trainable_parameters": trainable,
        "smoke_max_new_tokens": smoke_max_new_tokens,
        "positive_record_id": positive_record["id"],
        "positive_completion": positive_result.raw_completion,
        "negative_record_id": negative_record["id"],
        "negative_completion": negative_result.raw_completion,
        "peak_reserved_gib": peak_reserved,
        "memory_gate_gib": MEMORY_GATE_GIB,
        "memory": memory_points,
        "world_size": context.world_size,
        "rank_peak_reserved_gib_before_reload": rank_peaks,
        "excluded_overlength_records": excluded_overlength_records,
        "padded_record_ids": padded_record_ids,
        "rank_schedule_lengths_before_padding": schedule_lengths,
    }
    lifecycle_path = config.work_dir / "lifecycle.json"
    lifecycle_path.write_text(
        json.dumps(lifecycle, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    from conceptdet.acceptance import emit_hardware_gate_report

    emit_hardware_gate_report(
        gate="H1" if context.world_size == 1 else "D1",
        lifecycle_path=lifecycle_path,
        artifact_path=artifact.path,
        config_hash=config.config_hash,
        dataset_fingerprint=dataset.fingerprint,
        offline=config.runtime.local_files_only,
    )
    distributed_barrier(torch, context)
    return SFTResult(
        artifact,
        int(state["optimizer_step"]),
        int(state["micro_step"]),
        final_loss,
        peak_reserved,
        positive_result.raw_completion,
        negative_result.raw_completion,
        lifecycle_path,
    )
