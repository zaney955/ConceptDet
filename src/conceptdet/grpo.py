from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import math
import os
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

from PIL import Image, ImageOps
from transformers import TrainerCallback

from conceptdet.adapter import AdapterInput
from conceptdet.application import DetectionApplication
from conceptdet.artifact import (
    MODEL_ID,
    MODEL_REVISION,
    AdapterArtifact,
    initialize_artifact,
)
from conceptdet.config import ArtifactInitConfig, GRPOStageConfig
from conceptdet.dataset import (
    DatasetArtifact,
    detection_request,
    validate_training_dataset,
)
from conceptdet.errors import DatasetError, TrainingError
from conceptdet.evaluation import score_detection_reward
from conceptdet.model import (
    MAX_TOTAL_SEQUENCE_TOKENS,
    MAX_VISUAL_TOKENS,
    MIN_VISUAL_TOKENS,
    Qwen3VLAdapter,
    prepare_images,
    smart_image_size,
)
from conceptdet.peft_weights import load_exact_adapter_weights
from conceptdet.prompts import STRICT_DETECTION_PROMPT, build_messages
from conceptdet.protocol import parse_detection_set, serialize_detection_set
from conceptdet.run_state import (
    ProcessContext,
    RunIdentity,
    assert_distributed_consensus,
    code_fingerprint,
    distributed_barrier,
    distributed_objects,
    load_checkpoint_metadata,
    resolve_resume,
    write_checkpoint_metadata,
)
from conceptdet.types import Box

MEMORY_GATE_GIB = 44.0
NUM_GENERATIONS = 2
MAX_COMPLETION_TOKENS = 192
GRPO_TEMPERATURE = 1.2
EXPECTED_TRL_VERSION = "1.5.0"


@dataclass(frozen=True)
class GRPOBatchProvenance:
    record_id: str
    positive: bool
    prompt_tokens: int
    image_grids: tuple[tuple[int, int, int], ...]
    processor_pixels_equal: bool | None
    processor_pixel_max_abs_difference: float | None


@dataclass(frozen=True)
class GRPOResult:
    artifact: AdapterArtifact
    optimizer_steps: int
    reward_events: int
    nonzero_advantage_groups: int
    peak_reserved_gib: float
    positive_completion: str
    negative_completion: str
    lifecycle_report: Path


def _load_rgb(path: Path) -> Image.Image:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(path) as opened:
                return ImageOps.exif_transpose(opened).convert("RGB")
    except OSError as exc:
        raise DatasetError(f"Cannot read GRPO image: {path}") from exc


class GRPOBatchBuilder:
    """Own the shared model-visible record preparation used by stock TRL."""

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
            raise DatasetError(f"Cannot read GRPO image metadata: {path}") from exc
        if orientation in {5, 6, 7, 8}:
            width, height = height, width
        prepared = smart_image_size((width, height))
        self._prepared_size_cache[path] = prepared
        return prepared

    @staticmethod
    def _truth(record: dict[str, Any]) -> tuple[Any, ...]:
        raw_truth = record.get("detection_set")
        if not isinstance(raw_truth, list):
            raise DatasetError(f"GRPO record has no Detection Set: {record.get('id')}")
        truth_json = json.dumps(
            raw_truth, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        detections = parse_detection_set(truth_json)
        if serialize_detection_set(detections) != truth_json:
            raise DatasetError(f"GRPO truth is not canonical: {record.get('id')}")
        return detections

    def build(self, record: dict[str, Any]) -> tuple[dict[str, Any], GRPOBatchProvenance]:
        request = detection_request(self.dataset, record)
        reference, target = prepare_images(
            AdapterInput(
                _load_rgb(request.reference_image),
                request.reference_boxes,
                _load_rgb(request.target_image),
                request.query,
            )
        )
        detections = self._truth(record)

        encoded = self.processor.apply_chat_template(
            build_messages(reference, target, request.query),
            tokenize=True,
            add_generation_prompt=True,
            add_vision_id=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"do_resize": False},
        )
        prompt_tokens = int(encoded["input_ids"].shape[1])
        if prompt_tokens + MAX_COMPLETION_TOKENS > MAX_TOTAL_SEQUENCE_TOKENS:
            raise DatasetError(
                f"GRPO record {record.get('id')} prompt has {prompt_tokens} tokens; "
                f"prompt + {MAX_COMPLETION_TOKENS} must be <= {MAX_TOTAL_SEQUENCE_TOKENS} "
                "and truncation is forbidden"
            )
        grids = tuple(
            tuple(int(value) for value in row) for row in encoded["image_grid_thw"]
        )
        if len(grids) != 2:
            raise TrainingError(f"Expected two GRPO image grids, found {grids}")
        for grid in grids:
            visual_tokens = grid[0] * grid[1] * grid[2] // 4
            if not MIN_VISUAL_TOKENS <= visual_tokens <= MAX_VISUAL_TOKENS:
                raise TrainingError(
                    f"GRPO image grid {grid} produces {visual_tokens} tokens outside contract"
                )
        prompt = [
            {
                "role": "user",
                "content": STRICT_DETECTION_PROMPT.format(query=request.query),
            }
        ]
        prepared = {
            "prompt": prompt,
            "images": [reference, target],
            "ground_truth": [item.box.to_list(rounded=True) for item in detections],
            "record_id": str(record["id"]),
            "positive": bool(record["positive"]),
        }
        return prepared, GRPOBatchProvenance(
            str(record["id"]),
            bool(record["positive"]),
            prompt_tokens,
            grids,
            None,
            None,
        )

    def compatibility(self, record: dict[str, Any]) -> GRPOBatchProvenance:
        """Compute the exact prompt/grid contract without allocating pixel tensors."""
        request = detection_request(self.dataset, record)
        self._truth(record)
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
            raise TrainingError(f"GRPO preflight image grids produce invalid tokens: {grids}")

        placeholder = Image.new("RGB", (1, 1))
        messages = build_messages(placeholder, placeholder, request.query)
        rendered = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            add_vision_id=True,
        )
        token_ids = self.processor.tokenizer(rendered, add_special_tokens=False)[
            "input_ids"
        ]
        image_token_id = int(self.processor.image_token_id)
        if token_ids.count(image_token_id) != 2:
            raise TrainingError("GRPO preflight expected exactly two image placeholders")
        prompt_tokens = len(token_ids) - 2 + sum(visual_tokens)
        if prompt_tokens + MAX_COMPLETION_TOKENS > MAX_TOTAL_SEQUENCE_TOKENS:
            raise DatasetError(
                f"GRPO record {record.get('id')} prompt has {prompt_tokens} tokens; "
                f"prompt + {MAX_COMPLETION_TOKENS} must be <= {MAX_TOTAL_SEQUENCE_TOKENS} "
                "and truncation is forbidden"
            )
        return GRPOBatchProvenance(
            str(record["id"]),
            bool(record["positive"]),
            prompt_tokens,
            grids,
            None,
            None,
        )

    def preflight(
        self, record: dict[str, Any]
    ) -> tuple[dict[str, Any], GRPOBatchProvenance]:
        prepared, provenance = self.build(record)
        request = detection_request(self.dataset, record)
        messages = build_messages(
            prepared["images"][0], prepared["images"][1], request.query
        )
        common = {
            "tokenize": True,
            "add_generation_prompt": True,
            "add_vision_id": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        default = self.processor.apply_chat_template(messages, **common)
        fixed = self.processor.apply_chat_template(
            messages, **common, processor_kwargs={"do_resize": False}
        )
        default_pixels = default["pixel_values"].float()
        fixed_pixels = fixed["pixel_values"].float()
        same_shape = default_pixels.shape == fixed_pixels.shape
        maximum_difference = (
            float((default_pixels - fixed_pixels).abs().max())
            if same_shape
            else None
        )
        pixels_equal = bool(same_shape and maximum_difference == 0.0)
        if not pixels_equal:
            raise TrainingError(
                "Stock processor default path changes adapter-prepared GRPO pixels"
            )
        return prepared, replace(
            provenance,
            processor_pixels_equal=True,
            processor_pixel_max_abs_difference=maximum_difference,
        )


class _LazyGRPOTransform:
    def __init__(self, builder: GRPOBatchBuilder) -> None:
        self.builder = builder

    def __call__(self, batch: dict[str, list[str]]) -> dict[str, list[Any]]:
        raw_records = batch.get("record_json")
        if not isinstance(raw_records, list):
            raise DatasetError("Lazy GRPO dataset received an invalid record batch")
        prepared = [self.builder.build(json.loads(raw))[0] for raw in raw_records]
        return {
            key: [item[key] for item in prepared]
            for key in ("prompt", "images", "ground_truth", "record_id", "positive")
        }


def _stable_key(seed: int, kind: str, record: dict[str, Any]) -> str:
    return hashlib.sha256(f"{seed}:{kind}:{record['id']}".encode()).hexdigest()


def _ordered_records(records: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    positives = sorted(
        (record for record in records if bool(record["positive"])),
        key=lambda record: _stable_key(seed, "positive", record),
    )
    negatives = sorted(
        (record for record in records if not bool(record["positive"])),
        key=lambda record: _stable_key(seed, "negative", record),
    )
    if not positives or not negatives:
        raise DatasetError("GRPO training needs positive and negative records")
    ordered: list[dict[str, Any]] = []
    for index in range(max(len(positives), len(negatives))):
        if index < len(positives):
            ordered.append(positives[index])
        if index < len(negatives):
            ordered.append(negatives[index])
    return ordered


def _frontload_compatible_smoke_records(
    records: list[dict[str, Any]],
    builder: GRPOBatchBuilder,
    *,
    required_records: int | None,
) -> tuple[
    list[dict[str, Any]],
    tuple[GRPOBatchProvenance, GRPOBatchProvenance],
    list[str],
]:
    selected: dict[bool, tuple[dict[str, Any], GRPOBatchProvenance]] = {}
    compatible: list[dict[str, Any]] = []
    excluded: list[str] = []
    stopping_count = max(2, required_records) if required_records is not None else None
    for index, record in enumerate(records):
        kind = bool(record["positive"])
        try:
            provenance = builder.compatibility(record)
        except DatasetError as exc:
            if "prompt + 192" not in str(exc):
                raise
            excluded.append(str(record["id"]))
            continue
        compatible.append(record)
        selected.setdefault(kind, (record, provenance))
        if (
            stopping_count is not None
            and len(compatible) >= stopping_count
            and set(selected) == {False, True}
        ):
            compatible.extend(records[index + 1 :])
            break
    if set(selected) != {False, True}:
        raise TrainingError(
            "Could not find both positive and negative GRPO records within the 1,536-token contract"
        )
    positive_preflight = builder.preflight(selected[True][0])[1]
    negative_preflight = builder.preflight(selected[False][0])[1]
    for metadata_only, full in (
        (selected[True][1], positive_preflight),
        (selected[False][1], negative_preflight),
    ):
        if (
            metadata_only.prompt_tokens != full.prompt_tokens
            or metadata_only.image_grids != full.image_grids
        ):
            raise TrainingError(
                "GRPO metadata preflight disagrees with full processor preflight"
            )
    return compatible, (positive_preflight, negative_preflight), excluded


def _completion_text(completion: Any) -> str:
    content = completion[-1]["content"] if isinstance(completion, list) else completion
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ).strip()
    return str(content).strip()


class _RewardRecorder:
    def __init__(self) -> None:
        self.__name__ = "detection_set_reward"
        self.count = 0
        self.kinds: set[bool] = set()
        self.values: dict[str, list[float]] = defaultdict(list)
        self.events: list[dict[str, Any]] = []

    def __call__(
        self,
        prompts: list[Any],
        completions: list[Any],
        ground_truth: list[list[list[int]]],
        record_id: list[str],
        positive: list[bool],
        **_: Any,
    ) -> list[float]:
        del prompts
        rewards: list[float] = []
        for completion, raw_targets, identity, is_positive in zip(
            completions, ground_truth, record_id, positive, strict=True
        ):
            text = _completion_text(completion)
            targets = tuple(Box.from_sequence(box) for box in raw_targets)
            score = score_detection_reward(text, targets)
            rewards.append(score.total_reward)
            self.count += 1
            self.kinds.add(bool(is_positive))
            self.values[str(identity)].append(score.total_reward)
            if len(self.events) < 32:
                self.events.append(
                    {
                        "record_id": str(identity),
                        "positive": bool(is_positive),
                        "completion": text,
                        **asdict(score),
                    }
                )
        return rewards

    @property
    def nonzero_advantage_groups(self) -> int:
        return sum(
            len(values) >= NUM_GENERATIONS and max(values) - min(values) > 1e-8
            for values in self.values.values()
        )


class _CheckpointMetadataCallback(TrainerCallback):
    """Adds ConceptDet's fail-closed identity marker after stock Trainer saves state."""

    def __init__(self, identity: RunIdentity) -> None:
        self.identity = identity

    def on_save(self, args: Any, state: Any, control: Any, **_: Any) -> Any:
        if state.is_world_process_zero:
            checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            write_checkpoint_metadata(
                checkpoint,
                self.identity,
                {"optimizer_step": int(state.global_step)},
            )
            (checkpoint / "complete").write_text("complete\n", encoding="utf-8")
        return control


def _resolve_grpo_resume(
    work_dir: Path, resume: Literal["none", "auto"] | Path
) -> Path | None:
    if resume == "none":
        if work_dir.exists() and any(work_dir.iterdir()):
            raise TrainingError(
                f"GRPO work directory is not empty; use --resume auto or a checkpoint: {work_dir}"
            )
        return None
    if resume == "auto":
        return resolve_resume(work_dir / "trainer", "auto", stage="grpo")
    return resolve_resume(work_dir / "trainer", Path(resume), stage="grpo")


def _trainable_digest(model: Any) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        if parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _trainable_parameter_digests(model: Any) -> list[tuple[str, str]]:
    values: list[tuple[str, str]] = []
    for name, parameter in sorted(model.named_parameters()):
        if parameter.requires_grad:
            payload = parameter.detach().float().cpu().contiguous().numpy().tobytes()
            values.append((name, hashlib.sha256(payload).hexdigest()))
    return values


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


def _finite_metrics(metrics: dict[str, Any]) -> None:
    for key, value in metrics.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise TrainingError(f"TRL metric {key} is not finite: {value}")


def validate_grpo_inputs(
    config: GRPOStageConfig,
) -> tuple[DatasetArtifact, AdapterArtifact, dict[str, Any]]:
    dataset = DatasetArtifact.load(config.dataset_dir)
    validation = validate_training_dataset(dataset)
    parent = AdapterArtifact.load(config.parent_artifact)
    if parent.summary.get("stage") != "sft":
        raise TrainingError("GRPO parent Artifact must have stage=sft")
    if (parent.path / "optimizer.pt").exists():
        raise TrainingError("GRPO parent Artifact must not contain SFT optimizer state")
    if config.artifact_dir.exists():
        raise TrainingError(f"GRPO Artifact output already exists: {config.artifact_dir}")
    return dataset, parent, validation


def run_grpo(
    config: GRPOStageConfig,
    *,
    resume: Literal["none", "auto"] | Path = "none",
) -> GRPOResult:
    try:
        import torch
        import trl
        from accelerate import Accelerator
        from datasets import Dataset
        from peft import PeftModel
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise TrainingError(
            "GRPO dependencies are unavailable; install with pip install -e '.[grpo]'"
        ) from exc
    if trl.__version__ != EXPECTED_TRL_VERSION:
        raise TrainingError(
            f"ConceptDet requires TRL {EXPECTED_TRL_VERSION}, found {trl.__version__}"
        )
    if not torch.cuda.is_available():
        raise TrainingError("Qwen3-VL GRPO requires CUDA")
    context = ProcessContext.current()
    bootstrap_accelerator = Accelerator() if context.distributed else None
    device_name = context.cuda_device(config.runtime.device)
    if not device_name.startswith("cuda"):
        raise TrainingError("Qwen3-VL GRPO requires a CUDA device")
    if config.runtime.dtype not in {"auto", "bfloat16"}:
        raise TrainingError("Qwen3-VL GRPO supports bfloat16 only")
    resume_path = _resolve_grpo_resume(config.work_dir, resume)

    dataset, parent, validation = validate_grpo_inputs(config)
    identity = RunIdentity(
        "grpo",
        config.config_hash,
        dataset.fingerprint,
        str(parent.contract["contract_fingerprint"]),
        parent.fingerprint,
        code_fingerprint(
            Path(__file__),
            Path(__file__).with_name("evaluation.py"),
            Path(__file__).with_name("model.py"),
            Path(__file__).with_name("peft_weights.py"),
            Path(__file__).with_name("prompts.py"),
            Path(__file__).with_name("protocol.py"),
        ),
        context.world_size,
    )
    if resume_path is not None:
        load_checkpoint_metadata(resume_path, identity)
    device = (
        bootstrap_accelerator.device
        if bootstrap_accelerator is not None
        else torch.device(device_name)
    )
    torch.cuda.set_device(device)
    torch.manual_seed(config.optimization.seed)
    torch.cuda.manual_seed_all(config.optimization.seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    memory_points = [_memory(torch, device, "process_start")]
    config.work_dir.mkdir(parents=True, exist_ok=True)
    assert_distributed_consensus(
        torch,
        identity.fingerprint,
        context,
        name="run identity",
    )

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        min_pixels=65_536,
        max_pixels=655_360,
        local_files_only=config.runtime.local_files_only,
    )
    builder = GRPOBatchBuilder(dataset, processor)
    records = _ordered_records(
        list(dataset.iter_records("train")), config.optimization.seed
    )
    records, preflight, excluded_overlength_records = (
        _frontload_compatible_smoke_records(
            records,
            builder,
            required_records=config.optimization.max_steps,
        )
    )
    lazy_dataset = Dataset.from_dict(
        {
            "record_json": [
                json.dumps(record, sort_keys=True, separators=(",", ":"))
                for record in records
            ]
        }
    ).with_transform(_LazyGRPOTransform(builder))

    attention = config.runtime.attention
    if attention == "auto":
        attention = (
            "flash_attention_2"
            if importlib.util.find_spec("flash_attn") is not None
            else "sdpa"
        )
    try:
        base = Qwen3VLForConditionalGeneration.from_pretrained(
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
    base.config.use_cache = False
    model = PeftModel.from_pretrained(base, parent.path, is_trainable=True)
    try:
        source_adapter_digest = load_exact_adapter_weights(model, parent.path)
    except ValueError as exc:
        raise TrainingError(f"Cannot load exact GRPO parent weights: {exc}") from exc
    model.config.use_cache = False
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    expected_trainable = int(parent.contract["adapter"]["trainable_parameter_count"])
    if trainable != expected_trainable:
        raise TrainingError(
            f"Expected {expected_trainable} trainable GRPO parameters, found {trainable}"
        )
    unexpected = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" not in name
    ]
    if unexpected:
        raise TrainingError(f"Non-LoRA GRPO parameters are trainable: {unexpected[:10]}")
    initial_digest = _trainable_digest(model)
    initial_digests = distributed_objects(torch, initial_digest, context)
    if any(value != initial_digests[0] for value in initial_digests[1:]):
        parameter_digests = distributed_objects(
            torch, _trainable_parameter_digests(model), context
        )
        reference = dict(parameter_digests[0])
        differences = [
            name
            for rank_values in parameter_digests[1:]
            for name, digest in rank_values
            if reference.get(name) != digest
        ]
        raise TrainingError(
            "Cross-rank initial adapter checksum mismatch; differing parameters: "
            f"{sorted(set(differences))[:10]}"
        )
    memory_points.append(_memory(torch, device, "sft_parent_loaded"))

    recorder = _RewardRecorder()
    generation_batch_size = max(NUM_GENERATIONS, context.world_size)
    if generation_batch_size % NUM_GENERATIONS:
        raise TrainingError(
            f"GRPO world_size={context.world_size} is incompatible with "
            f"num_generations={NUM_GENERATIONS}"
        )
    trainer_args = GRPOConfig(
        output_dir=str(config.work_dir / "trainer"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=config.optimization.gradient_accumulation_steps,
        generation_batch_size=generation_batch_size,
        num_generations=NUM_GENERATIONS,
        temperature=GRPO_TEMPERATURE,
        num_train_epochs=config.optimization.epochs,
        max_steps=(config.optimization.max_steps or -1),
        max_completion_length=MAX_COMPLETION_TOKENS,
        learning_rate=config.optimization.learning_rate,
        weight_decay=config.optimization.weight_decay,
        warmup_steps=config.optimization.warmup_steps,
        beta=0.0,
        use_vllm=False,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_cache=False,
        shuffle_dataset=False,
        chat_template_kwargs={
            "add_vision_id": True,
            "processor_kwargs": {"do_resize": False},
        },
        logging_steps=1,
        logging_first_step=True,
        log_completions=True,
        num_completions_to_print=2,
        report_to="none",
        save_strategy="steps",
        save_steps=config.optimization.checkpoint_steps,
        save_only_model=False,
        disable_tqdm=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        seed=config.optimization.seed,
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=recorder,
        args=trainer_args,
        train_dataset=lazy_dataset,
        processing_class=processor,
        callbacks=[_CheckpointMetadataCallback(identity)],
    )
    if trainer.__class__ is not GRPOTrainer:
        raise TrainingError("GRPO must use the stock TRL GRPOTrainer class")
    if trainer.ref_model is not None or trainer_args.beta != 0.0:
        raise TrainingError("beta=0 GRPO must not allocate a reference model")
    if trainer.optimizer is not None:
        raise TrainingError("GRPO unexpectedly inherited an optimizer before training")
    memory_points.append(_memory(torch, device, "stock_trainer_ready"))
    try:
        train_output = trainer.train(
            resume_from_checkpoint=str(resume_path) if resume_path is not None else None
        )
    except torch.OutOfMemoryError as exc:
        raise TrainingError(
            "CUDA out of memory during native GRPO; one prompt group must fit per device"
        ) from exc
    metrics = dict(train_output.metrics)
    _finite_metrics(metrics)
    log_history = [dict(item) for item in trainer.state.log_history]
    gradient_norms = [
        float(item["grad_norm"])
        for item in log_history
        if isinstance(item.get("grad_norm"), (int, float))
    ]
    if not gradient_norms or not all(math.isfinite(value) for value in gradient_norms):
        raise TrainingError("Stock TRL did not report finite GRPO gradient norms")
    nonzero_gradient_norm = max(gradient_norms) > 0.0
    if not nonzero_gradient_norm:
        raise TrainingError("GRPO gradient norm remained zero")
    memory_points.append(_memory(torch, device, "native_grpo_complete"))
    synchronized_model = trainer.accelerator.unwrap_model(trainer.model)
    final_digest = _trainable_digest(synchronized_model)
    final_digests = distributed_objects(torch, final_digest, context)
    if any(value != final_digests[0] for value in final_digests[1:]):
        raise TrainingError(f"Final GRPO adapter checksum differs across ranks: {final_digests}")
    if initial_digest == final_digest:
        raise TrainingError("GRPO adapter parameters did not change")
    recorder_payloads = distributed_objects(
        torch,
        {"events": recorder.events, "values": dict(recorder.values)},
        context,
    )
    all_events = [
        event
        for payload in recorder_payloads
        for event in payload["events"]  # type: ignore[index]
    ]
    all_values: dict[str, list[float]] = defaultdict(list)
    for payload in recorder_payloads:
        for identity_key, values in payload["values"].items():  # type: ignore[union-attr]
            all_values[str(identity_key)].extend(float(value) for value in values)
    reward_event_count = len(all_events)
    reward_kinds = {bool(event["positive"]) for event in all_events}
    nonzero_advantage_groups = sum(
        len(values) >= NUM_GENERATIONS and max(values) - min(values) > 1e-8
        for values in all_values.values()
    )
    if reward_event_count < 4 or reward_kinds != {False, True}:
        raise TrainingError("GRPO reward did not observe four positive/negative completions")
    if nonzero_advantage_groups < 1:
        raise TrainingError("GRPO produced no reward group with nonzero advantage")

    final_peft = config.work_dir / "final-peft"
    trainer.save_model(str(final_peft))
    trainer.accelerator.wait_for_everyone()
    if not (final_peft / "adapter_model.safetensors").is_file():
        raise TrainingError("Stock TRL did not save the GRPO PEFT adapter")
    memory_points.append(_memory(torch, device, "grpo_adapter_saved"))
    local_peak_before_reload = max(
        float(point["peak_reserved_gib"]) for point in memory_points
    )
    rank_peaks = distributed_objects(torch, local_peak_before_reload, context)
    peak_before_reload = max(float(value) for value in rank_peaks)
    if peak_before_reload > MEMORY_GATE_GIB:
        raise TrainingError(
            f"GRPO peak reserved {peak_before_reload:.3f} GiB exceeds {MEMORY_GATE_GIB:.1f} GiB"
        )

    optimizer_steps = int(trainer.state.global_step)
    trainer_class = f"{trainer.__class__.__module__}.{trainer.__class__.__name__}"
    del trainer, model, base, processor, lazy_dataset
    gc.collect()
    torch.cuda.empty_cache()
    memory_points.append(_memory(torch, device, "training_model_released"))
    if not context.is_main:
        distributed_barrier(torch, context)
        artifact = AdapterArtifact.load(config.artifact_dir)
        lifecycle_path = config.work_dir / "lifecycle.json"
        lifecycle = json.loads(lifecycle_path.read_text(encoding="utf-8"))
        return GRPOResult(
            artifact,
            int(lifecycle["optimizer_steps"]),
            int(lifecycle["reward_event_count"]),
            int(lifecycle["nonzero_advantage_groups"]),
            float(lifecycle["peak_reserved_gib"]),
            str(lifecycle["positive_completion"]),
            str(lifecycle["negative_completion"]),
            lifecycle_path,
        )

    candidate = config.artifact_dir.parent / f".{config.artifact_dir.name}.candidate"
    if candidate.exists():
        raise TrainingError(f"GRPO candidate Artifact already exists: {candidate}")
    artifact_config = ArtifactInitConfig(
        1,
        "artifact.init",
        final_peft,
        candidate,
        "grpo",
        parent.path,
        config.config_path,
        config.config_hash,
    )
    artifact = initialize_artifact(
        artifact_config,
        provenance={
            **validation,
            "parent_artifact_fingerprint": parent.fingerprint,
            "parent_adapter_file_sha256": parent.summary["files"][
                "adapter_model.safetensors"
            ],
            "parent_loaded_adapter_digest": source_adapter_digest,
            "optimizer_inherited": False,
            "trainer_class": trainer_class,
            "trl_version": EXPECTED_TRL_VERSION,
            "beta": 0.0,
            "num_generations": NUM_GENERATIONS,
            "generation_batch_size": generation_batch_size,
            "temperature": GRPO_TEMPERATURE,
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
            "optimizer_steps": optimizer_steps,
            "reward_events": reward_event_count,
            "nonzero_advantage_groups": nonzero_advantage_groups,
            "nonzero_gradient_norm": nonzero_gradient_norm,
            "maximum_gradient_norm": max(gradient_norms),
            "initial_trainable_digest": initial_digest,
            "final_trainable_digest": final_digest,
            "peak_reserved_gib_before_reload": peak_before_reload,
            "trainable_parameters": trainable,
            "excluded_overlength_records": excluded_overlength_records,
        },
    )

    validation_records = list(dataset.iter_records("validation"))
    positive_record = next(row for row in validation_records if bool(row["positive"]))
    negative_record = next(row for row in validation_records if not bool(row["positive"]))
    inference = Qwen3VLAdapter.load(candidate, config.runtime)
    application = DetectionApplication(inference)
    smoke_max_new_tokens = min(config.runtime.max_new_tokens, 128)
    positive_result = application.detect(
        dataset.detection_request(positive_record),
        max_new_tokens=smoke_max_new_tokens,
        config_hash=config.config_hash,
    )
    negative_result = application.detect(
        dataset.detection_request(negative_record),
        max_new_tokens=smoke_max_new_tokens,
        config_hash=config.config_hash,
    )
    memory_points.append(_memory(torch, device, "strict_generation_complete"))
    peak_reserved = max(float(point["peak_reserved_gib"]) for point in memory_points)
    if peak_reserved > MEMORY_GATE_GIB:
        raise TrainingError(
            f"GRPO lifecycle peak reserved {peak_reserved:.3f} GiB exceeds "
            f"{MEMORY_GATE_GIB:.1f} GiB"
        )
    if config.artifact_dir.exists():
        raise TrainingError(f"GRPO Artifact output already exists: {config.artifact_dir}")
    os.replace(candidate, config.artifact_dir)
    artifact = AdapterArtifact.load(config.artifact_dir)

    lifecycle = {
        "schema_version": 1,
        "accepted": True,
        "dataset_fingerprint": dataset.fingerprint,
        "parent_artifact_fingerprint": parent.fingerprint,
        "artifact_fingerprint": artifact.fingerprint,
        "trainer_class": trainer_class,
        "trl_version": EXPECTED_TRL_VERSION,
        "beta": 0.0,
        "reference_model": False,
        "num_generations": NUM_GENERATIONS,
        "generation_batch_size": generation_batch_size,
        "temperature": GRPO_TEMPERATURE,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "optimizer_inherited": False,
        "parent_loaded_adapter_digest": source_adapter_digest,
        "optimizer_steps": optimizer_steps,
        "train_metrics": metrics,
        "trainer_log_history": log_history,
        "reward_events": all_events,
        "reward_event_count": reward_event_count,
        "nonzero_advantage_groups": nonzero_advantage_groups,
        "nonzero_gradient_norm": nonzero_gradient_norm,
        "maximum_gradient_norm": max(gradient_norms),
        "initial_trainable_digest": initial_digest,
        "final_trainable_digest": final_digest,
        "adapter_parameters_changed": initial_digest != final_digest,
        "trainable_parameters": trainable,
        "preflight": [asdict(item) for item in preflight],
        "excluded_overlength_records": excluded_overlength_records,
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
        "final_adapter_digests": final_digests,
    }
    lifecycle_path = config.work_dir / "lifecycle.json"
    lifecycle_path.write_text(
        json.dumps(lifecycle, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    from conceptdet.acceptance import emit_hardware_gate_report

    emit_hardware_gate_report(
        gate="H2" if context.world_size == 1 else "D1",
        lifecycle_path=lifecycle_path,
        artifact_path=artifact.path,
        config_hash=config.config_hash,
        dataset_fingerprint=dataset.fingerprint,
        offline=config.runtime.local_files_only,
    )
    distributed_barrier(torch, context)
    return GRPOResult(
        artifact,
        optimizer_steps,
        reward_event_count,
        nonzero_advantage_groups,
        peak_reserved,
        positive_result.raw_completion,
        negative_result.raw_completion,
        lifecycle_path,
    )
