from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, TypeAlias

import yaml

from conceptdet.errors import ConfigurationError
from conceptdet.types import Box

SCHEMA_VERSION = 1
SUPPORTED_KINDS = frozenset(
    {
        "infer.detect",
        "infer.batch",
        "artifact.init",
        "data.voc",
        "train.sft",
        "train.grpo",
        "evaluate",
    }
)
RESERVED_KINDS: frozenset[str] = frozenset()
LEGACY_KEYS = frozenset(
    {
        "model_path",
        "model_name_or_path",
        "input_size",
        "max_pixels",
        "max_seq_length",
        "packing",
        "resume_from_checkpoint",
        "use_vllm",
        "question_template",
        "is_grpo_train",
        "sam_path",
        "mask_path",
        "num_of_query",
        "learnable_query",
        "connector",
        "projection",
    }
)


class _StrictLoader(yaml.SafeLoader):
    pass


def _construct_mapping(loader: _StrictLoader, node: yaml.MappingNode, deep: bool = False) -> Any:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key == "<<":
            raise ConfigurationError("YAML merge keys are not supported")
        if key in mapping:
            raise ConfigurationError(f"Duplicate YAML key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "auto"
    dtype: Literal["auto", "float32", "float16", "bfloat16"] = "bfloat16"
    attention: Literal["auto", "eager", "sdpa", "flash_attention_2"] = (
        "flash_attention_2"
    )
    max_new_tokens: int = 192
    local_files_only: bool = False


@dataclass(frozen=True)
class OutputConfig:
    image: Path
    json: Path
    layout: Literal["annotated", "triptych"] = "annotated"


@dataclass(frozen=True)
class RequestConfig:
    reference_image: Path
    reference_boxes: tuple[Box, ...]
    target_image: Path
    query: str


@dataclass(frozen=True)
class DetectConfig:
    schema_version: int
    kind: Literal["infer.detect"]
    artifact: Path
    request: RequestConfig
    output: OutputConfig
    runtime: RuntimeConfig
    config_path: Path
    config_hash: str


@dataclass(frozen=True)
class BatchConfig:
    schema_version: int
    kind: Literal["infer.batch"]
    artifact: Path
    manifest: Path
    output_dir: Path
    layout: Literal["annotated", "triptych"]
    overwrite: bool
    runtime: RuntimeConfig
    config_path: Path
    config_hash: str


@dataclass(frozen=True)
class ArtifactInitConfig:
    schema_version: int
    kind: Literal["artifact.init"]
    source_adapter: Path
    output_dir: Path
    stage: Literal["sft", "grpo"]
    parent_artifact: Path | None
    config_path: Path
    config_hash: str


@dataclass(frozen=True)
class VocSourceConfig:
    name: str
    image_dir: Path
    annotation_dir: Path


@dataclass(frozen=True)
class SplitConfig:
    train: float = 0.9
    validation: float = 0.05
    test: float = 0.05
    seed: int = 17


@dataclass(frozen=True)
class DataVocConfig:
    schema_version: int
    kind: Literal["data.voc"]
    sources: tuple[VocSourceConfig, ...]
    classes: tuple[str, ...] | None
    output_dir: Path
    source_box_semantics: Literal["voc_inclusive", "xyxy_half_open"]
    negative_per_image: int
    splits: SplitConfig
    config_path: Path
    config_hash: str


@dataclass(frozen=True)
class OptimizationConfig:
    epochs: float = 1.0
    max_steps: int | None = None
    learning_rate: float = 2e-4
    gradient_accumulation_steps: int = 8
    weight_decay: float = 0.01
    warmup_steps: int = 0
    checkpoint_steps: int = 100
    seed: int = 17


@dataclass(frozen=True)
class SFTStageConfig:
    schema_version: int
    kind: Literal["train.sft"]
    dataset_dir: Path
    work_dir: Path
    artifact_dir: Path
    runtime: RuntimeConfig
    optimization: OptimizationConfig
    config_path: Path
    config_hash: str


@dataclass(frozen=True)
class GRPOOptimizationConfig:
    epochs: float = 1.0
    max_steps: int | None = None
    learning_rate: float = 1e-5
    gradient_accumulation_steps: int = 2
    weight_decay: float = 0.01
    warmup_steps: int = 0
    seed: int = 23
    checkpoint_steps: int = 100


@dataclass(frozen=True)
class GRPOStageConfig:
    schema_version: int
    kind: Literal["train.grpo"]
    dataset_dir: Path
    parent_artifact: Path
    work_dir: Path
    artifact_dir: Path
    runtime: RuntimeConfig
    optimization: GRPOOptimizationConfig
    config_path: Path
    config_hash: str


@dataclass(frozen=True)
class EvaluationConfig:
    schema_version: int
    kind: Literal["evaluate"]
    dataset_dir: Path
    artifact: Path
    predictions: Path
    split: Literal["train", "validation", "test"]
    output_dir: Path
    config_path: Path
    config_hash: str


ConceptDetConfig: TypeAlias = (  # noqa: UP040 - package supports Python 3.10
    DetectConfig
    | BatchConfig
    | ArtifactInitConfig
    | DataVocConfig
    | SFTStageConfig
    | GRPOStageConfig
    | EvaluationConfig
)


def _mapping(
    value: object,
    path: str,
    *,
    allowed: set[str],
    required: set[str] = frozenset(),
) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ConfigurationError(f"{path} must be a mapping")
    keys = set(value)
    missing = required - keys
    if missing:
        raise ConfigurationError(f"{path} is missing: {', '.join(sorted(missing))}")
    unknown = keys - allowed
    if unknown:
        legacy = unknown & LEGACY_KEYS
        if legacy:
            name = sorted(legacy)[0]
            raise ConfigurationError(
                f"{path}.{name} is an obsolete ConceptSeg/Qwen2.5 setting; "
                "ConceptDet v1 is a clean break"
            )
        raise ConfigurationError(f"{path} has unknown fields: {', '.join(sorted(unknown))}")
    return value


def _scan_legacy(value: object, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key in LEGACY_KEYS or key.startswith("mask") or key.startswith("sam_"):
                raise ConfigurationError(
                    f"{child_path} is an obsolete ConceptSeg/Qwen2.5 setting; "
                    "ConceptDet v1 is a clean break"
                )
            _scan_legacy(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _scan_legacy(child, f"{path}[{index}]")


def _path(value: object, path: str, base: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{path} must be a nonempty path string")
    candidate = Path(value).expanduser()
    return (candidate if candidate.is_absolute() else base / candidate).resolve()


def _text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{path} must be a nonempty string")
    return value.strip()


def _boolean(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigurationError(f"{path} must be true or false")
    return value


def _integer(value: object, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigurationError(f"{path} must be an integer >= {minimum}")
    return value


def _number(value: object, path: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < minimum:
        raise ConfigurationError(f"{path} must be a number >= {minimum}")
    return float(value)


def _runtime(value: object) -> RuntimeConfig:
    raw = _mapping(
        value or {},
        "$.runtime",
        allowed={"device", "dtype", "attention", "max_new_tokens", "local_files_only"},
    )
    device = raw.get("device", "auto")
    dtype = raw.get("dtype", "bfloat16")
    attention = raw.get("attention", "flash_attention_2")
    maximum = raw.get("max_new_tokens", 192)
    local = raw.get("local_files_only", False)
    if not isinstance(device, str) or not device:
        raise ConfigurationError("$.runtime.device must be a nonempty string")
    if dtype not in {"auto", "float32", "float16", "bfloat16"}:
        raise ConfigurationError("$.runtime.dtype is unsupported")
    if attention not in {"auto", "eager", "sdpa", "flash_attention_2"}:
        raise ConfigurationError("$.runtime.attention is unsupported")
    if isinstance(maximum, bool) or not isinstance(maximum, int) or not 1 <= maximum <= 192:
        raise ConfigurationError("$.runtime.max_new_tokens must be an integer from 1 to 192")
    return RuntimeConfig(
        device,
        dtype,
        attention,
        maximum,
        _boolean(local, "$.runtime.local_files_only"),
    )


def _boxes(value: object, path: str) -> tuple[Box, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigurationError(f"{path} must contain at least one XYXY box")
    try:
        return tuple(Box.from_sequence(box) for box in value)
    except Exception as exc:
        raise ConfigurationError(f"{path} contains an invalid XYXY box: {exc}") from exc


def _voc_sources(value: object, base: Path) -> tuple[VocSourceConfig, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigurationError("$.sources must contain at least one VOC source")
    sources: list[VocSourceConfig] = []
    names: set[str] = set()
    for index, item in enumerate(value):
        path = f"$.sources[{index}]"
        raw = _mapping(
            item,
            path,
            allowed={"name", "images", "annotations"},
            required={"name", "images", "annotations"},
        )
        name = _text(raw["name"], f"{path}.name")
        if name in names:
            raise ConfigurationError(f"Duplicate VOC source name: {name}")
        names.add(name)
        sources.append(
            VocSourceConfig(
                name,
                _path(raw["images"], f"{path}.images", base),
                _path(raw["annotations"], f"{path}.annotations", base),
            )
        )
    return tuple(sources)


def _splits(value: object) -> SplitConfig:
    raw = _mapping(
        value or {},
        "$.splits",
        allowed={"train", "validation", "test", "seed"},
    )
    train = _number(raw.get("train", 0.9), "$.splits.train")
    validation = _number(raw.get("validation", 0.05), "$.splits.validation")
    test = _number(raw.get("test", 0.05), "$.splits.test")
    if min(train, validation, test) <= 0 or abs(train + validation + test - 1.0) > 1e-9:
        raise ConfigurationError("$.splits ratios must be positive and sum to 1")
    return SplitConfig(
        train,
        validation,
        test,
        _integer(raw.get("seed", 17), "$.splits.seed"),
    )


def _optimization(value: object) -> OptimizationConfig:
    raw = _mapping(
        value or {},
        "$.optimization",
        allowed={
            "epochs",
            "max_steps",
            "learning_rate",
            "gradient_accumulation_steps",
            "weight_decay",
            "warmup_steps",
            "checkpoint_steps",
            "seed",
        },
    )
    maximum = raw.get("max_steps")
    if maximum is not None:
        maximum = _integer(maximum, "$.optimization.max_steps", minimum=1)
    epochs = _number(raw.get("epochs", 1.0), "$.optimization.epochs")
    if epochs <= 0:
        raise ConfigurationError("$.optimization.epochs must be > 0")
    learning_rate = _number(
        raw.get("learning_rate", 2e-4), "$.optimization.learning_rate"
    )
    if learning_rate <= 0:
        raise ConfigurationError("$.optimization.learning_rate must be > 0")
    return OptimizationConfig(
        epochs,
        maximum,
        learning_rate,
        _integer(
            raw.get("gradient_accumulation_steps", 8),
            "$.optimization.gradient_accumulation_steps",
            minimum=1,
        ),
        _number(raw.get("weight_decay", 0.01), "$.optimization.weight_decay"),
        _integer(raw.get("warmup_steps", 0), "$.optimization.warmup_steps"),
        _integer(
            raw.get("checkpoint_steps", 100),
            "$.optimization.checkpoint_steps",
            minimum=1,
        ),
        _integer(raw.get("seed", 17), "$.optimization.seed"),
    )


def _grpo_optimization(value: object) -> GRPOOptimizationConfig:
    raw = _mapping(
        value or {},
        "$.optimization",
        allowed={
            "epochs",
            "max_steps",
            "learning_rate",
            "gradient_accumulation_steps",
            "weight_decay",
            "warmup_steps",
            "seed",
            "checkpoint_steps",
        },
    )
    maximum = raw.get("max_steps")
    if maximum is not None:
        maximum = _integer(maximum, "$.optimization.max_steps", minimum=1)
    epochs = _number(raw.get("epochs", 1.0), "$.optimization.epochs")
    learning_rate = _number(
        raw.get("learning_rate", 1e-5), "$.optimization.learning_rate"
    )
    if epochs <= 0:
        raise ConfigurationError("$.optimization.epochs must be > 0")
    if learning_rate <= 0:
        raise ConfigurationError("$.optimization.learning_rate must be > 0")
    return GRPOOptimizationConfig(
        epochs,
        maximum,
        learning_rate,
        _integer(
            raw.get("gradient_accumulation_steps", 2),
            "$.optimization.gradient_accumulation_steps",
            minimum=1,
        ),
        _number(raw.get("weight_decay", 0.01), "$.optimization.weight_decay"),
        _integer(raw.get("warmup_steps", 0), "$.optimization.warmup_steps"),
        _integer(raw.get("seed", 23), "$.optimization.seed"),
        _integer(
            raw.get("checkpoint_steps", 100),
            "$.optimization.checkpoint_steps",
            minimum=1,
        ),
    )


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _finalize_hash(config: ConceptDetConfig) -> ConceptDetConfig:
    payload = config_to_dict(config)
    payload.pop("config_hash")
    return replace(config, config_hash=_canonical_hash(payload))


def load_config(path: str | Path) -> ConceptDetConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigurationError(f"Configuration file does not exist: {config_path}")
    try:
        raw = yaml.load(config_path.read_text(encoding="utf-8"), Loader=_StrictLoader)
    except ConfigurationError:
        raise
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Cannot parse YAML {config_path}: {exc}") from exc
    root = _mapping(
        raw,
        "$",
        allowed={
            "schema_version",
            "kind",
            "artifact",
            "request",
            "output",
            "runtime",
            "manifest",
            "output_dir",
            "layout",
            "overwrite",
            "source_adapter",
            "stage",
            "parent_artifact",
            "sources",
            "classes",
            "source_box_semantics",
            "negative_per_image",
            "splits",
            "dataset_dir",
            "work_dir",
            "artifact_dir",
            "optimization",
            "predictions",
            "split",
        },
        required={"schema_version", "kind"},
    )
    _scan_legacy(root)
    if root["schema_version"] != SCHEMA_VERSION:
        raise ConfigurationError(
            f"$.schema_version must be {SCHEMA_VERSION}, got {root['schema_version']!r}"
        )
    kind = root["kind"]
    if kind in RESERVED_KINDS:
        raise ConfigurationError(
            f"$.kind {kind!r} is specified but not implemented by this release"
        )
    if kind not in SUPPORTED_KINDS:
        raise ConfigurationError(f"$.kind is unsupported: {kind!r}")
    base = config_path.parent
    if kind == "infer.detect":
        _mapping(
            root,
            "$",
            allowed={"schema_version", "kind", "artifact", "request", "output", "runtime"},
            required={"schema_version", "kind", "artifact", "request", "output"},
        )
        request = _mapping(
            root["request"],
            "$.request",
            allowed={"reference_image", "reference_boxes", "target_image", "query"},
            required={"reference_image", "reference_boxes", "target_image", "query"},
        )
        output = _mapping(
            root["output"],
            "$.output",
            allowed={"image", "json", "layout"},
            required={"image"},
        )
        image_path = _path(output["image"], "$.output.image", base)
        json_path = _path(
            output.get("json", str(image_path.with_suffix(".json"))), "$.output.json", base
        )
        layout = output.get("layout", "annotated")
        if layout not in {"annotated", "triptych"}:
            raise ConfigurationError("$.output.layout must be annotated or triptych")
        return _finalize_hash(DetectConfig(
            1,
            "infer.detect",
            _path(root["artifact"], "$.artifact", base),
            RequestConfig(
                _path(request["reference_image"], "$.request.reference_image", base),
                _boxes(request["reference_boxes"], "$.request.reference_boxes"),
                _path(request["target_image"], "$.request.target_image", base),
                _text(request["query"], "$.request.query"),
            ),
            OutputConfig(image_path, json_path, layout),
            _runtime(root.get("runtime")),
            config_path,
            "",
        ))

    if kind == "infer.batch":
        _mapping(
            root,
            "$",
            allowed={
                "schema_version",
                "kind",
                "artifact",
                "manifest",
                "output_dir",
                "layout",
                "overwrite",
                "runtime",
            },
            required={"schema_version", "kind", "artifact", "manifest", "output_dir"},
        )
        layout = root.get("layout", "annotated")
        if layout not in {"annotated", "triptych"}:
            raise ConfigurationError("$.layout must be annotated or triptych")
        overwrite = root.get("overwrite", False)
        return _finalize_hash(BatchConfig(
            1,
            "infer.batch",
            _path(root["artifact"], "$.artifact", base),
            _path(root["manifest"], "$.manifest", base),
            _path(root["output_dir"], "$.output_dir", base),
            layout,
            _boolean(overwrite, "$.overwrite"),
            _runtime(root.get("runtime")),
            config_path,
            "",
        ))

    if kind == "data.voc":
        _mapping(
            root,
            "$",
            allowed={
                "schema_version",
                "kind",
                "sources",
                "classes",
                "output_dir",
                "source_box_semantics",
                "negative_per_image",
                "splits",
            },
            required={"schema_version", "kind", "sources", "output_dir"},
        )
        raw_classes = root.get("classes", "all")
        if raw_classes == "all":
            classes: tuple[str, ...] | None = None
        elif (
            isinstance(raw_classes, list)
            and raw_classes
            and all(isinstance(item, str) and item.strip() for item in raw_classes)
        ):
            classes = tuple(sorted({item.strip() for item in raw_classes}))
            if len(classes) != len(raw_classes):
                raise ConfigurationError("$.classes must not contain duplicates")
        else:
            raise ConfigurationError("$.classes must be 'all' or a nonempty string list")
        semantics = root.get("source_box_semantics", "voc_inclusive")
        if semantics not in {"voc_inclusive", "xyxy_half_open"}:
            raise ConfigurationError(
                "$.source_box_semantics must be voc_inclusive or xyxy_half_open"
            )
        return _finalize_hash(
            DataVocConfig(
                1,
                "data.voc",
                _voc_sources(root["sources"], base),
                classes,
                _path(root["output_dir"], "$.output_dir", base),
                semantics,
                _integer(
                    root.get("negative_per_image", 1),
                    "$.negative_per_image",
                    minimum=1,
                ),
                _splits(root.get("splits")),
                config_path,
                "",
            )
        )

    if kind == "train.sft":
        _mapping(
            root,
            "$",
            allowed={
                "schema_version",
                "kind",
                "dataset_dir",
                "work_dir",
                "artifact_dir",
                "runtime",
                "optimization",
            },
            required={
                "schema_version",
                "kind",
                "dataset_dir",
                "work_dir",
                "artifact_dir",
            },
        )
        return _finalize_hash(
            SFTStageConfig(
                1,
                "train.sft",
                _path(root["dataset_dir"], "$.dataset_dir", base),
                _path(root["work_dir"], "$.work_dir", base),
                _path(root["artifact_dir"], "$.artifact_dir", base),
                _runtime(root.get("runtime")),
                _optimization(root.get("optimization")),
                config_path,
                "",
            )
        )

    if kind == "train.grpo":
        _mapping(
            root,
            "$",
            allowed={
                "schema_version",
                "kind",
                "dataset_dir",
                "parent_artifact",
                "work_dir",
                "artifact_dir",
                "runtime",
                "optimization",
            },
            required={
                "schema_version",
                "kind",
                "dataset_dir",
                "parent_artifact",
                "work_dir",
                "artifact_dir",
            },
        )
        runtime = _runtime(root.get("runtime"))
        if runtime.max_new_tokens != 192:
            raise ConfigurationError(
                "$.runtime.max_new_tokens must be exactly 192 for native GRPO"
            )
        return _finalize_hash(
            GRPOStageConfig(
                1,
                "train.grpo",
                _path(root["dataset_dir"], "$.dataset_dir", base),
                _path(root["parent_artifact"], "$.parent_artifact", base),
                _path(root["work_dir"], "$.work_dir", base),
                _path(root["artifact_dir"], "$.artifact_dir", base),
                runtime,
                _grpo_optimization(root.get("optimization")),
                config_path,
                "",
            )
        )

    if kind == "evaluate":
        _mapping(
            root,
            "$",
            allowed={
                "schema_version",
                "kind",
                "dataset_dir",
                "artifact",
                "predictions",
                "split",
                "output_dir",
            },
            required={
                "schema_version",
                "kind",
                "dataset_dir",
                "artifact",
                "predictions",
                "output_dir",
            },
        )
        split = root.get("split", "test")
        if split not in {"train", "validation", "test"}:
            raise ConfigurationError("$.split must be train, validation, or test")
        return _finalize_hash(
            EvaluationConfig(
                1,
                "evaluate",
                _path(root["dataset_dir"], "$.dataset_dir", base),
                _path(root["artifact"], "$.artifact", base),
                _path(root["predictions"], "$.predictions", base),
                split,
                _path(root["output_dir"], "$.output_dir", base),
                config_path,
                "",
            )
        )

    _mapping(
        root,
        "$",
        allowed={
            "schema_version",
            "kind",
            "source_adapter",
            "output_dir",
            "stage",
            "parent_artifact",
        },
        required={"schema_version", "kind", "source_adapter", "output_dir", "stage"},
    )
    stage = root["stage"]
    if stage not in {"sft", "grpo"}:
        raise ConfigurationError("$.stage must be sft or grpo")
    parent = root.get("parent_artifact")
    if stage == "grpo" and parent is None:
        raise ConfigurationError("$.parent_artifact is required for a GRPO Artifact")
    return _finalize_hash(ArtifactInitConfig(
        1,
        "artifact.init",
        _path(root["source_adapter"], "$.source_adapter", base),
        _path(root["output_dir"], "$.output_dir", base),
        stage,
        _path(parent, "$.parent_artifact", base) if parent is not None else None,
        config_path,
        "",
    ))


def config_to_dict(config: ConceptDetConfig) -> dict[str, Any]:
    if isinstance(config, DetectConfig):
        return {
            "schema_version": 1,
            "kind": config.kind,
            "artifact": str(config.artifact),
            "request": {
                "reference_image": str(config.request.reference_image),
                "reference_boxes": [box.to_list() for box in config.request.reference_boxes],
                "target_image": str(config.request.target_image),
                "query": config.request.query,
            },
            "output": {
                "image": str(config.output.image),
                "json": str(config.output.json),
                "layout": config.output.layout,
            },
            "runtime": config.runtime.__dict__,
            "config_hash": config.config_hash,
        }
    if isinstance(config, BatchConfig):
        return {
            "schema_version": 1,
            "kind": config.kind,
            "artifact": str(config.artifact),
            "manifest": str(config.manifest),
            "output_dir": str(config.output_dir),
            "layout": config.layout,
            "overwrite": config.overwrite,
            "runtime": config.runtime.__dict__,
            "config_hash": config.config_hash,
        }
    if isinstance(config, DataVocConfig):
        return {
            "schema_version": 1,
            "kind": config.kind,
            "sources": [
                {
                    "name": source.name,
                    "images": str(source.image_dir),
                    "annotations": str(source.annotation_dir),
                }
                for source in config.sources
            ],
            "classes": list(config.classes) if config.classes is not None else "all",
            "output_dir": str(config.output_dir),
            "source_box_semantics": config.source_box_semantics,
            "negative_per_image": config.negative_per_image,
            "splits": config.splits.__dict__,
            "config_hash": config.config_hash,
        }
    if isinstance(config, SFTStageConfig):
        return {
            "schema_version": 1,
            "kind": config.kind,
            "dataset_dir": str(config.dataset_dir),
            "work_dir": str(config.work_dir),
            "artifact_dir": str(config.artifact_dir),
            "runtime": config.runtime.__dict__,
            "optimization": config.optimization.__dict__,
            "config_hash": config.config_hash,
        }
    if isinstance(config, GRPOStageConfig):
        return {
            "schema_version": 1,
            "kind": config.kind,
            "dataset_dir": str(config.dataset_dir),
            "parent_artifact": str(config.parent_artifact),
            "work_dir": str(config.work_dir),
            "artifact_dir": str(config.artifact_dir),
            "runtime": config.runtime.__dict__,
            "optimization": config.optimization.__dict__,
            "config_hash": config.config_hash,
        }
    if isinstance(config, EvaluationConfig):
        return {
            "schema_version": 1,
            "kind": config.kind,
            "dataset_dir": str(config.dataset_dir),
            "artifact": str(config.artifact),
            "predictions": str(config.predictions),
            "split": config.split,
            "output_dir": str(config.output_dir),
            "config_hash": config.config_hash,
        }
    return {
        "schema_version": 1,
        "kind": config.kind,
        "source_adapter": str(config.source_adapter),
        "output_dir": str(config.output_dir),
        "stage": config.stage,
        "parent_artifact": str(config.parent_artifact) if config.parent_artifact else None,
        "config_hash": config.config_hash,
    }
