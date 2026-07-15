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
SUPPORTED_KINDS = frozenset({"infer.detect", "infer.batch", "artifact.init"})
RESERVED_KINDS = frozenset({"train.sft", "train.grpo", "evaluate"})
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


ConceptDetConfig: TypeAlias = (  # noqa: UP040 - package supports Python 3.10
    DetectConfig | BatchConfig | ArtifactInitConfig
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
    return {
        "schema_version": 1,
        "kind": config.kind,
        "source_adapter": str(config.source_adapter),
        "output_dir": str(config.output_dir),
        "stage": config.stage,
        "parent_artifact": str(config.parent_artifact) if config.parent_artifact else None,
        "config_hash": config.config_hash,
    }
