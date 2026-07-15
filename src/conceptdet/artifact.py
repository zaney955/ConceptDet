from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from conceptdet.config import ArtifactInitConfig
from conceptdet.errors import ArtifactError

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
TARGET_MODULES_SHA256 = "fdff350e33d483666eb85ead6d1dc062df8739f2f6d78dc20663bf49fa755402"
TARGET_MODULE_PATTERNS = frozenset(
    {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "0.linear_fc1",
        "0.linear_fc2",
        "1.linear_fc1",
        "1.linear_fc2",
        "2.linear_fc1",
        "2.linear_fc2",
        "merger.linear_fc1",
        "merger.linear_fc2",
    }
)
CONTRACT_FILE = "conceptdet_contract.json"
SUMMARY_FILE = "training_summary.json"
WEIGHTS_FILE = "adapter_model.safetensors"
ADAPTER_CONFIG_FILE = "adapter_config.json"


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_contract() -> dict[str, Any]:
    contract: dict[str, Any] = {
        "artifact_schema_version": 1,
        "contract_id": "conceptdet.qwen3vl.reference-detection",
        "contract_version": 1,
        "artifact_kind": "peft_lora",
        "base": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_type": "qwen3_vl",
        },
        "processor": {
            "processor_id": MODEL_ID,
            "processor_revision": MODEL_REVISION,
            "add_vision_id": True,
        },
        "input": {
            "ordered_image_roles": ["reference", "target"],
            "max_images": 2,
            "exif_transpose": True,
            "color_mode": "RGB",
            "resize": {
                "algorithm": "qwen3_vl_smart_resize_v1",
                "factor": 32,
                "min_visual_tokens_per_image": 64,
                "max_visual_tokens_per_image": 640,
                "min_pixels_per_image": 65_536,
                "max_pixels_per_image": 655_360,
                "resampling": "bicubic",
                "processor_do_resize": False,
            },
            "reference_box_rendering": {
                "version": 1,
                "stage": "after_resize_before_normalize",
                "inner_color": "#ff2020",
                "inner_width": "clamp(round(shortest_side/256),2,4)",
                "outer_halo_color": "#ffffff",
                "outer_halo_width": 2,
            },
        },
        "prompt": {"version": 1},
        "output": {
            "schema_id": "conceptdet.detection-set",
            "schema_version": 1,
            "coordinate_space": "target_normalized_0_1000",
            "pixel_truth_box_semantics": "xyxy_half_open",
            "model_key": "bbox_2d",
            "empty_encoding": "[]",
        },
        "sequence": {
            "max_total_sequence_tokens": 1536,
            "max_completion_tokens": 192,
            "packing": False,
            "truncation": "error",
        },
        "adapter": {
            "peft_type": "LORA",
            "topology_id": "text-all+multimodal-mergers-v1",
            "rank": 16,
            "alpha": 32,
            "dropout": 0.05,
            "bias": "none",
            "target_module_count": 260,
            "target_modules_sha256": TARGET_MODULES_SHA256,
            "trainable_parameter_count": 44_793_856,
        },
    }
    contract["contract_fingerprint"] = hashlib.sha256(_canonical_json(contract)).hexdigest()
    return contract


def _verify_contract(contract: object) -> dict[str, Any]:
    if not isinstance(contract, dict):
        raise ArtifactError(f"{CONTRACT_FILE} must contain a JSON object")
    fingerprint = contract.get("contract_fingerprint")
    if not isinstance(fingerprint, str):
        raise ArtifactError("Artifact contract has no fingerprint")
    payload = dict(contract)
    del payload["contract_fingerprint"]
    actual = hashlib.sha256(_canonical_json(payload)).hexdigest()
    if actual != fingerprint:
        raise ArtifactError("Artifact contract fingerprint mismatch")
    expected = default_contract()
    if contract != expected:
        raise ArtifactError("Artifact contract is incompatible with ConceptDet v1")
    return contract


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"Cannot read {description}: {path}") from exc
    if not isinstance(value, dict):
        raise ArtifactError(f"{description} must contain a JSON object: {path}")
    return value


def _validate_peft_config(config: dict[str, Any]) -> None:
    expected = {
        "base_model_name_or_path": MODEL_ID,
        "peft_type": "LORA",
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "bias": "none",
    }
    for key, expected_value in expected.items():
        if config.get(key) != expected_value:
            raise ArtifactError(
                f"adapter_config.json {key}={config.get(key)!r} is incompatible; "
                f"expected {expected_value!r}"
            )
    targets = config.get("target_modules")
    if not isinstance(targets, list) or not all(isinstance(item, str) for item in targets):
        raise ArtifactError("adapter_config.json target_modules must be a string list")
    if set(targets) != TARGET_MODULE_PATTERNS:
        raise ArtifactError("adapter_config.json target_modules use an incompatible topology")


def validate_source_adapter(source: str | Path) -> dict[str, Any]:
    source_path = Path(source).expanduser().resolve()
    if not source_path.is_dir():
        raise ArtifactError(f"Source PEFT adapter does not exist: {source_path}")
    source_weights = source_path / WEIGHTS_FILE
    source_config = source_path / ADAPTER_CONFIG_FILE
    if not source_weights.is_file() or not source_config.is_file():
        raise ArtifactError(
            f"Source adapter needs {WEIGHTS_FILE} and {ADAPTER_CONFIG_FILE}: {source_path}"
        )
    adapter_config = _read_json(source_config, "source PEFT config")
    _validate_peft_config(adapter_config)
    return adapter_config


@dataclass(frozen=True)
class AdapterArtifact:
    path: Path
    contract: dict[str, Any]
    summary: dict[str, Any]
    adapter_config: dict[str, Any]

    @property
    def fingerprint(self) -> str:
        return str(self.summary["artifact_fingerprint"])

    @classmethod
    def load(cls, path: str | Path) -> AdapterArtifact:
        artifact_path = Path(path).expanduser().resolve()
        if not artifact_path.is_dir():
            raise ArtifactError(f"Artifact directory does not exist: {artifact_path}")
        required = (WEIGHTS_FILE, ADAPTER_CONFIG_FILE, CONTRACT_FILE, SUMMARY_FILE)
        missing = [name for name in required if not (artifact_path / name).is_file()]
        if missing:
            raise ArtifactError(f"Artifact is missing files: {', '.join(missing)}")
        contract = _verify_contract(_read_json(artifact_path / CONTRACT_FILE, "contract"))
        adapter_config = _read_json(artifact_path / ADAPTER_CONFIG_FILE, "PEFT config")
        _validate_peft_config(adapter_config)
        summary = _read_json(artifact_path / SUMMARY_FILE, "training summary")
        fingerprint = summary.get("artifact_fingerprint")
        if not isinstance(fingerprint, str):
            raise ArtifactError("training_summary.json has no Artifact fingerprint")
        expected_hashes = summary.get("files")
        if not isinstance(expected_hashes, dict):
            raise ArtifactError("training_summary.json has no file hashes")
        for name in (WEIGHTS_FILE, ADAPTER_CONFIG_FILE):
            if expected_hashes.get(name) != _sha256_file(artifact_path / name):
                raise ArtifactError(f"Artifact file hash mismatch: {name}")
        fingerprint_payload = dict(summary)
        del fingerprint_payload["artifact_fingerprint"]
        actual_fingerprint = hashlib.sha256(
            _canonical_json(
                {
                    "contract_fingerprint": contract["contract_fingerprint"],
                    "summary": fingerprint_payload,
                }
            )
        ).hexdigest()
        if fingerprint != actual_fingerprint:
            raise ArtifactError("Artifact fingerprint mismatch")
        return cls(artifact_path, contract, summary, adapter_config)


def initialize_artifact(config: ArtifactInitConfig) -> AdapterArtifact:
    source = config.source_adapter
    source_weights = source / WEIGHTS_FILE
    source_config = source / ADAPTER_CONFIG_FILE
    validate_source_adapter(source)
    if config.output_dir.exists():
        raise ArtifactError(f"Artifact output already exists: {config.output_dir}")
    if config.parent_artifact is not None:
        parent = AdapterArtifact.load(config.parent_artifact)
        parent_fingerprint: str | None = parent.fingerprint
    else:
        parent_fingerprint = None

    config.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{config.output_dir.name}.", dir=config.output_dir.parent)
    )
    try:
        shutil.copy2(source_weights, temporary / WEIGHTS_FILE)
        shutil.copy2(source_config, temporary / ADAPTER_CONFIG_FILE)
        contract = default_contract()
        (temporary / CONTRACT_FILE).write_text(
            json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        summary: dict[str, Any] = {
            "schema_version": 1,
            "stage": config.stage,
            "parent_artifact_fingerprint": parent_fingerprint,
            "source_adapter": str(source),
            "init_config_hash": config.config_hash,
            "files": {
                WEIGHTS_FILE: _sha256_file(temporary / WEIGHTS_FILE),
                ADAPTER_CONFIG_FILE: _sha256_file(temporary / ADAPTER_CONFIG_FILE),
            },
        }
        summary["artifact_fingerprint"] = hashlib.sha256(
            _canonical_json(
                {
                    "contract_fingerprint": contract["contract_fingerprint"],
                    "summary": summary,
                }
            )
        ).hexdigest()
        (temporary / SUMMARY_FILE).write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, config.output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return AdapterArtifact.load(config.output_dir)
