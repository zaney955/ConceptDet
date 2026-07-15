import json
from pathlib import Path

import pytest

from conceptdet.artifact import (
    ADAPTER_CONFIG_FILE,
    CONTRACT_FILE,
    TARGET_MODULE_PATTERNS,
    WEIGHTS_FILE,
    AdapterArtifact,
    initialize_artifact,
)
from conceptdet.config import ArtifactInitConfig
from conceptdet.errors import ArtifactError


def _source_adapter(path: Path) -> Path:
    path.mkdir()
    (path / WEIGHTS_FILE).write_bytes(b"fake-safe-tensors")
    (path / ADAPTER_CONFIG_FILE).write_text(
        json.dumps(
            {
                "base_model_name_or_path": "Qwen/Qwen3-VL-8B-Instruct",
                "peft_type": "LORA",
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "bias": "none",
                "target_modules": sorted(TARGET_MODULE_PATTERNS),
            }
        ),
        encoding="utf-8",
    )
    return path


def test_initialize_and_inspect_immutable_artifact(tmp_path: Path) -> None:
    source = _source_adapter(tmp_path / "source")
    output = tmp_path / "artifact"
    config = ArtifactInitConfig(
        1,
        "artifact.init",
        source,
        output,
        "sft",
        None,
        tmp_path / "config.yaml",
        "config-hash",
    )
    created = initialize_artifact(config)
    assert created.path == output
    assert created.summary["stage"] == "sft"
    assert len(created.fingerprint) == 64
    assert AdapterArtifact.load(output).fingerprint == created.fingerprint
    with pytest.raises(ArtifactError, match="already exists"):
        initialize_artifact(config)


def test_artifact_detects_contract_and_weight_tampering(tmp_path: Path) -> None:
    source = _source_adapter(tmp_path / "source")
    output = tmp_path / "artifact"
    config = ArtifactInitConfig(1, "artifact.init", source, output, "sft", None, tmp_path, "h")
    initialize_artifact(config)

    contract = json.loads((output / CONTRACT_FILE).read_text(encoding="utf-8"))
    contract["sequence"]["packing"] = True
    (output / CONTRACT_FILE).write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(ArtifactError, match="fingerprint"):
        AdapterArtifact.load(output)

    # Restore by recreating, then alter the immutable payload.
    for child in output.iterdir():
        child.unlink()
    output.rmdir()
    initialize_artifact(config)
    (output / WEIGHTS_FILE).write_bytes(b"tampered")
    with pytest.raises(ArtifactError, match="hash mismatch"):
        AdapterArtifact.load(output)
