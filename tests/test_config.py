from pathlib import Path

import pytest

from conceptdet.config import (
    ArtifactInitConfig,
    DatasetPredictionConfig,
    DataVocConfig,
    DetectConfig,
    EvaluationConfig,
    GRPOStageConfig,
    SFTStageConfig,
    config_to_dict,
    load_config,
)
from conceptdet.errors import ConfigurationError
from conceptdet.types import Box


def _write(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def _detect_yaml(extra: str = "") -> str:
    return f"""
schema_version: 1
kind: infer.detect
artifact: artifact
request:
  reference_image: reference.png
  reference_boxes: [[10, 20, 40, 60]]
  target_image: target.png
  query: matching bolt
output:
  image: outputs/result.png
runtime:
  device: cpu
  dtype: float32
  attention: eager
  max_new_tokens: 64
  local_files_only: true
{extra}
"""


def test_detect_config_is_typed_resolved_and_hashed(tmp_path: Path) -> None:
    config = load_config(_write(tmp_path / "detect.yaml", _detect_yaml()))
    assert isinstance(config, DetectConfig)
    assert config.request.reference_boxes == (Box(10, 20, 40, 60),)
    assert config.request.reference_image == (tmp_path / "reference.png").resolve()
    assert config.output.json == (tmp_path / "outputs/result.json").resolve()
    assert len(config.config_hash) == 64
    assert config_to_dict(config)["runtime"]["max_new_tokens"] == 64


def test_hash_uses_semantic_defaults_not_yaml_spelling(tmp_path: Path) -> None:
    implicit = load_config(_write(tmp_path / "implicit.yaml", _detect_yaml()))
    explicit_body = _detect_yaml().replace(
        "output:\n  image: outputs/result.png",
        "output:\n  image: outputs/result.png\n  json: outputs/result.json\n  layout: annotated",
    )
    explicit = load_config(_write(tmp_path / "explicit.yaml", explicit_body))
    assert implicit.config_hash == explicit.config_hash


def test_config_rejects_duplicates_unknown_and_legacy_fields(tmp_path: Path) -> None:
    duplicate = _detect_yaml() + "kind: infer.batch\n"
    with pytest.raises(ConfigurationError, match="Duplicate YAML key"):
        load_config(_write(tmp_path / "duplicate.yaml", duplicate))

    with pytest.raises(ConfigurationError, match="unknown fields"):
        load_config(_write(tmp_path / "unknown.yaml", _detect_yaml("mystery: 1")))

    with pytest.raises(ConfigurationError, match="clean break"):
        load_config(_write(tmp_path / "legacy.yaml", _detect_yaml("input_size: 600")))


def test_grpo_config_freezes_native_generation_contract(tmp_path: Path) -> None:
    body = """
schema_version: 1
kind: train.grpo
dataset_dir: compiled
parent_artifact: sft-artifact
work_dir: grpo-work
artifact_dir: grpo-artifact
runtime: {device: cuda:0, max_new_tokens: 192, local_files_only: true}
optimization: {max_steps: 2, gradient_accumulation_steps: 2}
"""
    config = load_config(_write(tmp_path / "grpo.yaml", body))
    assert isinstance(config, GRPOStageConfig)
    assert config.parent_artifact == (tmp_path / "sft-artifact").resolve()
    assert config.optimization.learning_rate == 1e-5
    assert config.runtime.max_new_tokens == 192

    with pytest.raises(ConfigurationError, match="exactly 192"):
        load_config(_write(tmp_path / "grpo-short.yaml", body.replace("192", "128")))


def test_artifact_init_requires_grpo_parent(tmp_path: Path) -> None:
    body = """
schema_version: 1
kind: artifact.init
source_adapter: source
output_dir: output
stage: grpo
"""
    with pytest.raises(ConfigurationError, match="parent_artifact"):
        load_config(_write(tmp_path / "artifact.yaml", body))

    body = body + "parent_artifact: parent\n"
    config = load_config(_write(tmp_path / "artifact-valid.yaml", body))
    assert isinstance(config, ArtifactInitConfig)
    assert config.parent_artifact == (tmp_path / "parent").resolve()


def test_data_voc_and_sft_configs_are_strict_and_resolved(tmp_path: Path) -> None:
    data = load_config(
        _write(
            tmp_path / "data.yaml",
            """
schema_version: 1
kind: data.voc
sources:
  - name: fixture
    images: images
    annotations: xml
classes: all
output_dir: compiled
negative_per_image: 1
splits: {train: 0.8, validation: 0.1, test: 0.1, seed: 17}
""",
        )
    )
    assert isinstance(data, DataVocConfig)
    assert data.sources[0].image_dir == (tmp_path / "images").resolve()
    assert data.classes is None

    sft = load_config(
        _write(
            tmp_path / "sft.yaml",
            """
schema_version: 1
kind: train.sft
dataset_dir: compiled
work_dir: work
artifact_dir: artifact
runtime: {device: cuda:0, local_files_only: true}
optimization: {epochs: 1, max_steps: 2, gradient_accumulation_steps: 2}
""",
        )
    )
    assert isinstance(sft, SFTStageConfig)
    assert sft.optimization.max_steps == 2
    assert sft.runtime.local_files_only is True


def test_evaluation_config_has_frozen_metrics_and_resolved_inputs(tmp_path: Path) -> None:
    config = load_config(
        _write(
            tmp_path / "evaluate.yaml",
            """
schema_version: 1
kind: evaluate
dataset_dir: compiled
artifact: adapter
predictions: predictions.jsonl
split: validation
output_dir: evaluation
""",
        )
    )
    assert isinstance(config, EvaluationConfig)
    assert config.dataset_dir == (tmp_path / "compiled").resolve()
    assert config.artifact == (tmp_path / "adapter").resolve()
    assert config.predictions == (tmp_path / "predictions.jsonl").resolve()
    assert config.split == "validation"
    assert set(config_to_dict(config)) == {
        "schema_version",
        "kind",
        "dataset_dir",
        "artifact",
        "predictions",
        "split",
        "output_dir",
        "config_hash",
    }


def test_dataset_prediction_config_is_typed_and_resolved(tmp_path: Path) -> None:
    config = load_config(
        _write(
            tmp_path / "predict.yaml",
            """
schema_version: 1
kind: predict.dataset
dataset_dir: compiled
artifact: adapter
predictions: predictions.jsonl
split: test
runtime: {device: auto, max_new_tokens: 128, local_files_only: true}
""",
        )
    )
    assert isinstance(config, DatasetPredictionConfig)
    assert config.dataset_dir == (tmp_path / "compiled").resolve()
    assert config.predictions == (tmp_path / "predictions.jsonl").resolve()
    assert config.runtime.max_new_tokens == 128
    assert config_to_dict(config)["kind"] == "predict.dataset"
