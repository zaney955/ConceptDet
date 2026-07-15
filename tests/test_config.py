from pathlib import Path

import pytest

from conceptdet.config import ArtifactInitConfig, DetectConfig, config_to_dict, load_config
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


def test_reserved_training_kind_has_targeted_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="not implemented"):
        load_config(
            _write(tmp_path / "sft.yaml", "schema_version: 1\nkind: train.sft\n")
        )


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
