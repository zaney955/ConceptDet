import json
from pathlib import Path

import pytest

from conceptdet.errors import ModelLoadError
from conceptdet.model import TransformersBackend


def test_checkpoint_validation_accepts_conceptseg_config(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen2_vl"}), encoding="utf-8")
    TransformersBackend._validate_checkpoint(tmp_path)


def test_checkpoint_validation_rejects_other_architecture(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}), encoding="utf-8")
    with pytest.raises(ModelLoadError):
        TransformersBackend._validate_checkpoint(tmp_path)


def test_loading_info_allows_only_removed_segmentation_weights() -> None:
    TransformersBackend._validate_loading_info(
        {
            "missing_keys": [],
            "unexpected_keys": [
                "learnable_query.param.weight",
                "connector.blocks.0.attn.q.weight",
                "proj_to_sam.0.weight",
                "conv_1d.weight",
                "conv_1d.bias",
            ],
        }
    )
    with pytest.raises(ModelLoadError):
        TransformersBackend._validate_loading_info(
            {"missing_keys": [], "unexpected_keys": ["unknown.weight"]}
        )
