import json
from pathlib import Path

from PIL import Image

import conceptdet.cli as cli
from conceptdet.adapter import FakeAdapter
from conceptdet.artifact import (
    ADAPTER_CONFIG_FILE,
    TARGET_MODULE_PATTERNS,
    WEIGHTS_FILE,
    initialize_artifact,
)
from conceptdet.config import ArtifactInitConfig, load_config


def _artifact(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    (source / WEIGHTS_FILE).write_bytes(b"fake-safe-tensors")
    (source / ADAPTER_CONFIG_FILE).write_text(
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
    artifact = tmp_path / "artifact"
    initialize_artifact(
        ArtifactInitConfig(1, "artifact.init", source, artifact, "sft", None, tmp_path, "h")
    )
    return artifact


def _detect_config(tmp_path: Path, artifact: Path) -> Path:
    Image.new("RGB", (100, 100), "white").save(tmp_path / "reference.png")
    Image.new("RGB", (200, 100), "white").save(tmp_path / "target.png")
    config = tmp_path / "detect.yaml"
    config.write_text(
        f"""
schema_version: 1
kind: infer.detect
artifact: {artifact}
request:
  reference_image: reference.png
  reference_boxes: [[10, 10, 40, 40]]
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
""",
        encoding="utf-8",
    )
    return config


def test_validate_checks_resources_without_loading_model(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    config = _detect_config(tmp_path, _artifact(tmp_path))
    monkeypatch.setattr(cli, "_load_adapter", lambda _: (_ for _ in ()).throw(AssertionError()))
    assert cli.main(["config", "validate", "--config", str(config)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["kind"] == "infer.detect"
    assert len(payload["config_hash"]) == 64

    rendered = tmp_path / "rendered.yaml"
    assert (
        cli.main(
            ["config", "render", "--config", str(config), "--output", str(rendered)]
        )
        == 0
    )
    assert "config_hash:" not in rendered.read_text(encoding="utf-8")
    assert load_config(rendered).config_hash == payload["config_hash"]


def test_detect_cli_uses_application_seam_and_keeps_stdout_strict(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    config = _detect_config(tmp_path, _artifact(tmp_path))
    monkeypatch.setattr(
        cli,
        "_load_adapter",
        lambda _: FakeAdapter('[{"bbox_2d":[100,200,500,600]}]'),
    )
    assert cli.main(["infer", "detect", "--config", str(config)]) == 0
    captured = capsys.readouterr()
    assert captured.out == '[{"bbox_2d":[100,200,500,600]}]\n'
    assert "Output image:" in captured.err
    assert (tmp_path / "outputs/result.png").is_file()
    payload = json.loads((tmp_path / "outputs/result.json").read_text(encoding="utf-8"))
    assert payload["detections"][0]["bbox_xyxy"] == [20, 20, 100, 60]


def test_validate_fails_closed_for_missing_artifact(tmp_path: Path, capsys) -> None:
    config = _detect_config(tmp_path, tmp_path / "missing")
    assert cli.main(["config", "validate", "--config", str(config)]) == 2
    assert "Artifact directory does not exist" in capsys.readouterr().err


def test_infer_fails_before_adapter_load_when_an_input_is_missing(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    config = _detect_config(tmp_path, _artifact(tmp_path))
    (tmp_path / "target.png").unlink()
    monkeypatch.setattr(cli, "_load_adapter", lambda _: (_ for _ in ()).throw(AssertionError()))
    assert cli.main(["infer", "detect", "--config", str(config)]) == 2
    assert "Target image does not exist" in capsys.readouterr().err


def test_batch_cli_validates_manifest_and_writes_summary(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    artifact = _artifact(tmp_path)
    Image.new("RGB", (100, 100), "white").save(tmp_path / "reference.png")
    Image.new("RGB", (200, 100), "white").save(tmp_path / "target.png")
    (tmp_path / "tasks.jsonl").write_text(
        json.dumps(
            {
                "id": "sample-1",
                "reference_image": "reference.png",
                "reference_boxes": [[10, 10, 40, 40]],
                "target_image": "target.png",
                "query": "matching bolt",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = tmp_path / "batch.yaml"
    config.write_text(
        f"""
schema_version: 1
kind: infer.batch
artifact: {artifact}
manifest: tasks.jsonl
output_dir: outputs
runtime:
  device: cpu
  dtype: float32
  attention: eager
  max_new_tokens: 64
  local_files_only: true
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "_load_adapter", lambda _: FakeAdapter("[]"))
    assert cli.main(["config", "validate", "--config", str(config)]) == 0
    capsys.readouterr()
    assert cli.main(["infer", "batch", "--config", str(config)]) == 0
    summary = [
        json.loads(line)
        for line in (tmp_path / "outputs/results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert summary[0]["status"] == "ok"
    assert summary[0]["detection_set"] == []
