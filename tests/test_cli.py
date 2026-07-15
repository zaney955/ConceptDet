import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

import conceptdet.cli as cli
from conceptdet.adapter import FakeAdapter
from conceptdet.artifact import (
    ADAPTER_CONFIG_FILE,
    TARGET_MODULE_PATTERNS,
    WEIGHTS_FILE,
    initialize_artifact,
)
from conceptdet.config import (
    ArtifactInitConfig,
    DatasetPredictionConfig,
    DataVocConfig,
    EvaluationConfig,
    GRPOOptimizationConfig,
    GRPOStageConfig,
    OptimizationConfig,
    RuntimeConfig,
    SFTStageConfig,
    SplitConfig,
    VocSourceConfig,
    load_config,
)


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


def test_data_voc_cli_reports_dataset_fingerprint(tmp_path: Path, monkeypatch, capsys) -> None:
    config = DataVocConfig(
        1,
        "data.voc",
        (VocSourceConfig("fixture", tmp_path, tmp_path),),
        None,
        tmp_path / "compiled",
        "xyxy_half_open",
        1,
        SplitConfig(),
        tmp_path / "data.yaml",
        "config-hash",
    )
    compiled = SimpleNamespace(
        path=config.output_dir,
        fingerprint="dataset-fingerprint",
        metadata={"files": {"train.jsonl": {"records": 3}}},
    )
    monkeypatch.setattr(cli, "load_config", lambda _: config)
    monkeypatch.setattr(cli, "compile_voc_dataset", lambda _: compiled)
    assert cli.main(["data", "voc", "--config", str(config.config_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dataset_fingerprint"] == "dataset-fingerprint"


def test_train_sft_cli_forwards_explicit_resume(tmp_path: Path, monkeypatch, capsys) -> None:
    import conceptdet.training as training

    config = SFTStageConfig(
        1,
        "train.sft",
        tmp_path / "dataset",
        tmp_path / "work",
        tmp_path / "artifact",
        RuntimeConfig(),
        OptimizationConfig(max_steps=2),
        tmp_path / "sft.yaml",
        "config-hash",
    )
    received: list[object] = []

    def fake_run(_: object, *, resume: object) -> SimpleNamespace:
        received.append(resume)
        return SimpleNamespace(
            artifact=SimpleNamespace(path=config.artifact_dir, fingerprint="artifact-hash"),
            optimizer_steps=2,
            micro_steps=6,
            final_loss=1.0,
            peak_reserved_gib=28.0,
            lifecycle_report=config.work_dir / "lifecycle.json",
        )

    monkeypatch.setattr(cli, "load_config", lambda _: config)
    monkeypatch.setattr(training, "run_sft", fake_run)
    checkpoint = tmp_path / "checkpoint-1"
    assert (
        cli.main(
            [
                "train",
                "sft",
                "--config",
                str(config.config_path),
                "--resume",
                str(checkpoint),
            ]
        )
        == 0
    )
    assert received == [checkpoint]
    assert json.loads(capsys.readouterr().out)["artifact_fingerprint"] == "artifact-hash"


def test_evaluate_cli_forwards_worker_count(tmp_path: Path, monkeypatch, capsys) -> None:
    import conceptdet.evaluation as evaluation

    config = EvaluationConfig(
        1,
        "evaluate",
        tmp_path / "dataset",
        tmp_path / "adapter",
        tmp_path / "predictions.jsonl",
        "test",
        tmp_path / "evaluation",
        tmp_path / "evaluate.yaml",
        "config-hash",
    )
    received: list[int] = []

    def fake_evaluate(_: object, *, workers: int) -> SimpleNamespace:
        received.append(workers)
        return SimpleNamespace(
            path=config.output_dir,
            fingerprint="evaluation-hash",
            report={"metrics": {"positive_macro_mean_set_f1_50_95": 0.5}},
        )

    monkeypatch.setattr(cli, "load_config", lambda _: config)
    monkeypatch.setattr(evaluation, "evaluate", fake_evaluate)
    assert (
        cli.main(
            [
                "evaluate",
                "--config",
                str(config.config_path),
                "--workers",
                "4",
            ]
        )
        == 0
    )
    assert received == [4]
    assert json.loads(capsys.readouterr().out)["evaluation_fingerprint"] == (
        "evaluation-hash"
    )


def test_predict_dataset_cli_reports_complete_output(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    import conceptdet.prediction as prediction

    config = DatasetPredictionConfig(
        1,
        "predict.dataset",
        tmp_path / "dataset",
        tmp_path / "adapter",
        tmp_path / "predictions.jsonl",
        "test",
        RuntimeConfig(max_new_tokens=128),
        tmp_path / "predict.yaml",
        "config-hash",
    )

    def fake_predict(_: object) -> SimpleNamespace:
        return SimpleNamespace(
            path=config.predictions,
            records=713,
            content_sha256="prediction-hash",
        )

    monkeypatch.setattr(cli, "load_config", lambda _: config)
    monkeypatch.setattr(prediction, "generate_dataset_predictions", fake_predict)
    assert (
        cli.main(["predict", "dataset", "--config", str(config.config_path)])
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "predictions": str(config.predictions),
        "records": 713,
        "content_sha256": "prediction-hash",
    }


def test_train_grpo_cli_uses_separate_stage_and_forwards_resume(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    import conceptdet.grpo as grpo

    config = GRPOStageConfig(
        1,
        "train.grpo",
        tmp_path / "dataset",
        tmp_path / "sft-artifact",
        tmp_path / "work",
        tmp_path / "grpo-artifact",
        RuntimeConfig(max_new_tokens=192),
        GRPOOptimizationConfig(max_steps=2),
        tmp_path / "grpo.yaml",
        "config-hash",
    )
    received: list[object] = []

    def fake_run(_: object, *, resume: object) -> SimpleNamespace:
        received.append(resume)
        return SimpleNamespace(
            artifact=SimpleNamespace(path=config.artifact_dir, fingerprint="grpo-hash"),
            optimizer_steps=2,
            reward_events=4,
            nonzero_advantage_groups=1,
            peak_reserved_gib=19.0,
            lifecycle_report=config.work_dir / "lifecycle.json",
        )

    monkeypatch.setattr(cli, "load_config", lambda _: config)
    monkeypatch.setattr(grpo, "run_grpo", fake_run)
    assert (
        cli.main(
            [
                "train",
                "grpo",
                "--config",
                str(config.config_path),
                "--resume",
                "none",
            ]
        )
        == 0
    )
    assert received == ["none"]
    payload = json.loads(capsys.readouterr().out)
    assert payload["artifact_fingerprint"] == "grpo-hash"
    assert payload["nonzero_advantage_groups"] == 1
