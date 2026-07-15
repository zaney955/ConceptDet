import json
from pathlib import Path
from runpy import run_path


def test_manual_config_script_has_valid_default_tasks() -> None:
    script = Path(__file__).parents[1] / "scripts" / "inference_config.py"
    namespace = run_path(str(script), run_name="manual_config_test")
    namespace["validate_config"](
        namespace["CONFIG"], namespace["TASKS"], namespace["BATCH_CONFIG"]
    )
    assert namespace["CONFIG"]["gpu_ids"]
    assert namespace["CONFIG"]["output_layout"] == "triptych"
    assert namespace["CONFIG"]["min_free_memory_gb"] > 0
    assert namespace["CONFIG"]["attention"] == "flash_attention_2"
    assert namespace["CONFIG"]["reference_box_width"] == 2
    assert namespace["BATCH_CONFIG"]["query"] == (
        "the same bolt as the red-boxed example in the reference image"
    )
    request = namespace["build_request"](namespace["TASKS"][0])
    assert request.reference_boxes
    assert request.query == "camouflaged object"
    assert request.reference_path.name == "cod_ref.png"
    specs = namespace["build_task_specs"](namespace["TASKS"])
    assert specs[0].index == 0
    assert specs[0].output_path.name == "config_demo.png"


def test_manual_config_builds_shared_reference_batch(tmp_path: Path) -> None:
    script = Path(__file__).parents[1] / "scripts" / "inference_config.py"
    namespace = run_path(str(script), run_name="manual_batch_config_test")
    reference = tmp_path / "reference.jpg"
    targets = tmp_path / "targets"
    output = tmp_path / "outputs"
    from PIL import Image

    Image.new("RGB", (20, 20), "white").save(reference)
    targets.mkdir()
    Image.new("RGB", (20, 20), "white").save(targets / "a.jpg")
    Image.new("RGB", (20, 20), "white").save(targets / "b.png")
    batch_config = {
        "input_paths": [str(targets)],
        "recursive": False,
        "reference_path": str(reference),
        "reference_boxes": "1,1,10,10",
        "query": "bolt",
        "output_dir": str(output),
    }
    specs = namespace["build_batch_task_specs"](batch_config)
    assert [spec.request.target_path.name for spec in specs] == ["a.jpg", "b.png"]
    assert [spec.output_path.name for spec in specs] == ["a.png", "b.png"]
    assert all(spec.request.reference_path == reference for spec in specs)


def test_batch_mode_can_skip_all_outputs_without_loading_cuda(tmp_path: Path) -> None:
    script = Path(__file__).parents[1] / "scripts" / "inference_config.py"
    namespace = run_path(str(script), run_name="manual_skip_config_test")
    from PIL import Image

    reference = tmp_path / "reference.jpg"
    targets = tmp_path / "targets"
    output = tmp_path / "outputs"
    log_path = output / "results.jsonl"
    Image.new("RGB", (20, 20), "white").save(reference)
    targets.mkdir()
    Image.new("RGB", (20, 20), "white").save(targets / "a.jpg")
    output.mkdir()
    Image.new("RGB", (20, 20), "white").save(output / "a.png")

    script_globals = namespace["main"].__globals__
    script_globals["CONFIG"] = {
        "mode": "batch",
        "model_path": str(tmp_path / "unused-model"),
        "gpu_ids": [0],
        "dtype": "bfloat16",
        "attention": "sdpa",
        "input_size": 600,
        "max_new_tokens": 32,
        "box_color": "red",
        "box_width": 2,
    }
    script_globals["BATCH_CONFIG"] = {
        "input_paths": [str(targets)],
        "recursive": False,
        "reference_path": str(reference),
        "reference_boxes": "1,1,10,10",
        "query": "bolt",
        "output_dir": str(output),
        "skip_existing": True,
        "log_path": str(log_path),
    }
    script_globals["TASKS"] = []
    namespace["main"]()

    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["status"] == "skipped"
    assert rows[0]["gpu_id"] is None
