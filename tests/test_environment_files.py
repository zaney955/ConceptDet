from pathlib import Path


REPO_ROOT = Path(__file__).parents[1]


def test_runtime_versions_are_locked_to_modern_stack() -> None:
    requirements = (
        REPO_ROOT / "requirements" / "runtime.txt"
    ).read_text(encoding="utf-8")
    assert "torch==2.13.0" in requirements
    assert "torchvision==0.28.0" in requirements
    assert "transformers==5.13.1" in requirements
    assert "accelerate==1.14.0" in requirements
    assert "pillow==12.3.0" in requirements
    assert "sentencepiece==0.2.2" in requirements


def test_environment_scripts_never_reference_conceptseg_runtime() -> None:
    for name in ("create_env.sh", "run_inference.sh", "check_environment.py"):
        content = (REPO_ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "ConceptSeg-R1/.venv" not in content
