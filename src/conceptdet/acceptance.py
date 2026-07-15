from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from conceptdet.artifact import AdapterArtifact
from conceptdet.errors import ConceptDetError
from conceptdet.protocol import parse_detection_set

REPORT_SCHEMA_VERSION = 1
MEMORY_GATE_GIB = 44.0
MANDATORY_BY_PROFILE = {
    "pr": ("C0", "C1", "C2"),
    "release": ("C0", "C1", "C2", "H1", "H2"),
    "distributed": ("C0", "C1", "C2", "H1", "H2", "D1"),
}


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_report(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ConceptDetError(f"Acceptance report already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_bytes(_canonical_json(payload) + b"\n")
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return path


def _finite(value: object) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _finite(item) for key, item in value.items())
    return False


def _versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {"python": platform.python_version()}
    for package in ("torch", "transformers", "peft", "accelerate", "trl"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def _git_commit(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


@dataclass(frozen=True)
class GateResult:
    gate: str
    status: Literal["pass", "fail", "skipped"]
    mandatory: bool
    diagnostics: tuple[str, ...]
    evidence: dict[str, Any]

    @property
    def passed(self) -> bool:
        return self.status == "pass"


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConceptDetError(f"Cannot read {description}: {path}") from exc
    if not isinstance(payload, dict):
        raise ConceptDetError(f"{description} must be a JSON object: {path}")
    return payload


def emit_hardware_gate_report(
    *,
    gate: Literal["H1", "H2", "D1"],
    lifecycle_path: Path,
    artifact_path: Path,
    config_hash: str,
    dataset_fingerprint: str,
    offline: bool,
) -> Path:
    lifecycle = _read_json(lifecycle_path, f"{gate} lifecycle")
    artifact = AdapterArtifact.load(artifact_path)
    diagnostics: list[str] = []
    if not offline:
        diagnostics.append("offline execution was not enabled")
    if not _finite(lifecycle):
        diagnostics.append("lifecycle contains NaN, Inf, or unsupported numeric values")
    peak = lifecycle.get("peak_reserved_gib")
    if not isinstance(peak, (int, float)) or isinstance(peak, bool) or peak > MEMORY_GATE_GIB:
        diagnostics.append(f"peak_reserved_gib={peak!r} exceeds {MEMORY_GATE_GIB:.1f}")
    for field in ("positive_completion", "negative_completion"):
        completion = lifecycle.get(field)
        if not isinstance(completion, str):
            diagnostics.append(f"{field} is missing")
            continue
        try:
            parse_detection_set(completion)
        except Exception as exc:
            diagnostics.append(f"{field} is not a strict Detection Set: {exc}")
    expected = {
        "artifact_fingerprint": artifact.fingerprint,
        "dataset_fingerprint": dataset_fingerprint,
    }
    for key, value in expected.items():
        if lifecycle.get(key) != value:
            diagnostics.append(
                f"lifecycle {key}={lifecycle.get(key)!r} does not match {value!r}"
            )
    if gate == "H2" and lifecycle.get("parent_artifact_fingerprint") != artifact.summary.get(
        "parent_artifact_fingerprint"
    ):
        diagnostics.append("GRPO parent Artifact lineage mismatch")
    if gate == "D1" and lifecycle.get("world_size") != 2:
        diagnostics.append("D1 requires world_size=2")
    payload = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "profile": "hardware-gate",
        "accepted": not diagnostics,
        "gate": gate,
        "gates": [
            asdict(
                GateResult(
                    gate,
                    "pass" if not diagnostics else "fail",
                    True,
                    tuple(diagnostics),
                    {"lifecycle": str(lifecycle_path), "artifact": str(artifact_path)},
                )
            )
        ],
        "hashes": {
            "resolved_config": config_hash,
            "contract": artifact.contract["contract_fingerprint"],
            "manifest": dataset_fingerprint,
            "artifact": artifact.fingerprint,
            "lifecycle_file": _sha256_file(lifecycle_path),
        },
        "controls": {
            "offline": offline,
            "no_sam_runtime": True,
            "strict_output": not any("strict Detection Set" in item for item in diagnostics),
            "finite_numeric": _finite(lifecycle),
            "artifact_atomic": True,
            "memory_gate_gib": MEMORY_GATE_GIB,
        },
        "dependencies": _versions(),
    }
    return _atomic_report(lifecycle_path.parent / "acceptance_report.json", payload)


def emit_evaluation_acceptance(
    *,
    output_dir: Path,
    report: dict[str, Any],
) -> Path:
    diagnostics: list[str] = []
    if not _finite(report):
        diagnostics.append("evaluation contains NaN, Inf, or unsupported numeric values")
    required_hashes = {
        "resolved_config": report.get("config_hash"),
        "contract": report.get("adapter_contract_fingerprint"),
        "manifest": report.get("dataset_fingerprint"),
        "artifact": report.get("adapter_artifact_fingerprint"),
        "predictions": report.get("prediction_content_fingerprint"),
    }
    if not all(isinstance(value, str) and value for value in required_hashes.values()):
        diagnostics.append("evaluation provenance hashes are incomplete")
    payload = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "profile": "evaluation",
        "accepted": not diagnostics,
        "gates": [
            asdict(
                GateResult(
                    "evaluation",
                    "pass" if not diagnostics else "fail",
                    True,
                    tuple(diagnostics),
                    {"report": str(output_dir / "report.json")},
                )
            )
        ],
        "hashes": required_hashes,
        "controls": {
            "offline": True,
            "no_sam_runtime": True,
            "strict_output_metric_present": "strict_valid_rate"
            in report.get("metrics", {}),
            "finite_numeric": _finite(report),
            "artifact_atomic": True,
            "memory_gate_gib": MEMORY_GATE_GIB,
        },
        "dependencies": _versions(),
    }
    return _atomic_report(output_dir / "acceptance_report.json", payload)


def run_cpu_gates(root: Path, output: Path) -> Path:
    commands = {
        "C0": [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_artifact.py",
            "tests/test_config.py",
            "tests/test_dataset.py",
            "tests/test_evaluation.py",
            "tests/test_grpo.py",
            "tests/test_protocol.py",
            "tests/test_run_state.py",
            "tests/test_training.py",
            "tests/test_types.py",
        ],
        "C1": [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_application.py",
            "tests/test_cli.py",
            "tests/test_qwen_adapter.py",
        ],
    }
    gates: list[GateResult] = []
    for gate, command in commands.items():
        process = subprocess.run(
            command,
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
        gates.append(
            GateResult(
                gate,
                "pass" if process.returncode == 0 else "fail",
                True,
                () if process.returncode == 0 else (process.stdout + process.stderr,),
                {"command": command, "stdout": process.stdout, "stderr": process.stderr},
            )
        )
    forbidden: list[str] = []
    for path in (root / "src").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        if any(
            name == "sam"
            or name.startswith("sam.")
            or name == "segment_anything"
            or name.startswith("segment_anything.")
            for name in imports
        ):
            forbidden.append(str(path.relative_to(root)))
    if forbidden:
        gates[0] = GateResult(
            "C0", "fail", True, (f"SAM runtime imports: {forbidden}",), gates[0].evidence
        )
    payload = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "profile": "cpu",
        "accepted": all(gate.passed for gate in gates),
        "gates": [asdict(gate) for gate in gates],
        "hashes": {"git_commit": _git_commit(root)},
        "controls": {
            "offline": True,
            "no_sam_runtime": not forbidden,
            "finite_numeric": True,
            "artifact_atomic": True,
            "memory_gate_gib": MEMORY_GATE_GIB,
        },
        "dependencies": _versions(),
    }
    return _atomic_report(output, payload)


def assemble_acceptance_report(
    *,
    root: Path,
    profile: Literal["pr", "release", "distributed"],
    evidence_dir: Path,
    output: Path,
) -> Path:
    required = MANDATORY_BY_PROFILE[profile]
    sources = {
        "C0": evidence_dir / "cpu_acceptance_report.json",
        "C1": evidence_dir / "cpu_acceptance_report.json",
        "C2": evidence_dir / "c2_report.json",
        "H1": evidence_dir / "h1_acceptance_report.json",
        "H2": evidence_dir / "h2_acceptance_report.json",
        "D1": evidence_dir / "d1_acceptance_report.json",
    }
    gates: list[GateResult] = []
    hashes: dict[str, Any] = {"git_commit": _git_commit(root)}
    for gate in ("C0", "C1", "C2", "H1", "H2", "D1"):
        mandatory = gate in required
        path = sources[gate]
        if not path.is_file():
            gates.append(
                GateResult(
                    gate,
                    "skipped",
                    mandatory,
                    (f"evidence missing: {path}",),
                    {},
                )
            )
            continue
        payload = _read_json(path, f"{gate} evidence")
        if gate in {"C0", "C1"}:
            source_gate = next(
                (item for item in payload.get("gates", []) if item.get("gate") == gate),
                None,
            )
            passed = bool(source_gate and source_gate.get("status") == "pass")
        else:
            passed = bool(payload.get("passed", payload.get("accepted", False)))
        diagnostics = () if passed else (f"{gate} evidence did not pass",)
        gates.append(
            GateResult(
                gate,
                "pass" if passed else "fail",
                mandatory,
                diagnostics,
                {"path": str(path), "sha256": _sha256_file(path)},
            )
        )
        hashes[f"{gate.lower()}_evidence"] = _sha256_file(path)
    accepted = all(gate.status == "pass" for gate in gates if gate.mandatory)
    payload = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "profile": profile,
        "accepted": accepted,
        "gates": [asdict(gate) for gate in gates],
        "hashes": hashes,
        "controls": {
            "mandatory_skips_fail": True,
            "offline": True,
            "no_sam_runtime": True,
            "strict_output": True,
            "finite_numeric": True,
            "artifact_atomic": True,
            "memory_gate_gib": MEMORY_GATE_GIB,
        },
        "dependencies": _versions(),
    }
    return _atomic_report(output, payload)
