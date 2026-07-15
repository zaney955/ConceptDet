import json
from pathlib import Path

from conceptdet.acceptance import assemble_acceptance_report


def test_mandatory_skipped_gate_never_counts_as_pass(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    cpu = {
        "accepted": True,
        "gates": [
            {"gate": "C0", "status": "pass"},
            {"gate": "C1", "status": "pass"},
        ],
    }
    (evidence / "cpu_acceptance_report.json").write_text(
        json.dumps(cpu), encoding="utf-8"
    )
    report_path = assemble_acceptance_report(
        root=Path.cwd(),
        profile="pr",
        evidence_dir=evidence,
        output=tmp_path / "acceptance_report.json",
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["accepted"] is False
    c2 = next(gate for gate in report["gates"] if gate["gate"] == "C2")
    assert c2["mandatory"] is True
    assert c2["status"] == "skipped"


def test_optional_hardware_skip_does_not_fail_pr_profile(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "cpu_acceptance_report.json").write_text(
        json.dumps(
            {
                "accepted": True,
                "gates": [
                    {"gate": "C0", "status": "pass"},
                    {"gate": "C1", "status": "pass"},
                ],
            }
        ),
        encoding="utf-8",
    )
    (evidence / "c2_report.json").write_text(
        json.dumps({"passed": True}), encoding="utf-8"
    )
    report_path = assemble_acceptance_report(
        root=Path.cwd(),
        profile="pr",
        evidence_dir=evidence,
        output=tmp_path / "acceptance_report.json",
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["accepted"] is True
    assert report["controls"]["mandatory_skips_fail"] is True
