import hashlib
import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

from conceptdet.config import DataVocConfig, SplitConfig, VocSourceConfig
from conceptdet.dataset import DatasetArtifact, compile_voc_dataset
from conceptdet.errors import DatasetError


def _voc(path: Path, filename: str, concept: str, *, second: bool = False) -> None:
    extra = f"""
  <object><name>{concept}</name><bndbox>
    <xmin>6</xmin><ymin>3</ymin><xmax>9</xmax><ymax>7</ymax>
  </bndbox></object>
""" if second else ""
    path.write_text(
        f"""<annotation>
  <filename>{filename}</filename>
  <size><width>12</width><height>10</height><depth>3</depth></size>
  <object><name>{concept}</name><bndbox>
    <xmin>2</xmin><ymin>2</ymin><xmax>5</xmax><ymax>6</ymax>
  </bndbox></object>{extra}
</annotation>
""",
        encoding="utf-8",
    )


def _empty_voc(path: Path, filename: str) -> None:
    path.write_text(
        f"""<annotation>
  <filename>{filename}</filename>
  <size><width>12</width><height>10</height><depth>3</depth></size>
</annotation>
""",
        encoding="utf-8",
    )


def _source(tmp_path: Path) -> VocSourceConfig:
    images = tmp_path / "images"
    annotations = tmp_path / "xml"
    images.mkdir()
    annotations.mkdir()
    for index in range(24):
        concept = "A" if index < 12 else "B"
        filename = f"image-{index:02d}.png"
        Image.new("RGB", (12, 10), (index * 10, index * 5, index)).save(images / filename)
        _voc(
            annotations / f"image-{index:02d}.xml",
            filename,
            concept,
            second=index == 0,
        )
    Image.new("RGB", (12, 10), (250, 100, 50)).save(images / "empty.png")
    _empty_voc(annotations / "empty.xml", "empty.png")
    Image.new("RGB", (12, 10), "white").save(images / "orphan.png")
    return VocSourceConfig("fixture", images, annotations)


def _config(tmp_path: Path, source: VocSourceConfig) -> DataVocConfig:
    return DataVocConfig(
        1,
        "data.voc",
        (source,),
        None,
        tmp_path / "compiled",
        "voc_inclusive",
        1,
        SplitConfig(0.8, 0.1, 0.1, 17),
        tmp_path / "config.yaml",
        "config-hash",
    )


def test_compile_voc_is_deterministic_complete_and_leakage_safe(tmp_path: Path) -> None:
    config = _config(tmp_path, _source(tmp_path))
    first = compile_voc_dataset(config)
    first_bytes = {
        path.name: path.read_bytes() for path in sorted(first.path.iterdir()) if path.is_file()
    }
    records = [
        row
        for split in ("train", "validation", "test")
        for row in first.iter_records(split)
    ]
    assert len(records) == 49
    assert sum(bool(row["positive"]) for row in records) == 24
    assert sum(not bool(row["positive"]) for row in records) == 25
    empty = next(row for row in records if row["target"]["path"] == "empty.png")
    assert empty["detection_set"] == []
    multi = next(
        row
        for row in records
        if row["target"]["path"] == "image-00.png" and row["positive"]
    )
    assert multi["target"]["boxes_xyxy"] == [[1, 1, 5, 6], [5, 2, 9, 7]]
    assert len(multi["detection_set"]) == 2
    for row in records:
        assert row["reference"]["path"] != row["target"]["path"]
        assert row["query"] == "the same Visual Concept as the boxed Reference Instances"
    audit = json.loads((first.path / "audit.json").read_text(encoding="utf-8"))
    assert audit["orphan_images"] == {"fixture": ["orphan"]}

    shutil.rmtree(config.output_dir)
    second = compile_voc_dataset(config)
    second_bytes = {
        path.name: path.read_bytes() for path in sorted(second.path.iterdir()) if path.is_file()
    }
    assert {
        name: hashlib.sha256(content).hexdigest() for name, content in first_bytes.items()
    } == {name: hashlib.sha256(content).hexdigest() for name, content in second_bytes.items()}


def test_compiled_dataset_detects_manifest_tampering(tmp_path: Path) -> None:
    artifact = compile_voc_dataset(_config(tmp_path, _source(tmp_path)))
    with (artifact.path / "train.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(DatasetError, match="hash mismatch"):
        DatasetArtifact.load(artifact.path)
