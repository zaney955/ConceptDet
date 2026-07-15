from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import unicodedata
import warnings
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from conceptdet.config import DataVocConfig, SplitConfig, VocSourceConfig
from conceptdet.errors import DatasetError
from conceptdet.protocol import ProtocolDetection, encode_pixel_box
from conceptdet.types import Box

DATASET_FILE = "dataset.json"
AUDIT_FILE = "audit.json"
SPLITS = ("train", "validation", "test")
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})
QUERY = "the same Visual Concept as the boxed Reference Instances"


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(*parts: object) -> str:
    return hashlib.sha256("\x1f".join(str(part) for part in parts).encode()).hexdigest()


@dataclass(frozen=True)
class VocObject:
    visual_concept: str
    box: Box


@dataclass(frozen=True)
class VocImage:
    record_id: str
    source: str
    relative_path: str
    path: Path
    width: int
    height: int
    objects: tuple[VocObject, ...]
    content_sha256: str
    canonical_group_key: str

    @property
    def concepts(self) -> frozenset[str]:
        return frozenset(item.visual_concept for item in self.objects)

    def boxes_for(self, visual_concept: str) -> tuple[Box, ...]:
        return tuple(
            item.box for item in self.objects if item.visual_concept == visual_concept
        )


class _Groups:
    def __init__(self, identifiers: list[str]) -> None:
        self.parent = {identifier: identifier for identifier in identifiers}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root))
        self.parent[second] = first


def _canonical_group_key(source: str, stem: str) -> str:
    normalized = unicodedata.normalize("NFKC", stem).lower()
    normalized = re.sub(r"^(?:gx|rxxg|rxzc)cam_", "", normalized)
    normalized = re.sub(r"_artificial_(?:obstacles|scene(?:_aug\d+)?)$", "", normalized)
    operational = re.search(r"_(k\d+)_([^_]+)_([^_]+)_(?:0[1-9]-c\d+)$", normalized)
    if operational:
        parts = (operational.group(1), operational.group(2), operational.group(3))
        return f"{source}:sequence:{':'.join(parts)}"
    return f"{source}:name:{normalized}"


def _image_index(source: VocSourceConfig) -> tuple[dict[str, Path], list[str]]:
    if not source.image_dir.is_dir():
        raise DatasetError(f"VOC image directory does not exist: {source.image_dir}")
    if not source.annotation_dir.is_dir():
        raise DatasetError(f"VOC annotation directory does not exist: {source.annotation_dir}")
    indexed: dict[str, Path] = {}
    duplicates: list[str] = []
    for path in sorted(source.image_dir.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if path.stem in indexed:
            duplicates.append(path.stem)
        indexed[path.stem] = path.resolve()
    if duplicates:
        raise DatasetError(
            f"VOC source {source.name} has duplicate image stems: {', '.join(duplicates[:10])}"
        )
    return indexed, sorted(indexed)


def _required_text(parent: ET.Element, name: str, context: str) -> str:
    value = parent.findtext(name)
    if value is None or not value.strip():
        raise DatasetError(f"{context}: missing <{name}>")
    return value.strip()


def _required_int(parent: ET.Element, name: str, context: str) -> int:
    value = _required_text(parent, name, context)
    try:
        return int(value)
    except ValueError as exc:
        raise DatasetError(f"{context}: <{name}> must be an integer, got {value!r}") from exc


def _parse_voc(
    xml_path: Path,
    image_path: Path,
    source: VocSourceConfig,
    semantics: str,
) -> VocImage:
    context = f"{source.name}/{xml_path.name}"
    try:
        xml_bytes = xml_path.read_bytes()
        if b"<!DOCTYPE" in xml_bytes or b"<!ENTITY" in xml_bytes:
            raise DatasetError(f"{context}: XML entities and doctypes are forbidden")
        root = ET.fromstring(xml_bytes)
    except (OSError, ET.ParseError) as exc:
        raise DatasetError(f"{context}: malformed XML: {exc}") from exc
    size = root.find("size")
    if size is None:
        raise DatasetError(f"{context}: missing <size>")
    xml_width = _required_int(size, "width", context)
    xml_height = _required_int(size, "height", context)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(image_path) as image:
                actual_width, actual_height = image.size
                image.verify()
    except OSError as exc:
        raise DatasetError(f"{context}: unreadable image {image_path}: {exc}") from exc
    if (xml_width, xml_height) != (actual_width, actual_height):
        raise DatasetError(
            f"{context}: XML size {(xml_width, xml_height)} != image size "
            f"{(actual_width, actual_height)}"
        )

    objects: list[VocObject] = []
    for object_index, item in enumerate(root.findall("object")):
        object_context = f"{context}:object[{object_index}]"
        visual_concept = _required_text(item, "name", object_context)
        bounds = item.find("bndbox")
        if bounds is None:
            raise DatasetError(f"{object_context}: missing <bndbox>")
        xmin = _required_int(bounds, "xmin", object_context)
        ymin = _required_int(bounds, "ymin", object_context)
        xmax = _required_int(bounds, "xmax", object_context)
        ymax = _required_int(bounds, "ymax", object_context)
        if semantics == "voc_inclusive":
            coordinates = (xmin - 1, ymin - 1, xmax, ymax)
        else:
            coordinates = (xmin, ymin, xmax, ymax)
        try:
            box = Box(*coordinates).clamp(actual_width, actual_height)
        except Exception as exc:
            raise DatasetError(
                f"{object_context}: invalid {semantics} bbox {(xmin, ymin, xmax, ymax)}: {exc}"
            ) from exc
        if tuple(box.to_list()) != tuple(float(value) for value in coordinates):
            raise DatasetError(
                f"{object_context}: bbox {(xmin, ymin, xmax, ymax)} exceeds image bounds"
            )
        objects.append(VocObject(visual_concept, box))
    objects.sort(key=lambda item: (item.visual_concept, *item.box.to_list()))
    relative_path = image_path.name
    record_id = f"{source.name}/{image_path.stem}"
    return VocImage(
        record_id,
        source.name,
        relative_path,
        image_path,
        actual_width,
        actual_height,
        tuple(objects),
        _sha256_file(image_path),
        _canonical_group_key(source.name, image_path.stem),
    )


def _load_voc_images(
    config: DataVocConfig,
) -> tuple[list[VocImage], dict[str, list[str]], list[str]]:
    images: list[VocImage] = []
    orphan_images: dict[str, list[str]] = {}
    errors: list[str] = []
    for source in config.sources:
        indexed, image_stems = _image_index(source)
        annotation_paths = sorted(
            (
                path
                for path in source.annotation_dir.iterdir()
                if path.is_file() and path.suffix.lower() == ".xml"
            ),
            key=lambda item: item.name,
        )
        annotation_stems = {path.stem for path in annotation_paths}
        orphan_images[source.name] = sorted(set(image_stems) - annotation_stems)
        for xml_path in annotation_paths:
            image_path = indexed.get(xml_path.stem)
            if image_path is None:
                errors.append(f"{source.name}/{xml_path.name}: corresponding image is missing")
                continue
            try:
                images.append(
                    _parse_voc(
                        xml_path,
                        image_path,
                        source,
                        config.source_box_semantics,
                    )
                )
            except DatasetError as exc:
                errors.append(str(exc))
    if errors:
        preview = "\n  ".join(errors[:20])
        suffix = f"\n  ... and {len(errors) - 20} more" if len(errors) > 20 else ""
        raise DatasetError(f"VOC validation failed for {len(errors)} records:\n  {preview}{suffix}")
    if not images:
        raise DatasetError("VOC sources contain no valid annotations")
    identifiers = [image.record_id for image in images]
    if len(set(identifiers)) != len(identifiers):
        raise DatasetError("VOC source names and stems do not form unique record IDs")
    return sorted(images, key=lambda item: item.record_id), orphan_images, errors


def _group_images(images: list[VocImage]) -> tuple[dict[str, str], dict[str, list[VocImage]]]:
    groups = _Groups([image.record_id for image in images])
    by_content: dict[str, str] = {}
    by_name: dict[str, str] = {}
    for image in images:
        keys = (
            (by_content, image.content_sha256),
            (by_name, image.canonical_group_key),
        )
        for index, key in keys:
            previous = index.get(key)
            if previous is None:
                index[key] = image.record_id
            else:
                groups.union(previous, image.record_id)
    members: dict[str, list[VocImage]] = defaultdict(list)
    for image in images:
        members[groups.find(image.record_id)].append(image)
    stable_members: dict[str, list[VocImage]] = {}
    image_groups: dict[str, str] = {}
    for group_images in members.values():
        ordered = sorted(group_images, key=lambda item: item.record_id)
        group_id = _stable_hash(*(image.record_id for image in ordered))[:24]
        stable_members[group_id] = ordered
        for image in ordered:
            image_groups[image.record_id] = group_id
    return image_groups, dict(sorted(stable_members.items()))


def _assign_splits(
    groups: dict[str, list[VocImage]], classes: tuple[str, ...], splits: SplitConfig
) -> dict[str, str]:
    group_classes = {
        group_id: frozenset().union(*(image.concepts for image in images))
        for group_id, images in groups.items()
    }
    available = {
        visual_concept: sum(visual_concept in concepts for concepts in group_classes.values())
        for visual_concept in classes
    }
    insufficient = {name: count for name, count in available.items() if count < 6}
    if insufficient:
        raise DatasetError(
            "Each Visual Concept needs at least six leakage groups for train/validation/test "
            f"references; insufficient={insufficient}"
        )
    ratios = {"train": splits.train, "validation": splits.validation, "test": splits.test}
    target_counts = {
        split: round(len(groups) * ratio) for split, ratio in ratios.items()
    }
    target_counts["train"] += len(groups) - sum(target_counts.values())
    assigned: dict[str, str] = {}
    concept_counts = {split: Counter() for split in SPLITS}

    for required_count in (1, 2):
        for split in SPLITS:
            while True:
                deficits = {
                    name for name in classes if concept_counts[split][name] < required_count
                }
                if not deficits:
                    break
                candidates = [
                    group_id
                    for group_id, concepts in group_classes.items()
                    if group_id not in assigned and concepts & deficits
                ]
                if not candidates:
                    raise DatasetError(
                        f"Cannot place two reference groups per Visual Concept in {split}; "
                        f"missing={sorted(deficits)}"
                    )
                candidates.sort(
                    key=lambda group_id: (
                        -len(group_classes[group_id] & deficits),
                        _stable_hash(splits.seed, required_count, split, group_id),
                    )
                )
                selected = candidates[0]
                assigned[selected] = split
                concept_counts[split].update(group_classes[selected])

    remaining = sorted(
        (group_id for group_id in groups if group_id not in assigned),
        key=lambda group_id: _stable_hash(splits.seed, "remaining", group_id),
    )
    split_counts = Counter(assigned.values())
    for group_id in remaining:
        ranked = sorted(
            SPLITS,
            key=lambda split: (
                -(target_counts[split] - split_counts[split]),
                _stable_hash(splits.seed, group_id, split),
            ),
        )
        selected = ranked[0]
        assigned[group_id] = selected
        split_counts[selected] += 1
        concept_counts[selected].update(group_classes[group_id])
    return assigned


def _model_detection_set(boxes: tuple[Box, ...], size: tuple[int, int]) -> list[dict[str, Any]]:
    detections = [
        ProtocolDetection(encode_pixel_box(box, size)).to_model_dict() for box in boxes
    ]
    return sorted(detections, key=lambda item: item["bbox_2d"])


def _image_payload(image: VocImage, boxes: tuple[Box, ...]) -> dict[str, Any]:
    return {
        "source": image.source,
        "path": image.relative_path,
        "width": image.width,
        "height": image.height,
        "boxes_xyxy": [box.to_list(rounded=True) for box in boxes],
    }


def _select_reference(
    candidates: list[VocImage], target: VocImage, group_ids: dict[str, str], seed: int, concept: str
) -> VocImage:
    choices = [
        image
        for image in candidates
        if group_ids[image.record_id] != group_ids[target.record_id]
    ]
    if not choices:
        raise DatasetError(
            f"No leakage-safe Reference Image for {target.record_id} / {concept}"
        )
    return min(
        choices,
        key=lambda image: _stable_hash(seed, target.record_id, concept, image.record_id),
    )


def _records(
    images: list[VocImage],
    classes: tuple[str, ...],
    image_groups: dict[str, str],
    group_splits: dict[str, str],
    negative_per_image: int,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    by_split_class: dict[tuple[str, str], list[VocImage]] = defaultdict(list)
    for image in images:
        split = group_splits[image_groups[image.record_id]]
        for concept in image.concepts & set(classes):
            by_split_class[(split, concept)].append(image)
    records: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    allowed = set(classes)
    for target in images:
        split = group_splits[image_groups[target.record_id]]
        positive_concepts = sorted(target.concepts & allowed)
        requested: list[tuple[str, bool]] = [(concept, True) for concept in positive_concepts]
        absent = sorted(
            allowed - target.concepts,
            key=lambda concept: _stable_hash(seed, split, target.record_id, "negative", concept),
        )
        requested.extend((concept, False) for concept in absent[:negative_per_image])
        for concept, positive in requested:
            reference = _select_reference(
                by_split_class[(split, concept)], target, image_groups, seed, concept
            )
            reference_boxes = reference.boxes_for(concept)
            target_boxes = target.boxes_for(concept) if positive else ()
            record_id = _stable_hash(
                "conceptdet.dataset.record.v1",
                split,
                target.record_id,
                concept,
                reference.record_id,
                positive,
            )[:24]
            records[split].append(
                {
                    "schema_version": 1,
                    "id": record_id,
                    "split": split,
                    "group_id": image_groups[target.record_id],
                    "visual_concept": concept,
                    "query": QUERY,
                    "positive": positive,
                    "reference": _image_payload(reference, reference_boxes),
                    "target": _image_payload(target, target_boxes),
                    "detection_set": _model_detection_set(
                        target_boxes, (target.width, target.height)
                    ),
                }
            )
    for split in SPLITS:
        records[split].sort(key=lambda row: str(row["id"]))
    return records


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


@dataclass(frozen=True)
class DatasetArtifact:
    path: Path
    metadata: dict[str, Any]

    @property
    def fingerprint(self) -> str:
        return str(self.metadata["dataset_fingerprint"])

    @classmethod
    def load(cls, path: str | Path) -> DatasetArtifact:
        dataset_path = Path(path).expanduser().resolve()
        metadata_path = dataset_path / DATASET_FILE
        if not metadata_path.is_file():
            raise DatasetError(f"Compiled dataset is missing {DATASET_FILE}: {dataset_path}")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DatasetError(f"Cannot read compiled dataset metadata: {metadata_path}") from exc
        if not isinstance(metadata, dict) or metadata.get("dataset_schema_version") != 1:
            raise DatasetError("Compiled dataset schema is incompatible")
        fingerprint = metadata.get("dataset_fingerprint")
        if not isinstance(fingerprint, str):
            raise DatasetError("Compiled dataset has no fingerprint")
        payload = dict(metadata)
        del payload["dataset_fingerprint"]
        if hashlib.sha256(_canonical_json(payload)).hexdigest() != fingerprint:
            raise DatasetError("Compiled dataset fingerprint mismatch")
        files = metadata.get("files")
        if not isinstance(files, dict):
            raise DatasetError("Compiled dataset has no file manifest")
        for name, info in files.items():
            file_path = dataset_path / name
            if not isinstance(info, dict) or not file_path.is_file():
                raise DatasetError(f"Compiled dataset file is missing: {name}")
            if info.get("sha256") != _sha256_file(file_path):
                raise DatasetError(f"Compiled dataset file hash mismatch: {name}")
        return cls(dataset_path, metadata)

    def iter_records(self, split: str) -> Iterator[dict[str, Any]]:
        if split not in SPLITS:
            raise DatasetError(f"Unknown dataset split: {split}")
        path = self.path / f"{split}.jsonl"
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict) or row.get("split") != split:
                raise DatasetError(f"Invalid record at {path}:{line_number}")
            yield row

    def resolve_image(self, payload: dict[str, Any]) -> Path:
        source_name = payload.get("source")
        relative_path = payload.get("path")
        sources = {
            item["name"]: Path(item["image_dir"])
            for item in self.metadata["sources"]
        }
        if source_name not in sources or not isinstance(relative_path, str):
            raise DatasetError(f"Record has invalid image source: {payload}")
        path = (sources[source_name] / relative_path).resolve()
        if not path.is_file():
            raise DatasetError(f"Dataset image no longer exists: {path}")
        return path


def compile_voc_dataset(config: DataVocConfig) -> DatasetArtifact:
    if config.output_dir.exists():
        raise DatasetError(f"Compiled dataset output already exists: {config.output_dir}")
    images, orphan_images, _ = _load_voc_images(config)
    discovered_classes = tuple(sorted(set().union(*(image.concepts for image in images))))
    classes = config.classes or discovered_classes
    missing_classes = sorted(set(classes) - set(discovered_classes))
    if missing_classes:
        raise DatasetError(f"Requested Visual Concepts are absent: {missing_classes}")
    image_groups, groups = _group_images(images)
    group_splits = _assign_splits(groups, classes, config.splits)
    records = _records(
        images,
        classes,
        image_groups,
        group_splits,
        config.negative_per_image,
        config.splits.seed,
    )

    config.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{config.output_dir.name}.", dir=config.output_dir.parent)
    )
    try:
        file_metadata: dict[str, dict[str, Any]] = {}
        for split in SPLITS:
            manifest = temporary / f"{split}.jsonl"
            manifest.write_text(
                "".join(
                    json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                    + "\n"
                    for row in records[split]
                ),
                encoding="utf-8",
            )
            file_metadata[manifest.name] = {
                "sha256": _sha256_file(manifest),
                "records": len(records[split]),
                "positive": sum(bool(row["positive"]) for row in records[split]),
                "negative": sum(not bool(row["positive"]) for row in records[split]),
            }
        class_instances = Counter(
            item.visual_concept for image in images for item in image.objects
        )
        class_images = Counter(
            concept for image in images for concept in image.concepts
        )
        split_images = Counter(group_splits[image_groups[image.record_id]] for image in images)
        split_groups = Counter(group_splits.values())
        audit = {
            "schema_version": 1,
            "annotation_records": len(images),
            "orphan_images": orphan_images,
            "orphan_image_count": sum(len(items) for items in orphan_images.values()),
            "class_instances": dict(sorted(class_instances.items())),
            "class_images": dict(sorted(class_images.items())),
            "split_images": {name: split_images[name] for name in SPLITS},
            "split_groups": {name: split_groups[name] for name in SPLITS},
            "exact_or_canonical_duplicate_groups": sum(
                len(group_images) > 1 for group_images in groups.values()
            ),
        }
        audit_path = temporary / AUDIT_FILE
        _write_json(audit_path, audit)
        file_metadata[AUDIT_FILE] = {"sha256": _sha256_file(audit_path)}
        metadata: dict[str, Any] = {
            "dataset_schema_version": 1,
            "contract_id": "conceptdet.reference-detection-dataset",
            "contract_version": 1,
            "config_hash": config.config_hash,
            "source_box_semantics": config.source_box_semantics,
            "pixel_box_semantics": "xyxy_half_open",
            "model_coordinate_space": "target_normalized_0_1000",
            "query": QUERY,
            "classes": list(classes),
            "negative_per_image": config.negative_per_image,
            "splits": config.splits.__dict__,
            "sources": [
                {
                    "name": source.name,
                    "image_dir": str(source.image_dir),
                    "annotation_dir": str(source.annotation_dir),
                }
                for source in config.sources
            ],
            "files": dict(sorted(file_metadata.items())),
        }
        metadata["dataset_fingerprint"] = hashlib.sha256(_canonical_json(metadata)).hexdigest()
        _write_json(temporary / DATASET_FILE, metadata)
        os.replace(temporary, config.output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return DatasetArtifact.load(config.output_dir)
