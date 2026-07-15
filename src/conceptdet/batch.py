from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from conceptdet.errors import InputError


IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})


@dataclass(frozen=True)
class DiscoveredImage:
    path: Path
    relative_path: Path


def _is_under(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


def discover_images(
    input_paths: Sequence[Path],
    *,
    recursive: bool,
    exclude_paths: Iterable[Path] = (),
    exclude_directories: Iterable[Path] = (),
) -> list[DiscoveredImage]:
    excluded_files = {path.expanduser().resolve() for path in exclude_paths}
    excluded_directories = tuple(
        path.expanduser().resolve() for path in exclude_directories
    )
    discovered: list[DiscoveredImage] = []
    seen: set[Path] = set()

    for raw_input_path in input_paths:
        input_path = raw_input_path.expanduser().resolve()
        if input_path.is_file():
            candidates = [(input_path, Path(input_path.name))]
        elif input_path.is_dir():
            iterator = input_path.rglob("*") if recursive else input_path.iterdir()
            candidates = [
                (candidate.resolve(), candidate.relative_to(input_path))
                for candidate in iterator
                if candidate.is_file()
            ]
        else:
            raise InputError(f"批量输入路径不存在: {input_path}")

        for candidate, relative_path in sorted(candidates, key=lambda item: str(item[0])):
            if candidate.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if candidate in excluded_files:
                continue
            if any(_is_under(candidate, directory) for directory in excluded_directories):
                continue
            if candidate in seen:
                continue
            seen.add(candidate)
            discovered.append(DiscoveredImage(candidate, relative_path))

    return discovered


def plan_output_paths(
    images: Sequence[DiscoveredImage], output_directory: Path
) -> list[Path]:
    output_directory = output_directory.expanduser().resolve()
    used: set[Path] = set()
    planned: list[Path] = []

    for image in images:
        relative_output = image.relative_path.with_suffix(".png")
        candidate = output_directory / relative_output
        collision_index = 2
        while candidate in used:
            candidate = (
                output_directory
                / relative_output.parent
                / f"{relative_output.stem}__{collision_index}.png"
            )
            collision_index += 1
        used.add(candidate)
        planned.append(candidate)

    return planned
