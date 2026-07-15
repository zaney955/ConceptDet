from pathlib import Path

from PIL import Image

from conceptdet.batch import discover_images, plan_output_paths


def _image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (10, 10), "white").save(path)


def test_discover_images_supports_files_directories_recursion_and_exclusions(
    tmp_path: Path,
) -> None:
    input_directory = tmp_path / "inputs"
    reference_path = input_directory / "reference.jpg"
    output_directory = input_directory / "outputs"
    _image(reference_path)
    _image(input_directory / "a.jpg")
    _image(input_directory / "nested" / "b.png")
    _image(output_directory / "old.png")
    (input_directory / "notes.txt").write_text("ignore", encoding="utf-8")

    flat = discover_images(
        [input_directory],
        recursive=False,
        exclude_paths=(reference_path,),
        exclude_directories=(output_directory,),
    )
    assert [image.relative_path.as_posix() for image in flat] == ["a.jpg"]

    recursive = discover_images(
        [input_directory],
        recursive=True,
        exclude_paths=(reference_path,),
        exclude_directories=(output_directory,),
    )
    assert [image.relative_path.as_posix() for image in recursive] == [
        "a.jpg",
        "nested/b.png",
    ]


def test_discover_images_deduplicates_and_output_planner_avoids_collisions(
    tmp_path: Path,
) -> None:
    first_directory = tmp_path / "first"
    second_directory = tmp_path / "second"
    first = first_directory / "same.jpg"
    second = second_directory / "same.webp"
    _image(first)
    _image(second)

    images = discover_images(
        [first_directory, first, second_directory],
        recursive=False,
    )
    assert [image.path for image in images] == [first.resolve(), second.resolve()]
    outputs = plan_output_paths(images, tmp_path / "outputs")
    assert [path.name for path in outputs] == ["same.png", "same__2.png"]
