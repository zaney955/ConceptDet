from PIL import Image

from conceptdet.geometry import ImageTransform, prepare_reference
from conceptdet.types import Box


def test_image_transform_round_trip() -> None:
    transform = ImageTransform((1200, 300), (600, 600))
    source = Box(120, 30, 720, 240)
    assert transform.to_source(transform.to_model(source)) == source


def test_prepare_reference_full_resizes_and_draws_boxes() -> None:
    image = Image.new("RGB", (200, 100), "white")
    prepared = prepare_reference(
        image,
        (Box(20, 10, 80, 50),),
        model_size=(600, 600),
        box_width=3,
    )
    assert prepared.crop_box == (0, 0, 200, 100)
    assert prepared.model_boxes == (Box(60, 60, 240, 300),)
    assert prepared.image.getpixel((60, 60)) == (255, 0, 0)


def test_prepare_reference_crop_stays_inside_image() -> None:
    image = Image.new("RGB", (1000, 500), "white")
    prepared = prepare_reference(
        image,
        (Box(900, 450, 980, 490),),
        model_size=(600, 600),
        crop_mode="crop",
        context_scale=4,
    )
    assert prepared.crop_box == (680, 340, 1000, 500)
    model_box = prepared.model_boxes[0]
    assert model_box.to_list(rounded=True) == [412, 412, 562, 562]


def test_prepare_reference_rounds_prompt_box_before_drawing() -> None:
    image = Image.new("RGB", (1000, 1000), "white")
    prepared = prepare_reference(
        image,
        (Box(101, 101, 201, 201),),
        model_size=(600, 600),
    )
    # 101 * 0.6 = 60.6, matching ConceptSeg's int(round(...)) behavior.
    assert prepared.image.getpixel((61, 61)) == (255, 0, 0)
    assert prepared.image.getpixel((60, 60)) == (255, 255, 255)
