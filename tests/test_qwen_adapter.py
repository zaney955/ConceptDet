from PIL import Image

from conceptdet.adapter import AdapterInput
from conceptdet.model import prepare_images
from conceptdet.prompts import build_messages
from conceptdet.types import Box


def test_qwen_preprocessing_is_dynamic_and_only_decorates_reference() -> None:
    reference_source = Image.new("RGB", (1600, 1200), "gray")
    target_source = Image.new("RGB", (800, 1600), "gray")
    prepared_reference, prepared_target = prepare_images(
        AdapterInput(
            reference_source,
            (Box(160, 120, 480, 360),),
            target_source,
            "matching bolt",
        )
    )
    assert prepared_reference.size != (600, 600)
    assert prepared_target.size != (600, 600)
    assert prepared_reference.size[0] % 32 == 0
    assert prepared_target.size[1] % 32 == 0
    assert prepared_reference.getbbox() is not None
    assert set(prepared_target.get_flattened_data()) == {(128, 128, 128)}
    assert any(
        red > green + 100 for red, green, _ in prepared_reference.get_flattened_data()
    )


def test_qwen_prompt_has_ordered_roles_and_strict_json_instruction() -> None:
    messages = build_messages(
        Image.new("RGB", (32, 32)), Image.new("RGB", (32, 32)), "bolt"
    )
    content = messages[0]["content"]
    assert content[0]["type"] == "image"
    assert content[1]["type"] == "image"
    text = content[2]["text"]
    assert "Picture 1: Reference Image" in text
    assert "Picture 2: Target Image" in text
    assert "No Markdown" in text
