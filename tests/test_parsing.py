import pytest

from conceptdet.errors import OutputFormatError
from conceptdet.parsing import parse_completion
from conceptdet.types import Box


def test_parse_completion_extracts_protocol_fields() -> None:
    parsed = parse_completion(
        "<think>compare shapes</think><rule>six-sided head</rule>"
        "<bbox>[10.5, 20, 30, 40]</bbox><answer>bolt</answer>"
    )
    assert parsed.boxes == (Box(10.5, 20, 30, 40),)
    assert parsed.answer == "bolt"
    assert parsed.rule == "six-sided head"
    assert parsed.reasoning == "compare shapes"


def test_parse_completion_allows_multiple_bbox_tags() -> None:
    parsed = parse_completion("<bbox>[1,2,3,4]</bbox><bbox>5,6,7,8</bbox>")
    assert parsed.boxes == (Box(1, 2, 3, 4), Box(5, 6, 7, 8))


@pytest.mark.parametrize(
    "text",
    [
        "no box here",
        "<bbox>[1,2,3]</bbox>",
        "<bbox>[4,3,2,1]</bbox>",
    ],
)
def test_parse_completion_rejects_missing_or_invalid_box(text: str) -> None:
    with pytest.raises(OutputFormatError):
        parse_completion(text)
