import pytest

from conceptdet.errors import InputError
from conceptdet.types import Box, parse_boxes


def test_parse_boxes_accepts_semicolon_and_whitespace() -> None:
    boxes = parse_boxes("1, 2, 11, 12; 20 30 40 50")
    assert boxes == (Box(1, 2, 11, 12), Box(20, 30, 40, 50))


def test_parse_boxes_accepts_single_and_nested_sequences() -> None:
    assert parse_boxes([1, 2, 3, 4]) == (Box(1, 2, 3, 4),)
    assert parse_boxes([[1, 2, 3, 4], [5, 6, 7, 8]]) == (
        Box(1, 2, 3, 4),
        Box(5, 6, 7, 8),
    )


@pytest.mark.parametrize("value", ["", "1,2,3", "1,2,1,4", [[1, 2, 3]]])
def test_parse_boxes_rejects_invalid_input(value: object) -> None:
    with pytest.raises(InputError):
        parse_boxes(value)  # type: ignore[arg-type]


def test_box_clamp_rejects_fully_outside_box() -> None:
    with pytest.raises(InputError):
        Box(20, 20, 30, 30).clamp(10, 10)
