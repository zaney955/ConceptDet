import pytest

from conceptdet.errors import InputError
from conceptdet.types import Box


def test_box_accepts_numeric_xyxy_and_exposes_area() -> None:
    box = Box.from_sequence([1, 2.5, 11, 12.5])
    assert box == Box(1, 2.5, 11, 12.5)
    assert box.area == 100


@pytest.mark.parametrize(
    ("values", "message"),
    [([True, 2, 3, 4], "must be numeric"), (["1", 2, 3, 4], "must be numeric"), (7, "sequence")],
)
def test_box_rejects_non_numeric_yaml_values(values: object, message: str) -> None:
    with pytest.raises(InputError, match=message):
        Box.from_sequence(values)  # type: ignore[arg-type]


def test_box_clamp_rejects_fully_outside_box() -> None:
    with pytest.raises(InputError, match="outside image"):
        Box(20, 20, 30, 30).clamp(10, 10)
