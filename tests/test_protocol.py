import pytest

from conceptdet.errors import OutputFormatError
from conceptdet.protocol import (
    ProtocolDetection,
    decode_model_box,
    encode_pixel_box,
    hard_set_counts,
    parse_detection_set,
    serialize_detection_set,
)
from conceptdet.types import Box


def test_strict_detection_set_round_trip_is_order_preserving_but_order_free_semantically() -> None:
    parsed = parse_detection_set(
        '[{"bbox_2d":[700,200,800,400]},'
        '{"bbox_2d":[100,100,300,300],"label":"bolt"}]'
    )
    assert parsed == (
        ProtocolDetection(Box(700, 200, 800, 400)),
        ProtocolDetection(Box(100, 100, 300, 300), "bolt"),
    )
    assert serialize_detection_set(parsed) == (
        '[{"bbox_2d":[700,200,800,400]},'
        '{"bbox_2d":[100,100,300,300],"label":"bolt"}]'
    )
    assert parse_detection_set("[]") == ()


@pytest.mark.parametrize(
    "raw",
    [
        "```json\n[]\n```",
        '{"bbox_2d":[1,2,3,4]}',
        '[{"bbox_2d":[1.0,2,3,4]}]',
        '[{"bbox_2d":[1,2,3,4],"score":0.9}]',
        '[{"bbox_2d":[3,2,1,4]}]',
        '[{"bbox_2d":[0,0,1001,10]}]',
    ],
)
def test_strict_detection_set_rejects_non_contract_output(raw: str) -> None:
    with pytest.raises(OutputFormatError):
        parse_detection_set(raw)


def test_protocol_coordinate_round_trip_uses_normalized_grid() -> None:
    source = Box(100, 50, 500, 300)
    encoded = encode_pixel_box(source, (1000, 500))
    assert encoded == Box(100, 100, 500, 600)
    assert decode_model_box(encoded, (1000, 500)) == source


def test_encoding_rejects_boxes_that_collapse_on_normalized_grid() -> None:
    with pytest.raises(OutputFormatError, match="collapses"):
        encode_pixel_box(Box(1, 1, 2, 2), (10000, 10000))


def test_exact_matching_penalizes_duplicates_and_is_order_independent() -> None:
    targets = [Box(0, 0, 100, 100), Box(200, 200, 300, 300)]
    predictions = [targets[1], targets[0], targets[0]]
    assert hard_set_counts(predictions, targets, 0.5) == (2, 1, 0)
    assert hard_set_counts(list(reversed(predictions)), targets, 0.5) == (2, 1, 0)


def test_exact_matching_has_no_small_detection_set_limit() -> None:
    targets = [Box(index * 30, 0, index * 30 + 20, 20) for index in range(25)]
    predictions = list(reversed(targets))
    assert hard_set_counts(predictions, targets, 0.5) == (25, 0, 0)
