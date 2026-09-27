"""Tests of the pure parts of the embedder. No model is loaded."""

import pytest
import torch

from ai.embedder import frame_size, is_small_box, padded_square, pool_box
from ai.types import Box


def test_frame_size_scales_the_short_edge_and_snaps_to_the_patch_size() -> None:
    assert frame_size(640, 480, 448, 784) == (588, 448)
    assert frame_size(480, 640, 448, 784) == (448, 588)
    for width, height in [(1920, 1080), (4000, 3000), (333, 777)]:
        w, h = frame_size(width, height, 448, 784)
        assert w % 14 == 0 and h % 14 == 0 and w >= 14 and h >= 14


def test_frame_size_cuts_the_long_edge() -> None:
    width, height = frame_size(4000, 1000, 448, 784)
    assert width == 784 and height <= 224


def test_pool_box_averages_the_patches_inside_the_box() -> None:
    tokens: torch.Tensor = torch.zeros(4, 4, 2)
    tokens[:2, :2] = torch.tensor([1.0, 0.0])  # Top-left quarter
    tokens[2:, 2:] = torch.tensor([0.0, 1.0])  # Bottom-right quarter
    top_left: torch.Tensor = pool_box(tokens, Box(0, 0, 100, 100), 200, 200)
    bottom_right: torch.Tensor = pool_box(tokens, Box(100, 100, 200, 200), 200, 200)
    assert top_left.tolist() == [1.0, 0.0]
    assert bottom_right.tolist() == [0.0, 1.0]
    assert pool_box(tokens, Box(0, 0, 200, 200), 200, 200).tolist() == [0.25, 0.25]


def test_pool_box_uses_the_patch_under_a_box_that_holds_no_patch_centre() -> None:
    tokens: torch.Tensor = torch.arange(16, dtype=torch.float32).reshape(4, 4, 1)
    # Patch centres are at 25, 75, 125, 175. This box lies between two centres.
    vector: torch.Tensor = pool_box(tokens, Box(30, 130, 40, 140), 200, 200)
    assert vector.tolist() == [tokens[2, 0, 0].item()]


def test_pool_box_at_the_image_border() -> None:
    tokens: torch.Tensor = torch.ones(2, 2, 3)
    assert pool_box(tokens, Box(199, 199, 200, 200), 200, 200).tolist() == [1.0] * 3


@pytest.mark.parametrize(
    ("box", "expected"),
    [
        (
            Box(100, 100, 120, 140),
            Box(84, 94, 136, 146),
        ),  # Padding 0.15 of the long side, square
        (Box(0, 0, 20, 20), Box(0, 0, 26, 26)),  # Moved inside at the corner
        (Box(180, 180, 200, 200), Box(174, 174, 200, 200)),
    ],
)
def test_padded_square_stays_inside_the_frame(box: Box, expected: Box) -> None:
    result: Box = padded_square(box, 0.15, 200, 200)
    assert result.width == result.height
    assert result.x0 >= 0 and result.y0 >= 0 and result.x1 <= 200 and result.y1 <= 200
    assert result == expected


def test_padded_square_is_cut_to_a_small_frame() -> None:
    result: Box = padded_square(Box(0, 0, 90, 90), 0.5, 100, 80)
    assert result.width == result.height == 80


@pytest.mark.parametrize(
    ("frame", "expected"),
    [
        ((968, 728), False),  # Scale 0.61: 100 px become 61 px
        ((1920, 1080), True),  # Scale 0.41: 100 px become 41 px
        ((4032, 3024), True),  # Scale 0.15: 100 px become 15 px
    ],
)
def test_small_box_cutoff_uses_the_model_pixels(
    frame: tuple[int, int], expected: bool
) -> None:
    width, height = frame
    size: tuple[int, int] = frame_size(width, height, 448, 784)
    box: Box = Box(100, 100, 100 + 100, 100 + 100)  # 100 px in the original frame
    assert is_small_box(box, width, height, size, 50) is expected


def test_small_box_cutoff_checks_the_shorter_side() -> None:
    size: tuple[int, int] = frame_size(640, 480, 448, 784)
    assert is_small_box(Box(0, 0, 400, 20), 640, 480, size, 64) is True
    assert is_small_box(Box(0, 0, 400, 300), 640, 480, size, 64) is False
