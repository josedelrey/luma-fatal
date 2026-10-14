"""Validate the explicit resolution profile and interpolation arithmetic."""

import json
import hashlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from luma_fatal import FilterParams, apply_filter
from luma_fatal.cli import main
from luma_fatal.processing import (
    enlargement_sizes,
    nearest_coordinates,
    resize_bilinear_fixed,
    resize_linear_floor,
    working_size,
)


SCALE_REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/bilinear-fixed.json").read_text()
)


@pytest.mark.parametrize("row", SCALE_REFERENCE["trials"])
def test_bilinear_sampling_matches_reference_pixels(row):
    width, height = row["source_size"]
    pixels = np.random.default_rng(row["seed"]).integers(
        0, 256, (height, width, 3), np.uint8
    )
    assert hashlib.sha256(pixels.tobytes()).hexdigest() == row["source_rgb_sha256"]
    result = resize_bilinear_fixed(pixels, tuple(row["destination_size"]))
    assert hashlib.sha256(result.tobytes()).hexdigest() == row["expected_rgb_sha256"]


@pytest.mark.parametrize(
    "source,target,expected",
    [
        ((1536, 1152), (2048, 1536), [(2048, 1536)]),
        ((1536, 1152), (3264, 2448), [(3072, 2304), (3264, 2448)]),
        ((1536, 1152), (3072, 2304), [(3072, 2304)]),
        ((1536, 1152), (7000, 5250), [(3072, 2304), (6144, 4608), (7000, 5250)]),
    ],
)
def test_progressive_enlargement(source, target, expected):
    assert enlargement_sizes(source, target) == expected


def test_nearest_indices_match_coordinate_atlas():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/nearest-coordinates.json").read_text()
    )
    x, y = nearest_coordinates(
        tuple(fixture["source_size"]), tuple(fixture["target_size"])
    )
    assert (
        hashlib.sha256(x.astype("<i4").tobytes()).hexdigest()
        == fixture["expected_x_sha256"]
    )
    assert (
        hashlib.sha256(y.astype("<i4").tobytes()).hexdigest()
        == fixture["expected_y_sha256"]
    )


@pytest.mark.parametrize(
    "size,expected",
    [
        ((1536, 1152), (1536, 1152)),
        ((2048, 1536), (1536, 1152)),
        ((3264, 2448), (1536, 1152)),
        ((2040, 2040), (1536, 1536)),
        ((2016, 1344), (1536, 1024)),
        ((1344, 2016), (1024, 1536)),
    ],
)
def test_working_dimensions_preserve_orientation(size, expected):
    assert working_size(size) == expected


@pytest.mark.parametrize(
    "source,expected",
    [
        ((2241, 2801), (1229, 1536)),
        ((3024, 1998), (1536, 1015)),
    ],
)
def test_working_dimensions_match_reference_geometries(source, expected):
    # Fixed reference sizes cover odd portrait and landscape dimensions.
    assert working_size(source) == expected


@pytest.mark.parametrize(
    "row",
    json.loads((Path(__file__).parent / "fixtures/working-sizes.json").read_text())[
        "trials"
    ],
)
def test_working_dimensions_match_rounding_references(row):
    # References cover odd dimensions, half ties and float32 limits.
    assert (
        list(working_size(tuple(row["source_size"]), row["max_edge"]))
        == row["expected_size"]
    )


def test_half_pixel_interpolation_truncates_after_both_axes():
    source = np.array([[0, 5], [10, 15]], np.uint8)
    rgb = np.repeat(source[..., None], 3, axis=2)
    expected = np.array(
        [[0, 1, 3, 5], [2, 3, 6, 7], [7, 8, 11, 12], [10, 11, 13, 15]], np.uint8
    )
    np.testing.assert_array_equal(resize_linear_floor(rgb, (4, 4))[..., 0], expected)


def test_profile_preserves_identity_at_large_resolution_and_records_choice(
    tmp_path, capsys
):
    source = tmp_path / "source.png"
    destination = tmp_path / "output.png"
    pixels = np.random.default_rng(3095).integers(0, 256, (17, 1601, 3), np.uint8)
    Image.fromarray(pixels).save(source)
    main(
        [
            "apply",
            str(source),
            str(destination),
            "--fade",
            "100",
            "--output-format",
            "png",
        ]
    )
    np.testing.assert_array_equal(np.asarray(Image.open(destination)), pixels)
    assert (
        json.loads(destination.with_suffix(".meta.json").read_text())[
            "processing_profile"
        ]
        == "bounded-1536"
    )


def test_profile_rejects_unvalidated_transparent_route():
    with pytest.raises(ValueError, match="opaque"):
        apply_filter(np.zeros((32, 32, 4), np.uint8))


def test_default_large_route_reduces_before_filtering(monkeypatch):
    from luma_fatal import core
    from luma_fatal.processing import resize_nearest

    pixels = np.random.default_rng(55).integers(0, 256, (32, 2048, 3), np.uint8)
    calls = []
    original_apply = apply_filter

    def record_core_call(image, variant, params, numerics):
        calls.append((image.copy(), variant, params.fade, numerics))
        return image.copy()

    monkeypatch.setattr(core, "apply_core", record_core_call)
    result = original_apply(pixels, 2, FilterParams(fade=0))
    assert result.shape == pixels.shape
    assert len(calls) == 1
    work, variant, fade, numerics = calls[0]
    np.testing.assert_array_equal(work, resize_nearest(pixels, (1536, 24)))
    assert (variant, fade) == (2, 0)
    assert numerics.iterations == 80


def test_unknown_profile_is_rejected():
    with pytest.raises(ValueError, match="profile"):
        apply_filter(np.zeros((32, 32, 3), np.uint8), profile="unknown")
