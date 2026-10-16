import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from luma_fatal import FilterParams, apply_filter, fattal1, fattal2
from luma_fatal.core import _pyramid, _normalize
from luma_fatal.numerics import (
    fast_pow, gradient_scale, luminance_fractional, luminance_u8,
)


FADE_CASES = json.loads(
    (Path(__file__).parent / "fixtures/fade-flat.json").read_text()
)["cases"]


@pytest.mark.parametrize(
    "case",
    FADE_CASES,
    ids=[
        f"gray{case['input_gray']}-fattal{case['variant']}-fade{case['fade']}"
        for case in FADE_CASES
    ],
)
def test_pipeline_matches_flat_field_fade_references(case):
    image = np.full((case["height"], case["width"], 3), case["input_gray"], np.uint8)
    output = apply_filter(image, case["variant"], FilterParams(fade=case["fade"]))
    assert np.all(output == case["expected_gray"])


@pytest.mark.parametrize("function", [fattal1, fattal2])
def test_identity_fade_is_exact_and_input_immutable(function):
    image = np.random.default_rng(42).integers(0, 256, (67, 93, 4), dtype=np.uint8)
    snapshot = image.copy()
    output = function(image, fade=100)
    np.testing.assert_array_equal(output, image)
    np.testing.assert_array_equal(image, snapshot)
    assert output is not image


@pytest.mark.parametrize("function", [fattal1, fattal2])
def test_black_and_low_light_gate(function):
    for value in [0, 1, 5]:
        image = np.full((32, 48, 3), value, np.uint8)
        np.testing.assert_array_equal(function(image, fade=0), image)


@pytest.mark.parametrize("function", [fattal1, fattal2])
def test_uniform_midgray_zero_range_behavior(function):
    image = np.full((32, 48, 3), 128, np.uint8)
    assert np.all(function(image, fade=0) == 255)
    assert np.all(function(image) == 191)


def test_rectangular_odd_image_variants_deterministic_and_distinct():
    image = np.random.default_rng(7).integers(0, 256, (129, 193, 3), dtype=np.uint8)
    original = image.copy()
    outputs = [apply_filter(image, variant, FilterParams(fade=0)) for variant in [1, 2]]
    for variant, output in enumerate(outputs, 1):
        assert output.shape == image.shape and output.dtype == np.uint8
        np.testing.assert_array_equal(
            apply_filter(image, variant, FilterParams(fade=0)), output
        )
    assert np.mean(np.abs(outputs[0].astype(float) - outputs[1])) > 1
    np.testing.assert_array_equal(image, original)


def test_pillow_rgba_alpha_preserved_as_explicit_extension():
    image = Image.fromarray(
        np.random.default_rng(4).integers(0, 256, (32, 40, 4), np.uint8)
    )
    output = fattal2(image, profile="core")
    assert isinstance(output, Image.Image)
    assert output.mode == "RGBA"
    np.testing.assert_array_equal(np.asarray(output)[..., 3], np.asarray(image)[..., 3])


def test_small_image_extension():
    image = np.full((1, 5, 3), 128, np.uint8)
    with pytest.warns(RuntimeWarning, match="identity extension"):
        np.testing.assert_array_equal(fattal1(image), image)


@pytest.mark.parametrize(
    "parameters", [{"alpha": 0}, {"beta": 101}, {"noise": -1}, {"fade": float("nan")}]
)
def test_invalid_parameters(parameters):
    with pytest.raises(ValueError):
        FilterParams(**parameters)


@pytest.mark.parametrize(
    "image",
    [np.zeros((0, 3, 3), np.uint8), np.zeros((3, 3)), np.zeros((3, 3, 2), np.uint8)],
)
def test_invalid_images(image):
    with pytest.raises(ValueError):
        fattal1(image)


def test_fixed_point_luminance_known_primaries():
    image = np.array(
        [[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]]], np.uint8
    )
    np.testing.assert_array_equal(luminance_u8(image), [[76, 150, 29, 255]])


def test_fractional_luminance_uses_reference_weights_without_byte_rounding():
    image = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]], np.uint8)
    expected = np.array([[4899, 9617, 1868, 16384]], np.float32)
    expected /= np.float32(16384 * 255)
    np.testing.assert_array_equal(luminance_fractional(image), expected)
    np.testing.assert_array_equal(luminance_u8(image), [[0, 1, 0, 1]])


def test_power_preserves_affine_approximation():
    # Bias-based approximation maps exponent zero to ~0.971, rather than 1.
    output = fast_pow(np.array([0.5, 1, 2, 10], np.float32), 0)
    assert np.all(output == np.float32(0.9710078239440918))
    assert gradient_scale(1) == np.float32(2)


def test_pyramid_stops_above_32_and_rounds_down():
    assert [a.shape for a in _pyramid(np.zeros((257, 513), np.float32))] == [
        (257, 513),
        (128, 256),
        (64, 128),
        (32, 64),
    ]


def test_normalization_keeps_positive_subfloor_values():
    values = np.concatenate(
        (
            np.array([0, 0.5, 1, 1.000001], np.float32),
            np.arange(2, 1001, dtype=np.float32),
        )
    ).reshape(1, -1)
    normalized = _normalize(values)
    assert normalized[0, 0] == np.float32(1e-4)
    assert normalized.max() > 1
    assert 0 < normalized[0, 3] < np.float32(1e-4)
