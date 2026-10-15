"""Optional shadow blending retains the original reference path by default."""

from dataclasses import replace
import hashlib

import numpy as np
import pytest

from luma_fatal import Config, FilterParams, NumericsConfig, ProcessingConfig, apply_filter
from luma_fatal.core import _recombine
from luma_fatal.numerics import luminance_u8


@pytest.mark.parametrize("value", [-1, 256, True, 1.5, "6", float("nan")])
def test_shadow_softness_requires_an_integer_luminance_width(value):
    with pytest.raises(ValueError, match="shadow_softness"):
        NumericsConfig(shadow_softness=value)


def test_smooth_shadow_transition_has_known_endpoints_and_midpoint():
    gray = np.array([0, 3, 6, 9, 12, 24], np.uint8)
    rgb = np.repeat(gray[None, :, None], 3, axis=2)
    output = _recombine(
        rgb, gray[None, :].astype(np.float32) / 255,
        np.ones((1, len(gray)), np.float32), 0,
        power_mode="standard", shadow_softness=6,
    )
    np.testing.assert_array_equal(output[0, :, 0], [0, 42, 131, 217, 255, 255])
    np.testing.assert_array_equal(output[..., 0], output[..., 1])
    np.testing.assert_array_equal(output[..., 0], output[..., 2])


def test_smooth_shadow_ramp_removes_the_reference_cutoff_jump():
    gray = np.arange(25, dtype=np.uint8)
    rgb = np.repeat(gray[None, :, None], 3, axis=2)
    args = (rgb, gray[None, :].astype(np.float32) / 255, np.ones((1, 25), np.float32), 0)
    reference = _recombine(*args, power_mode="standard")
    smooth = _recombine(*args, power_mode="standard", shadow_softness=6)
    reference_steps = np.diff(reference[0, :, 0].astype(int))
    smooth_steps = np.diff(smooth[0, :, 0].astype(int))
    assert reference_steps.max() >= 240
    assert 0 <= smooth_steps.min() <= smooth_steps.max() <= 34


@pytest.mark.parametrize("threshold", [0, 6, 255])
@pytest.mark.parametrize("softness", [1, 6, 255])
def test_black_remains_black_at_shadow_control_extremes(threshold, softness):
    output = _recombine(
        np.zeros((2, 3, 3), np.uint8), np.zeros((2, 3), np.float32),
        np.ones((2, 3), np.float32), 0, shadow_threshold=threshold,
        power_mode="standard", shadow_softness=softness,
    )
    np.testing.assert_array_equal(output, 0)


def test_shadow_mask_uses_fractional_luminance_within_one_integer_bin():
    rgb = np.array([[[0, 7, 0], [1, 7, 0]]], np.uint8)
    luma = luminance_u8(rgb)
    assert luma[0, 0] == luma[0, 1]
    output = _recombine(
        rgb, luma.astype(np.float32) / 255, np.ones((1, 2), np.float32), 0,
        power_mode="standard", shadow_softness=6,
    )
    assert output[0, 0, 2] < output[0, 1, 2]


@pytest.mark.parametrize("gamma,mapped,expected", [(1, 10.5 / 255, 8), (2, 0.25, 67)])
def test_shadow_blend_follows_gamma_and_precedes_byte_quantization(gamma, mapped, expected):
    output = _recombine(
        np.full((1, 1, 3), 6, np.uint8), np.full((1, 1), 6 / 255, np.float32),
        np.full((1, 1), mapped, np.float32), 0,
        power_mode="standard", output_gamma=gamma, shadow_softness=6,
    )
    np.testing.assert_array_equal(output, expected)


def test_shadow_blend_bounds_overshoot_and_nonfinite_candidates():
    output = _recombine(
        np.full((1, 4, 3), 6, np.uint8), np.full((1, 4), 6 / 255, np.float32),
        np.array([[-1, 1, 4, np.nan]], np.float32), 0,
        power_mode="standard", shadow_softness=6,
    )
    np.testing.assert_array_equal(output[0, :, 0], [3, 131, 131, 131])


@pytest.mark.parametrize("power_mode", ["approximate", "standard"])
@pytest.mark.parametrize("gamma", [1, 1.7])
def test_pixels_above_transition_match_reference_bytes(power_mode, gamma):
    rgb = np.random.default_rng(40).integers(12, 256, (17, 23, 3), np.uint8)
    luma = luminance_u8(rgb).astype(np.float32) / 255
    mapped = np.random.default_rng(41).random((17, 23), dtype=np.float32) * 2
    args = (rgb, luma, mapped, 80)
    reference = _recombine(*args, power_mode=power_mode, output_gamma=gamma)
    smooth = _recombine(*args, power_mode=power_mode, output_gamma=gamma, shadow_softness=6)
    np.testing.assert_array_equal(smooth, reference)


@pytest.mark.parametrize("variant", [1, 2])
@pytest.mark.parametrize("max_edge", [None, 32])
def test_shadow_option_changes_dark_pixels_in_both_resolution_routes(variant, max_edge):
    pixels = np.random.default_rng(42).integers(0, 40, (47, 63, 3), np.uint8)
    original = pixels.copy()
    config = Config(
        filter=f"fattal{variant}", parameters=FilterParams(fade=0),
        processing=ProcessingConfig(resize=max_edge is not None, max_edge=max_edge),
        numerics=NumericsConfig(iterations=10),
    )
    reference = apply_filter(pixels, config=config)
    soft_config = replace(config, numerics=replace(config.numerics, shadow_softness=6))
    smooth = apply_filter(pixels, config=soft_config)
    assert smooth.shape == pixels.shape
    assert smooth.dtype == np.uint8
    assert np.any(smooth != reference)
    np.testing.assert_array_equal(pixels, original)
    bypass = replace(soft_config, parameters=replace(config.parameters, fade=100))
    np.testing.assert_array_equal(apply_filter(pixels, config=bypass), pixels)


def test_fade_blends_the_completed_smooth_shadow_result():
    pixels = np.random.default_rng(43).integers(0, 40, (17, 23, 3), np.uint8)
    config = Config(
        parameters=FilterParams(fade=0), processing=ProcessingConfig(resize=False),
        numerics=NumericsConfig(iterations=10, shadow_softness=6),
    )
    smooth = apply_filter(pixels, config=config)
    faded = apply_filter(pixels, config=replace(config, parameters=FilterParams(fade=25)))
    expected = (3 * smooth.astype(np.uint16) + pixels.astype(np.uint16) + 1) // 4
    np.testing.assert_array_equal(faded, expected)


def test_existing_positional_numerics_arguments_keep_their_meaning():
    config = NumericsConfig(5, 1.2, 16, 0.01, 0.9, 0.001, 0.001, 0.02, 8, 2, "standard")
    assert config.shadow_threshold == 8
    assert config.detail_level == 2
    assert config.power_mode == "standard"
    assert config.shadow_softness == 0


def test_existing_schema_one_configuration_keeps_reference_shadow_behavior():
    config = Config.from_dict({"schema_version": 1, "numerics": {"shadow_threshold": 6}})
    assert config.numerics.shadow_softness == 0


@pytest.mark.parametrize(
    "power_mode,gamma,threshold,softness,expected_sha256",
    [
        ('approximate', 1, 6, 6,
         '0cd6e52543c21500883b9170cf775e9fe645312719e61d715a85d0edcbed9ab3'),
        ('standard', 1.7, 6, 12,
         'd6594f9853d020c64d72cede2d5ac0da9f410e0c828b6d5066bbecf360094fde'),
        ('approximate', 0.2, 0, 1,
         'f09ce0c218f883dff3b87eedd7c590e06a16f73acc1db21356db058237271baf'),
        ('standard', 4.0, 255, 255,
         '73fcf8598becb22f3f5bbc9fcb01147ee90d375ba1d357e9b43891119a91a5e8'),
        ('approximate', 1, 6, 0,
         '4df051d59d09d5f2bc4cb6d0d7d74617bc16562dfb5c5788f18092edfd55e0ef'),
        ('standard', 1, 255, 1,
         '51ea5de13cc53707594a1460e207f5d90c26400fc97d20767925aeb0142f1eab'),
    ],
)
def test_shadow_blending_retains_frozen_rgb_bytes(
    power_mode, gamma, threshold, softness, expected_sha256,
):
    # Captured from the NumPy shadow blend before its fused-loop optimization.
    rng = np.random.default_rng(44)
    rgb = rng.integers(0, 256, (31, 47, 3), np.uint8)
    rgb[:16] = rng.integers(0, 25, (16, 47, 3), np.uint8)
    mapped = rng.uniform(-0.5, 2, (31, 47)).astype(np.float32)
    mapped[0, :7] = [np.nan, np.inf, -np.inf, 0, -0., 1, 10.5 / 255]
    luminance = luminance_u8(rgb).astype(np.float32) / 255
    output = _recombine(
        rgb, luminance, mapped, 80, threshold, power_mode, gamma, softness,
    )
    assert hashlib.sha256(output.tobytes()).hexdigest() == expected_sha256
