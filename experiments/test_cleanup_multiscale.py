"""Safety and strength contracts for the experimental correction."""

import numpy as np
import pytest

from cleanup_multiscale import ScaleSeparatedCleanup
from luma_fatal.cleanup import SourceCleanup
from luma_fatal.numerics import luminance_u8


def sample(shape=(101, 397)):
    y, x = np.indices(shape)
    ramp = 70 + x * 0.025 + y * 0.015
    noise = np.random.default_rng(42).normal(0, 0.7, shape)
    gray = np.clip(np.round(ramp + noise), 0, 255).astype(np.uint8)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    # A strong stripe provides protected pixels inside the same frame.
    rgb[:, shape[1] // 2:shape[1] // 2 + 6] = (200, 30, 20)
    luminance = luminance_u8(rgb).astype(np.float32) * np.float32(1 / 255)
    return rgb, luminance


@pytest.mark.parametrize("strength", [0, 0.25, 0.75, 1])
def test_equal_strengths_recover_existing_cleanup(strength):
    rgb, luminance = sample()
    expected = SourceCleanup(strength)(rgb, luminance)
    actual = ScaleSeparatedCleanup(strength, strength)(rgb, luminance)
    np.testing.assert_allclose(actual[0], expected[0], rtol=0, atol=2e-5)
    np.testing.assert_allclose(actual[1], expected[1], rtol=0, atol=1e-7)


@pytest.mark.parametrize("shape", [(1, 1), (11, 17), (101, 397), (129, 769)])
def test_bounds_and_protected_pixels(shape):
    rgb, luminance = sample(shape)
    cleanup = ScaleSeparatedCleanup(0.5, 1, 0.7)
    color, corrected = cleanup(rgb, luminance)
    assert np.isfinite(color).all() and np.isfinite(corrected).all()
    assert color.shape == rgb.shape and corrected.shape == luminance.shape
    assert color.min() >= 0 and color.max() <= 255
    assert corrected.min() >= 0 and corrected.max() <= 1
    assert np.max(np.abs(color - rgb)) <= 4.00002
    assert np.max(np.abs(corrected - luminance) * 255) <= 4.50002
    protected = cleanup.mask == 0
    if shape[1] >= 397:
        assert protected.any()
    np.testing.assert_array_equal(color[protected], rgb[protected])
    np.testing.assert_array_equal(corrected[protected], luminance[protected])


def test_zero_strength_is_identity():
    rgb, luminance = sample()
    color, corrected = ScaleSeparatedCleanup(0, 0)(rgb, luminance)
    np.testing.assert_array_equal(color, rgb)
    np.testing.assert_array_equal(corrected, luminance)


def test_constant_image_does_not_acquire_a_pattern():
    rgb = np.full((101, 397, 3), (74, 91, 103), np.uint8)
    luminance = luminance_u8(rgb).astype(np.float32) * np.float32(1 / 255)
    color, corrected = ScaleSeparatedCleanup()(rgb, luminance)
    np.testing.assert_allclose(color, rgb, rtol=0, atol=2e-5)
    assert np.ptp(corrected) < 1e-7


def test_repeated_frame_is_stable():
    rgb, luminance = sample()
    cleanup = ScaleSeparatedCleanup()
    first = cleanup(rgb, luminance)
    for _ in range(4):
        result = cleanup(rgb, luminance)
        np.testing.assert_allclose(result[0], first[0], rtol=0, atol=2e-5)
        np.testing.assert_allclose(result[1], first[1], rtol=0, atol=1e-7)


@pytest.mark.parametrize("arguments", [(-0.1, 1, 1), (0.5, 1.1, 1),
                                        (0.5, 1, 0), (0.5, 1, float("nan"))])
def test_invalid_settings_fail(arguments):
    with pytest.raises(ValueError):
        ScaleSeparatedCleanup(*arguments)
