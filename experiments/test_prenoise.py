"""Behavioral checks for the opt-in pre-filter noise experiment.

Run explicitly with pytest experiments/test_prenoise.py from the repository.
The experiment and these checks are excluded from release distributions.
"""

from dataclasses import replace

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from experiments.prenoise import SourceNoise, noise_field
from luma_fatal.cleanup import SourceCleanup
from luma_fatal.configuration import Config
from luma_fatal.core import apply_core
from luma_fatal.numerics import luminance_fractional


def source(shape=(97, 193)):
    y, x = np.indices(shape)
    gray = np.round(80 + 0.03 * x + 0.02 * y).astype(np.uint8)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    return rgb, luminance_fractional(rgb)


@pytest.mark.parametrize("kind", ["highpass", "fbm"])
def test_noise_is_centered_normalized_and_reproducible(kind):
    field = noise_field((193, 257), kind, 42)
    np.testing.assert_array_equal(field, noise_field(field.shape, kind, 42))
    assert abs(float(field.mean())) < 1e-6
    assert float(np.mean(field * field)) == pytest.approx(1, abs=1e-6)
    assert not np.array_equal(field, noise_field(field.shape, kind, 43))


def test_fbm_contains_more_broad_variation_than_highpass():
    fine = noise_field((193, 257), "highpass", 42)
    broad = noise_field(fine.shape, "fbm", 42)
    assert np.std(gaussian_filter(broad, 2)) > 5 * np.std(gaussian_filter(fine, 2))


@pytest.mark.parametrize("kind", ["highpass", "fbm"])
@pytest.mark.parametrize("shape", [(1, 1), (7, 13), (97, 193), (99, 397)])
def test_finite_bounded_achromatic_fractional_noise(kind, shape):
    rgb, luminance = source(shape)
    rgb[0, 0] = (0, 100, 255)
    luminance = luminance_fractional(rgb)
    original = rgb.copy()
    callback = SourceNoise(4, kind=kind)
    color, changed = callback(rgb, luminance)
    assert color.dtype == changed.dtype == np.float32
    assert color.shape == rgb.shape and changed.shape == luminance.shape
    assert np.isfinite(color).all() and np.isfinite(changed).all()
    assert color.min() >= 0 and color.max() <= 255
    assert changed.min() >= 0 and changed.max() <= 1
    delta = color - rgb
    assert np.abs(delta).max() <= 4.00001
    np.testing.assert_allclose(delta[..., 0], delta[..., 1], atol=2e-5, rtol=0)
    np.testing.assert_allclose(delta[..., 0], delta[..., 2], atol=2e-5, rtol=0)
    np.testing.assert_allclose(delta[..., 0], (changed - luminance) * 255,
                               atol=2e-5, rtol=0)
    np.testing.assert_array_equal(rgb, original)
    np.testing.assert_array_equal(color[0, 0], rgb[0, 0])
    if min(shape) > 1:
        assert np.any(color != np.round(color))


def test_zero_noise_recovers_selected_cleanup():
    rgb, luminance = source()
    for strength in (0, 0.75):
        expected = SourceCleanup(strength)(rgb, luminance)
        actual = SourceNoise(0, cleanup=strength)(rgb, luminance)
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, b)


def test_mask_protects_texture_and_noise_does_not_change_detection():
    rgb, luminance = source((97, 397))
    rgb[:, 250::2] = 30
    rgb[:, 251::2] = 190
    luminance = luminance_fractional(rgb)
    plain = SourceNoise(0)
    noisy = SourceNoise(0.75)
    plain(rgb, luminance)
    color, changed = noisy(rgb, luminance)
    np.testing.assert_array_equal(plain.mask, noisy.mask)
    assert noisy.mask[:, :100].mean() > 0.9
    assert noisy.mask[:, 300:].max() == 0
    protected = noisy.mask == 0
    np.testing.assert_array_equal(color[protected], rgb[protected])
    np.testing.assert_array_equal(changed[protected], luminance[protected])


def test_identical_frames_are_stable_and_shape_changes_reset():
    callback = SourceNoise(0.75)
    rgb, luminance = source()
    first = callback(rgb, luminance)
    for _ in range(3):
        repeated = callback(rgb, luminance)
        for a, b in zip(first, repeated):
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-5)
    other, other_y = source((51, 71))
    callback(other, other_y)
    assert callback.field.shape == other_y.shape
    for a, b in zip(callback(rgb, luminance), first):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("variant", [1, 2])
def test_core_zero_noise_and_original_fade_shadows(variant):
    rgb, _ = source((65, 97))
    rgb[:20] = 2
    config = Config()
    expected = apply_core(rgb, variant, config.parameters, config.numerics)
    actual = apply_core(rgb, variant, config.parameters, config.numerics,
                        source_cleanup=SourceNoise(0))
    np.testing.assert_array_equal(actual, expected)
    noisy = apply_core(rgb, variant, config.parameters, config.numerics,
                       source_cleanup=SourceNoise(0.75))
    np.testing.assert_array_equal(noisy[:20], rgb[:20])
    faded = apply_core(rgb, variant, replace(config.parameters, fade=100),
                       config.numerics, source_cleanup=SourceNoise(0.75))
    np.testing.assert_array_equal(faded, rgb)


@pytest.mark.parametrize("settings", [
    {"rms": -1}, {"rms": np.nan}, {"rms": np.inf}, {"rms": True},
    {"rms": 4.1}, {"kind": "unknown"}, {"cleanup": -1},
    {"masked": "yes"}, {"seed": -1}, {"seed": True},
])
def test_invalid_settings_fail(settings):
    with pytest.raises(ValueError):
        SourceNoise(**settings)
