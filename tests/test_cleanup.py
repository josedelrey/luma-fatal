"""Source cleanup protects structure and keeps temporal history bounded."""

from dataclasses import replace

import numpy as np
import pytest

from luma_fatal import Config, FilterParams, NumericsConfig, ProcessingConfig, process_frames
from luma_fatal.cleanup import SourceCleanup
from luma_fatal.numerics import luminance_u8


def reference(rgb):
    return luminance_u8(rgb).astype(np.float32) * np.float32(1 / 255)


@pytest.mark.parametrize("strength", [True, -0.1, 1.01, float("nan"), float("inf"), "0.2"])
def test_cleanup_rejects_invalid_strength(strength):
    with pytest.raises(ValueError):
        SourceCleanup(strength)


def test_mask_selects_flat_block_noise_and_protects_texture():
    yy, xx = np.indices((128, 256))
    gray = np.full((128, 256), 100, np.uint8)
    gray[:, :128] += ((xx[:, :128] // 8 + yy[:, :128] // 8) % 2).astype(np.uint8)
    gray[:, 128:] = 80 + 60 * ((xx[:, 128:] // 2 + yy[:, 128:] // 2) % 2)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    cleanup = SourceCleanup(0.25)
    cleaned, luminance = cleanup(rgb, reference(rgb))
    assert cleanup.mask[16:-16, 16:100].mean() > 0.7
    assert cleanup.mask[16:-16, 155:-16].max() < 0.05
    assert cleaned.dtype == luminance.dtype == np.float32
    assert np.isfinite(cleaned).all() and np.isfinite(luminance).all()
    assert np.any(cleaned[:, :100] != rgb[:, :100])
    np.testing.assert_array_equal(cleaned[:, 160:], rgb[:, 160:])


def test_reduced_detection_preserves_subpixel_texture_and_bounds_changes():
    yy, xx = np.indices((397, 769))
    gray = np.where((xx + yy) % 2, 60, 180).astype(np.uint8)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    snapshot = rgb.copy()
    cleanup = SourceCleanup(0.25)
    cleaned, luminance = cleanup(rgb, reference(rgb))
    assert max(cleanup.mask.shape) <= 384
    assert cleanup.mask.max() == 0
    np.testing.assert_array_equal(cleaned, rgb)
    np.testing.assert_array_equal(luminance, reference(rgb))
    np.testing.assert_array_equal(rgb, snapshot)


@pytest.mark.parametrize("width", [320, 640, 1280])
@pytest.mark.parametrize("transpose", [False, True])
def test_smooth_shading_qualifies_at_multiple_detection_scales(width, transpose):
    yy, xx = np.indices((128, width))
    start = width // 2 - 45
    gray = np.rint(np.clip(40 + 2 * (xx - start) + yy / 16, 40, 220)).astype(np.uint8)
    rgb = np.repeat(gray[..., None], 3, 2)
    if transpose:
        rgb = rgb.transpose(1, 0, 2).copy()
    cleanup = SourceCleanup(0.25)
    cleanup(rgb, reference(rgb))
    factor = max(1, (width + 383) // 384)
    mask = cleanup.mask.T if transpose else cleanup.mask
    interior = mask[8:-8, (start + 24) // factor:(start + 64) // factor]
    assert interior.mean() > 0.9


def test_gradient_detrending_preserves_stripes_hidden_by_area_reduction():
    yy, xx = np.indices((128, 640))
    rgb = np.repeat(np.where(xx % 2, 80, 140).astype(np.uint8)[..., None], 3, 2)
    cleanup = SourceCleanup(0.25)
    cleaned, luminance = cleanup(rgb, reference(rgb))
    assert cleanup.mask.max() == 0
    np.testing.assert_array_equal(cleaned, rgb)
    np.testing.assert_array_equal(luminance, reference(rgb))


def test_color_edges_are_protected_at_matching_luminance():
    rgb = np.empty((64, 128, 3), np.uint8)
    rgb[:, :64] = [80, 100, 100]
    rgb[:, 64:] = [139, 70, 100]
    assert np.unique(luminance_u8(rgb)).size == 1
    cleanup = SourceCleanup(0.25)
    cleanup(rgb, reference(rgb))
    assert cleanup.mask[:, 62:65].max() == 0


def test_fractional_corrections_stay_bounded_without_byte_rounding():
    yy, xx = np.indices((65, 97))
    gray = (60 + xx // 8 + (yy // 9) % 2).astype(np.uint8)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    cleaned, luminance = SourceCleanup(0.25)(rgb, reference(rgb))
    assert np.max(np.abs(cleaned - rgb)) <= 1
    assert np.any(cleaned != np.floor(cleaned))
    assert np.min(luminance) >= 0 and np.max(luminance) <= 1


@pytest.mark.parametrize("value", [0, 64, 128, 255])
def test_constant_color_stays_constant(value):
    rgb = np.full((17, 23, 3), value, np.uint8)
    cleaned, luminance = SourceCleanup(1)(rgb, reference(rgb))
    np.testing.assert_allclose(cleaned, rgb, atol=1e-5, rtol=0)
    np.testing.assert_allclose(luminance, reference(rgb), atol=1e-7, rtol=0)


def test_identical_frames_have_identical_masks_and_pixels():
    rgb = np.random.default_rng(17).integers(60, 65, (96, 128, 3), np.uint8)
    cleanup = SourceCleanup(0.25)
    first = cleanup(rgb, reference(rgb))
    first_mask = cleanup.mask.copy()
    for _ in range(3):
        actual = cleanup(rgb, reference(rgb))
        np.testing.assert_array_equal(cleanup.mask, first_mask)
        for expected, output in zip(first, actual):
            np.testing.assert_array_equal(output, expected)


def test_scene_cut_discards_mask_history():
    old = np.full((64, 96, 3), 35, np.uint8)
    yy, xx = np.indices(old.shape[:2])
    current = np.repeat((150 + 30 * ((xx // 3 + yy // 3) % 2))[..., None], 3, 2).astype(np.uint8)
    cleanup = SourceCleanup(0.5)
    cleanup(old, reference(old))
    actual = cleanup(current, reference(current))
    fresh = SourceCleanup(0.5)
    expected = fresh(current, reference(current))
    assert cleanup.resets == 1
    np.testing.assert_array_equal(cleanup.mask, fresh.mask)
    for output, target in zip(actual, expected):
        np.testing.assert_array_equal(output, target)


def test_new_moving_edges_immediately_override_old_flat_mask():
    rgb = np.full((64, 128, 3), 80, np.uint8)
    cleanup = SourceCleanup(0.5)
    cleanup(rgb, reference(rgb))
    current = rgb.copy()
    current[:, 50:80] = 180
    cleanup(current, reference(current))
    assert cleanup.mask[:, 48:51].max() == 0
    assert cleanup.mask[:, 78:81].max() == 0


@pytest.mark.parametrize("variant", ["fattal1", "fattal2"])
@pytest.mark.parametrize("resize", [True, False])
def test_zero_strength_preserves_frame_pixels_and_fade(variant, resize):
    rgb = np.random.default_rng(19).integers(0, 256, (35, 47, 3), np.uint8)
    config = Config(
        filter=variant, parameters=FilterParams(fade=0),
        processing=ProcessingConfig(resize=resize),
        numerics=NumericsConfig(iterations=11, shadow_softness=6),
    )
    baseline = next(process_frames([rgb], config=config))
    actual = next(process_frames([rgb], config=config, source_cleanup=SourceCleanup(0)))
    np.testing.assert_array_equal(actual, baseline)
    cleanup = SourceCleanup(0.25)
    faded = next(process_frames([rgb], config=replace(config, parameters=FilterParams(fade=100)),
                                source_cleanup=cleanup))
    np.testing.assert_array_equal(faded, rgb)
    assert cleanup.frames == 0


def test_cleanup_is_applied_at_working_resolution():
    rgb = np.full((80, 120, 3), 80, np.uint8)
    cleanup = SourceCleanup(0.25)
    config = Config(processing=ProcessingConfig(max_edge=60),
                    numerics=NumericsConfig(iterations=11))
    output = next(process_frames([rgb], config=config, source_cleanup=cleanup))
    assert cleanup.shape == (40, 60, 3)
    assert output.shape == rgb.shape


def test_cleanup_preserves_original_hard_shadow_protection():
    yy, xx = np.indices((48, 64))
    gray = (5 + (xx // 8 + yy // 8) % 2).astype(np.uint8)
    rgb = np.repeat(gray[..., None], 3, 2)
    rgb[:8, :8] = 100
    config = Config(parameters=FilterParams(fade=0), numerics=NumericsConfig(iterations=11))
    output = next(process_frames([rgb], config=config, source_cleanup=SourceCleanup(1)))
    # Stay strictly inside the existing float32 shadow cutoff.
    shadows = np.all(rgb == 5, axis=2)
    np.testing.assert_array_equal(output[shadows], rgb[shadows])
