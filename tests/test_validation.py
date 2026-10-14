import json

import numpy as np
import pytest
from PIL import Image

from luma_fatal.validation import compare_images, load_rgb, write_difference_panel


def test_exact_match_is_json_safe_and_has_perfect_metrics():
    rng = np.random.default_rng(99)
    image = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    report = compare_images(image, image)
    assert report["mae"] == report["rmse"] == report["max_abs_error"] == 0
    assert report["psnr_db"] is None
    assert report["psnr_infinite"] is True
    assert report["ssim"] == pytest.approx(1)
    assert report["exact_pixel_fraction"] == report["exact_channel_fraction"] == 1
    json.dumps(report, allow_nan=False)


def test_known_rgb_error_metrics_and_sign():
    reference = np.full((16, 16, 3), 100, dtype=np.uint8)
    candidate = reference.copy()
    candidate[:, :, 0] += 3
    candidate[:, :, 1] -= 4
    report = compare_images(reference, candidate)
    assert report["mae"] == pytest.approx(7 / 3)
    assert report["rmse"] == pytest.approx(np.sqrt(25 / 3))
    assert report["max_abs_error"] == 4
    assert report["psnr_db"] == pytest.approx(10 * np.log10(255**2 / (25 / 3)))
    assert report["exact_pixel_fraction"] == 0
    assert report["exact_channel_fraction"] == pytest.approx(1 / 3)
    assert report["per_channel"]["red"]["mean_signed_error"] == 3
    assert report["per_channel"]["green"]["mean_signed_error"] == -4
    assert report["per_channel"]["blue"]["mae"] == 0


def test_single_pixel_error_counts_pixels_and_samples_separately():
    reference = np.zeros((4, 5, 3), dtype=np.uint8)
    candidate = reference.copy()
    candidate[0, 0, 1] = 255
    report = compare_images(reference, candidate)
    assert report["exact_pixel_fraction"] == pytest.approx(19 / 20)
    assert report["exact_channel_fraction"] == pytest.approx(59 / 60)
    assert report["mae"] == pytest.approx(255 / 60)


def test_ssim_constant_image_has_analytical_value():
    reference = np.full((20, 20, 3), 40, dtype=np.uint8)
    candidate = np.full((20, 20, 3), 80, dtype=np.uint8)
    c1 = (0.01 * 255) ** 2
    expected = (2 * 40 * 80 + c1) / (40**2 + 80**2 + c1)
    report = compare_images(reference, candidate)
    assert report["ssim"] == pytest.approx(expected, abs=1e-12)


def test_ssim_detects_spatial_change():
    yy, xx = np.indices((32, 32))
    checker = np.repeat((((xx + yy) % 2) * 255)[:, :, None], 3, axis=2).astype(np.uint8)
    report = compare_images(checker, 255 - checker)
    assert report["ssim"] < 0


def test_dimensions_are_never_automatically_aligned():
    with pytest.raises(ValueError, match="dimensions must match"):
        compare_images(
            np.zeros((16, 16, 3), dtype=np.uint8), np.zeros((16, 15, 3), dtype=np.uint8)
        )


@pytest.mark.parametrize(
    "array",
    [np.zeros((3, 3)), np.zeros((3, 3, 3)), np.zeros((0, 3, 3), dtype=np.uint8)],
)
def test_invalid_arrays_fail(array):
    with pytest.raises(ValueError):
        load_rgb(array)


def test_path_loading_and_rgb_conversion(tmp_path):
    path = tmp_path / "gray.png"
    Image.new("L", (8, 9), 128).save(path)
    rgb = load_rgb(path)
    assert rgb.shape == (9, 8, 3)
    assert rgb.dtype == np.uint8
    assert np.all(rgb == 128)


def test_difference_panel_preserves_resolution_and_amplifies_error(tmp_path):
    reference = np.full((16, 20, 3), 100, dtype=np.uint8)
    candidate = reference.copy()
    candidate[5, 6] = (102, 90, 150)
    output = write_difference_panel(
        reference, reference, candidate, tmp_path / "panels" / "difference.png", gain=8
    )
    with Image.open(output) as panel:
        assert panel.size == (80, 44)
        assert panel.getpixel((60 + 6, 28 + 5)) == (16, 80, 255)
        assert panel.getpixel((20 + 6, 28 + 5)) == (100, 100, 100)


@pytest.mark.parametrize("gain", [0, -1, float("nan"), float("inf")])
def test_invalid_difference_gain_fails(tmp_path, gain):
    image = np.zeros((3, 3, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="finite and positive"):
        write_difference_panel(
            image, image, image, tmp_path / "difference.png", gain=gain
        )
