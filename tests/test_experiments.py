"""Experimental controls preserve the reference path and have explicit semantics."""

from dataclasses import replace
import json

import numpy as np
from PIL import Image
import pytest

from luma_fatal import Config, FilterParams, NumericsConfig, ProcessingConfig, apply_filter
from luma_fatal.cli import main
from luma_fatal.core import _phi, _recombine
from luma_fatal.numerics import power64
from luma_fatal.processing import resize_working


@pytest.mark.parametrize("suffix", ["yaml", "toml", "json"])
def test_experimental_configuration_round_trips(suffix, tmp_path):
    config = Config(
        parameters=FilterParams(output_gamma=1.6),
        processing=ProcessingConfig(working_resampler="bicubic"),
        numerics=NumericsConfig(detail_level=2, power_mode="standard"),
    )
    path = tmp_path / f"settings.{suffix}"
    content = {
        "yaml": config.to_yaml(),
        "toml": config.to_toml(),
        "json": json.dumps(config.to_dict()),
    }[suffix]
    path.write_text(content)
    assert Config.load(path) == config


@pytest.mark.parametrize(
    "values",
    [
        {"parameters": {"output_gamma": value}}
        for value in (0, -1, True, "1.2", float("nan"), float("inf"), 5e-324)
    ]
    + [{"numerics": {"detail_level": value}} for value in (-1, True, 1.5)]
    + [{"numerics": {"power_mode": value}} for value in (None, True, "exact")]
    + [{"processing": {"working_resampler": value}} for value in (None, "lanczos")],
)
def test_invalid_experimental_controls_are_rejected(values):
    with pytest.raises(ValueError):
        Config.from_dict(values)


@pytest.mark.parametrize(
    "exponent,expected", [(0, [1, 1, 1]), (0.5, [0.5, 1, 2]), (2, [0.0625, 1, 16])]
)
def test_standard_power_has_the_expected_mathematical_values(exponent, expected):
    output = power64(np.array([0.25, 1, 4], np.float32), exponent, "standard")
    assert output.dtype == np.float64
    np.testing.assert_array_equal(output, expected)


@pytest.mark.parametrize("variant", [1, 2])
def test_standard_power_gives_unit_attenuation_at_beta_100(variant):
    pyramid = [
        np.arange(64, dtype=np.float32).reshape(8, 8),
        np.arange(16, dtype=np.float32).reshape(4, 4),
    ]
    params = FilterParams(beta=100)
    standard = _phi(pyramid, variant, params, power_mode="standard")
    np.testing.assert_array_equal(standard, np.ones((8, 8), np.float32))
    assert np.any(_phi(pyramid, variant, params) != standard)


@pytest.mark.parametrize("variant", [1, 2])
def test_detail_level_excludes_the_finest_attenuation_field(variant):
    rng = np.random.default_rng(23)
    fine = rng.random((8, 8), dtype=np.float32)
    other_fine = rng.random((8, 8), dtype=np.float32)
    coarse = rng.random((4, 4), dtype=np.float32)
    params = FilterParams()
    first = _phi([fine, coarse], variant, params, detail_level=1)
    second = _phi([other_fine, coarse], variant, params, detail_level=1)
    np.testing.assert_array_equal(first, second)
    assert first.shape == fine.shape
    assert np.any(first != 1)
    assert np.any(
        _phi([fine, coarse], variant, params)
        != _phi([other_fine, coarse], variant, params)
    )
    for excluded in (2, 20):
        np.testing.assert_array_equal(
            _phi([fine, coarse], variant, params, detail_level=excluded),
            np.ones_like(fine),
        )


@pytest.mark.parametrize(
    "gamma,expected", [(1, [0, 64, 255]), (2, [0, 128, 255]), (0.5, [0, 16, 255])]
)
def test_output_gamma_curve_precedes_rgb8_quantization(gamma, expected):
    rgb = np.full((1, 3, 3), 128, np.uint8)
    luminance = np.full((1, 3), 128 / 255, np.float32)
    mapped = np.array([[0, 0.25, 1]], np.float32)
    output = _recombine(
        rgb, luminance, mapped, 0, power_mode="standard", output_gamma=gamma
    )
    np.testing.assert_array_equal(
        output, np.repeat(np.array(expected)[None, :, None], 3, axis=2)
    )


def test_output_gamma_preserves_the_shadow_gate():
    rgb = np.full((2, 3, 3), 2, np.uint8)
    output = _recombine(
        rgb,
        np.full((2, 3), 2 / 255, np.float32),
        np.ones((2, 3), np.float32),
        100,
        output_gamma=2,
    )
    np.testing.assert_array_equal(output, rgb)


def test_output_gamma_clips_overshoot_and_preserves_nan_white_behavior():
    output = _recombine(
        np.full((1, 3, 3), 128, np.uint8),
        np.full((1, 3), 128 / 255, np.float32),
        np.array([[-1, 2, np.nan]], np.float32),
        0,
        power_mode="standard",
        output_gamma=1.5,
    )
    np.testing.assert_array_equal(output[0, :, 0], [0, 255, 255])


@pytest.mark.parametrize(
    "method,expected",
    [
        ("nearest", [[0, 8], [32, 40]]),
        ("bilinear", [[10, 18], [42, 50]]),
        ("bicubic", [[8, 16], [43, 51]]),
    ],
)
def test_working_resamplers_match_known_sampling_values(method, expected):
    plane = np.arange(4, dtype=np.uint8)[:, None] * 16 + np.arange(4, dtype=np.uint8) * 4
    rgb = np.repeat(plane[..., None], 3, axis=2)
    original = rgb.copy()
    output = resize_working(rgb, (2, 2), method)
    assert output.dtype == np.uint8
    np.testing.assert_array_equal(
        output, np.repeat(np.array(expected)[..., None], 3, axis=2)
    )
    np.testing.assert_array_equal(rgb, original)


def test_bicubic_working_resampler_clips_step_edge_overshoot():
    rgb = np.zeros((2, 8, 3), np.uint8)
    rgb[:, 4:] = 255
    output = resize_working(rgb, (6, 2), "bicubic")
    np.testing.assert_array_equal(output[..., 0], [[0, 0, 0, 255, 255, 255]] * 2)


@pytest.mark.parametrize("variant", [1, 2])
@pytest.mark.parametrize("max_edge", [None, 40])
@pytest.mark.parametrize("control", ["detail_level", "power_mode", "output_gamma"])
def test_experiments_change_pixels_in_both_resolution_routes(
    variant, max_edge, control
):
    pixels = np.random.default_rng(19).integers(0, 256, (67, 91, 3), np.uint8)
    original = pixels.copy()
    config = Config(
        filter=f"fattal{variant}",
        parameters=FilterParams(fade=0),
        processing=ProcessingConfig(resize=max_edge is not None, max_edge=max_edge),
        numerics=NumericsConfig(iterations=10, pyramid_min_size=8),
    )
    changes = {
        "detail_level": replace(
            config, numerics=replace(config.numerics, detail_level=1)
        ),
        "power_mode": replace(
            config, numerics=replace(config.numerics, power_mode="standard")
        ),
        "output_gamma": replace(
            config, parameters=replace(config.parameters, output_gamma=1.8)
        ),
    }
    reference = apply_filter(pixels, config=config)
    output = apply_filter(pixels, config=changes[control])
    assert output.dtype == np.uint8
    assert output.shape == pixels.shape
    assert np.any(output != reference)
    np.testing.assert_array_equal(pixels, original)


@pytest.mark.parametrize("variant", [1, 2])
@pytest.mark.parametrize("method", ["bilinear", "bicubic"])
def test_working_resampler_changes_bounded_filter_outputs(variant, method):
    pixels = np.random.default_rng(21).integers(0, 256, (67, 91, 3), np.uint8)
    config = Config(
        filter=f"fattal{variant}",
        parameters=FilterParams(fade=0),
        processing=ProcessingConfig(max_edge=40),
    )
    expected = apply_filter(pixels, config=config)
    output = apply_filter(
        pixels,
        config=replace(
            config, processing=replace(config.processing, working_resampler=method)
        ),
    )
    assert output.shape == pixels.shape
    assert output.dtype == np.uint8
    assert np.any(output != expected)


@pytest.mark.parametrize("max_edge", [None, 40])
def test_gamma_is_applied_before_fade_and_bypassed_at_fade_100(max_edge):
    pixels = np.random.default_rng(25).integers(0, 256, (67, 91, 3), np.uint8)
    config = Config(
        parameters=FilterParams(fade=0, output_gamma=1.7),
        processing=ProcessingConfig(resize=max_edge is not None, max_edge=max_edge),
    )
    filtered = apply_filter(pixels, config=config)
    mixed = apply_filter(
        pixels, config=replace(config, parameters=replace(config.parameters, fade=25))
    )
    expected = (3 * filtered.astype(np.uint16) + pixels.astype(np.uint16) + 1) // 4
    np.testing.assert_array_equal(mixed, expected)
    identity = replace(config, parameters=replace(config.parameters, fade=100))
    np.testing.assert_array_equal(apply_filter(pixels, config=identity), pixels)


def test_cli_overrides_experimental_settings_and_records_resolved_values(
    tmp_path, capsys
):
    pixels = np.random.default_rng(27).integers(0, 256, (67, 91, 3), np.uint8)
    source, output = tmp_path / "source.png", tmp_path / "output.png"
    Image.fromarray(pixels).save(source)
    config = Config(
        parameters=FilterParams(fade=0, output_gamma=0.8),
        processing=ProcessingConfig(max_edge=40, working_resampler="bicubic"),
        numerics=NumericsConfig(iterations=7, pyramid_min_size=8, detail_level=2),
    )
    path = tmp_path / "settings.yaml"
    path.write_text(config.to_yaml())
    main(
        [
            "apply", str(source), str(output), "--config", str(path),
            "--output-format", "png", "--output-gamma", "1.6", "--detail-level", "1",
            "--power-mode", "standard", "--working-resampler", "bilinear",
        ]
    )
    record = json.loads(output.with_suffix(".meta.json").read_text())
    resolved = Config.from_dict(record["configuration"])
    assert resolved.parameters == replace(config.parameters, output_gamma=1.6)
    assert resolved.processing == replace(config.processing, working_resampler="bilinear")
    assert resolved.numerics == replace(
        config.numerics, detail_level=1, power_mode="standard"
    )
    with Image.open(output) as image:
        np.testing.assert_array_equal(
            np.asarray(image), apply_filter(pixels, config=resolved)
        )
