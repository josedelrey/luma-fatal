"""Vibrance boosts muted color without inventing hues or changing brightness."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from luma_fatal import Config, FilterParams, ProcessingConfig, apply_filter
from luma_fatal.cli import main
from luma_fatal.core import _boost_vibrance, _recombine
from luma_fatal.pipeline import process_frames
from test_video import tools


WEIGHTS = np.array([4899, 9617, 1868], np.float64) / 16384


@pytest.mark.parametrize("value", [-1, 100.1, True, "50", None,
                                    float("nan"), float("inf")])
def test_invalid_vibrance_is_rejected(value):
    with pytest.raises(ValueError, match="vibrance"):
        FilterParams(vibrance=value)


@pytest.mark.parametrize("suffix", ["yaml", "toml", "json"])
def test_vibrance_configuration_roundtrips(tmp_path, suffix):
    config = Config(parameters=FilterParams(vibrance=62.5))
    content = {"yaml": config.to_yaml(), "toml": config.to_toml(),
               "json": json.dumps(config.to_dict())}[suffix]
    path = tmp_path / f"settings.{suffix}"
    path.write_text(content)
    assert Config.load(path) == config


@pytest.mark.parametrize("strength,expected", [
    (0, [150, 100, 100]), (25, [154, 98, 98]),
    (50, [158, 97, 97]), (100, [166, 93, 93]),
])
def test_vibrance_has_known_quantized_color_values(strength, expected):
    rgb = np.array([[[150, 100, 100]]], np.uint8)
    luminance = np.array([[100 / 255]], np.float32)
    result = _recombine(rgb, luminance, luminance, 100, power_mode="standard",
                        vibrance=strength)
    np.testing.assert_array_equal(result, [[expected]])


@pytest.mark.parametrize("strength", [0, 50, 100])
def test_neutral_grays_are_unchanged(strength):
    rgb = np.repeat(np.linspace(0, 255, 257)[None, :, None], 3, axis=2)
    np.testing.assert_array_equal(_boost_vibrance(rgb, strength), rgb)


def test_fully_saturated_colors_are_unchanged():
    rgb = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255],
                     [255, 255, 0], [255, 0, 255], [0, 255, 255]]], np.float64)
    np.testing.assert_array_equal(_boost_vibrance(rgb, 100), rgb)


def test_muted_colors_receive_more_relative_chroma_boost():
    rgb = np.array([[[150, 140, 140], [200, 100, 100]]], np.float64)
    result = _boost_vibrance(rgb, 100)
    gain = (result[0, :, 0] - result[0, :, 1]) / (rgb[0, :, 0] - rgb[0, :, 1])
    np.testing.assert_allclose(gain, [1.8711111111111112, 1.25], rtol=0, atol=1e-14)


def test_vibrance_preserves_weighted_brightness_hue_and_gamut():
    rgb = np.random.default_rng(96).uniform(0, 255, (31, 47, 3))
    before = rgb.copy()
    result = _boost_vibrance(rgb, 100)
    assert np.isfinite(result).all()
    assert result.min() >= 0 and result.max() <= 255
    brightness = rgb @ WEIGHTS
    np.testing.assert_allclose(result @ WEIGHTS, brightness, rtol=0, atol=1e-12)
    chroma = rgb - brightness[..., None]
    changed = result - brightness[..., None]
    gain = np.sum(changed * chroma, axis=2) / np.sum(chroma * chroma, axis=2)
    assert gain.min() >= 1 - 1e-14 and gain.max() <= 2
    np.testing.assert_allclose(changed, chroma * gain[..., None], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(rgb, before)


def test_gamut_boundary_limits_boost_without_channel_clipping():
    rgb = np.array([[[254, 200, 150]]], np.float64)
    result = _boost_vibrance(rgb, 100)
    assert result[0, 0, 0] == 255
    assert result[0, 0, 2] > 0
    np.testing.assert_allclose(result @ WEIGHTS, rgb @ WEIGHTS, rtol=0, atol=1e-12)
    actual_gain = (result[0, 0, 0] - result[0, 0, 2]) / (254 - 150)
    assert 1 < actual_gain < 1.1


def test_nonfinite_and_overrange_values_follow_existing_clipping_convention():
    rgb = np.array([[[np.nan, np.inf, -np.inf], [-1, 128, 300]]], np.float64)
    result = _boost_vibrance(rgb, 100)
    np.testing.assert_array_equal(result, [[[255, 255, 0], [0, 128, 255]]])


def test_vibrance_follows_gamma():
    rgb = np.array([[[150, 100, 100]]], np.uint8)
    luminance = np.array([[100 / 255]], np.float32)
    result = _recombine(rgb, luminance, luminance, 100, power_mode="standard",
                        output_gamma=2, vibrance=100)
    np.testing.assert_array_equal(result, [[[212, 153, 153]]])


def test_output_levels_follow_vibrance():
    rgb = np.array([[[150, 100, 100]]], np.uint8)
    luminance = np.array([[100 / 255]], np.float32)
    result = _recombine(rgb, luminance, luminance, 100, power_mode="standard",
                        vibrance=100, black_level=20, white_level=80)
    np.testing.assert_array_equal(result, [[[150, 107, 107]]])


@pytest.mark.parametrize("softness", [0, 6])
def test_protected_shadows_keep_original_color(softness):
    rgb = np.zeros((2, 3, 3), np.uint8)
    luminance = np.full((2, 3), 1 / 255, np.float32)
    result = _recombine(rgb, luminance, np.ones_like(luminance), 100,
                        power_mode="standard", vibrance=100, shadow_softness=softness)
    np.testing.assert_array_equal(result, rgb)


@pytest.mark.parametrize("variant", [1, 2])
@pytest.mark.parametrize("max_edge", [None, 40])
def test_vibrance_works_in_both_resolution_routes_and_preserves_fade_and_alpha(
    variant, max_edge,
):
    pixels = np.random.default_rng(97).integers(20, 240, (67, 91, 4), np.uint8)
    before = pixels.copy()
    config = Config(
        filter=f"fattal{variant}",
        parameters=FilterParams(fade=0, vibrance=80, tone_strength=60,
                                highlight_compression=75, black_level=5, white_level=95),
        processing=ProcessingConfig(resize=max_edge is not None, max_edge=max_edge,
                                    transparency="straight"),
    )
    result = apply_filter(pixels, config=config)
    disabled = apply_filter(pixels, config=replace(
        config, parameters=replace(config.parameters, vibrance=0)))
    assert np.any(result != disabled)
    assert result[..., :3].min() >= 13 and result[..., :3].max() <= 242
    np.testing.assert_array_equal(result[..., 3], pixels[..., 3])
    faded = apply_filter(pixels, config=replace(
        config, parameters=replace(config.parameters, fade=25)))
    expected = (3 * result[..., :3].astype(np.uint16)
                + pixels[..., :3].astype(np.uint16) + 1) // 4
    np.testing.assert_array_equal(faded[..., :3], expected)
    np.testing.assert_array_equal(faded[..., 3], pixels[..., 3])
    bypass = apply_filter(pixels, config=replace(
        config, parameters=replace(config.parameters, fade=100)))
    np.testing.assert_array_equal(bypass, pixels)
    np.testing.assert_array_equal(pixels, before)


def test_image_cli_overrides_vibrance_and_records_metadata(tmp_path, capsys):
    pixels = np.random.default_rng(98).integers(20, 240, (35, 47, 3), np.uint8)
    source, output = tmp_path / "input.png", tmp_path / "output.png"
    Image.fromarray(pixels).save(source)
    config = Config(parameters=FilterParams(fade=0, vibrance=20))
    path = tmp_path / "settings.yaml"
    path.write_text(config.to_yaml())
    main(["apply", str(source), str(output), "--config", str(path),
          "--vibrance", "75", "--output-format", "png"])
    record = json.loads(output.with_suffix(".meta.json").read_text())
    assert record["configuration"]["parameters"]["vibrance"] == 75
    assert record["parameters"]["vibrance"] == 75
    expected = apply_filter(pixels, config=replace(
        config, parameters=replace(config.parameters, vibrance=75)))
    np.testing.assert_array_equal(np.array(Image.open(output)), expected)


def test_video_cli_vibrance_works_with_tone_controls_and_fbm(tmp_path, tools, capsys):
    from luma_fatal.prenoise import SourceNoise
    from test_video import make_clip, reference_rgb

    preset = Path(__file__).resolve().parents[1] / "configs/fattal-2-compressed-fbm.yaml"
    config = Config.load(preset)
    source = make_clip(tmp_path / "input.mkv", tools, width=48, height=36)
    output = tmp_path / "output.mp4"
    main(["video", str(source), str(output), "--config", str(preset),
          "--vibrance", "75", "--tone-strength", "60", "--highlight-compression", "75",
          "--black-level", "5", "--white-level", "95",
          "--ffmpeg", tools[0], "--ffprobe", tools[1], "--json"])
    record = json.loads(capsys.readouterr().out)
    parameters = record["configuration"]["parameters"]
    assert parameters["vibrance"] == 75
    assert record["source_noise"]["kind"] == "fbm"
    frames = reference_rgb(source, tools, 48, 36, deband="gentle")
    expected = hashlib.sha256()
    config = replace(config, parameters=replace(
        config.parameters, vibrance=75, tone_strength=60, highlight_compression=75,
        black_level=5, white_level=95))
    for frame in process_frames(frames, config=config,
                                source_cleanup=SourceNoise(0.75, kind="fbm")):
        expected.update(frame.tobytes())
    assert record["processed_rgb_sha256"] == expected.hexdigest()
