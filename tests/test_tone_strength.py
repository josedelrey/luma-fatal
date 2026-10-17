"""Tone blending occurs before color reconstruction and output clipping."""

from dataclasses import replace
import hashlib
import json

import numpy as np
from PIL import Image
import pytest

from luma_fatal import Config, FilterParams, NumericsConfig, ProcessingConfig, apply_filter
from luma_fatal.cli import main
from luma_fatal.core import _recombine
from luma_fatal.pipeline import process_frames
from test_video import tools


@pytest.mark.parametrize("value", [-1, 100.1, True, "50", None,
                                    float("nan"), float("inf")])
def test_invalid_tone_strength_is_rejected(value):
    with pytest.raises(ValueError, match="tone_strength"):
        FilterParams(tone_strength=value)


@pytest.mark.parametrize("suffix", ["yaml", "toml", "json"])
def test_tone_strength_configuration_roundtrips(tmp_path, suffix):
    config = Config(parameters=FilterParams(tone_strength=62.5))
    content = {"yaml": config.to_yaml(), "toml": config.to_toml(),
               "json": json.dumps(config.to_dict())}[suffix]
    path = tmp_path / f"config.{suffix}"
    path.write_text(content)
    assert Config.load(path) == config


@pytest.mark.parametrize("strength,expected", [(0, 64), (25, 96), (50, 128), (100, 192)])
def test_tone_strength_interpolates_source_and_filtered_brightness(strength, expected):
    rgb = np.full((2, 3, 3), 64, np.uint8)
    luminance = np.full((2, 3), 64 / 255, np.float32)
    mapped = np.full((2, 3), 192 / 255, np.float32)
    result = _recombine(rgb, luminance, mapped, 100, power_mode="standard",
                        tone_strength=strength)
    np.testing.assert_array_equal(result, np.full_like(rgb, expected))


def test_tone_blends_before_color_channels_clip():
    rgb = np.array([[[25, 50, 75]]], np.uint8)
    luminance = np.array([[50 / 255]], np.float32)
    result = _recombine(rgb, luminance, np.ones_like(luminance), 100,
                        power_mode="standard", tone_strength=50)
    np.testing.assert_array_equal(result, [[[76, 153, 229]]])


def test_tone_blending_precedes_gamma():
    rgb = np.full((2, 3, 3), 64, np.uint8)
    luminance = np.full((2, 3), 64 / 255, np.float32)
    mapped = np.full((2, 3), 192 / 255, np.float32)
    result = _recombine(rgb, luminance, mapped, 100, power_mode="standard",
                        tone_strength=50, output_gamma=2)
    np.testing.assert_array_equal(result, np.full_like(rgb, 181))


def test_zero_tone_strength_uses_original_brightness_with_source_treatment():
    rgb = np.full((2, 3, 3), 64, np.uint8)
    original = np.full((2, 3), 64 / 255, np.float32)
    treated = np.full((2, 3), 128 / 255, np.float32)
    result = _recombine(
        rgb, treated, np.ones_like(treated), 100, power_mode="standard",
        color_rgb=np.full(rgb.shape, 128, np.float32),
        shadow_luminance=original, tone_strength=0,
    )
    np.testing.assert_array_equal(result, rgb)


@pytest.mark.parametrize("strength,expected", [(0, [64, 64, 64]), (50, [160, 160, 32])])
def test_nonfinite_mapping_has_defined_partial_strength_behavior(strength, expected):
    rgb = np.full((1, 3, 3), 64, np.uint8)
    luminance = np.full((1, 3), 64 / 255, np.float32)
    mapped = np.array([[np.nan, np.inf, -np.inf]], np.float32)
    before = mapped.copy()
    result = _recombine(rgb, luminance, mapped, 100, power_mode="standard",
                        tone_strength=strength)
    np.testing.assert_array_equal(result[0, :, 0], expected)
    np.testing.assert_array_equal(mapped, before)


@pytest.mark.parametrize("variant", [1, 2])
def test_zero_strength_handles_degenerate_flat_mapping(variant):
    pixels = np.full((35, 47, 3), (64, 96, 128), np.uint8)
    config = Config(
        filter=f"fattal{variant}", parameters=FilterParams(fade=0, tone_strength=0),
        numerics=NumericsConfig(power_mode="standard"),
    )
    np.testing.assert_array_equal(apply_filter(pixels, config=config), pixels)


@pytest.mark.parametrize("variant", [1, 2])
@pytest.mark.parametrize("max_edge", [None, 40])
def test_tone_strength_changes_both_resolution_routes_and_retains_fade(variant, max_edge):
    pixels = np.random.default_rng(83).integers(20, 220, (67, 91, 3), np.uint8)
    original = pixels.copy()
    config = Config(
        filter=f"fattal{variant}", parameters=FilterParams(fade=0, tone_strength=40),
        processing=ProcessingConfig(resize=max_edge is not None, max_edge=max_edge),
    )
    result = apply_filter(pixels, config=config)
    full = apply_filter(pixels, config=replace(
        config, parameters=replace(config.parameters, tone_strength=100)))
    assert np.any(result != full)
    faded = apply_filter(pixels, config=replace(
        config, parameters=replace(config.parameters, fade=25)))
    expected = (3 * result.astype(np.uint16) + pixels.astype(np.uint16) + 1) // 4
    np.testing.assert_array_equal(faded, expected)
    bypass = apply_filter(pixels, config=replace(
        config, parameters=replace(config.parameters, fade=100)))
    np.testing.assert_array_equal(bypass, pixels)
    np.testing.assert_array_equal(pixels, original)


@pytest.mark.parametrize("strength", [0, 50, 100])
def test_tone_strength_preserves_original_shadow_protection(strength):
    rgb = np.full((2, 3, 3), 2, np.uint8)
    luminance = np.full((2, 3), 2 / 255, np.float32)
    result = _recombine(rgb, luminance, np.ones_like(luminance), 100,
                        power_mode="standard", tone_strength=strength)
    np.testing.assert_array_equal(result, rgb)


def test_image_cli_overrides_tone_strength_and_records_metadata(tmp_path, capsys):
    pixels = np.random.default_rng(84).integers(20, 220, (35, 47, 3), np.uint8)
    source, output = tmp_path / "input.png", tmp_path / "output.png"
    Image.fromarray(pixels).save(source)
    config = Config(parameters=FilterParams(fade=0, tone_strength=80))
    path = tmp_path / "settings.yaml"
    path.write_text(config.to_yaml())
    main(["apply", str(source), str(output), "--config", str(path),
          "--tone-strength", "45", "--output-format", "png"])
    record = json.loads(output.with_suffix(".meta.json").read_text())
    assert record["configuration"]["parameters"]["tone_strength"] == 45
    assert record["parameters"]["tone_strength"] == 45
    expected = apply_filter(pixels, config=replace(
        config, parameters=replace(config.parameters, tone_strength=45)))
    np.testing.assert_array_equal(np.array(Image.open(output)), expected)


def test_video_cli_tone_strength_works_with_noise_preset(tmp_path, tools, capsys):
    from pathlib import Path
    from luma_fatal.cleanup import SourceCleanup
    from luma_fatal.prenoise import SourceNoise
    from test_video import make_clip, reference_rgb

    preset = Path(__file__).resolve().parents[1] / "configs/fattal-1-compressed-noise.yaml"
    config = Config.load(preset)
    source = make_clip(tmp_path / "input.mkv", tools, width=48, height=36, neutral=True)
    output = tmp_path / "output.mp4"
    main(["video", str(source), str(output), "--config", str(preset),
          "--tone-strength", "45", "--ffmpeg", tools[0], "--ffprobe", tools[1], "--json"])
    record = json.loads(capsys.readouterr().out)
    assert record["configuration"]["parameters"]["tone_strength"] == 45
    assert record["source_noise"]["rms"] == 0.35
    frames = reference_rgb(source, tools, 48, 36, deband="gentle")
    expected = hashlib.sha256()
    callback = SourceNoise(0.35, source_cleanup=SourceCleanup(0.75))
    for frame in process_frames(frames, config=replace(
        config, parameters=replace(config.parameters, tone_strength=45)),
        source_cleanup=callback,
    ):
        expected.update(frame.tobytes())
    assert record["processed_rgb_sha256"] == expected.hexdigest()
