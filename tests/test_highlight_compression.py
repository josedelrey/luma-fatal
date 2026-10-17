"""Highlight roll-off preserves color ratios before clipping and quantization."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from luma_fatal import Config, FilterParams, ProcessingConfig, apply_filter
from luma_fatal.cli import main
from luma_fatal.core import _compress_highlights, _recombine
from luma_fatal.pipeline import process_frames
from test_video import tools


@pytest.mark.parametrize("value", [-1, 100.1, True, "50", None,
                                    float("nan"), float("inf")])
def test_invalid_compression_is_rejected(value):
    with pytest.raises(ValueError, match="highlight_compression"):
        FilterParams(highlight_compression=value)


@pytest.mark.parametrize("suffix", ["yaml", "toml", "json"])
def test_compression_configuration_roundtrips(tmp_path, suffix):
    config = Config(parameters=FilterParams(highlight_compression=62.5))
    content = {"yaml": config.to_yaml(), "toml": config.to_toml(),
               "json": json.dumps(config.to_dict())}[suffix]
    path = tmp_path / f"settings.{suffix}"
    path.write_text(content)
    assert Config.load(path) == config


@pytest.mark.parametrize("strength", [0, 25, 50, 100])
def test_compression_preserves_pixels_below_knee(strength):
    rgb = np.random.default_rng(91).uniform(0, 191.25, (13, 17, 3))
    before = rgb.copy()
    np.testing.assert_array_equal(_compress_highlights(rgb, strength), rgb)
    np.testing.assert_array_equal(rgb, before)


def test_full_compression_is_monotonic_bounded_and_preserves_color_ratios():
    peak = np.concatenate((np.linspace(0, 1024, 1025), [1e5, 1e100, 1e308]))
    rgb = peak[None, :, None] * np.array([0.25, 0.5, 1.0])
    before = rgb.copy()
    result = _compress_highlights(rgb, 100)
    assert np.isfinite(result).all()
    assert result.min() >= 0 and result.max() <= 255
    assert np.all(np.diff(result[0, :, 2]) >= 0)
    np.testing.assert_allclose(result[..., 0], result[..., 2] * 0.25, rtol=1e-15)
    np.testing.assert_allclose(result[..., 1], result[..., 2] * 0.5, rtol=1e-15)
    np.testing.assert_array_equal(rgb, before)


@pytest.mark.parametrize("strength,expected", [
    (0, [64, 128, 255]), (50, [60, 120, 239]), (100, [56, 112, 223]),
])
def test_compression_has_known_prequantization_results(strength, expected):
    rgb = np.array([[[25, 50, 100]]], np.uint8)
    luminance = np.array([[100 / 255]], np.float32)
    result = _recombine(rgb, luminance, np.ones_like(luminance), 100,
                        power_mode="standard", highlight_compression=strength)
    np.testing.assert_array_equal(result, [[expected]])


def test_full_compression_preserves_color_in_overrange_highlights():
    rgb = np.array([[[25, 50, 100]]], np.uint8)
    luminance = np.array([[100 / 255]], np.float32)
    mapped = np.full_like(luminance, 2)
    clipped = _recombine(rgb, luminance, mapped, 100, power_mode="standard")
    compressed = _recombine(rgb, luminance, mapped, 100,
                            power_mode="standard", highlight_compression=100)
    np.testing.assert_array_equal(clipped, [[[128, 255, 255]]])
    np.testing.assert_array_equal(compressed, [[[61, 122, 244]]])


def test_strength_progressively_reduces_bright_values():
    rgb = np.array([[[128.0, 256.0, 512.0]]])
    peaks = [_compress_highlights(rgb, strength)[0, 0, 2]
             for strength in (0, 25, 50, 75, 100)]
    assert all(a > b for a, b in zip(peaks, peaks[1:]))


def test_compression_is_smooth_at_knee():
    knee = 191.25
    delta = 1e-3
    values = np.array([knee - delta, knee, knee + delta])
    rgb = np.repeat(values[None, :, None], 3, axis=2)
    result = _compress_highlights(rgb, 100)[0, :, 0]
    np.testing.assert_array_equal(result[:2], values[:2])
    assert (result[2] - result[1]) / delta == pytest.approx(1, abs=2e-5)


def test_nonfinite_pixels_retain_existing_quantization_policy():
    rgb = np.array([[[np.nan, 255, 128], [np.inf, 255, 128], [64, -np.inf, 128]]])
    np.testing.assert_array_equal(_compress_highlights(rgb, 100), rgb)


def test_compression_precedes_gamma():
    rgb = np.full((2, 3, 3), 64, np.uint8)
    luminance = np.full((2, 3), 64 / 255, np.float32)
    result = _recombine(rgb, luminance, np.ones_like(luminance), 100,
                        power_mode="standard", output_gamma=2,
                        highlight_compression=100)
    np.testing.assert_array_equal(result, np.full_like(rgb, 239))


def test_tone_strength_precedes_compression():
    rgb = np.full((2, 3, 3), 64, np.uint8)
    luminance = np.full((2, 3), 64 / 255, np.float32)
    # The halfway brightness is below the knee, so compression leaves it intact.
    result = _recombine(rgb, luminance, np.ones_like(luminance), 100,
                        power_mode="standard", tone_strength=50,
                        highlight_compression=100)
    np.testing.assert_array_equal(result, np.full_like(rgb, 160))


@pytest.mark.parametrize("softness", [0, 6])
def test_compression_retains_original_shadow_protection(softness):
    rgb = np.full((2, 3, 3), 0, np.uint8)
    luminance = np.full((2, 3), 1 / 255, np.float32)
    result = _recombine(rgb, luminance, np.ones_like(luminance), 100,
                        power_mode="standard", shadow_softness=softness,
                        highlight_compression=100)
    np.testing.assert_array_equal(result, rgb)


@pytest.mark.parametrize("variant", [1, 2])
@pytest.mark.parametrize("max_edge", [None, 40])
def test_compression_works_in_both_resolution_routes_and_retains_fade(variant, max_edge):
    pixels = np.random.default_rng(92).integers(20, 240, (67, 91, 3), np.uint8)
    before = pixels.copy()
    config = Config(
        filter=f"fattal{variant}",
        parameters=FilterParams(fade=0, highlight_compression=80, tone_strength=70),
        processing=ProcessingConfig(resize=max_edge is not None, max_edge=max_edge),
    )
    compressed = apply_filter(pixels, config=config)
    disabled = apply_filter(pixels, config=replace(
        config, parameters=replace(config.parameters, highlight_compression=0)))
    assert np.any(compressed != disabled)
    faded = apply_filter(pixels, config=replace(
        config, parameters=replace(config.parameters, fade=25)))
    expected = (3 * compressed.astype(np.uint16) + pixels.astype(np.uint16) + 1) // 4
    np.testing.assert_array_equal(faded, expected)
    bypass = apply_filter(pixels, config=replace(
        config, parameters=replace(config.parameters, fade=100)))
    np.testing.assert_array_equal(bypass, pixels)
    np.testing.assert_array_equal(pixels, before)


def test_image_cli_overrides_compression_and_records_metadata(tmp_path, capsys):
    pixels = np.random.default_rng(93).integers(20, 240, (35, 47, 3), np.uint8)
    source, output = tmp_path / "input.png", tmp_path / "output.png"
    Image.fromarray(pixels).save(source)
    config = Config(parameters=FilterParams(fade=0, highlight_compression=20))
    path = tmp_path / "settings.yaml"
    path.write_text(config.to_yaml())
    main(["apply", str(source), str(output), "--config", str(path),
          "--highlight-compression", "75", "--output-format", "png"])
    record = json.loads(output.with_suffix(".meta.json").read_text())
    assert record["configuration"]["parameters"]["highlight_compression"] == 75
    assert record["parameters"]["highlight_compression"] == 75
    expected = apply_filter(pixels, config=replace(
        config, parameters=replace(config.parameters, highlight_compression=75)))
    np.testing.assert_array_equal(np.array(Image.open(output)), expected)


def test_video_cli_compression_works_with_tone_strength_and_fbm(tmp_path, tools, capsys):
    from luma_fatal.prenoise import SourceNoise
    from test_video import make_clip, reference_rgb

    preset = Path(__file__).resolve().parents[1] / "configs/fattal-2-compressed-fbm.yaml"
    config = Config.load(preset)
    source = make_clip(tmp_path / "input.mkv", tools, width=48, height=36, neutral=True)
    output = tmp_path / "output.mp4"
    main(["video", str(source), str(output), "--config", str(preset),
          "--highlight-compression", "75", "--tone-strength", "60",
          "--ffmpeg", tools[0], "--ffprobe", tools[1], "--json"])
    record = json.loads(capsys.readouterr().out)
    assert record["configuration"]["parameters"]["highlight_compression"] == 75
    assert record["configuration"]["parameters"]["tone_strength"] == 60
    assert record["source_noise"]["kind"] == "fbm"
    frames = reference_rgb(source, tools, 48, 36, deband="gentle")
    expected = hashlib.sha256()
    config = replace(config, parameters=replace(
        config.parameters, highlight_compression=75, tone_strength=60))
    for frame in process_frames(frames, config=config,
                                source_cleanup=SourceNoise(0.75, kind="fbm")):
        expected.update(frame.tobytes())
    assert record["processed_rgb_sha256"] == expected.hexdigest()
