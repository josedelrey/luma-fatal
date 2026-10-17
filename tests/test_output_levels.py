"""Output levels remap floating RGB after shadow protection and before Fade."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from luma_fatal import Config, FilterParams, ProcessingConfig, apply_filter
from luma_fatal.cli import main
from luma_fatal.core import _recombine
from luma_fatal.pipeline import process_frames
from test_video import tools


@pytest.mark.parametrize("field", ["black_level", "white_level"])
@pytest.mark.parametrize("value", [-1, 100.1, True, "50", None,
                                    float("nan"), float("inf")])
def test_invalid_output_levels_are_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        FilterParams(**{field: value})


@pytest.mark.parametrize("black,white", [(0, 0), (100, 100), (80, 20), (50, 50)])
def test_black_level_must_be_below_white(black, white):
    with pytest.raises(ValueError, match="black_level must be below white_level"):
        FilterParams(black_level=black, white_level=white)


@pytest.mark.parametrize("suffix", ["yaml", "toml", "json"])
def test_output_level_configuration_roundtrips(tmp_path, suffix):
    config = Config(parameters=FilterParams(black_level=5.5, white_level=94.5))
    content = {"yaml": config.to_yaml(), "toml": config.to_toml(),
               "json": json.dumps(config.to_dict())}[suffix]
    path = tmp_path / f"settings.{suffix}"
    path.write_text(content)
    assert Config.load(path) == config


def test_output_levels_map_endpoints_midtones_and_nonfinite_values():
    rgb = np.full((1, 8, 3), 128, np.uint8)
    luminance = np.full((1, 8), 128 / 255, np.float32)
    mapped = np.array([[0, 0.25, 0.5, 1, 2, np.nan, -np.inf, np.inf]], np.float32)
    result = _recombine(rgb, luminance, mapped, 100, power_mode="standard",
                        black_level=10, white_level=90)
    expected = np.repeat(np.array([[26, 77, 128, 230, 230, 230, 26, 230]],
                                  np.uint8)[..., None], 3, axis=2)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("black,white,expected", [
    (0, 80, [0, 204]), (20, 100, [51, 255]), (20, 80, [51, 204]),
])
def test_black_and_white_levels_can_be_adjusted_independently(black, white, expected):
    rgb = np.full((1, 2, 3), 128, np.uint8)
    luminance = np.full((1, 2), 128 / 255, np.float32)
    result = _recombine(rgb, luminance, np.array([[0, 1]], np.float32), 100,
                        power_mode="standard", black_level=black, white_level=white)
    np.testing.assert_array_equal(result[0, :, 0], expected)


def test_output_levels_do_not_round_intermediate_rgb():
    rgb = np.full((2, 3, 3), 64, np.uint8)
    luminance = np.full((2, 3), 64 / 255, np.float32)
    # 1.4 maps to 26.62 and rounds to 27. Rounding it first would yield 26.
    result = _recombine(rgb, luminance, np.full((2, 3), 1.4 / 255, np.float32), 100,
                        power_mode="standard", black_level=10, white_level=90)
    np.testing.assert_array_equal(result, np.full_like(rgb, 27))


@pytest.mark.parametrize("options,mapped,expected", [
    ({"output_gamma": 2}, 0.25, 128),
    ({"highlight_compression": 100}, 1, 185),
    ({"tone_strength": 50}, 1, 147),
])
def test_levels_follow_gamma_compression_and_tone_blending(options, mapped, expected):
    rgb = np.full((2, 3, 3), 64, np.uint8)
    luminance = np.full((2, 3), 64 / 255, np.float32)
    result = _recombine(rgb, luminance, np.full((2, 3), mapped, np.float32), 100,
                        power_mode="standard", black_level=20, white_level=80, **options)
    np.testing.assert_array_equal(result, np.full_like(rgb, expected))


@pytest.mark.parametrize("softness,expected", [(0, 55), (6, 129)])
def test_levels_follow_hard_and_soft_shadow_protection(softness, expected):
    rgb = np.full((2, 3, 3), 6, np.uint8)
    luminance = np.full((2, 3), 6 / 255, np.float32)
    result = _recombine(rgb, luminance, np.ones_like(luminance), 100,
                        power_mode="standard", shadow_softness=softness,
                        black_level=20, white_level=80)
    np.testing.assert_array_equal(result, np.full_like(rgb, expected))


def test_shadow_protection_uses_original_source_with_output_levels():
    rgb = np.full((2, 3, 3), 2, np.uint8)
    reference = np.full((2, 3), 2 / 255, np.float32)
    treated = np.full((2, 3), 128 / 255, np.float32)
    result = _recombine(rgb, treated, np.ones_like(treated), 100,
                        power_mode="standard", color_rgb=np.full(rgb.shape, 128),
                        shadow_luminance=reference, black_level=20, white_level=80)
    np.testing.assert_array_equal(result, np.full_like(rgb, 52))


@pytest.mark.parametrize("variant", [1, 2])
def test_all_black_images_receive_output_levels(variant):
    pixels = np.zeros((35, 47, 3), np.uint8)
    config = Config(filter=f"fattal{variant}",
                    parameters=FilterParams(fade=0, black_level=20, white_level=80))
    np.testing.assert_array_equal(apply_filter(pixels, config=config), np.full_like(pixels, 51))


@pytest.mark.parametrize("shape", [(1, 1, 3), (1, 5, 3), (5, 1, 3)])
def test_one_pixel_wide_identity_extension_receives_levels(shape):
    pixels = np.zeros(shape, np.uint8)
    config = Config(parameters=FilterParams(fade=0, black_level=20, white_level=80))
    with pytest.warns(RuntimeWarning, match="identity extension"):
        result = apply_filter(pixels, config=config)
    np.testing.assert_array_equal(result, np.full_like(pixels, 51))


@pytest.mark.parametrize("variant", [1, 2])
@pytest.mark.parametrize("max_edge", [None, 40])
def test_levels_bound_full_effect_and_preserve_fade_and_alpha(variant, max_edge):
    pixels = np.random.default_rng(94).integers(0, 256, (67, 91, 4), np.uint8)
    before = pixels.copy()
    config = Config(
        filter=f"fattal{variant}",
        parameters=FilterParams(fade=0, black_level=20, white_level=80,
                                tone_strength=60, highlight_compression=75),
        processing=ProcessingConfig(resize=max_edge is not None, max_edge=max_edge,
                                    transparency="straight"),
    )
    result = apply_filter(pixels, config=config)
    assert result[..., :3].min() >= 51 and result[..., :3].max() <= 204
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


def test_image_cli_overrides_output_levels_and_records_metadata(tmp_path, capsys):
    pixels = np.random.default_rng(95).integers(0, 256, (35, 47, 3), np.uint8)
    source, output = tmp_path / "input.png", tmp_path / "output.png"
    Image.fromarray(pixels).save(source)
    config = Config(parameters=FilterParams(fade=0, black_level=10, white_level=90))
    path = tmp_path / "settings.yaml"
    path.write_text(config.to_yaml())
    main(["apply", str(source), str(output), "--config", str(path),
          "--black-level", "20", "--white-level", "80", "--output-format", "png"])
    record = json.loads(output.with_suffix(".meta.json").read_text())
    assert record["configuration"]["parameters"]["black_level"] == 20
    assert record["parameters"]["white_level"] == 80
    expected = apply_filter(pixels, config=replace(
        config, parameters=replace(config.parameters, black_level=20, white_level=80)))
    np.testing.assert_array_equal(np.array(Image.open(output)), expected)


def test_video_cli_levels_work_with_other_tone_controls_and_noise(tmp_path, tools, capsys):
    from luma_fatal.cleanup import SourceCleanup
    from luma_fatal.prenoise import SourceNoise
    from test_video import make_clip, reference_rgb

    preset = Path(__file__).resolve().parents[1] / "configs/fattal-1-compressed-noise.yaml"
    config = Config.load(preset)
    source = make_clip(tmp_path / "input.mkv", tools, width=48, height=36, neutral=True)
    output = tmp_path / "output.mp4"
    main(["video", str(source), str(output), "--config", str(preset),
          "--black-level", "20", "--white-level", "80", "--tone-strength", "60",
          "--highlight-compression", "75", "--ffmpeg", tools[0], "--ffprobe", tools[1], "--json"])
    record = json.loads(capsys.readouterr().out)
    parameters = record["configuration"]["parameters"]
    assert parameters["black_level"] == 20 and parameters["white_level"] == 80
    assert parameters["tone_strength"] == 60 and parameters["highlight_compression"] == 75
    assert record["source_noise"]["rms"] == 0.35
    frames = reference_rgb(source, tools, 48, 36, deband="gentle")
    expected = hashlib.sha256()
    config = replace(config, parameters=replace(
        config.parameters, black_level=20, white_level=80,
        tone_strength=60, highlight_compression=75))
    for frame in process_frames(frames, config=config,
                                source_cleanup=SourceNoise(0.35, source_cleanup=SourceCleanup(0.75))):
        assert frame.min() >= 51 and frame.max() <= 204
        expected.update(frame.tobytes())
    assert record["processed_rgb_sha256"] == expected.hexdigest()
