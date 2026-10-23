"""Noise suppression, structure preservation and video integration contracts."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from luma_fatal.configuration import Config, ProcessingConfig, VideoConfig
from luma_fatal.core import apply_core
from luma_fatal.denoise import SourceDenoise
from luma_fatal.numerics import luminance_fractional, luminance_u8
from luma_fatal.pipeline import process_frames
from luma_fatal.prenoise import SourceNoise
from test_video import tools


def noisy_source(sigma=3, correlated=False):
    y, x = np.indices((128, 192))
    clean = np.repeat((65 + 0.12 * x + 0.07 * y)[..., None], 3, axis=2)
    clean[:, 96:] += 70
    noise = np.random.default_rng(48).normal(size=clean.shape)
    if correlated:
        noise = gaussian_filter(noise, (0.8, 0.8, 0))
        noise /= np.std(noise, axis=(0, 1))
    rgb = np.clip(np.round(clean + sigma * noise), 0, 255).astype(np.uint8)
    return rgb, clean


@pytest.mark.parametrize("correlated", [False, True])
@pytest.mark.parametrize("sigma", [1, 3, 8])
def test_suppresses_fine_and_correlated_noise_without_flattening_edges(sigma, correlated):
    rgb, clean = noisy_source(sigma, correlated)
    original = rgb.copy()
    color, luminance = SourceDenoise(1)(rgb, luminance_fractional(rgb))
    interior = np.s_[8:-8, 8:80, :]
    input_error = np.mean((rgb[interior] - clean[interior]) ** 2)
    output_error = np.mean((color[interior] - clean[interior]) ** 2)
    assert output_error < input_error * 0.15
    # A noisy 70-code edge remains localized rather than being blurred away.
    assert np.min(np.mean(color[:, 96] - color[:, 95], axis=0)) > 50
    assert np.max(np.abs(color[interior].mean((0, 1)) - clean[interior].mean((0, 1)))) < 0.5
    np.testing.assert_array_equal(rgb, original)
    np.testing.assert_allclose(
        luminance, np.sum(color * np.array([4899, 9617, 1868]) / (16384 * 255), axis=2),
        rtol=0, atol=1e-7,
    )
    assert np.any(color != np.round(color))


@pytest.mark.parametrize("variant", [1, 2])
@pytest.mark.parametrize("correlated", [False, True])
def test_denoising_reduces_amplified_output_noise(variant, correlated):
    rgb, clean = noisy_source(correlated=correlated)
    config = Config()
    reference = apply_core(np.round(clean).astype(np.uint8), variant,
                           config.parameters, config.numerics).astype(np.float32)
    before = apply_core(rgb, variant, config.parameters, config.numerics).astype(np.float32)
    after = apply_core(rgb, variant, config.parameters, config.numerics,
                       source_cleanup=SourceDenoise(1)).astype(np.float32)
    interior = np.s_[8:-8, 8:80, :]
    assert np.mean((after[interior] - reference[interior]) ** 2) < (
        0.5 * np.mean((before[interior] - reference[interior]) ** 2)
    )


def test_strength_is_a_continuous_blend_and_zero_is_exact():
    rgb, _ = noisy_source()
    luminance = luminance_u8(rgb).astype(np.float32) / np.float32(255)
    full = SourceDenoise(1)(rgb, luminance)
    half = SourceDenoise(0.5)(rgb, luminance)
    disabled = SourceDenoise(0)(rgb, luminance)
    for original, a, b, off in zip((rgb, luminance), full, half, disabled):
        np.testing.assert_array_equal(off, original)
        np.testing.assert_allclose(b, original + 0.5 * (a - original), rtol=0, atol=1e-5)


@pytest.mark.parametrize("shape", [(1, 1), (1, 17), (17, 1), (2, 3), (11, 17)])
def test_tiny_frames_remain_finite_and_in_range(shape):
    rgb = np.random.default_rng(48).integers(0, 256, (*shape, 3), dtype=np.uint8)
    color, luminance = SourceDenoise(1)(rgb, luminance_fractional(rgb))
    assert color.shape == rgb.shape and luminance.shape == shape
    assert np.isfinite(color).all() and np.isfinite(luminance).all()
    assert color.min() >= 0 and color.max() <= 255
    assert luminance.min() >= 0 and luminance.max() <= 1


@pytest.mark.parametrize("pattern", ["black", "white", "color", "ramp", "stripes"])
def test_clean_flat_colors_shading_and_directional_texture_are_preserved(pattern):
    y, x = np.indices((97, 129))
    rgb = np.full((97, 129, 3), (71, 83, 112), np.uint8)
    if pattern in ("black", "white"):
        rgb[:] = 0 if pattern == "black" else 255
    elif pattern == "ramp":
        rgb = np.repeat((40 + x + y // 4)[..., None], 3, axis=2).astype(np.uint8)
    elif pattern == "stripes":
        rgb[:, ::2] = 200
        rgb[:, 1::2] = 30
    luminance = luminance_u8(rgb).astype(np.float32) / np.float32(255)
    color, corrected = SourceDenoise(1)(rgb, luminance)
    np.testing.assert_array_equal(color, rgb)
    np.testing.assert_array_equal(corrected, luminance)


def test_frames_have_no_color_history_or_scene_cut_trails():
    rgb, _ = noisy_source()
    callback = SourceDenoise(1)
    luminance = luminance_fractional(rgb)
    first = callback(rgb, luminance)
    other = np.full((47, 51, 3), 230, np.uint8)
    callback(other, luminance_fractional(other))
    repeated = callback(rgb, luminance)
    for a, b in zip(first, repeated):
        np.testing.assert_array_equal(a, b)
    assert callback.frames == 3


@pytest.mark.parametrize("variant", [1, 2])
def test_original_fade_shadows_and_working_resolution_are_preserved(variant):
    rgb, _ = noisy_source()
    rgb[:10] = 2
    config = Config(filter=f"fattal{variant}", processing=ProcessingConfig(max_edge=96))
    shapes = []
    callback = SourceDenoise(1)

    def capture(color, luminance):
        shapes.append(color.shape)
        return callback(color, luminance)

    result = list(process_frames([rgb], config=config, source_cleanup=capture))[0]
    assert shapes == [(64, 96, 3)]
    assert result.shape == rgb.shape
    full = apply_core(rgb, variant, config.parameters, config.numerics,
                      source_cleanup=callback)
    np.testing.assert_array_equal(full[:10], rgb[:10])
    faded = list(process_frames(
        [rgb], config=replace(config, parameters=replace(config.parameters, fade=100)),
        source_cleanup=capture,
    ))[0]
    np.testing.assert_array_equal(faded, rgb)
    assert len(shapes) == 1


def test_denoising_precedes_cleanup_and_intentional_noise():
    rgb, _ = noisy_source()
    luminance = luminance_fractional(rgb)
    denoised = SourceDenoise(1)(rgb, luminance)
    expected = SourceNoise(0.75, cleanup=0.75)(*denoised)
    actual = SourceDenoise(1, source_cleanup=SourceNoise(0.75, cleanup=0.75))(rgb, luminance)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)
    assert not np.array_equal(actual[0], denoised[0])


@pytest.mark.parametrize("value", [-1, 1.1, True, "auto", None, float("nan"), float("inf")])
def test_invalid_strengths_are_rejected(value):
    with pytest.raises(ValueError):
        SourceDenoise(value)
    with pytest.raises(ValueError):
        VideoConfig(denoise=value)


@pytest.mark.parametrize("suffix", ["yaml", "toml", "json"])
def test_denoise_configuration_roundtrips(tmp_path, suffix):
    config = Config(video=VideoConfig(denoise=0.7))
    text = {"yaml": config.to_yaml(), "toml": config.to_toml(),
            "json": json.dumps(config.to_dict())}[suffix]
    path = tmp_path / f"denoise.{suffix}"
    path.write_text(text)
    assert Config.load(path) == config


@pytest.mark.parametrize("path", sorted(
    (Path(__file__).resolve().parents[1] / "configs").glob("*.yaml")
), ids=lambda p: p.stem)
def test_only_compressed_presets_enable_denoising_by_default(path):
    assert Config.load(path).video.denoise == (1.0 if "compressed" in path.stem else 0.0)


@pytest.mark.parametrize("variant", [1, 2])
@pytest.mark.parametrize("strength", [0, 0.6, 1])
def test_video_cli_denoise_override_matches_pixels_and_records_metadata(
    tmp_path, tools, capsys, variant, strength,
):
    from luma_fatal.cli import main
    from test_video import run

    rgb, _ = noisy_source()
    # Lossless RGB storage keeps controlled noise without codec artifacts.
    frames = [rgb, np.roll(rgb, 3, axis=1), 255 - rgb]
    source = tmp_path / "input.mkv"
    run([
        tools[0], "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", "192x128", "-r", "25", "-i", "pipe:0", "-c:v", "ffv1",
        "-pix_fmt", "gbrp", "-color_trc", "iec61966-2-1", "-color_primaries", "bt709",
        "-colorspace", "rgb", "-color_range", "pc", str(source),
    ], b"".join(f.tobytes() for f in frames))
    data = run([
        tools[0], "-v", "error", "-i", str(source), "-an", "-vf",
        "format=gbrp16le,zscale=matrixin=gbr:transferin=iec61966-2-1:"
        "primariesin=709:rangein=full:matrix=gbr:transfer=iec61966-2-1:"
        "primaries=709:range=full,format=gbrp,format=rgb24",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ])
    decoded = np.frombuffer(data, np.uint8).reshape(3, 128, 192, 3)
    preset = Path(__file__).resolve().parents[1] / "configs" / f"fattal-{variant}-compressed.yaml"
    output = tmp_path / "out.mp4"
    main([
        "video", str(source), str(output), "--config", str(preset),
        "--denoise", str(strength), "--flat-cleanup", "0", "--deband", "off",
        "--encoder", "libx264", "--ffmpeg", tools[0], "--ffprobe", tools[1], "--json",
    ])
    record = json.loads(capsys.readouterr().out)
    config = Config.load(preset)
    callback = SourceDenoise(strength) if strength else None
    digest = hashlib.sha256()
    for frame in process_frames(decoded, config=config, source_cleanup=callback):
        digest.update(frame.tobytes())
    assert record["processed_rgb_sha256"] == digest.hexdigest()
    assert record["video_configuration"]["denoise"] == strength
    assert record["configuration"]["video"]["denoise"] == strength
    assert ("source_denoise" in record) == bool(strength)
    assert "denoise.py" in record["implementation_sources_sha256"]
    if strength:
        assert record["source_denoise"]["frames"] == 3
        assert record["source_denoise"]["strength"] == strength
    assert json.loads(output.with_suffix(".meta.json").read_text()) == record
