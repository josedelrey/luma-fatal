"""Behavioral checks for optional noise before video filtering."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from luma_fatal.prenoise import SourceNoise, noise_field
from luma_fatal.cleanup import MultiscaleSourceCleanup, SourceCleanup
from luma_fatal.configuration import Config, VideoConfig
from luma_fatal.core import apply_core
from luma_fatal.numerics import luminance_fractional
from luma_fatal.pipeline import process_frames
from test_video import tools


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


@pytest.mark.parametrize("values", [
    {"pre_noise_rms": -0.1}, {"pre_noise_rms": 4.1},
    {"pre_noise_rms": True}, {"pre_noise_rms": float("nan")},
    {"pre_noise_rms": float("inf")}, {"pre_noise_kind": "white"},
    {"pre_noise_kind": True}, {"pre_noise_masked": "false"},
    {"pre_noise_masked": 0}, {"pre_noise_seed": -1},
    {"pre_noise_seed": True}, {"pre_noise_seed": 1.5},
])
def test_video_rejects_invalid_noise_settings(values):
    with pytest.raises(ValueError):
        VideoConfig(**values)


def test_noise_configuration_roundtrips(tmp_path):
    config = Config(video=VideoConfig(
        pre_noise_rms=0.35, pre_noise_kind="fbm",
        pre_noise_masked=False, pre_noise_seed=42,
    ))
    for suffix, content in (("yaml", config.to_yaml()), ("toml", config.to_toml()),
                            ("json", json.dumps(config.to_dict()))):
        path = tmp_path / f"settings.{suffix}"
        path.write_text(content)
        assert Config.load(path) == config


@pytest.mark.parametrize("factory", [SourceCleanup, MultiscaleSourceCleanup])
def test_noise_wraps_selected_cleanup_without_changing_zero_noise(factory):
    rgb, luminance = source((101, 397))
    expected = factory(0.75)(rgb, luminance)
    callback = SourceNoise(0, source_cleanup=factory(0.75))
    actual = callback(rgb, luminance)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)
    noisy = SourceNoise(0.35, source_cleanup=factory(0.75))
    color, changed = noisy(rgb, luminance)
    assert np.isfinite(color).all() and np.isfinite(changed).all()
    assert noisy.mask.shape == luminance.shape
    assert np.any(color != expected[0])
    assert noisy.frames == 1
    assert noisy.seconds >= noisy.cleanup.seconds


@pytest.mark.parametrize("variant", [1, 2])
@pytest.mark.parametrize("approach", ["noise", "fbm"])
def test_noise_presets_match_requested_treatments(variant, approach):
    root = Path(__file__).resolve().parents[1] / "configs"
    parent = root / f"fattal-{variant}-compressed.yaml"
    child = root / f"fattal-{variant}-compressed-{approach}.yaml"
    base = Config.load(parent)
    config = Config.load(child)
    is_fine = approach == "noise"
    assert config == replace(base, video=replace(
        base.video, flat_cleanup=0.75 if is_fine else 0,
        flat_cleanup_mode="standard", pre_noise_rms=0.35 if is_fine else 0.75,
        pre_noise_kind="highpass" if is_fine else "fbm",
        pre_noise_masked=True, pre_noise_seed=20261017,
    ))
    assert Config.source_paths(child) == (child.resolve(), parent.resolve())


@pytest.mark.parametrize("variant", [1, 2])
@pytest.mark.parametrize("approach", ["noise", "fbm"])
def test_video_cli_noise_presets_match_frame_pipeline_and_metadata(
    tmp_path, tools, capsys, variant, approach,
):
    from luma_fatal.cli import main
    from test_video import make_clip, reference_rgb

    root = Path(__file__).resolve().parents[1]
    preset = root / "configs" / f"fattal-{variant}-compressed-{approach}.yaml"
    config = Config.load(preset)
    source = make_clip(tmp_path / "input.mkv", tools, width=48, height=36, neutral=True)
    frames = reference_rgb(source, tools, 48, 36, deband="gentle")
    callback = SourceNoise(
        rms=0.35 if approach == "noise" else 0.75,
        kind="highpass" if approach == "noise" else "fbm",
        cleanup=0.75 if approach == "noise" else 0,
    )
    expected = hashlib.sha256()
    for frame in process_frames(frames, config=config, source_cleanup=callback):
        expected.update(frame.tobytes())
    destination = tmp_path / "noise.mp4"
    main([
        "video", str(source), str(destination), "--config", str(preset),
        "--ffmpeg", tools[0], "--ffprobe", tools[1], "--json",
    ])
    record = json.loads(capsys.readouterr().out)
    assert record["processed_rgb_sha256"] == expected.hexdigest()
    assert record["source_noise"]["frames"] == len(frames)
    assert record["source_noise"]["kind"] == config.video.pre_noise_kind
    assert record["source_noise"]["rms"] == config.video.pre_noise_rms
    assert record["source_noise"]["masked"] is True
    assert record["source_noise"]["seed"] == 20261017
    assert record["video_configuration"] == config.to_dict()["video"]
    assert ("source_cleanup" in record) == (approach == "noise")
    assert "prenoise.py" in record["implementation_sources_sha256"]
    inherited = root / "configs" / f"fattal-{variant}-compressed.yaml"
    assert record["config_source"]["inherits"] == [{
        "path": str(inherited.resolve()),
        "sha256": hashlib.sha256(inherited.read_bytes()).hexdigest(),
    }]


def test_cli_noise_overrides_and_disabled_path(tmp_path, tools, capsys):
    from luma_fatal.cli import main
    from test_video import make_clip, reference_rgb

    preset = Path(__file__).resolve().parents[1] / "configs/fattal-1-compressed-fbm.yaml"
    source = make_clip(tmp_path / "input.mkv", tools, width=48, height=36, neutral=True)
    frames = reference_rgb(source, tools, 48, 36, deband="gentle")
    config = Config.load(preset)
    for rms in (0.2, 0):
        destination = tmp_path / f"noise-{rms}.mp4"
        main([
            "video", str(source), str(destination), "--config", str(preset),
            "--pre-noise-rms", str(rms), "--pre-noise-kind", "highpass",
            "--no-pre-noise-masked", "--pre-noise-seed", "42",
            "--ffmpeg", tools[0], "--ffprobe", tools[1], "--json",
        ])
        record = json.loads(capsys.readouterr().out)
        callback = SourceNoise(rms, masked=False, seed=42) if rms else None
        expected = hashlib.sha256()
        for frame in process_frames(frames, config=config, source_cleanup=callback):
            expected.update(frame.tobytes())
        assert record["processed_rgb_sha256"] == expected.hexdigest()
        assert record["video_configuration"]["pre_noise_rms"] == rms
        assert record["video_configuration"]["pre_noise_kind"] == "highpass"
        assert record["video_configuration"]["pre_noise_masked"] is False
        assert record["video_configuration"]["pre_noise_seed"] == 42
        assert ("source_noise" in record) == bool(rms)
        assert "source_cleanup" not in record
