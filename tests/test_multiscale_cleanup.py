"""Safety and strength contracts for the experimental correction."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from luma_fatal.cleanup import MultiscaleSourceCleanup as ScaleSeparatedCleanup
from luma_fatal.configuration import Config, VideoConfig
from luma_fatal.cleanup import SourceCleanup
from luma_fatal.denoise import SourceDenoise
from luma_fatal.pipeline import process_frames
from luma_fatal.numerics import luminance_u8
from test_video import tools


def sample(shape=(101, 397)):
    y, x = np.indices(shape)
    ramp = 70 + x * 0.025 + y * 0.015
    noise = np.random.default_rng(42).normal(0, 0.7, shape)
    gray = np.clip(np.round(ramp + noise), 0, 255).astype(np.uint8)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    # A strong stripe provides protected pixels inside the same frame.
    rgb[:, shape[1] // 2:shape[1] // 2 + 6] = (200, 30, 20)
    luminance = luminance_u8(rgb).astype(np.float32) * np.float32(1 / 255)
    return rgb, luminance


@pytest.mark.parametrize("strength", [0, 0.25, 0.75, 1])
def test_equal_strengths_recover_existing_cleanup(strength):
    rgb, luminance = sample()
    expected = SourceCleanup(strength)(rgb, luminance)
    actual = ScaleSeparatedCleanup(strength, strength)(rgb, luminance)
    np.testing.assert_allclose(actual[0], expected[0], rtol=0, atol=2e-5)
    np.testing.assert_allclose(actual[1], expected[1], rtol=0, atol=1e-7)


@pytest.mark.parametrize("shape", [(1, 1), (11, 17), (101, 397), (129, 769)])
def test_bounds_and_protected_pixels(shape):
    rgb, luminance = sample(shape)
    cleanup = ScaleSeparatedCleanup(0.5, 1, 0.7)
    color, corrected = cleanup(rgb, luminance)
    assert np.isfinite(color).all() and np.isfinite(corrected).all()
    assert color.shape == rgb.shape and corrected.shape == luminance.shape
    assert color.min() >= 0 and color.max() <= 255
    assert corrected.min() >= 0 and corrected.max() <= 1
    assert np.max(np.abs(color - rgb)) <= 4.00002
    assert np.max(np.abs(corrected - luminance) * 255) <= 4.50002
    protected = cleanup.mask == 0
    if shape[1] >= 397:
        assert protected.any()
    np.testing.assert_array_equal(color[protected], rgb[protected])
    np.testing.assert_array_equal(corrected[protected], luminance[protected])


def test_zero_strength_is_identity():
    rgb, luminance = sample()
    color, corrected = ScaleSeparatedCleanup(0, 0)(rgb, luminance)
    np.testing.assert_array_equal(color, rgb)
    np.testing.assert_array_equal(corrected, luminance)


def test_constant_image_does_not_acquire_a_pattern():
    rgb = np.full((101, 397, 3), (74, 91, 103), np.uint8)
    luminance = luminance_u8(rgb).astype(np.float32) * np.float32(1 / 255)
    color, corrected = ScaleSeparatedCleanup()(rgb, luminance)
    np.testing.assert_allclose(color, rgb, rtol=0, atol=2e-5)
    assert np.ptp(corrected) < 1e-7


def test_repeated_frame_is_stable():
    rgb, luminance = sample()
    cleanup = ScaleSeparatedCleanup()
    first = cleanup(rgb, luminance)
    for _ in range(4):
        result = cleanup(rgb, luminance)
        np.testing.assert_allclose(result[0], first[0], rtol=0, atol=2e-5)
        np.testing.assert_allclose(result[1], first[1], rtol=0, atol=1e-7)


@pytest.mark.parametrize("arguments", [(-0.1, 1, 1), (0.5, 1.1, 1),
                                        (0.5, 1, 0), (0.5, 1, float("nan"))])
def test_invalid_settings_fail(arguments):
    with pytest.raises(ValueError):
        ScaleSeparatedCleanup(*arguments)


@pytest.mark.parametrize("variant", [1, 2])
def test_multiscale_presets_inherit_compressed_settings(variant):
    root = Path(__file__).resolve().parents[1] / "configs"
    parent = root / f"fattal-{variant}-compressed.yaml"
    child = root / f"fattal-{variant}-compressed-multiscale.yaml"
    standard = Config.load(parent)
    multiscale = Config.load(child)
    assert multiscale == replace(
        standard,
        video=replace(standard.video, flat_cleanup_mode="multiscale"),
    )
    assert Config.source_paths(child) == (child.resolve(), parent.resolve())
    assert multiscale.video.flat_cleanup == 0.75
    assert multiscale.video.flat_cleanup_coarse_strength == 1
    assert multiscale.video.flat_cleanup_sigma == 1.2


def test_inherited_config_overrides_nested_fields_and_checks_cycles(tmp_path):
    parent = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    parent.write_text(
        "schema_version: 1\nfilter: fattal2\nvideo:\n"
        "  flat_cleanup: 0.6\n  deband: gentle\n"
    )
    child.write_text("extends: base.yaml\nvideo:\n  flat_cleanup_mode: multiscale\n")
    loaded = Config.load(child)
    assert loaded.filter == "fattal2"
    assert loaded.video == VideoConfig(
        deband="gentle", flat_cleanup=0.6, flat_cleanup_mode="multiscale"
    )
    assert Config.load(tmp_path / "child.yaml") == loaded
    child.write_text("extends: child.yaml\n")
    with pytest.raises(ValueError, match="cycle"):
        Config.load(child)
    child.write_text("extends: null\n")
    with pytest.raises(ValueError, match="extends"):
        Config.load(child)


@pytest.mark.parametrize("values", [
    {"flat_cleanup_mode": "unknown"},
    {"flat_cleanup_coarse_strength": True},
    {"flat_cleanup_coarse_strength": 1.1},
    {"flat_cleanup_sigma": 0},
    {"flat_cleanup_sigma": 17},
    {"flat_cleanup_sigma": float("nan")},
])
def test_multiscale_config_rejects_invalid_values(values):
    with pytest.raises(ValueError):
        VideoConfig(**values)


@pytest.mark.parametrize("variant", [1, 2])
def test_cli_multiscale_preset_matches_working_frame_pipeline_and_metadata(
    tmp_path, tools, capsys, variant,
):
    from luma_fatal.cli import main
    from test_video import make_clip, reference_rgb

    root = Path(__file__).resolve().parents[1]
    preset = root / "configs" / f"fattal-{variant}-compressed-multiscale.yaml"
    source = make_clip(tmp_path / "input.mkv", tools, width=48, height=36)
    expected = hashlib.sha256()
    config = Config.load(preset)
    frames = reference_rgb(source, tools, 48, 36, deband="gentle")
    callback = ScaleSeparatedCleanup(0.75, 1.0, 1.2)
    callback = SourceDenoise(config.video.denoise, source_cleanup=callback)
    for frame in process_frames(frames, config=config, source_cleanup=callback):
        expected.update(frame.tobytes())

    destination = tmp_path / "multiscale.mp4"
    main([
        "video", str(source), str(destination), "--config", str(preset),
        "--ffmpeg", tools[0], "--ffprobe", tools[1], "--json",
    ])
    record = json.loads(capsys.readouterr().out)
    assert record["processed_rgb_sha256"] == expected.hexdigest()
    assert record["source_cleanup"]["mode"] == "multiscale"
    assert record["source_cleanup"]["frames"] == len(frames)
    assert record["video_configuration"]["flat_cleanup"] == 0.75
    assert record["video_configuration"]["flat_cleanup_mode"] == "multiscale"
    assert record["config_source"]["inherits"] == [{
        "path": str((root / "configs" / f"fattal-{variant}-compressed.yaml").resolve()),
        "sha256": hashlib.sha256((root / "configs" / f"fattal-{variant}-compressed.yaml").read_bytes()).hexdigest(),
    }]


def test_cli_can_override_multiscale_settings_and_disable_cleanup(
    tmp_path, tools, capsys,
):
    from luma_fatal.cli import main
    from test_video import make_clip, reference_rgb

    preset = Path(__file__).resolve().parents[1] / "configs/fattal-1-compressed-multiscale.yaml"
    source = make_clip(tmp_path / "input.mkv", tools, width=48, height=36)
    destination = tmp_path / "override.mp4"
    main([
        "video", str(source), str(destination), "--config", str(preset),
        "--flat-cleanup", "0.5", "--flat-cleanup-coarse-strength", "0.9",
        "--flat-cleanup-sigma", "2", "--ffmpeg", tools[0],
        "--ffprobe", tools[1], "--json",
    ])
    record = json.loads(capsys.readouterr().out)
    assert record["video_configuration"]["flat_cleanup"] == 0.5
    assert record["video_configuration"]["flat_cleanup_coarse_strength"] == 0.9
    assert record["video_configuration"]["flat_cleanup_sigma"] == 2
    assert record["source_cleanup"]["mode"] == "multiscale"

    standard = tmp_path / "standard.mp4"
    main([
        "video", str(source), str(standard), "--config", str(preset),
        "--flat-cleanup-mode", "standard", "--ffmpeg", tools[0],
        "--ffprobe", tools[1], "--json",
    ])
    record = json.loads(capsys.readouterr().out)
    expected = hashlib.sha256()
    frames = reference_rgb(source, tools, 48, 36, deband="gentle")
    for frame in process_frames(
        frames, config=Config.load(preset),
        source_cleanup=SourceCleanup(0.75),
    ):
        expected.update(frame.tobytes())
    assert record["processed_rgb_sha256"] == expected.hexdigest()
    assert record["source_cleanup"]["mode"] == "standard"

    disabled = tmp_path / "disabled.mp4"
    main([
        "video", str(source), str(disabled), "--config", str(preset),
        "--flat-cleanup", "0", "--ffmpeg", tools[0],
        "--ffprobe", tools[1], "--json",
    ])
    record = json.loads(capsys.readouterr().out)
    assert "source_cleanup" not in record
