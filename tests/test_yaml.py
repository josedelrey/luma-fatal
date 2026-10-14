"""Configuration safety and experiment reproducibility through the public API."""

from dataclasses import replace
import json

import numpy as np
from PIL import Image
import pytest

from luma_fatal import (
    Config,
    FilterParams,
    NumericsConfig,
    ProcessingConfig,
    apply_filter,
)
from luma_fatal.cli import main
from luma_fatal.images import process_image


@pytest.mark.parametrize("suffix", ["yaml", "yml"])
def test_yaml_round_trip_preserves_nulls_and_numeric_controls(tmp_path, suffix):
    config = Config(numerics=NumericsConfig(iterations=12, relaxation=1.4))
    path = tmp_path / f"config.{suffix}"
    path.write_text(config.to_yaml())
    assert Config.load(path) == config
    assert "jpeg_background: null" in path.read_text()
    assert "max_edge: null" in path.read_text()


@pytest.mark.parametrize(
    "text",
    [
        "",
        "[]",
        "true",
        "filter: fattal1\nfilter: fattal2",
        "parameters:\n  fade: 0\n  fade: 100",
        "parameters: [0, 1]",
        "parameters:\n  fdae: 20",
        "processing:\n  resize: on",
        'processing:\n  resize: "true"',
        "schema_version: 2",
        "schema_version: true",
        "numerics:\n  iterations: 0",
        "numerics:\n  iterations: true",
        "numerics:\n  relaxation: 2",
        "numerics:\n  relaxation: .nan",
        "numerics:\n  normalization_low: 0.8\n  normalization_high: 0.2",
        "numerics:\n  pyramid_min_size: 1",
        "numerics:\n  gradient_threshold: 1e-100",
        "numerics:\n  log_offset: 0",
        "numerics:\n  shadow_threshold: 256",
        '!!python/object/apply:os.system ["echo unsafe"]',
        "1: value",
        "parameters:\n  1: value",
        "parameters: [",
    ],
)
def test_invalid_yaml_is_rejected_before_processing(tmp_path, text):
    path = tmp_path / "invalid.yaml"
    path.write_text(text)
    destination = tmp_path / "output.jpg"
    with pytest.raises(ValueError):
        process_image(tmp_path / "missing.png", destination, config_path=path)
    assert not destination.exists()


def test_yaml_scientific_notation_is_numeric(tmp_path):
    path = tmp_path / "settings.yaml"
    path.write_text("numerics:\n  gradient_threshold: 1e-4\n  log_offset: 1e-2\n")
    assert Config.load(path).numerics == NumericsConfig()


@pytest.mark.parametrize(
    "numerics",
    [
        NumericsConfig(iterations=3),
        NumericsConfig(relaxation=1.2),
        NumericsConfig(pyramid_min_size=8),
        NumericsConfig(normalization_low=0.05, normalization_high=0.9),
        NumericsConfig(normalization_floor=0.1),
        NumericsConfig(gradient_threshold=0.5),
        NumericsConfig(log_offset=0.5),
        NumericsConfig(shadow_threshold=255),
    ],
)
@pytest.mark.parametrize("max_edge", [None, 32])
def test_numeric_controls_change_actual_pixels_in_both_resolution_routes(
    numerics, max_edge
):
    pixels = np.random.default_rng(19).integers(0, 256, (67, 91, 3), np.uint8)
    original = pixels.copy()
    config = Config(
        parameters=FilterParams(fade=0),
        processing=ProcessingConfig(resize=max_edge is not None, max_edge=max_edge),
    )
    reference = apply_filter(pixels, config=config)
    candidate = apply_filter(pixels, config=replace(config, numerics=numerics))
    assert candidate.shape == pixels.shape
    assert candidate.dtype == np.uint8
    assert np.any(candidate != reference)
    np.testing.assert_array_equal(pixels, original)


def test_cli_overrides_yaml_without_discarding_numeric_settings(tmp_path, capsys):
    pixels = np.random.default_rng(20).integers(0, 256, (37, 49, 3), np.uint8)
    source, destination = tmp_path / "source.png", tmp_path / "result.png"
    Image.fromarray(pixels).save(source)
    path = tmp_path / "experiment.yaml"
    config = Config(numerics=NumericsConfig(iterations=7, relaxation=1.5))
    path.write_text(config.to_yaml())
    main(
        [
            "apply",
            str(source),
            str(destination),
            "--config",
            str(path),
            "--fade",
            "0",
            "--filter",
            "fattal2",
            "--output-format",
            "png",
        ]
    )
    expected = apply_filter(
        pixels,
        config=replace(config, filter="fattal2", parameters=FilterParams(fade=0)),
    )
    with Image.open(destination) as result:
        np.testing.assert_array_equal(np.asarray(result), expected)
    record = json.loads(destination.with_suffix(".meta.json").read_text())
    assert record["configuration"]["numerics"] == config.to_dict()["numerics"]


def test_cli_exports_reusable_yaml_and_refuses_to_overwrite_it(tmp_path, capsys):
    path = tmp_path / "settings.yaml"
    main(["config", "--output", str(path)])
    assert Config.load(path) == Config()
    before = path.read_bytes()
    with pytest.raises(SystemExit):
        main(["config", "--output", str(path)])
    assert path.read_bytes() == before
