"""Configuration safety and experiment reproducibility through the public API."""

from dataclasses import replace
import hashlib
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


@pytest.mark.parametrize("mode", ["file", "resolved", "cli"])
def test_image_records_all_inherited_config_fingerprints(tmp_path, capsys, mode):
    source = tmp_path / "source.png"
    Image.new("RGB", (17, 19), (80, 90, 100)).save(source)
    base = tmp_path / "base.yaml"
    base.write_text("parameters:\n  fade: 100\noutput:\n  format: png\n")
    presets = tmp_path / "presets"
    presets.mkdir()
    parent = presets / "parent.yaml"
    child = tmp_path / "child.yaml"
    child.write_text("extends: presets/parent.yaml\nfilter: fattal2\n")
    records = []
    for index, beta in enumerate((60, 70)):
        parent.write_text(f"extends: ../base.yaml\nparameters:\n  beta: {beta}\n")
        destination = tmp_path / f"result-{index}.png"
        if mode == "cli":
            main(["apply", str(source), str(destination), "--config", str(child),
                  "--saturation", "75", "--json"])
            record = json.loads(capsys.readouterr().out)[0]
        else:
            options = {}
            if mode == "resolved":
                config = Config.load(child)
                options["config"] = replace(
                    config, parameters=replace(config.parameters, saturation=75)
                )
            record = process_image(source, destination, config_path=child, **options)
        expected = {
            "path": str(child),
            "sha256": hashlib.sha256(child.read_bytes()).hexdigest(),
            "inherits": [
                {"path": str(path.resolve()),
                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for path in (parent, base)
            ],
        }
        assert record["config_source"] == expected
        saved = json.loads(destination.with_suffix(".meta.json").read_text())
        assert saved["config_source"] == expected
        assert record["configuration"]["parameters"]["beta"] == beta
        assert record["configuration"]["parameters"]["saturation"] == (
            100 if mode == "file" else 75
        )
        records.append(record)
    before, after = [record["config_source"] for record in records]
    assert before["sha256"] == after["sha256"]
    assert before["inherits"][0]["sha256"] != after["inherits"][0]["sha256"]
    assert before["inherits"][1] == after["inherits"][1]


@pytest.mark.parametrize("section,values", [
    ("parameters", {"fade": 25}),
    ("processing", {"max_edge": 512}),
    ("image", {"color_space": "encoded"}),
    ("output", {"format": "png"}),
    ("numerics", {"iterations": 12}),
    ("video", {"crf": 22}),
])
def test_child_config_can_add_sections_missing_from_parent(tmp_path, section, values):
    parent = tmp_path / "parent.yaml"
    child = tmp_path / "child.yaml"
    grandchild = tmp_path / "grandchild.yaml"
    parent.write_text('filter: fattal2\n')
    child.write_text(json.dumps({"extends": "parent.yaml", section: values}))
    grandchild.write_text('extends: child.yaml\nparameters:\n  saturation: 75\n')
    expected = Config.from_dict({"filter": "fattal2", section: values})
    assert Config.load(child) == expected
    assert Config.load(grandchild) == replace(
        expected, parameters=replace(expected.parameters, saturation=75)
    )
    assert Config.source_paths(grandchild) == (
        grandchild.resolve(), child.resolve(), parent.resolve()
    )


@pytest.mark.parametrize("invalid", [None, [], "invalid", 1])
@pytest.mark.parametrize("invalid_parent", [False, True])
def test_inherited_sections_still_require_objects(tmp_path, invalid, invalid_parent):
    parent = tmp_path / "parent.yaml"
    child = tmp_path / "child.yaml"
    parent.write_text(json.dumps({"video": invalid} if invalid_parent else {}))
    child.write_text(json.dumps({
        "extends": "parent.yaml", "video": {} if invalid_parent else invalid
    }))
    with pytest.raises(ValueError, match="Inherited video section must be an object"):
        Config.load(child)


NUMERIC_OVERRIDES = {
    "iterations": 7,
    "relaxation": 1.4,
    "pyramid_min_size": 8,
    "normalization_low": 0.02,
    "normalization_high": 0.98,
    "normalization_floor": 0.002,
    "gradient_threshold": 0.003,
    "log_offset": 0.04,
    "shadow_threshold": 12,
}


@pytest.mark.parametrize("command", ["apply", "evaluate", "video"])
@pytest.mark.parametrize("overrides", [{}, NUMERIC_OVERRIDES])
def test_all_processing_commands_resolve_numeric_cli_overrides(
    tmp_path, monkeypatch, capsys, command, overrides
):
    config = Config(
        parameters=FilterParams(fade=25),
        numerics=NumericsConfig(iterations=11, detail_level=1, shadow_softness=3),
    )
    path = tmp_path / "settings.yaml"
    path.write_text(config.to_yaml())
    source = tmp_path / "input.png"
    Image.new("RGB", (17, 19)).save(source)
    captured = []

    def process(source, destination, *, config, **kwargs):
        captured.append(config)
        return {}

    monkeypatch.setattr("luma_fatal.cli.process_image", process)
    monkeypatch.setattr("luma_fatal.cli.process_video", process)
    if command == "evaluate":
        arguments = [command, str(tmp_path), str(tmp_path / "references"),
                     str(tmp_path / "evaluation")]
    else:
        arguments = [command, str(source), str(tmp_path / "result"), "--json"]
    arguments += ["--config", str(path)]
    for name, value in overrides.items():
        arguments += ["--" + name.replace("_", "-"), str(value)]
    main(arguments)
    assert captured == [replace(config, numerics=replace(config.numerics, **overrides))]


@pytest.mark.parametrize("name,value", [
    ("iterations", "0"),
    ("relaxation", "2"),
    ("pyramid_min_size", "1"),
    ("normalization_low", "0.999"),
    ("normalization_high", "0"),
    ("normalization_floor", "0"),
    ("gradient_threshold", "1e-100"),
    ("log_offset", "nan"),
    ("shadow_threshold", "256"),
])
def test_invalid_numeric_cli_overrides_fail_before_processing(
    tmp_path, capsys, name, value
):
    destination = tmp_path / "output.jpg"
    with pytest.raises(SystemExit) as error:
        main(["apply", str(tmp_path / "missing.png"), str(destination),
              "--" + name.replace("_", "-"), value])
    assert error.value.code == 2
    assert name in capsys.readouterr().err
    assert not destination.exists()


def test_numeric_cli_overrides_match_image_api_pixels(tmp_path, capsys):
    source = tmp_path / "source.png"
    destination = tmp_path / "result.png"
    pixels = np.random.default_rng(21).integers(0, 256, (37, 49, 3), np.uint8)
    Image.fromarray(pixels).save(source)
    path = tmp_path / "settings.yaml"
    config = Config(parameters=FilterParams(fade=0))
    path.write_text(config.to_yaml())
    arguments = ["apply", str(source), str(destination), "--config", str(path),
                 "--output-format", "png"]
    for name, value in NUMERIC_OVERRIDES.items():
        arguments += ["--" + name.replace("_", "-"), str(value)]
    main(arguments)
    expected_config = replace(config, numerics=NumericsConfig(**NUMERIC_OVERRIDES))
    with Image.open(destination) as image:
        np.testing.assert_array_equal(
            np.asarray(image), apply_filter(pixels, config=expected_config)
        )
    record = json.loads(destination.with_suffix(".meta.json").read_text())
    assert record["configuration"]["numerics"] == expected_config.to_dict()["numerics"]
