"""Configuration precedence, image interpretation and export behavior."""

from dataclasses import replace
import hashlib
from io import BytesIO
import json

import numpy as np
from PIL import Image, ImageCms, JpegImagePlugin
import pytest

from luma_fatal import (
    Config,
    FilterParams,
    ImageConfig,
    NumericsConfig,
    OutputConfig,
    ProcessingConfig,
    VideoConfig,
    apply_filter,
    fattal2,
)
from luma_fatal.cli import main
from luma_fatal.processing import working_size


@pytest.mark.parametrize(
    "device,edge", [("low", 1024), ("medium", 1536), ("high", 2048)]
)
def test_device_and_custom_limits(device, edge):
    config = ProcessingConfig(device_class=device)
    assert working_size((3264, 2448), config.effective_max_edge) == (
        edge,
        edge * 3 // 4,
    )
    assert ProcessingConfig(device_class=device, max_edge=512).effective_max_edge == 512
    assert working_size(
        (3264, 2448), ProcessingConfig(resize=False).effective_max_edge
    ) == (3264, 2448)


@pytest.mark.parametrize(
    "values",
    [
        {"unexpected": {}},
        {"parameters": {"fdae": 0}},
        {"processing": {"max_edge": True}},
        {"processing": {"max_edge": 1}},
        {"processing": {"device_class": "unknown"}},
        {"processing": {"resize": False, "max_edge": 1024}},
        {"processing": {"blend_mode": "multiply"}},
        {"processing": {"transparency": "premultiplied"}},
        {"parameters": {"fade": "50"}},
        {"parameters": {"noise": True}},
        {"output": {"jpeg_quality": 101}},
        {"output": {"jpeg_background": [0, 256, 0]}},
        {"image": {"color_space": ["srgb"]}},
        {"filter": "fattal3"},
        {"video": {"deblock": "strong"}},
        {"video": {"deband": "strong"}},
        {"video": {"flat_cleanup": True}},
        {"video": {"flat_cleanup": 1.1}},
        {"video": {"crf": True}},
    ],
)
def test_invalid_config_is_rejected(values):
    with pytest.raises(ValueError):
        Config.from_dict(values)


@pytest.mark.parametrize("relaxation", [1e-50, 1e-8, 1.999999999])
def test_relaxation_rejects_float32_solver_boundary_weights(relaxation):
    with pytest.raises(ValueError, match="relaxation must keep float32 solver weights"):
        NumericsConfig(relaxation=relaxation)


@pytest.mark.parametrize("relaxation", [3e-8, 0.5, 1, 1.9, 1.9999999])
def test_relaxation_accepts_weights_inside_float32_bounds(relaxation):
    assert NumericsConfig(relaxation=relaxation).relaxation == relaxation


@pytest.mark.parametrize("command", ["apply", "evaluate", "video"])
@pytest.mark.parametrize("relaxation", ["1e-50", "1e-8", "1.999999999"])
def test_invalid_relaxation_cli_fails_before_processing(
    tmp_path, capsys, command, relaxation
):
    destination = tmp_path / "output"
    arguments = [command, str(tmp_path / "missing")]
    if command == "evaluate":
        arguments.append(str(tmp_path / "references"))
    arguments += [str(destination), "--relaxation", relaxation]
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2
    assert "relaxation must keep float32 solver weights" in capsys.readouterr().err
    assert not destination.exists()


@pytest.mark.parametrize("suffix", ["yaml", "toml", "json"])
def test_invalid_relaxation_in_config_is_rejected(tmp_path, suffix):
    path = tmp_path / f"settings.{suffix}"
    path.write_text({
        "yaml": "numerics:\n  relaxation: 1.999999999\n",
        "toml": "[numerics]\nrelaxation = 1.999999999\n",
        "json": '{"numerics": {"relaxation": 1.999999999}}',
    }[suffix])
    with pytest.raises(ValueError, match="relaxation must keep float32 solver weights"):
        Config.load(path)


def test_tiny_alpha_cli_processes_image_without_numeric_warnings(tmp_path):
    source = tmp_path / "input.png"
    pixels = np.random.default_rng(9).integers(0, 256, (33, 41, 3), np.uint8)
    Image.fromarray(pixels).save(source)
    source_bytes = source.read_bytes()
    destination = tmp_path / "result.png"
    with np.errstate(divide="raise", over="raise", invalid="raise"):
        main(["apply", str(source), str(destination), "--alpha", "1e-42",
              "--fade", "0", "--output-format", "png"])
    with Image.open(destination) as output:
        expected = apply_filter(pixels, config=Config(parameters=FilterParams(alpha=1e-42)))
        np.testing.assert_array_equal(np.asarray(output), expected)
    record = json.loads(destination.with_suffix(".meta.json").read_text())
    assert record["configuration"]["parameters"]["alpha"] == 1e-42
    assert source.read_bytes() == source_bytes


@pytest.mark.parametrize("alpha", [1e-50, 1e-44])
def test_alpha_that_underflows_during_scaling_is_rejected(alpha):
    with pytest.raises(ValueError, match="alpha must remain positive in float32"):
        FilterParams(alpha=alpha)


@pytest.mark.parametrize("alpha", [1e-20, 0.01, 1, 100])
def test_representable_positive_alpha_is_accepted(alpha):
    assert FilterParams(alpha=alpha).alpha == alpha


@pytest.mark.parametrize("suffix", ["yaml", "json", "toml"])
@pytest.mark.parametrize("alpha", ["1e-50", "1e-44"])
def test_alpha_underflow_in_config_is_rejected_before_processing(
    tmp_path, capsys, suffix, alpha
):
    content = {
        "yaml": f"parameters:\n  alpha: {alpha}\n",
        "json": f'{{"parameters": {{"alpha": {alpha}}}}}',
        "toml": f"[parameters]\nalpha = {alpha}\n",
    }
    path = tmp_path / f"settings.{suffix}"
    path.write_text(content[suffix])
    with pytest.raises(ValueError, match="alpha must remain positive in float32"):
        Config.load(path)
    destination = tmp_path / "output.jpg"
    with pytest.raises(SystemExit) as error:
        main(["apply", str(tmp_path / "missing.png"), str(destination),
              "--config", str(path)])
    assert error.value.code == 2
    assert "alpha must remain positive in float32" in capsys.readouterr().err
    assert not destination.exists()
    assert not destination.with_suffix(".meta.json").exists()


@pytest.mark.parametrize("alpha", ["1e-50", "1e-44"])
@pytest.mark.parametrize("command", ["apply", "evaluate", "video"])
def test_alpha_underflow_cli_override_is_rejected_before_processing(
    tmp_path, capsys, alpha, command
):
    destination = tmp_path / "output"
    arguments = [command, str(tmp_path / "missing")]
    if command == "evaluate":
        arguments.append(str(tmp_path / "references"))
    arguments += [str(destination), "--alpha", alpha]
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2
    assert "alpha must remain positive in float32" in capsys.readouterr().err
    assert not destination.exists()


@pytest.mark.parametrize("text,key", [
    ('{"filter": "fattal1", "filter": "fattal2"}', "filter"),
    ('{"parameters": {"fade": 0}, "parameters": {"fade": 100}}', "parameters"),
    ('{"parameters": {"fade": 0, "fade": 100}}', "fade"),
    ('{"parameters": {"fade": 0, "fade": 0}}', "fade"),
    (r'{"parameters": {"fade": 0, "\u0066ade": 100}}', "fade"),
])
@pytest.mark.parametrize("inherited", [False, True])
def test_duplicate_json_keys_fail_before_processing(
    tmp_path, capsys, text, key, inherited
):
    path = tmp_path / "settings.json"
    path.write_text(text)
    if inherited:
        child = tmp_path / "child.yaml"
        child.write_text("extends: settings.json\nparameters:\n  fade: 50\n")
        path = child
    message = f"Duplicate configuration key: {key}"
    with pytest.raises(ValueError, match=message):
        Config.load(path)
    with pytest.raises(ValueError, match=message):
        Config.source_paths(path)
    destination = tmp_path / "output.jpg"
    with pytest.raises(SystemExit) as error:
        main(["apply", str(tmp_path / "missing.png"), str(destination),
              "--config", str(path)])
    assert error.value.code == 2
    assert message in capsys.readouterr().err
    assert not destination.exists()
    assert not destination.with_suffix(".meta.json").exists()


def test_json_inheritance_allows_overrides_between_files(tmp_path):
    parent = tmp_path / "parent.json"
    child = tmp_path / "child.json"
    parent.write_text('{"parameters": {"fade": 0, "beta": 60}}')
    child.write_text(
        '{"extends": "parent.json", "parameters": {"fade": 100}, '
        '"output": {"format": "png"}}'
    )
    assert Config.load(child) == Config(
        parameters=FilterParams(fade=100, beta=60), output=OutputConfig(format="png")
    )


def test_config_round_trip_and_checked_in_defaults(tmp_path):
    config = Config(
        output=OutputConfig(jpeg_background=(10, 20, 30)),
        video=VideoConfig(deblock="weak8", deband="gentle", flat_cleanup=0.25),
    )
    for suffix, content in [
        ("toml", config.to_toml()),
        ("json", json.dumps(config.to_dict())),
    ]:
        path = tmp_path / f"config.{suffix}"
        path.write_text(content)
        assert Config.load(path) == config
    assert Config.load("configs/default.yaml") == Config(video=VideoConfig(encoder="h264_nvenc"))


def test_api_uses_filter_selection_and_convenience_overrides():
    rgb = np.random.default_rng(15).integers(0, 256, (35, 47, 3), np.uint8)
    config = Config(filter="fattal2", parameters=FilterParams(fade=25))
    np.testing.assert_array_equal(
        apply_filter(rgb, config=config), apply_filter(rgb, 2, FilterParams(fade=25))
    )
    np.testing.assert_array_equal(
        fattal2(rgb, config=config, fade=0), apply_filter(rgb, 2, FilterParams(fade=0))
    )
    with pytest.raises(ValueError, match="one image"):
        apply_filter(rgb, config=replace(config, filter="both"))
    with pytest.raises(ValueError, match="not both"):
        apply_filter(rgb, config=config, profile="core")
    with pytest.raises(ValueError, match="FilterParams"):
        apply_filter(rgb, params={})


def test_custom_limit_changes_real_processing_without_changing_dimensions():
    rgb = np.random.default_rng(23).integers(0, 256, (72, 96, 3), np.uint8)
    config = Config(
        parameters=FilterParams(fade=0), processing=ProcessingConfig(max_edge=48)
    )
    small = apply_filter(rgb, config=config)
    full_size = apply_filter(rgb, 1, FilterParams(fade=0), profile="core")
    assert small.shape == rgb.shape
    assert np.mean(np.abs(small.astype(float) - full_size)) > 1
    np.testing.assert_array_equal(
        rgb, np.random.default_rng(23).integers(0, 256, rgb.shape, np.uint8)
    )


def test_cli_overrides_file_and_records_effective_configuration(tmp_path, capsys):
    source, destination = tmp_path / "input.png", tmp_path / "outputs"
    Image.new("RGB", (33, 25), (80, 90, 100)).save(source)
    path = tmp_path / "settings.toml"
    path.write_text(
        'filter = "both"\n[parameters]\nfade = 0\n[processing]\nmax_edge = 32\n[output]\nformat = "jpeg"\njpeg_quality = 89\n'
    )
    main(
        [
            "apply",
            str(source),
            str(destination),
            "--config",
            str(path),
            "--fade",
            "100",
            "--device-class",
            "high",
            "--jpeg-quality",
            "95",
        ]
    )
    for variant in (1, 2):
        image = destination / f"fattal{variant}.jpg"
        assert Image.open(image).format == "JPEG"
        record = json.loads(image.with_suffix(".meta.json").read_text())
        assert record["configuration"]["filter"] == f"fattal{variant}"
        assert record["configuration"]["parameters"]["fade"] == 100
        assert record["configuration"]["processing"]["max_edge"] is None
        assert record["resolved_processing"]["max_edge"] == 2048
        assert record["resolved_processing"]["working_dimensions"] is None
        assert record["configuration"]["output"]["jpeg_quality"] == 95
        assert (
            record["config_source"]["sha256"]
            == hashlib.sha256(path.read_bytes()).hexdigest()
        )


@pytest.mark.parametrize("fade", [0, 100])
def test_orientation_is_configurable_and_recorded(tmp_path, capsys, fade):
    source = tmp_path / "rotated.jpg"
    image = Image.new("RGB", (35, 27), (70, 110, 30))
    exif = Image.Exif()
    exif[274] = 6
    image.save(source, exif=exif)
    for policy, size in [("ignore", (35, 27)), ("apply", (27, 35))]:
        output = tmp_path / f"{policy}.jpg"
        main(
            [
                "apply",
                str(source),
                str(output),
                "--fade",
                str(fade),
                "--exif-orientation",
                policy,
            ]
        )
        assert Image.open(output).size == ((35, 27) if fade == 100 else size)
        record = json.loads(output.with_suffix(".meta.json").read_text())
        assert record["source_dimensions"] == [35, 27]
        assert record["source_mode"] == "RGB"
        assert record["source_exif_orientation"] == 6
        assert record["configuration"]["image"]["exif_orientation"] == policy


def test_icc_interpretation_and_optional_profile_embedding(tmp_path, capsys):
    source = tmp_path / "profile.png"
    # LAB input with an embedded LAB profile exercises an actual color transform.
    image = Image.new("LAB", (16, 16), (140, 130, 120))
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("LAB")).tobytes()
    with pytest.raises(ValueError, match="embedded ICC"):
        bad = Image.new("RGB", (16, 16))
        bad.info["icc_profile"] = b"invalid"
        apply_filter(bad, config=Config(image=ImageConfig(color_space="srgb")))
    transformed = ImageCms.profileToProfile(
        image,
        ImageCms.ImageCmsProfile(ImageCms.createProfile("LAB")),
        ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")),
        outputMode="RGB",
    )
    image.info["icc_profile"] = profile
    actual = apply_filter(
        image,
        config=Config(
            image=ImageConfig(color_space="srgb")
        ),
    )
    np.testing.assert_array_equal(
        np.asarray(actual), apply_filter(np.asarray(transformed))
    )
    rgb = Image.new("RGB", (16, 16), (90, 100, 110))
    srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    rgb.save(source, icc_profile=srgb)
    output = tmp_path / "output.jpg"
    main(
        [
            "apply",
            str(source),
            str(output),
            "--fade",
            "100",
            "--color-space",
            "srgb",
            "--embed-icc",
        ]
    )
    assert Image.open(output).info["icc_profile"]
    default = tmp_path / "default.jpg"
    main(["apply", str(source), str(default), "--fade", "100"])
    assert "icc_profile" not in Image.open(default).info


def test_arrays_are_already_interpreted_pixels_under_photo_defaults():
    pixels = np.random.default_rng(61).integers(0, 256, (33, 35, 3), np.uint8)
    explicit = Config(
        image=ImageConfig(color_space="encoded", exif_orientation="ignore")
    )
    np.testing.assert_array_equal(
        apply_filter(pixels), apply_filter(pixels, config=explicit)
    )


@pytest.mark.parametrize("fade", [0, 100])
def test_photo_export_applies_exif_and_requested_jpeg_codec(tmp_path, capsys, fade):
    source = tmp_path / "phone.jpg"
    pixels = np.random.default_rng(30).integers(0, 256, (27, 35, 3), np.uint8)
    exif = Image.Exif()
    exif[274] = 6
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    Image.fromarray(pixels).save(source, exif=exif, icc_profile=profile)
    original_bytes = source.read_bytes()
    outputs = tmp_path / "outputs"
    main(["apply", str(source), str(outputs), "--filter", "both", "--fade", str(fade)])
    with Image.open(source) as decoded:
        pixels = np.asarray(decoded).copy()
    if fade < 100:
        pixels = np.rot90(pixels, k=3).copy()
    for variant in (1, 2):
        filtered = apply_filter(pixels, variant, FilterParams(fade=fade))
        control = BytesIO()
        Image.fromarray(filtered).save(control, format="JPEG", quality=100, subsampling=2)
        control.seek(0)
        with Image.open(control) as expected:
            output = outputs / f"fattal{variant}.jpg"
            with Image.open(output) as actual:
                assert actual.format == "JPEG"
                assert actual.size == ((35, 27) if fade == 100 else (27, 35))
                assert JpegImagePlugin.get_sampling(actual) == 2
                assert all(
                    value == 1
                    for table in actual.quantization.values()
                    for value in table
                )
                assert "icc_profile" not in actual.info
                assert actual.getexif().get(274) is None
                np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
            record = json.loads(output.with_suffix(".meta.json").read_text())
            assert record["color_policy"] == {
                "color_space": "srgb",
                "color_backend": "matrix_trc",
                "exif_orientation": "apply",
                "decode_sampling": True,
            }
    assert source.read_bytes() == original_bytes


def test_png_is_explicit_and_wrong_extension_cannot_hide_jpeg(tmp_path, capsys):
    source, output = tmp_path / "source.png", tmp_path / "output.png"
    pixels = np.random.default_rng(62).integers(0, 256, (16, 18, 3), np.uint8)
    Image.fromarray(pixels).save(source)
    with pytest.raises(SystemExit):
        main(["apply", str(source), str(output), "--fade", "100"])
    assert not output.exists()
    main(["apply", str(source), str(output), "--fade", "100", "--output-format", "png"])
    with Image.open(output) as image:
        assert image.format == "PNG"
        np.testing.assert_array_equal(np.asarray(image), pixels)


def test_transparency_extension_and_explicit_jpeg_background(tmp_path, capsys):
    source, output = tmp_path / "alpha.png", tmp_path / "output.jpg"
    rgba = np.zeros((32, 40, 4), np.uint8)
    rgba[..., :3] = [255, 0, 0]
    rgba[..., 3] = 128
    Image.fromarray(rgba).save(source)
    config = Config(
        processing=ProcessingConfig(transparency="straight"),
        parameters=FilterParams(fade=0),
    )
    np.testing.assert_array_equal(
        apply_filter(rgba, config=config)[..., 3], rgba[..., 3]
    )
    with pytest.raises(SystemExit):
        main(
            [
                "apply",
                str(source),
                str(output),
                "--fade",
                "100",
                "--output-format",
                "jpeg",
            ]
        )
    assert not output.exists()
    main(
        [
            "apply",
            str(source),
            str(output),
            "--fade",
            "100",
            "--output-format",
            "jpeg",
            "--jpeg-subsampling",
            "444",
            "--jpeg-background",
            "255",
            "255",
            "255",
        ]
    )
    rgb = np.asarray(Image.open(output))
    assert np.max(np.abs(rgb.astype(int) - np.array([255, 127, 127]))) <= 2


def test_evaluate_reuses_configuration_and_exports_requested_format(tmp_path, capsys):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    Image.new("RGB", (32, 24), "red").save(inputs / "image.png")
    config = tmp_path / "config.toml"
    config.write_text(
        'filter = "fattal2"\n[parameters]\nfade = 100\n[output]\nformat = "jpeg"\n'
    )
    output = tmp_path / "evaluation"
    main(
        [
            "evaluate",
            str(inputs),
            str(tmp_path / "references"),
            str(output),
            "--config",
            str(config),
        ]
    )
    assert (output / "fattal_2/image.jpg").exists()
    report = json.loads((output / "evaluation.json").read_text())
    assert len(report["records"]) == 1
    assert report["configuration"]["output"]["format"] == "jpeg"


def test_exported_configuration_is_reusable_and_preserves_existing_files(
    tmp_path, capsys
):
    path = tmp_path / "config.toml"
    main(["config", "--output", str(path)])
    assert Config.load(path) == Config()
    with pytest.raises(SystemExit):
        main(["config", "--output", str(path)])
    assert Config.load(path) == Config()
