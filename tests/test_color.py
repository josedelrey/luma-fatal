"""Matrix/TRC conversion over complete RGB8 domains and profile references."""

import base64
from dataclasses import replace
import hashlib
from io import BytesIO
import json
from pathlib import Path
import struct

import numpy as np
from PIL import Image, ImageCms
import pytest

from luma_fatal import Config, FilterParams, ImageConfig, apply_filter
from luma_fatal.cli import main
from luma_fatal.color import (
    cached_profile,
    MalformedProfileError,
    parse_profile,
    UnsupportedProfileError,
)
from luma_fatal.image_io import prepare_image

FIXTURES = Path(__file__).parent / "fixtures/color"
GOLDENS = json.loads((FIXTURES / "goldens.json").read_text())


def profile(row):
    return base64.b64decode(GOLDENS["profiles"][row["profile"]])


def source(row):
    if "file" in row:
        path = FIXTURES / row["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["file_sha256"]
        with Image.open(path) as image:
            return image.copy()
    if "cube_block" in row:
        block = row["cube_block"]
        indices = np.arange(block * 1048576, (block + 1) * 1048576, dtype=np.uint32)
        rgb = np.stack(
            (indices >> 16, (indices >> 8) & 255, indices & 255), axis=1
        ).astype(np.uint8)
        image = Image.fromarray(rgb.reshape(1024, 1024, 3))
    else:
        width, height = row["dimensions"]
        pixels = np.random.default_rng(row["seed"]).integers(
            0, 256, (height, width, 3), np.uint8
        )
        image = Image.fromarray(pixels)
    image.info["icc_profile"] = profile(row)
    return image


@pytest.mark.parametrize("row", GOLDENS["cases"], ids=lambda row: row["id"])
def test_matrix_conversion_matches_full_domain_and_profile_references(row):
    image = source(row)
    before = image.tobytes()
    assert hashlib.sha256(before).hexdigest() == row["source_rgb_sha256"]
    converted = prepare_image(image, ImageConfig())
    assert hashlib.sha256(converted.tobytes()).hexdigest() == row["expected_rgb_sha256"]
    assert converted.info["_color_conversion"]["backend"] == "matrix_trc"
    assert image.tobytes() == before
    assert image.info["icc_profile"] == profile(row)


def test_lcms_backend_preserves_its_color_interpretation():
    row = next(r for r in GOLDENS["cases"] if r["id"] == "noise_p3_car")
    image = source(row)
    expected = ImageCms.profileToProfile(
        image,
        ImageCms.ImageCmsProfile(BytesIO(profile(row))),
        ImageCms.createProfile("sRGB"),
        outputMode="RGB",
    )
    lcms_result = prepare_image(image, ImageConfig(color_backend="lcms"))
    np.testing.assert_array_equal(np.asarray(lcms_result), np.asarray(expected))
    assert lcms_result.info["_color_conversion"]["backend"] == "lcms"
    assert lcms_result.tobytes() != prepare_image(image, ImageConfig()).tobytes()


def test_unsupported_lab_profile_uses_a_recorded_lcms_extension():
    image = Image.new("LAB", (16, 16), (140, 130, 120))
    image.info["icc_profile"] = ImageCms.ImageCmsProfile(
        ImageCms.createProfile("LAB")
    ).tobytes()
    result = prepare_image(image, ImageConfig())
    metadata = result.info["_color_conversion"]
    assert metadata["requested_backend"] == "matrix_trc"
    assert metadata["backend"] == "lcms"
    assert metadata["fallback_reason"]


def test_opaque_rgba_conversion_keeps_alpha_and_transparent_extension_is_explicit():
    row = next(r for r in GOLDENS["cases"] if r["id"] == "noise_p3_car")
    image = source(row)
    opaque = image.convert("RGBA")
    opaque.info["icc_profile"] = profile(row)
    result = prepare_image(opaque, ImageConfig())
    assert (
        hashlib.sha256(np.asarray(result)[..., :3].tobytes()).hexdigest()
        == row["expected_rgb_sha256"]
    )
    assert np.all(np.asarray(result)[..., 3] == 255)
    pixels = np.asarray(opaque).copy()
    pixels[..., 3] = 128
    transparent = Image.fromarray(pixels)
    transparent.info["icc_profile"] = profile(row)
    result = prepare_image(transparent, ImageConfig())
    assert result.info["_color_conversion"]["backend"] == "lcms"
    assert "Transparent" in result.info["_color_conversion"]["fallback_reason"]
    np.testing.assert_array_equal(np.asarray(result)[..., 3], pixels[..., 3])


def test_encoded_and_untagged_inputs_bypass_color_conversion():
    row = next(r for r in GOLDENS["cases"] if r["id"] == "noise_p3_car")
    image = source(row)
    encoded = prepare_image(image, ImageConfig(color_space="encoded"))
    assert encoded.tobytes() == image.tobytes()
    assert encoded.info["_color_conversion"]["backend"] == "none"
    image.info.pop("icc_profile")
    untagged = prepare_image(image, ImageConfig())
    assert untagged.tobytes() == image.tobytes()
    assert untagged.info["_color_conversion"]["backend"] == "none"


@pytest.mark.parametrize(
    "mutation", ["signature", "length", "offset", "missing", "curve-count", "duplicate"]
)
def test_malformed_profiles_are_rejected_without_a_silent_fallback(mutation):
    row = next(r for r in GOLDENS["cases"] if r["id"] == "noise_p3_car")
    data = bytearray(profile(row))
    if mutation == "signature":
        data[36:40] = b"nope"
    elif mutation == "length":
        data[:4] = struct.pack(">I", len(data) + 1)
    elif mutation == "offset":
        data[136:140] = struct.pack(">I", len(data) + 1)
    elif mutation == "missing":
        data[128:132] = struct.pack(">I", 0)
    elif mutation == "duplicate":
        data[132:136] = b"rTRC"
    else:
        # Redirect all TRCs to a malformed parametric function type.
        count = struct.unpack_from(">I", data, 128)[0]
        for i in range(count):
            name, offset, _ = struct.unpack_from(">4sII", data, 132 + i * 12)
            if name == b"rTRC":
                data[offset + 8 : offset + 10] = struct.pack(">H", 5)
                break
    with pytest.raises(MalformedProfileError):
        parse_profile(bytes(data))
    image = source(row)
    image.info["icc_profile"] = bytes(data)
    with pytest.raises(ValueError, match="embedded ICC"):
        prepare_image(image, ImageConfig())


def test_matrix_backend_rejects_multidimensional_lut_and_caches_read_only_profiles():
    row = next(r for r in GOLDENS["cases"] if r["id"] == "noise_p3_car")
    data = bytearray(profile(row))
    data[132:136] = b"A2B0"
    with pytest.raises(UnsupportedProfileError, match="LUT"):
        parse_profile(bytes(data))
    parsed = cached_profile(profile(row))
    assert cached_profile(profile(row)) is parsed
    with pytest.raises(ValueError, match="read-only"):
        parsed.matrix[0, 0] = 0


def test_cli_override_and_effect_metadata_record_the_actual_backend(tmp_path, capsys):
    row = next(r for r in GOLDENS["cases"] if r["id"] == "noise_p3_car")
    image = source(row)
    source_path = tmp_path / "p3.png"
    image.save(source_path, icc_profile=profile(row))
    settings = tmp_path / "settings.toml"
    settings.write_text('[image]\ncolor_backend = "lcms"\n')
    output = tmp_path / "converted.png"
    main(
        [
            "apply",
            str(source_path),
            str(output),
            "--fade",
            "100",
            "--output-format",
            "png",
            "--config",
            str(settings),
            "--color-backend",
            "matrix_trc",
        ]
    )
    with Image.open(output) as actual:
        assert (
            hashlib.sha256(actual.tobytes()).hexdigest() == row["expected_rgb_sha256"]
        )
    metadata = json.loads(output.with_suffix(".meta.json").read_text())
    assert metadata["color_conversion"]["backend"] == "matrix_trc"
    assert metadata["configuration"]["image"]["color_backend"] == "matrix_trc"
    assert "color.py" in metadata["implementation_sources_sha256"]
    result = apply_filter(image, config=Config(parameters=FilterParams(fade=100)))
    assert result.info["_color_conversion"]["backend"] == "matrix_trc"
    with pytest.raises(ValueError, match="color_backend"):
        replace(ImageConfig(), color_backend="unknown")
