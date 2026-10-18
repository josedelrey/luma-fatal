"""Scalar precision and complete deterministic gradient-domain regressions."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from luma_fatal import Config, FilterParams, ProcessingConfig, apply_filter
from luma_fatal.core import _pyramid, _phi, _divergence, _sor, _normalize, _recombine
from luma_fatal.numerics import fast_pow64, luminance_u8, scalar_math_f32
from luma_fatal.probes import _probe_arrays


FIXTURES = Path(__file__).parent / "fixtures"
MATH = json.loads((FIXTURES / "scalar-math.json").read_text())["trials"]
PIPELINE = json.loads((FIXTURES / "core-supplied-floor.json").read_text())
RAW_PIPELINE = json.loads((FIXTURES / "core-zero-luminance.json").read_text())


@pytest.mark.parametrize("variant", [1, 2])
@pytest.mark.parametrize("power_mode", ["approximate", "standard"])
@pytest.mark.parametrize("alpha,peak", [(1e-42, 1), (1e-35, 100)])
def test_attenuation_rejects_alpha_that_would_underflow_or_overflow(
    variant, power_mode, alpha, peak
):
    log_luminance = np.zeros((33, 41), np.float32)
    log_luminance[16, 20] = peak
    with np.errstate(divide="raise", over="raise", invalid="raise", under="raise"):
        with pytest.raises(ValueError, match="alpha is too small for the current image gradients"):
            _phi([log_luminance], variant, FilterParams(alpha=alpha), power_mode=power_mode)


@pytest.mark.parametrize("variant", [1, 2])
@pytest.mark.parametrize("low_contrast", [False, True])
def test_image_processing_rejects_unsafe_alpha_without_numeric_warnings(
    variant, low_contrast
):
    if low_contrast:
        pixels = np.full((33, 41, 3), 128, np.uint8)
        pixels[16, 20] = 129
    else:
        pixels = np.random.default_rng(9).integers(0, 256, (33, 41, 3), np.uint8)
    original = pixels.copy()
    config = Config(
        parameters=FilterParams(alpha=1e-42, fade=0),
        processing=ProcessingConfig(resize=False),
    )
    with np.errstate(divide="raise", over="raise", invalid="raise"):
        with pytest.raises(ValueError, match="increase parameters.alpha"):
            apply_filter(pixels, variant, config=config)
    np.testing.assert_array_equal(pixels, original)


@pytest.mark.parametrize("trial", MATH, ids=[trial["function"] for trial in MATH])
def test_scalar_math_matches_reference_bits(trial):
    inputs = np.array(
        [int(value, 16) for value in trial["input_bits"]], np.uint32
    ).view(np.float32)
    expected = np.array([int(value, 16) for value in trial["output_bits"]], np.uint32)
    output = scalar_math_f32(inputs, trial["function"] == "expf")
    assert output.dtype == np.float32
    np.testing.assert_array_equal(output.view(np.uint32), expected)


@pytest.mark.parametrize(
    "exponent,expected_bits",
    [
        (0.75, [
            "0ffbc49f00000000", "6ffbc49f00000000", "3f4f6dc900000000",
            "3fd7c49f00000000", "3fe3c49f00000000", "3fefc49f00000000",
            "3ffbc49f00000000", "4016c49f00000000", "9fefc49f00000000",
            "6fefc49f00000000", "6ff5c49f00000000",
        ]),
        (-0.5, [
            "5fe69bbe00000000", "1fe69bbe00000000", "4059804d00000000",
            "3ffe9bbe00000000", "3ff69bbe00000000", "3fee9bbe00000000",
            "3fe69bbe00000000", "3fd49bbe00000000", "ffee9bbf00000000",
            "1fee9bbe00000000", "1fea9bbe00000000",
        ]),
    ],
)
def test_affine_power_matches_frozen_high_word_bits(exponent, expected_bits):
    values = np.array(
        [0, -0.0, 1e-4, 0.25, 0.5, 1, 2, 10, -1, np.inf, np.nan],
        np.float64,
    )
    actual = fast_pow64(values, exponent)
    expected = np.array([int(bits, 16) for bits in expected_bits], np.uint64)
    np.testing.assert_array_equal(actual.view(np.uint64), expected)


@pytest.mark.parametrize(
    "row",
    PIPELINE["rows"],
    ids=[f"{row['probe']}-fattal{row['variant']}" for row in PIPELINE["rows"]],
)
def test_core_matches_supplied_luminance_floor_reference(row):
    # This reference supplies a luminance floor before the logarithm.
    # Keep it separate from the pipeline contract that preserves zero luminance.
    source = _probe_arrays(PIPELINE["size"], PIPELINE["seed"])[row["probe"]][0]
    assert hashlib.sha256(source.tobytes()).hexdigest() == row["source_pixel_sha256"]
    luminance = luminance_u8(source).astype(np.float32) * np.float32(1 / 255)
    luminance = np.maximum(luminance, np.float32(1e-4))
    argument = (
        (luminance * np.float32(100) / luminance.max()).astype(np.float64) + 0.01
    ).astype(np.float32)
    log = scalar_math_f32(argument, False)
    phi = _phi(_pyramid(log), row["variant"], FilterParams(fade=0))
    mapped = _normalize(
        scalar_math_f32(_sor(_divergence(log, phi)), True) + np.float32(-0.01)
    )
    output = _recombine(source, luminance, mapped, 100)
    assert (
        hashlib.sha256(output.tobytes()).hexdigest() == row["expected_rgb_pixel_sha256"]
    )


@pytest.mark.parametrize(
    "row",
    RAW_PIPELINE["rows"],
    ids=[f"{row['probe']}-fattal{row['variant']}" for row in RAW_PIPELINE["rows"]],
)
def test_pipeline_matches_zero_luminance_reference(row):
    source = _probe_arrays(RAW_PIPELINE["size"], RAW_PIPELINE["seed"])[row["probe"]][0]
    assert hashlib.sha256(source.tobytes()).hexdigest() == row["source_pixel_sha256"]
    output = apply_filter(source, row["variant"], FilterParams(fade=0))
    assert (
        hashlib.sha256(output.tobytes()).hexdigest() == row["expected_rgb_pixel_sha256"]
    )
