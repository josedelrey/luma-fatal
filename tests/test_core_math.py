"""Scalar precision and complete deterministic gradient-domain regressions."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from luma_fatal import Config, FilterParams, NumericsConfig, ProcessingConfig, apply_filter
from luma_fatal.core import _pyramid, _phi, _divergence, _sor, _normalize, _recombine
from luma_fatal.numerics import fast_pow64, luminance_u8, scalar_math_f32
from luma_fatal.probes import _probe_arrays


FIXTURES = Path(__file__).parent / "fixtures"
MATH = json.loads((FIXTURES / "scalar-math.json").read_text())["trials"]
PIPELINE = json.loads((FIXTURES / "core-supplied-floor.json").read_text())
RAW_PIPELINE = json.loads((FIXTURES / "core-zero-luminance.json").read_text())


@pytest.mark.parametrize("variant", [1, 2])
@pytest.mark.parametrize("power_mode", ["approximate", "standard"])
@pytest.mark.parametrize("alpha,peak", [(1e-43, 1), (1e-42, 1), (1e-35, 100)])
@pytest.mark.parametrize("beta", [0, 80, 100])
def test_attenuation_handles_alpha_with_unsafe_float32_division(
    variant, power_mode, alpha, peak, beta
):
    log_luminance = np.zeros((33, 41), np.float32)
    log_luminance[16, 20] = peak
    with np.errstate(divide="raise", over="raise", invalid="raise", under="raise"):
        phi = _phi([log_luminance], variant, FilterParams(alpha=alpha, beta=beta),
                   power_mode=power_mode)
    assert phi.dtype == np.float32
    assert np.isfinite(phi).all()
    assert np.all((phi >= 0) & (phi <= 1))
    affected = np.zeros_like(log_luminance, dtype=bool)
    affected[[15, 17, 16, 16], [20, 20, 19, 21]] = True
    np.testing.assert_array_equal(phi[~affected], 1)
    if power_mode == "standard":
        # An isolated peak has four gradients of peak / 2 and mean 2 * peak / N.
        with np.errstate(under="ignore"):
            scaled_alpha = np.float32(alpha) / np.float32(100)
        average = np.float32(2 * peak) / np.float32(log_luminance.size)
        exponent = np.float32(np.float32(beta) / np.float32(100) - np.float32(1))
        expected = ((float(peak) / 2) / (float(scaled_alpha) * float(average))) ** float(exponent)
        np.testing.assert_allclose(phi[affected], expected, rtol=1e-6,
                                   atol=float(np.nextafter(np.float32(0), np.float32(1))))


@pytest.mark.parametrize("variant", [1, 2])
@pytest.mark.parametrize("low_contrast", [False, True])
@pytest.mark.parametrize("power_mode", ["approximate", "standard"])
@pytest.mark.parametrize("alpha,beta,shape", [
    (1e-42, 80, (33, 41)), (1e-43, 0, (129, 161)), (1e-35, 100, (65, 81)),
])
def test_image_processing_handles_tiny_alpha_without_numeric_warnings(
    variant, low_contrast, power_mode, alpha, beta, shape
):
    if low_contrast:
        pixels = np.full((*shape, 3), 128, np.uint8)
        pixels[16, 20] = 129
    else:
        pixels = np.random.default_rng(9).integers(0, 256, (*shape, 3), np.uint8)
    original = pixels.copy()
    config = Config(
        parameters=FilterParams(alpha=alpha, beta=beta, fade=0),
        processing=ProcessingConfig(resize=False),
        numerics=NumericsConfig(power_mode=power_mode),
    )
    with np.errstate(divide="raise", over="raise", invalid="raise"):
        result = apply_filter(pixels, variant, config=config)
        repeated = apply_filter(pixels, variant, config=config)
    assert result.dtype == np.uint8
    assert result.shape == pixels.shape
    np.testing.assert_array_equal(result, repeated)
    np.testing.assert_array_equal(pixels, original)
    if beta == 100:
        # A zero exponent makes attenuation independent of alpha in either mode.
        control = replace(config, parameters=replace(config.parameters, alpha=1))
        np.testing.assert_array_equal(result, apply_filter(pixels, variant, config=control))


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
