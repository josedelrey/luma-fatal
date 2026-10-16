"""Float32 scalar math, fixed-point luminance and affine power approximations.

Operation ordering is part of the frozen numerical contract.
"""

from __future__ import annotations

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray

F32 = NDArray[np.float32]
_BIAS = 1072632447


@njit(cache=True)
def scalar_math_f32(source: F32, exponential: bool) -> F32:
    """Evaluate scalar float32 log or exp without fastmath.

    Scalar host libm is covered by fixed serial traces on the locked Linux
    runtime. Cross-platform bit equality is outside the current contract.
    """
    result = np.empty_like(source)
    values, output = source.ravel(), result.ravel()
    for index in range(values.size):
        if exponential:
            output[index] = np.exp(values[index])
        else:
            output[index] = np.log(values[index])
    return result


def fast_pow64(base: ArrayLike, exponent: float) -> NDArray[np.float64]:
    """Approximate a power using an affine transform of the float64 high word."""
    value = np.ascontiguousarray(base, dtype=np.float64)
    return _fast_pow64_values(value, np.float64(exponent))


@njit(cache=True)
def _fast_pow64_values(
    value: NDArray[np.float64], exponent: np.float64
) -> NDArray[np.float64]:
    """Transform high words in one pass without intermediate full-size arrays."""
    result = np.empty(value.shape, dtype=np.float64)
    source = value.view(np.uint64).ravel()
    output = result.view(np.uint64).ravel()
    for index in range(source.size):
        high = np.int64(source[index] >> np.uint64(32))
        word = np.int64(np.trunc(exponent * (high - _BIAS) + _BIAS))
        output[index] = np.uint64(word) << np.uint64(32)
    return result


def power64(
    base: ArrayLike, exponent: float, mode: str = "approximate"
) -> NDArray[np.float64]:
    """Select the affine approximation or NumPy float64 exponentiation."""
    if mode == "approximate":
        return fast_pow64(base, exponent)
    if mode == "standard":
        return np.power(np.asarray(base, dtype=np.float64), np.float64(exponent))
    raise ValueError("power_mode must be approximate or standard")


def fast_pow(base: ArrayLike, exponent: float) -> F32:
    """Evaluate the affine power approximation and narrow to float32."""
    return fast_pow64(base, exponent).astype(np.float32)


def gradient_scale(value: int) -> np.float32:
    """Approximate powers of two for multiscale gradient normalization."""
    word = int(1109377.0 * value + _BIAS)
    bits = np.array(word << 32, dtype=np.uint64)
    return np.float32(bits.view(np.float64))


def luminance_u8(rgb: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Convert RGB8 to luminance using 14-bit fixed-point weights."""
    channels = rgb.astype(np.int32)
    result = (
        4899 * channels[..., 0]
        + 9617 * channels[..., 1]
        + 1868 * channels[..., 2]
        + 8192
    ) >> 14
    return result.astype(np.uint8)


def luminance_fractional(rgb: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Use the reference RGB weights without rounding luminance to a byte."""
    channels = rgb.astype(np.int32)
    weighted = (
        4899 * channels[..., 0]
        + 9617 * channels[..., 1]
        + 1868 * channels[..., 2]
    )
    return weighted.astype(np.float32) / np.float32(16384 * 255)


def attenuation(gradient: F32, alpha: float, beta: float, noise: float) -> F32:
    """Compute a gradient attenuation field from its mean magnitude."""
    row_sums = np.cumsum(gradient, axis=1, dtype=np.float32)[:, -1]
    average = np.cumsum(row_sums, dtype=np.float32)[-1] / np.float32(gradient.size)
    denominator = np.float32(np.float32(alpha) * average)
    result = np.ones_like(gradient)
    mask = gradient > np.float32(1e-4)
    if not np.any(mask) or denominator <= 0:
        return result
    ratio = (gradient[mask] + np.float32(noise)) / denominator
    exponent = np.float32(np.float32(beta) - np.float32(1))
    result[mask] = fast_pow(ratio, exponent)
    return result
