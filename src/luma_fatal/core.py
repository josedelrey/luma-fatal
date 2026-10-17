"""Deterministic gradient-domain operators over RGB8 arrays."""

from __future__ import annotations

import warnings
from functools import lru_cache

import numpy as np
from numba import njit

from .numerics import (
    gradient_scale, luminance_fractional, luminance_u8, power64, scalar_math_f32,
)
from .configuration import FilterParams, NumericsConfig


def _gaussian3(source: np.ndarray) -> np.ndarray:
    return _gaussian3_pass(np.pad(source, 1, mode="edge"))


@njit(cache=True)
def _gaussian3_pass(padded: np.ndarray) -> np.ndarray:
    """Accumulate the nine taps in the reference float32 order."""
    height, width = padded.shape[0] - 2, padded.shape[1] - 2
    result = np.empty((height, width), dtype=np.float32)
    weights = (1 / 16, 2 / 16, 1 / 16, 2 / 16, 4 / 16, 2 / 16, 1 / 16, 2 / 16, 1 / 16)
    for y in range(height):
        for x in range(width):
            value = np.float32(0)
            for index, weight in enumerate(weights):
                row, col = divmod(index, 3)
                value = np.float32(value + np.float32(weight) * padded[y + row, x + col])
            result[y, x] = value
    return result


@lru_cache(maxsize=64)
def _cubic_axis(
    length: int, output_length: int, horizontal: bool
) -> tuple[np.ndarray, np.ndarray]:
    # Evaluate coordinates in float64 and narrow to float32 before flooring.
    coords = (
        (np.arange(output_length, dtype=np.float64) + 0.5)
        * (1 / (output_length / length))
        - 0.5
    ).astype(np.float32)
    base = np.floor(coords).astype(np.int64)
    t = coords - base.astype(np.float32)
    if horizontal:
        t = np.where((base < 0) | (base >= length - 1), np.float32(0), t)
        base = np.clip(base, 0, length - 1)
    t1 = t + np.float32(1)
    u = np.float32(1) - t
    w0 = (
        (t1 * np.float32(-0.75) + np.float32(3.75)) * t1 + np.float32(-6)
    ) * t1 + np.float32(3)
    w1 = t * (t * np.float32(1.25) + np.float32(-2.25)) * t + np.float32(1)
    w2 = u * (u * np.float32(1.25) + np.float32(-2.25)) * u + np.float32(1)
    w3 = ((np.float32(1) - w0) - w1) - w2
    indices = np.clip(base[:, None] + np.arange(-1, 3), 0, length - 1)
    weights = np.stack((w0, w1, w2, w3), axis=1)
    indices.flags.writeable = False
    weights.flags.writeable = False
    return indices, weights


def _resize_cubic(source: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resample scalar fields with half-pixel cubic weights (a = -0.75).

    No antialias prefilter is applied. Fixed numerical references cover
    bounded contiguous buffers. Other strides and nonfinite inputs are
    outside the established scope.
    """
    if source.shape == shape:
        return source.copy()
    y_index, y_weight = _cubic_axis(source.shape[0], shape[0], False)
    x_index, x_weight = _cubic_axis(source.shape[1], shape[1], True)
    return _resize_cubic_pass(source, y_index, y_weight, x_index, x_weight)


@njit(cache=True)
def _resize_cubic_pass(source, y_index, y_weight, x_index, x_weight):
    """Apply cached cubic axes with the original four-tap accumulation order."""
    horizontal = np.empty((source.shape[0], x_index.shape[0]), dtype=np.float32)
    for y in range(source.shape[0]):
        for x in range(x_index.shape[0]):
            value = np.float32(0)
            for k in range(4):
                value = np.float32(value + source[y, x_index[x, k]] * x_weight[x, k])
            horizontal[y, x] = value
    result = np.empty((y_index.shape[0], x_index.shape[0]), dtype=np.float32)
    for y in range(y_index.shape[0]):
        for x in range(x_index.shape[0]):
            value = np.float32(0)
            for k in range(4):
                value = np.float32(value + horizontal[y_index[y, k], x] * y_weight[y, k])
            result[y, x] = value
    return result


def _pyramid(log_luminance: np.ndarray, minimum_size: int = 32) -> list[np.ndarray]:
    pyramid = [log_luminance]
    minimum = min(log_luminance.shape)
    levels = 0
    while minimum >= minimum_size:
        levels += 1
        minimum //= 2
    for _ in range(max(1, levels) - 1):
        previous = pyramid[-1]
        shape = (previous.shape[0] // 2, previous.shape[1] // 2)
        pyramid.append(_resize_cubic(_gaussian3(previous), shape))
    return pyramid


def _gradient(source: np.ndarray, level: int) -> tuple[np.ndarray, np.float32]:
    padded = np.pad(source, 1, mode="edge")
    denominator = gradient_scale(level + 1)
    dx = (padded[1:-1, :-2] - padded[1:-1, 2:]) / denominator
    dy = (padded[2:, 1:-1] - padded[:-2, 1:-1]) / denominator
    magnitude = np.sqrt(dy * dy + dx * dx)
    # Float32 row sums precede the row-total sum to preserve accumulation order.
    row_sums = np.cumsum(magnitude, axis=1, dtype=np.float32)[:, -1]
    average = np.cumsum(row_sums, dtype=np.float32)[-1] / np.float32(source.size)
    return magnitude, average


def _phi(
    pyramid: list[np.ndarray],
    variant: int,
    params: FilterParams,
    gradient_threshold: float = 1e-4,
    detail_level: int = 0,
    power_mode: str = "approximate",
) -> np.ndarray:
    """Build attenuation while omitting the selected finest pyramid levels."""
    if detail_level >= len(pyramid):
        return np.ones_like(pyramid[0])
    phi = np.ones_like(pyramid[-1])
    alpha = np.float32(params.alpha) / np.float32(100)
    exponent = np.float32(np.float32(params.beta) / np.float32(100) - np.float32(1))
    noise = np.float32(params.noise) / np.float32(100)
    for level in range(len(pyramid) - 1, -1, -1):
        if level >= detail_level:
            gradient, average = _gradient(pyramid[level], level)
            value = np.ones_like(gradient)
            mask = gradient > np.float32(gradient_threshold)
            if np.any(mask) and average > 0:
                ratio = (gradient[mask] + noise) / np.float32(alpha * average)
                value[mask] = power64(ratio, exponent, power_mode).astype(np.float32)
            if variant == 1:
                phi *= value
            else:
                # Variant 2 replaces the attenuation field at each finer level.
                phi = value
        if level and (variant == 1 or level <= detail_level):
            phi = _gaussian3(_resize_cubic(phi, pyramid[level - 1].shape))
    return phi


def _divergence(log_luminance: np.ndarray, phi: np.ndarray) -> np.ndarray:
    gx = np.zeros_like(phi)
    gy = np.zeros_like(phi)
    gx[:, :-1] = (log_luminance[:, 1:] - log_luminance[:, :-1]) * phi[:, :-1]
    gy[:-1, :] = (log_luminance[1:, :] - log_luminance[:-1, :]) * phi[:-1, :]
    result = gx + gy
    result[:, 1:] -= gx[:, :-1]
    result[1:, :] -= gy[:-1, :]
    result[0] = 0  # The first row is fixed by the solver boundary condition.
    return result


@njit(cache=True)
def _sor(rhs: np.ndarray, iterations: int = 80, relaxation: float = 1.9) -> np.ndarray:
    """Solve the discrete gradient field with paired SOR sweeps.

    Rows advance serially, with the first row fixed at zero. The last column
    is updated in both half-sweeps. Float32 operation order is preserved,
    with no fastmath or parallel reassociation.
    """
    height, width = rhs.shape
    result = np.zeros_like(rhs)
    old_weight, new_weight = np.float32(1 - relaxation), np.float32(relaxation)
    for _ in range(iterations):
        for color in (1, 0):
            for y in range(1, height):
                start = 1
                if (color ^ y) & 1:
                    start = 2
                    value = np.float32(rhs[y, 0] - result[y - 1, 0])
                    value = np.float32(value - result[y, 1])
                    if y < height - 1:
                        value = np.float32(value - result[y + 1, 0])
                        value = np.float32(value / np.float32(-3))
                    else:
                        value = np.float32(value * np.float32(-0.5))
                    result[y, 0] = np.float32(
                        result[y, 0] * old_weight + value * new_weight
                    )
                for x in range(start, width - 1, 2):
                    value = np.float32(rhs[y, x] - result[y - 1, x])
                    value = np.float32(value - result[y, x + 1])
                    if y < height - 1:
                        value = np.float32(value - result[y + 1, x])
                        value = np.float32(value - result[y, x - 1])
                        value = np.float32(value * np.float32(-0.25))
                    else:
                        value = np.float32(value - result[y, x - 1])
                        value = np.float32(value / np.float32(-3))
                    result[y, x] = np.float32(
                        result[y, x] * old_weight + value * new_weight
                    )
                x = width - 1
                value = np.float32(rhs[y, x] - result[y - 1, x])
                if y < height - 1:
                    value = np.float32(value - result[y + 1, x])
                    value = np.float32(value - result[y, x - 1])
                    value = np.float32(value / np.float32(-3))
                else:
                    value = np.float32(value - result[y, x - 1])
                    value = np.float32(value * np.float32(-0.5))
                result[y, x] = np.float32(
                    result[y, x] * old_weight + value * new_weight
                )
    return result


def _normalize(
    source: np.ndarray,
    low_percentile: float = 0.001,
    high_percentile: float = 0.995,
    floor: float = 1e-4,
) -> np.ndarray:
    values = np.sort(source[source != 0])
    if not len(values):
        return np.full_like(source, np.float32(floor))
    low = values[
        min(len(values) - 1, int(np.float32(len(values)) * np.float32(low_percentile)))
    ]
    high = values[
        min(len(values) - 1, int(np.float32(len(values)) * np.float32(high_percentile)))
    ]
    with np.errstate(divide="ignore", invalid="ignore"):
        value = (source - low) / np.float32(high - low)
    # Preserve NaNs with the canonical positive quiet-NaN bit pattern.
    # This avoids platform-dependent sign bits after zero-range division.
    nan = np.array(0x7FC00000, dtype=np.uint32).view(np.float32)
    value = np.where(np.isnan(value), nan, value)
    return np.where((value > 0) | np.isnan(value), value, np.float32(floor))


@njit(cache=True)
def _quantize_smooth_shadows(
    rgb: np.ndarray,
    filtered: np.ndarray,
    shadow_threshold: int,
    shadow_softness: int,
) -> np.ndarray:
    """Blend smooth shadow protection and quantize in one serial pixel pass."""
    lower = max(0, shadow_threshold - shadow_softness)
    upper = min(255, shadow_threshold + shadow_softness)
    output = np.empty(rgb.shape, np.uint8)
    for y in range(rgb.shape[0]):
        for x in range(rgb.shape[1]):
            # Match the fractional mask's float64 operation order without fastmath.
            luminance = (
                4899 * np.float64(rgb[y, x, 0])
                + 9617 * np.float64(rgb[y, x, 1])
                + 1868 * np.float64(rgb[y, x, 2])
            ) / 16384
            if luminance <= lower:
                for channel in range(3):
                    output[y, x, channel] = rgb[y, x, channel]
                continue
            transition = luminance < upper
            weight = 1.0
            if transition:
                t = (luminance - lower) / (upper - lower)
                weight = t * t * (3 - 2 * t)
            for channel in range(3):
                value = filtered[y, x, channel]
                if transition:
                    # Preserve fmin's NaN-to-white rule before shadow blending.
                    if np.isnan(value) or value > 255:
                        value = 255.0
                    elif value < 0:
                        value = 0.0
                    original = np.float64(rgb[y, x, channel])
                    value = original + weight * (value - original)
                value = value + 0.5
                if np.isnan(value) or value > 255:
                    value = 255.0
                elif value < 0:
                    value = 0.0
                output[y, x, channel] = np.uint8(value)
    return output


@njit(cache=True)
def _compress_highlights(rgb: np.ndarray, strength: float) -> np.ndarray:
    """Roll bright RGB peaks off with a common gain before channel clipping."""
    result = np.empty_like(rgb)
    amount = strength / 100.0
    knee = 0.75 * 255.0
    headroom = 255.0 - knee
    for y in range(rgb.shape[0]):
        for x in range(rgb.shape[1]):
            peak = max(rgb[y, x, 0], rgb[y, x, 1], rgb[y, x, 2])
            finite = (np.isfinite(rgb[y, x, 0]) and np.isfinite(rgb[y, x, 1])
                      and np.isfinite(rgb[y, x, 2]))
            if finite and peak > knee and amount > 0:
                excess = peak - knee
                compressed = knee + headroom / (1.0 + headroom / excess)
                target = (1.0 - amount) * peak + amount * compressed
                for channel in range(3):
                    result[y, x, channel] = (rgb[y, x, channel] / peak) * target
            else:
                for channel in range(3):
                    result[y, x, channel] = rgb[y, x, channel]
    return result


def _recombine(
    rgb: np.ndarray,
    luminance: np.ndarray,
    mapped: np.ndarray,
    saturation: float,
    shadow_threshold: int = 6,
    power_mode: str = "approximate",
    output_gamma: float = 1.0,
    shadow_softness: int = 0,
    color_rgb: np.ndarray | None = None,
    shadow_luminance: np.ndarray | None = None,
    tone_strength: float = 100,
    highlight_compression: float = 0,
) -> np.ndarray:
    if tone_strength < 100:
        source_luminance = luminance if shadow_luminance is None else shadow_luminance
        if tone_strength == 0:
            mapped = source_luminance
        else:
            # Degenerate normalization maps to white on the reference path.
            target = np.nan_to_num(mapped, nan=1.0, posinf=1.0, neginf=0.0)
            amount = np.float32(tone_strength) / np.float32(100)
            mapped = source_luminance + amount * (target - source_luminance)
    # Keep color powers in float64 through final RGB scaling.
    original_scale = luminance * np.float32(255)
    ratio = np.divide(
        rgb.astype(np.float32) if color_rgb is None else color_rgb,
        original_scale[..., None],
        out=np.zeros(rgb.shape, np.float32),
        where=original_scale[..., None] > 0,
    )
    chroma = power64(ratio, np.float32(saturation) / np.float32(100), power_mode)
    result = chroma * (mapped * np.float32(255)).astype(np.float64)[..., None]
    if highlight_compression > 0:
        result = _compress_highlights(result, highlight_compression)
    if output_gamma != 1.0:
        # Apply the curve to filtered RGB before byte quantization and fade.
        unit = np.maximum(np.fmin(result, 255), 0) / 255
        result = np.power(unit, 1.0 / output_gamma) * 255
    if shadow_softness:
        return _quantize_smooth_shadows(rgb, result, shadow_threshold, shadow_softness)
    result = result + 0.5
    # fmin maps NaNs to the upper limit before byte quantization.
    result = np.maximum(np.fmin(result, 255), 0).astype(np.uint8)
    shadow_scale = original_scale if shadow_luminance is None else shadow_luminance * np.float32(255)
    result[shadow_scale <= shadow_threshold] = rgb[shadow_scale <= shadow_threshold]
    return result


def apply_core(
    array: np.ndarray, variant: int, params: FilterParams, numerics: NumericsConfig,
    *, source_cleanup=None,
) -> np.ndarray:
    """Apply the effect to decoded pixels with no file or resizing policies."""
    result = array.copy()
    if params.fade < 100:
        if min(array.shape[:2]) < 2:
            warnings.warn(
                "One-pixel-wide images use an identity extension of the solver",
                RuntimeWarning,
            )
        else:
            rgb = np.ascontiguousarray(array[..., :3])
            if numerics.luminance_mode == "fractional":
                luminance = luminance_fractional(rgb)
            else:
                luminance = luminance_u8(rgb).astype(np.float32) * np.float32(1 / 255)
            color_rgb = None
            shadow_luminance = None
            if source_cleanup is not None:
                shadow_luminance = luminance
                color_rgb, luminance = source_cleanup(rgb, luminance)
            maximum = np.max(luminance)
            if maximum > 0:
                # Add the offset in float64, then narrow before the scalar logarithm.
                log_source = scalar_math_f32(
                    (
                        (luminance * np.float32(100) / maximum).astype(np.float64)
                        + numerics.log_offset
                    ).astype(np.float32),
                    False,
                )
                pyramid = _pyramid(log_source, numerics.pyramid_min_size)
                phi = _phi(
                    pyramid,
                    variant,
                    params,
                    numerics.gradient_threshold,
                    numerics.detail_level,
                    numerics.power_mode,
                )
                solution = _sor(
                    _divergence(log_source, phi),
                    numerics.iterations,
                    numerics.relaxation,
                )
                mapped = _normalize(
                    scalar_math_f32(solution, True) + np.float32(-numerics.log_offset),
                    numerics.normalization_low,
                    numerics.normalization_high,
                    numerics.normalization_floor,
                )
                filtered = _recombine(
                    rgb,
                    luminance,
                    mapped,
                    params.saturation,
                    numerics.shadow_threshold,
                    numerics.power_mode,
                    params.output_gamma,
                    numerics.shadow_softness,
                    color_rgb,
                    shadow_luminance,
                    params.tone_strength,
                    params.highlight_compression,
                )
                # Opaque Normal Fade uses nearest rounding with downward ties.
                amount = np.float32(params.fade) / np.float32(100)
                mixed = (
                    filtered.astype(np.float32) * (np.float32(1) - amount)
                    + rgb.astype(np.float32) * amount
                )
                result[..., :3] = np.clip(
                    np.ceil(mixed - np.float32(0.5)), 0, 255
                ).astype(np.uint8)
    return result
