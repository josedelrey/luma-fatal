"""Optional source-guided cleanup with bounded, per-stream mask history."""

from __future__ import annotations

import time
from numbers import Real

import numpy as np
from numba import njit
from scipy.ndimage import gaussian_filter, map_coordinates, maximum_filter, uniform_filter


def _smoothstep(low, high, values):
    values = np.clip((values - low) / (high - low), 0, 1)
    return values * values * (3 - 2 * values)


@njit(cache=True)
def _reduce(rgb, factor):
    height, width = rgb.shape[:2]
    small = np.empty(((height + factor - 1) // factor,
                      (width + factor - 1) // factor, 3), np.float32)
    variance = np.empty(small.shape, np.float32)
    for y in range(small.shape[0]):
        for x in range(small.shape[1]):
            bottom = min((y + 1) * factor, height)
            right = min((x + 1) * factor, width)
            count = (bottom - y * factor) * (right - x * factor)
            for c in range(3):
                total = 0.0
                squares = 0.0
                for yy in range(y * factor, bottom):
                    for xx in range(x * factor, right):
                        value = float(rgb[yy, xx, c])
                        total += value
                        squares += value * value
                small[y, x, c] = total / count
                variance[y, x, c] = max(squares / count - (total / count) ** 2, 0.0)
    return small, variance


@njit(cache=True)
def _detrended_activity(small, variance, factor, height, width):
    """Subtract smooth planar shading from within-tile texture energy.

    Neighboring tile means estimate the shading slope. Using those means keeps
    aliased fine stripes protected even when each tile itself looks linear.
    Partial border tiles use their actual centers and spatial variances.
    """
    activity = np.empty(small.shape[:2], np.float32)
    for y in range(small.shape[0]):
        top = max(y - 1, 0)
        bottom = min(y + 1, small.shape[0] - 1)
        top_center = (top * factor + min((top + 1) * factor, height) - 1) * 0.5
        bottom_center = (bottom * factor + min((bottom + 1) * factor, height) - 1) * 0.5
        tile_height = min((y + 1) * factor, height) - y * factor
        spatial_y = (tile_height * tile_height - 1) / 12.0
        for x in range(small.shape[1]):
            left = max(x - 1, 0)
            right = min(x + 1, small.shape[1] - 1)
            left_center = (left * factor + min((left + 1) * factor, width) - 1) * 0.5
            right_center = (right * factor + min((right + 1) * factor, width) - 1) * 0.5
            tile_width = min((x + 1) * factor, width) - x * factor
            spatial_x = (tile_width * tile_width - 1) / 12.0
            residual = 0.0
            for c in range(3):
                dx = 0.0 if left == right else (
                    (small[y, right, c] - small[y, left, c]) / (right_center - left_center)
                )
                dy = 0.0 if top == bottom else (
                    (small[bottom, x, c] - small[top, x, c]) / (bottom_center - top_center)
                )
                shading = dx * dx * spatial_x + dy * dy * spatial_y
                residual = max(residual, variance[y, x, c] - shading)
            activity[y, x] = np.sqrt(max(residual, 0.0))
    return activity


@njit(cache=True)
def _compose(rgb, reference_luminance, fields, factor, strength):
    """Upsample coefficients and blend in one pass without RGB8 requantization."""
    height, width = rgb.shape[:2]
    clean = np.empty(rgb.shape, np.float32)
    luminance = np.empty((height, width), np.float32)
    for y in range(height):
        fy = min(max((y + 0.5) / factor - 0.5, 0.0), fields.shape[0] - 1.0)
        y0 = int(fy)
        y1 = min(y0 + 1, fields.shape[0] - 1)
        wy = fy - y0
        for x in range(width):
            fx = min(max((x + 0.5) / factor - 0.5, 0.0), fields.shape[1] - 1.0)
            x0 = int(fx)
            x1 = min(x0 + 1, fields.shape[1] - 1)
            wx = fx - x0
            weights = ((1 - wy) * (1 - wx), (1 - wy) * wx,
                       wy * (1 - wx), wy * wx)
            mask = (fields[y0, x0, 0] * weights[0]
                    + fields[y0, x1, 0] * weights[1]
                    + fields[y1, x0, 0] * weights[2]
                    + fields[y1, x1, 0] * weights[3])
            amount = strength * min(max(mask, 0.0), 1.0)
            if amount == 0:
                for c in range(3):
                    clean[y, x, c] = rgb[y, x, c]
                luminance[y, x] = reference_luminance[y, x]
                continue
            guide = (4899.0 * rgb[y, x, 0] + 9617.0 * rgb[y, x, 1]
                     + 1868.0 * rgb[y, x, 2]) / 16384.0
            candidate_luminance = 0.0
            for c in range(3):
                a = (fields[y0, x0, c + 1] * weights[0]
                     + fields[y0, x1, c + 1] * weights[1]
                     + fields[y1, x0, c + 1] * weights[2]
                     + fields[y1, x1, c + 1] * weights[3])
                b = (fields[y0, x0, c + 4] * weights[0]
                     + fields[y0, x1, c + 4] * weights[1]
                     + fields[y1, x0, c + 4] * weights[2]
                     + fields[y1, x1, c + 4] * weights[3])
                original = float(rgb[y, x, c])
                # Bound damage from an uncertain mask near low-contrast detail.
                candidate = min(max(a * guide + b, original - 4.0), original + 4.0)
                candidate = min(max(candidate, 0.0), 255.0)
                clean[y, x, c] = original + amount * (candidate - original)
                candidate_luminance += candidate * (4899 if c == 0 else 9617 if c == 1 else 1868)
            reference = reference_luminance[y, x]
            luminance[y, x] = reference + amount * (
                candidate_luminance / (16384.0 * 255.0) - reference
            )
    return clean, luminance


class SourceCleanup:
    """Apply guided correction to flat source regions at working resolution.

    Strength is in [0, 1]. Detection and guided coefficients use an area-reduced
    image with longest edge at most 384 pixels. History contains only that small
    RGB guide and mask. RGB frames are never temporally averaged. The mask is
    exposed at reduced resolution for diagnostics, with white selecting cleanup.
    """

    def __init__(self, strength):
        if (isinstance(strength, bool) or not isinstance(strength, Real)
                or not np.isfinite(strength) or not 0 <= strength <= 1):
            raise ValueError("Cleanup strength must be finite and between 0 and 1")
        self.strength = float(strength)
        self.guide = None
        self.mask = None
        self.shape = None
        self.frames = 0
        self.resets = 0
        self.seconds = 0.0

    def __call__(self, rgb, reference_luminance):
        fields, factor = self.prepare(rgb)
        start = time.perf_counter()
        result = _compose(rgb, reference_luminance, fields, factor, self.strength)
        self.seconds += time.perf_counter() - start
        return result

    def prepare(self, rgb):
        """Update guide/mask history and return reduced cleanup coefficients.

        Device backends can compose these fields without downloading a full
        floating RGB frame. One call advances the history by one source frame.
        """
        start = time.perf_counter()
        factor = max(1, (max(rgb.shape[:2]) + 383) // 384)
        small, variance_rgb = _reduce(rgb, factor)
        within_activity = _detrended_activity(
            small, variance_rgb, factor, rgb.shape[0], rgb.shape[1]
        )
        fields = self.prepare_fields(small, within_activity, rgb.shape)
        self.seconds += time.perf_counter() - start
        return fields, factor

    def prepare_fields(self, small, within_activity, shape, *, backend=np, filters=None,
                       device_decisions=False):
        """Build reduced guide fields using NumPy or a compatible device backend."""
        gaussian, uniform, maximum = filters or (gaussian_filter, uniform_filter, maximum_filter)

        def smoothstep(low, high, values):
            values = backend.clip((values - low) / (high - low), 0, 1)
            return values * values * (3 - 2 * values)

        guide_rgb = gaussian(small, sigma=(0.5, 0.5, 0), mode="reflect")
        fine = guide_rgb - gaussian(guide_rgb, sigma=(1, 1, 0), mode="reflect")
        coarse = guide_rgb - gaussian(guide_rgb, sigma=(2, 2, 0), mode="reflect")
        fine_activity = backend.sqrt(backend.maximum(uniform(fine * fine, (3, 3, 1)), 0)).max(2)
        coarse_activity = backend.sqrt(backend.maximum(uniform(coarse * coarse, (3, 3, 1)), 0)).max(2)
        # High-pass gradients reject abrupt edges without penalizing a smooth slope.
        edge = backend.float32(2) * backend.maximum(
            backend.abs(backend.diff(fine, axis=0, append=fine[-1:])),
            backend.abs(backend.diff(fine, axis=1, append=fine[:, -1:])),
        ).max(2)
        edge_guard = ((1 - smoothstep(3, 10, maximum(edge, 3)))
                      * (1 - smoothstep(1, 5, within_activity)))
        mask = ((1 - smoothstep(0.7, 3, fine_activity))
                * (1 - smoothstep(2, 7, coarse_activity)) * edge_guard)
        guide = (small[..., 0] * backend.float32(4899 / 16384)
                 + small[..., 1] * backend.float32(9617 / 16384)
                 + small[..., 2] * backend.float32(1868 / 16384))
        mean = uniform(guide, 5)
        variance = backend.maximum(uniform(guide * guide, 5) - mean * mean, 0)
        mean_mask = uniform(mask, 5)
        a = (uniform(guide * mask, 5) - mean * mean_mask) / (variance + 9)
        b = mean_mask - a * mean
        mask = backend.minimum(
            backend.clip(uniform(a, 5) * guide + uniform(b, 5), 0, 1), edge_guard
        )
        if self.shape == shape:
            change = backend.abs(guide_rgb - self.guide).max(2)
            if device_decisions:
                cut = (backend.mean(change) > 12) | (backend.mean(change > 12) > 0.3)
                persistence = backend.float32(0.7) * (1 - smoothstep(1, 6, change))
                mask = backend.where(cut, mask, mask + persistence * (self.mask - mask))
                self.resets = backend.asarray(self.resets, dtype=backend.int64) + cut
            elif not (backend.mean(change) > 12 or backend.mean(change > 12) > 0.3):
                persistence = backend.float32(0.7) * (1 - smoothstep(1, 6, change))
                mask = mask + persistence * (self.mask - mask)
            else:
                self.resets += 1
        elif self.shape is not None:
            self.resets += 1
        # Newly appearing edges immediately override mask history.
        mask = backend.minimum(mask, edge_guard).astype(backend.float32)
        self.guide, self.mask, self.shape = guide_rgb, mask, shape
        mean_rgb = uniform(small, (5, 5, 1))
        cov = uniform(guide[..., None] * small, (5, 5, 1)) - mean[..., None] * mean_rgb
        a = cov / (variance[..., None] + 16)
        b = mean_rgb - a * mean[..., None]
        fields = backend.concatenate((mask[..., None], uniform(a, (5, 5, 1)),
                                 uniform(b, (5, 5, 1))), axis=2)
        self.frames += 1
        return fields


class MultiscaleSourceCleanup:
    """Split masked source correction into broad and fine components.

    The existing full-strength cleanup supplies a bounded correction and soft
    mask. A normalized Gaussian split gives broad and fine strengths without
    processing independent patches. The mask retains its existing history.
    Sigma is measured in effect working pixels.
    """

    def __init__(self, fine_strength=0.75, coarse_strength=1.0, sigma=1.2):
        for name, value in (("fine_strength", fine_strength),
                            ("coarse_strength", coarse_strength)):
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not np.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f"{name} must be finite and between 0 and 1")
        if (isinstance(sigma, bool) or not isinstance(sigma, Real)
                or not np.isfinite(sigma) or not 0.01 <= sigma <= 16):
            raise ValueError("sigma must be finite and between 0.01 and 16")
        self.fine_strength = float(fine_strength)
        self.coarse_strength = float(coarse_strength)
        self.sigma = float(sigma)
        self.cleanup = SourceCleanup(1.0)
        self.mask = None
        self.seconds = 0.0

    @property
    def frames(self):
        return self.cleanup.frames

    @property
    def resets(self):
        return self.cleanup.resets

    def __call__(self, rgb, reference_luminance):
        start = time.perf_counter()
        clean_rgb, clean_luminance = self.cleanup(rgb, reference_luminance)
        height, width = rgb.shape[:2]
        factor = max(1, (max(height, width) + 383) // 384)
        y, x = np.meshgrid(
            (np.arange(height, dtype=np.float32) + 0.5) / factor - 0.5,
            (np.arange(width, dtype=np.float32) + 0.5) / factor - 0.5,
            indexing="ij",
        )
        self.mask = np.clip(map_coordinates(
            self.cleanup.mask, (y, x), order=1, mode="nearest", prefilter=False,
        ), 0, 1)
        original = np.concatenate((
            rgb.astype(np.float32), (reference_luminance * 255)[..., None],
        ), axis=2)
        cleaned = np.concatenate((
            clean_rgb, (clean_luminance * 255)[..., None],
        ), axis=2)
        correction = cleaned - original
        denominator = np.maximum(
            gaussian_filter(self.mask, self.sigma, mode="reflect"), 1e-6,
        )
        broad = gaussian_filter(
            correction, (self.sigma, self.sigma, 0), mode="reflect",
        ) * (self.mask / denominator)[..., None]
        correction = (
            self.fine_strength * correction
            + (self.coarse_strength - self.fine_strength) * broad
        )
        # Luminance also accounts for the original half-byte rounding error.
        limit = self.mask[..., None] * np.array([4, 4, 4, 4.5], np.float32)
        result = np.clip(original + np.clip(correction, -limit, limit), 0, 255)
        result_rgb = result[..., :3].copy()
        result_luminance = result[..., 3] / np.float32(255)
        # Preserve zero-mask pixels exactly, including rounded luminance.
        protected = self.mask == 0
        result_rgb[protected] = rgb[protected]
        result_luminance[protected] = reference_luminance[protected]
        if self.fine_strength == self.coarse_strength == 0:
            output = rgb.astype(np.float32), reference_luminance.copy()
        else:
            output = result_rgb, result_luminance
        self.seconds += time.perf_counter() - start
        return output
