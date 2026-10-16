"""Optional source-guided cleanup with bounded, per-stream mask history."""

from __future__ import annotations

import time
from numbers import Real

import numpy as np
from numba import njit
from scipy.ndimage import gaussian_filter, maximum_filter, uniform_filter


def _smoothstep(low, high, values):
    values = np.clip((values - low) / (high - low), 0, 1)
    return values * values * (3 - 2 * values)


@njit(cache=True)
def _reduce(rgb, factor):
    height, width = rgb.shape[:2]
    small = np.empty(((height + factor - 1) // factor,
                      (width + factor - 1) // factor, 3), np.float32)
    activity = np.zeros(small.shape[:2], np.float32)
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
                deviation = np.sqrt(max(squares / count - (total / count) ** 2, 0.0))
                activity[y, x] = max(activity[y, x], deviation)
    return small, activity


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
    """Detect flat source regions and gently clean them at working resolution.

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
        start = time.perf_counter()
        factor = max(1, (max(rgb.shape[:2]) + 383) // 384)
        small, within_activity = _reduce(rgb, factor)
        guide_rgb = gaussian_filter(small, sigma=(0.5, 0.5, 0), mode="reflect")
        fine = guide_rgb - gaussian_filter(guide_rgb, sigma=(1, 1, 0), mode="reflect")
        coarse = guide_rgb - gaussian_filter(guide_rgb, sigma=(2, 2, 0), mode="reflect")
        fine_activity = np.sqrt(np.maximum(uniform_filter(fine * fine, (3, 3, 1)), 0)).max(2)
        coarse_activity = np.sqrt(np.maximum(uniform_filter(coarse * coarse, (3, 3, 1)), 0)).max(2)
        edge = np.maximum(
            np.abs(np.diff(guide_rgb, axis=0, append=guide_rgb[-1:])),
            np.abs(np.diff(guide_rgb, axis=1, append=guide_rgb[:, -1:])),
        ).max(2)
        edge_guard = ((1 - _smoothstep(3, 10, maximum_filter(edge, 3)))
                      * (1 - _smoothstep(1, 5, within_activity)))
        mask = ((1 - _smoothstep(0.7, 3, fine_activity))
                * (1 - _smoothstep(2, 7, coarse_activity)) * edge_guard)
        guide = (small[..., 0] * np.float32(4899 / 16384)
                 + small[..., 1] * np.float32(9617 / 16384)
                 + small[..., 2] * np.float32(1868 / 16384))
        mean = uniform_filter(guide, 5)
        variance = np.maximum(uniform_filter(guide * guide, 5) - mean * mean, 0)
        mean_mask = uniform_filter(mask, 5)
        a = (uniform_filter(guide * mask, 5) - mean * mean_mask) / (variance + 9)
        b = mean_mask - a * mean
        mask = np.minimum(
            np.clip(uniform_filter(a, 5) * guide + uniform_filter(b, 5), 0, 1), edge_guard
        )
        if self.shape == rgb.shape:
            change = np.abs(guide_rgb - self.guide).max(2)
            cut = np.mean(change) > 12 or np.mean(change > 12) > 0.3
            if not cut:
                persistence = np.float32(0.7) * (1 - _smoothstep(1, 6, change))
                mask = mask + persistence * (self.mask - mask)
            else:
                self.resets += 1
        elif self.shape is not None:
            self.resets += 1
        # Newly appearing edges immediately override mask history.
        mask = np.minimum(mask, edge_guard).astype(np.float32)
        self.guide, self.mask, self.shape = guide_rgb, mask, rgb.shape
        mean_rgb = uniform_filter(small, (5, 5, 1))
        cov = uniform_filter(guide[..., None] * small, (5, 5, 1)) - mean[..., None] * mean_rgb
        a = cov / (variance[..., None] + 16)
        b = mean_rgb - a * mean[..., None]
        fields = np.concatenate((mask[..., None], uniform_filter(a, (5, 5, 1)),
                                 uniform_filter(b, (5, 5, 1))), axis=2)
        result = _compose(rgb, reference_luminance, fields, factor, self.strength)
        self.frames += 1
        self.seconds += time.perf_counter() - start
        return result
