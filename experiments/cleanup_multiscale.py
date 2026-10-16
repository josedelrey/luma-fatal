"""Experimental scale-separated source cleanup, outside the production pipeline.

Use the existing full-strength cleanup to estimate a bounded correction D and
its continuous mask m. Split that correction with normalized Gaussian filtering:

    broad = m * G_sigma(D) / max(G_sigma(m), epsilon)
    correction = fine_strength * D + (coarse_strength - fine_strength) * broad

This increases correction of broad variations while retaining more fine residual
than uniform full-strength cleanup. It does not classify compression artifacts.
Small steps can occur in both frequency bands, and Fattal is nonlinear.

The Gaussian operates continuously over the whole working frame. There are no
independently processed patches. The original cleanup owns mask history, with
its existing motion and scene-cut behavior. Additional correction is bounded
and masked again so it cannot modify source pixels with zero cleanup weight.

Sigma is measured in effect working pixels, not decoded video pixels. This
research implementation allocates full-frame float fields and is not optimized.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from luma_fatal.cleanup import SourceCleanup


class ScaleSeparatedCleanup:
    """Keep separate cleanup strengths for broad and fine source variations."""

    def __init__(self, fine_strength=0.75, coarse_strength=1.0, sigma=1.2):
        for name, value in (("fine_strength", fine_strength),
                            ("coarse_strength", coarse_strength)):
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and between 0 and 1")
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("sigma must be positive and finite")
        self.fine_strength = float(fine_strength)
        self.coarse_strength = float(coarse_strength)
        self.sigma = float(sigma)
        self.cleanup = SourceCleanup(1.0)
        self.mask = None

    def __call__(self, rgb, reference_luminance):
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
            return rgb.astype(np.float32), reference_luminance.copy()
        return result_rgb, result_luminance
