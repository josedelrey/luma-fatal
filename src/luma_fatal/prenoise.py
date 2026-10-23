"""Optional masked noise in floating RGB/luminance before Fattal.

Noise is fixed in working-frame coordinates. It is deterministic, but is not
motion compensated. RMS describes the field before masking and range limits.
"""

from numbers import Real
import time

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from luma_fatal.cleanup import SourceCleanup


def noise_field(shape, kind, seed):
    """Make unit-RMS high-pass noise or an fBm-like Gaussian octave sum."""
    rng = np.random.default_rng(seed)
    if kind == "highpass":
        white = rng.standard_normal(shape, dtype=np.float32)
        field = white - gaussian_filter(white, 0.8, mode="reflect")
    elif kind == "fbm":
        field = np.zeros(shape, np.float32)
        for sigma, weight in ((4, 1), (2, 0.5), (1, 0.25), (0.5, 0.125)):
            octave = gaussian_filter(
                rng.standard_normal(shape, dtype=np.float32), sigma, mode="reflect",
            )
            octave -= octave.mean()
            rms = np.sqrt(np.mean(octave * octave))
            if rms > 0:
                field += np.float32(weight) * octave / rms
    else:
        raise ValueError("kind must be highpass or fbm")
    field -= field.mean()
    rms = np.sqrt(np.mean(field * field))
    return field / rms if rms > 0 else field


class SourceNoise:
    """Add achromatic noise under the existing smooth-source mask.

    The same delta enters all RGB channels and luminance, with a common range
    limit to avoid colored clipping. Noise never passes through RGB8 quantization.
    The mask sees incoming pixels before cleanup and noise injection.
    Original source pixels still control core shadow protection and Fade.
    """

    def __init__(self, rms=0.5, kind="highpass", cleanup=0, masked=True, seed=20261017,
                 *, source_cleanup=None):
        if (isinstance(rms, bool) or not isinstance(rms, Real)
                or not np.isfinite(rms) or not 0 <= rms <= 4):
            raise ValueError("rms must be finite and between 0 and 4")
        if kind not in ("highpass", "fbm"):
            raise ValueError("kind must be highpass or fbm")
        if not isinstance(masked, bool):
            raise ValueError("masked must be boolean")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.rms, self.kind, self.masked, self.seed = float(rms), kind, masked, seed
        self.cleanup = SourceCleanup(cleanup) if source_cleanup is None else source_cleanup
        self.field = None
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
        color, luminance = self.cleanup(rgb, reference_luminance)
        height, width = rgb.shape[:2]
        if self.masked:
            factor = (1 if self.cleanup.mask.shape == (height, width)
                      else max(1, (max(height, width) + 383) // 384))
            y, x = np.meshgrid(
                (np.arange(height, dtype=np.float32) + 0.5) / factor - 0.5,
                (np.arange(width, dtype=np.float32) + 0.5) / factor - 0.5,
                indexing="ij",
            )
            self.mask = np.clip(map_coordinates(
                self.cleanup.mask, (y, x), order=1, mode="nearest", prefilter=False,
            ), 0, 1)
        else:
            self.mask = np.ones((height, width), np.float32)
        if self.rms == 0:
            self.seconds += time.perf_counter() - start
            return color, luminance
        if self.field is None or self.field.shape != (height, width):
            self.field = noise_field((height, width), self.kind, self.seed)
        delta = self.field * np.float32(self.rms) * self.mask
        # Bound perturbations to four source code values even in the tails.
        limit = np.float32(4) * self.mask
        lower = np.maximum(-color.min(2), -luminance * np.float32(255))
        upper = np.minimum(255 - color.max(2), (1 - luminance) * np.float32(255))
        delta = np.clip(delta, np.maximum(lower, -limit), np.minimum(upper, limit))
        result = color + delta[..., None], luminance + delta / np.float32(255)
        self.seconds += time.perf_counter() - start
        return result
