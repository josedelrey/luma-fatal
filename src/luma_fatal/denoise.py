"""Noise-adaptive spatial denoising before gradient amplification."""

from numbers import Real
import time

import numpy as np
from scipy.ndimage import convolve1d, uniform_filter


def _noise_sigma(rgb, backend=np):
    """Estimate RGB noise from diagonal differences that cancel planar shading."""
    if min(rgb.shape[:2]) < 2:
        return backend.zeros(3, backend.float32)
    diagonal = (rgb[1:, 1:] - rgb[1:, :-1]
                - rgb[:-1, 1:] + rgb[:-1, :-1])
    # The four independent samples give twice the source standard deviation.
    return backend.median(backend.abs(diagonal), axis=(0, 1)) / backend.float32(1.3489795)


def _bands():
    """Calibrate residual noise energy for the separable smoothing kernels."""
    previous = np.array([1.0])
    bands = []
    for step in (1, 2, 4):
        kernel = np.zeros(2 * step + 1)
        kernel[::step] = (0.25, 0.5, 0.25)
        current = np.convolve(previous, kernel)
        padded = np.pad(previous, step)
        energy = (np.dot(previous, previous) ** 2
                  + np.dot(current, current) ** 2
                  - 2 * np.dot(padded, current) ** 2)
        bands.append((kernel.astype(np.float32), np.float32(energy)))
        previous = current
    return bands


_BANDS = _bands()
_LUMA = np.array([4899, 9617, 1868], np.float32) / np.float32(16384 * 255)


def denoise_rgb(rgb, *, backend=np, filters=None):
    """Shrink noise-dominated detail while retaining locally coherent structure.

    Undecimated bands avoid block boundaries and retain full pixel alignment.
    Each channel uses its estimated noise floor. Local band energy retains
    edges and texture above that floor without applying a smooth-region mask.
    """
    convolve, uniform = filters or (convolve1d, uniform_filter)
    color = rgb.astype(backend.float32)
    sigma = _noise_sigma(color, backend)
    if not backend.any(sigma):
        return color, sigma
    coarse = color
    result = backend.zeros_like(color)
    for kernel, energy in _BANDS:
        kernel = backend.asarray(kernel)
        smooth = convolve(coarse, kernel, axis=0, mode="reflect")
        smooth = convolve(smooth, kernel, axis=1, mode="reflect")
        detail = coarse - smooth
        # Correlated grain can grow at coarser scales, unlike white noise.
        # Estimate each band separately so compressed/resampled noise is caught.
        center = backend.median(detail, axis=(0, 1))
        band_sigma = (
            backend.median(backend.abs(detail - center), axis=(0, 1)) / backend.float32(0.67448975)
        )
        noise = backend.maximum(sigma * sigma * energy, band_sigma * band_sigma)
        activity = uniform(detail * detail, size=(5, 5, 1), mode="reflect")
        gain = backend.maximum(activity - backend.float32(1.5) * noise, 0) / backend.maximum(
            activity, backend.maximum(noise, backend.float32(1e-12)),
        )
        result += detail * gain
        coarse = smooth
    return backend.clip(result + coarse, 0, 255), sigma


class SourceDenoise:
    """Denoise floating RGB/luminance, then run optional cleanup and source noise.

    Frames are independent, so motion and scene cuts cannot leave color trails.
    The core retains original pixels for shadow protection and Fade.
    """

    def __init__(self, strength, *, source_cleanup=None):
        if (isinstance(strength, bool) or not isinstance(strength, Real)
                or not np.isfinite(strength) or not 0 <= strength <= 1):
            raise ValueError("Denoise strength must be finite and between 0 and 1")
        self.strength = float(strength)
        self.cleanup = source_cleanup
        self.frames = 0
        self.seconds = 0.0

    def __call__(self, rgb, reference_luminance):
        start = time.perf_counter()
        color, luminance = rgb, reference_luminance
        if self.strength:
            candidate, sigma = denoise_rgb(rgb)
            if np.any(sigma):
                amount = np.float32(self.strength)
                color = rgb + amount * (candidate - rgb)
                luminance = reference_luminance + amount * (
                    np.sum(candidate * _LUMA, axis=2) - reference_luminance
                )
        self.frames += 1
        self.seconds += time.perf_counter() - start
        if self.cleanup is not None:
            return self.cleanup(color, luminance)
        return color, luminance
