"""Strict pixel comparisons of exported RGB images.

No image registration, resizing, EXIF rotation, or color-profile conversion is
performed. Pillow converts images to RGB, discarding any alpha channel. Metrics
use encoded 8-bit RGB values, rather than linear-light or perceptual values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

ImageInput = str | Path | Image.Image | np.ndarray


def load_rgb(image: ImageInput) -> np.ndarray:
    """Return an H×W×3 uint8 copy, without modifying image geometry."""
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                "Image arrays must have shape (height, width, 3) and dtype uint8"
            )
        if image.shape[0] == 0 or image.shape[1] == 0:
            raise ValueError("Images must have nonzero width and height")
        return image.copy()
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    with Image.open(image) as opened:
        return np.asarray(opened.convert("RGB"), dtype=np.uint8).copy()


def _same_geometry(*images: np.ndarray) -> None:
    if any(image.shape != images[0].shape for image in images[1:]):
        shapes = ", ".join(str(image.shape) for image in images)
        raise ValueError(f"Image dimensions must match exactly, received {shapes}")


def _ssim(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Compute per-channel SSIM using Gaussian population moments.

    This is a local RGB SSIM variant. Each channel uses a Gaussian window with
    sigma 1.5, radius 5, reflect boundaries, and constants (0.01*255)^2 and
    (0.03*255)^2. The spatial average includes boundary pixels. Covariance uses
    population normalization. The three channel means receive equal weight.
    """
    sigma = (1.5, 1.5, 0)
    mean_reference = gaussian_filter(reference, sigma, mode="reflect", truncate=3.5)
    mean_candidate = gaussian_filter(candidate, sigma, mode="reflect", truncate=3.5)
    variance_reference = np.maximum(
        gaussian_filter(reference * reference, sigma, mode="reflect", truncate=3.5)
        - mean_reference * mean_reference,
        0,
    )
    variance_candidate = np.maximum(
        gaussian_filter(candidate * candidate, sigma, mode="reflect", truncate=3.5)
        - mean_candidate * mean_candidate,
        0,
    )
    covariance = (
        gaussian_filter(reference * candidate, sigma, mode="reflect", truncate=3.5)
        - mean_reference * mean_candidate
    )
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    numerator = (2 * mean_reference * mean_candidate + c1) * (2 * covariance + c2)
    denominator = (
        mean_reference * mean_reference + mean_candidate * mean_candidate + c1
    ) * (variance_reference + variance_candidate + c2)
    return np.mean(numerator / denominator, axis=(0, 1))


def compare_images(reference: ImageInput, candidate: ImageInput) -> dict[str, Any]:
    """Return JSON-safe metrics for strictly corresponding RGB pixels.

    MAE, RMSE and maximum errors are in 8-bit code values. PSNR uses a peak
    value of 255. An exact match has psnr_db=None and psnr_infinite=True.
    exact_pixel_fraction requires all three channels to match at a pixel.
    exact_channel_fraction counts matching individual RGB channel samples.
    """
    reference_rgb = load_rgb(reference)
    candidate_rgb = load_rgb(candidate)
    _same_geometry(reference_rgb, candidate_rgb)
    reference_float = reference_rgb.astype(np.float64)
    candidate_float = candidate_rgb.astype(np.float64)
    delta = candidate_float - reference_float
    absolute = np.abs(delta)
    squared = delta * delta
    mse = float(np.mean(squared))
    ssim_channels = _ssim(reference_float, candidate_float)
    return {
        "width": int(reference_rgb.shape[1]),
        "height": int(reference_rgb.shape[0]),
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(mse)),
        "max_abs_error": int(np.max(absolute)),
        "psnr_db": float(10 * np.log10(255**2 / mse)) if mse else None,
        "psnr_infinite": mse == 0,
        "ssim": float(np.mean(ssim_channels)),
        "exact_pixel_fraction": float(np.mean(np.all(delta == 0, axis=2))),
        "exact_channel_fraction": float(np.mean(delta == 0)),
        "per_channel": {
            name: {
                "mae": float(np.mean(absolute[:, :, index])),
                "rmse": float(np.sqrt(np.mean(squared[:, :, index]))),
                "max_abs_error": int(np.max(absolute[:, :, index])),
                "mean_signed_error": float(np.mean(delta[:, :, index])),
                "ssim": float(ssim_channels[index]),
                "exact_fraction": float(np.mean(delta[:, :, index] == 0)),
            }
            for index, name in enumerate(("red", "green", "blue"))
        },
        "metric_definition": {
            "space": "8-bit encoded RGB, no ICC transform or alpha compositing",
            "ssim": "RGB mean, Gaussian sigma=1.5 radius=5, population moments, reflect borders, full-image mean",
            "geometry": "strict equal dimensions, no registration or resampling",
            "signed_error": "candidate minus reference",
        },
    }


def write_difference_panel(
    original: ImageInput,
    reference: ImageInput,
    candidate: ImageInput,
    output_path: str | Path,
    gain: float = 8,
) -> Path:
    """Save input, reference, candidate, and amplified absolute RGB error.

    Panels retain full resolution. The last panel uses round-and-clip of
    abs(candidate-reference)*gain. A 28-pixel label strip is added above them.
    """
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError("Difference amplification must be finite and positive")
    arrays = [load_rgb(image) for image in (original, reference, candidate)]
    _same_geometry(*arrays)
    difference = np.abs(arrays[2].astype(np.float64) - arrays[1].astype(np.float64))
    arrays.append(np.clip(np.rint(difference * gain), 0, 255).astype(np.uint8))
    height, width = arrays[0].shape[:2]
    panel = Image.new("RGB", (4 * width, height + 28), (24, 24, 24))
    draw = ImageDraw.Draw(panel)
    labels = ("Original", "Reference", "Candidate", f"Absolute difference x{gain:g}")
    for index, (array, label) in enumerate(zip(arrays, labels, strict=True)):
        panel.paste(Image.fromarray(array), (index * width, 28))
        draw.text((index * width + 6, 7), label, fill=(255, 255, 255))
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    panel.save(destination, format="PNG")
    return destination
