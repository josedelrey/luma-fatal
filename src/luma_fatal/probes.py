"""Deterministic synthetic images for identifying image-processing behavior."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def _probe_arrays(size: int, seed: int) -> dict[str, tuple[np.ndarray, str]]:
    ramp = np.rint(np.linspace(0, 255, size)).astype(np.uint8)
    horizontal = np.broadcast_to(ramp[None, :, None], (size, size, 3)).copy()
    probes = {
        "ramp_gray_horizontal": (horizontal, "Horizontal grayscale ramp from 0 to 255"),
        "ramp_gray_vertical": (
            horizontal.transpose(1, 0, 2).copy(),
            "Vertical grayscale ramp from 0 to 255",
        ),
    }
    for index, channel in enumerate(("red", "green", "blue")):
        array = np.zeros_like(horizontal)
        array[:, :, index] = horizontal[:, :, index]
        probes[f"ramp_{channel}"] = (
            array,
            f"Horizontal {channel} ramp with the other channels zero",
        )
    for value in (0, 16, 32, 64, 96, 128, 160, 192, 224, 240, 255):
        probes[f"flat_gray_{value:03d}"] = (
            np.full((size, size, 3), value, dtype=np.uint8),
            f"Spatially constant neutral RGB ({value}, {value}, {value})",
        )
    patches = np.zeros((size, size, 3), dtype=np.uint8)
    colors = (
        (0, 0, 0),
        (255, 255, 255),
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (128, 128, 128),
        (128, 0, 0),
        (0, 128, 0),
        (0, 0, 128),
        (192, 128, 96),
        (240, 208, 176),
        (96, 64, 48),
        (32, 64, 96),
    )
    for index, color in enumerate(colors):
        row, column = divmod(index, 4)
        patches[
            row * size // 4 : (row + 1) * size // 4,
            column * size // 4 : (column + 1) * size // 4,
        ] = color
    probes["color_patches"] = (
        patches,
        "4x4 chart with primaries, secondary colors, neutrals and skin-like RGB colors",
    )
    cube = np.empty_like(patches)
    for row in range(16):
        for column in range(16):
            color = (
                round((column % 8) * 255 / 7),
                round((row % 8) * 255 / 7),
                (column // 8 + 2 * (row // 8)) * 85,
            )
            cube[
                row * size // 16 : (row + 1) * size // 16,
                column * size // 16 : (column + 1) * size // 16,
            ] = color
    probes["rgb_cube"] = (cube, "256 patches sampling an 8x8x4 RGB cube")
    impulse = np.zeros_like(patches)
    impulse[size // 2, size // 2] = 255
    impulse[size // 4, size // 4] = (255, 0, 0)
    impulse[size // 4, 3 * size // 4] = (0, 255, 0)
    impulse[3 * size // 4, size // 4] = (0, 0, 255)
    probes["impulses_black"] = (impulse, "Single-pixel white and RGB impulses on black")
    inverse_impulse = 255 - impulse
    probes["impulses_white"] = (
        inverse_impulse,
        "Single-pixel black and complementary-color impulses on white",
    )
    checker = np.zeros_like(patches)
    for index, period in enumerate((1, 2, 4, 8, 16, 32)):
        row, column = divmod(index, 3)
        y0, y1 = row * size // 2, (row + 1) * size // 2
        x0, x1 = column * size // 3, (column + 1) * size // 3
        yy, xx = np.indices((y1 - y0, x1 - x0))
        values = ((xx // period + yy // period) % 2 * 255).astype(np.uint8)
        checker[y0:y1, x0:x1] = values[:, :, None]
    probes["checker_multiscale"] = (
        checker,
        "Six checkerboard regions with square widths 1, 2, 4, 8, 16 and 32 pixels",
    )
    rng = np.random.Generator(np.random.PCG64(seed))
    noise = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    probes["noise_rgb"] = (
        noise,
        f"Independent uniformly distributed RGB noise from PCG64 seed {seed}",
    )
    gray_noise = rng.integers(0, 256, (size, size, 1), dtype=np.uint8)
    probes["noise_gray"] = (
        np.repeat(gray_noise, 3, axis=2),
        f"Uniform grayscale noise from the same PCG64 seed {seed}",
    )
    yy, xx = np.indices((size, size))
    scene = np.stack(
        (xx / (size - 1) * 255, yy / (size - 1) * 255, (1 - xx / (size - 1)) * 255),
        axis=2,
    )
    texture = np.sin(xx * 0.3) * np.cos(yy * 0.19) * 20
    scene = np.clip(np.rint(scene + texture[:, :, None]), 0, 255).astype(np.uint8)
    scene_image = Image.fromarray(scene)
    draw = ImageDraw.Draw(scene_image)
    draw.rectangle((size // 12, size // 12, size // 3, size // 3), fill=(32, 32, 32))
    draw.ellipse(
        (size // 2, size // 12, 11 * size // 12, 5 * size // 12), fill=(232, 180, 136)
    )
    draw.polygon(
        (
            (size // 12, 11 * size // 12),
            (size // 3, size // 2),
            (size // 2, 11 * size // 12),
        ),
        fill=(240, 240, 240),
    )
    for offset in range(0, size // 3, 8):
        draw.line(
            (
                2 * size // 3 + offset,
                size // 2,
                2 * size // 3 + offset,
                11 * size // 12,
            ),
            fill=(16, 64, 192),
            width=2,
        )
    probes["test_scene"] = (
        np.asarray(scene_image),
        "Composite gradients, sinusoidal texture, a neutral square, warm circle, neutral triangle and narrow blue lines",
    )
    return probes


def generate_probes(
    output_dir: str | Path,
    size: int = 512,
    seed: int = 3095,
) -> dict[str, Any]:
    """Write reproducible square RGB PNGs and evaluation instructions.

    Use size >=256 to retain spatial structure for multiscale filters. The
    default 512 preserves five pyramid levels down to 32 pixels. PNG
    files have no EXIF or ICC metadata. Existing probe files are overwritten.
    """
    if isinstance(size, bool) or not isinstance(size, int) or size < 256:
        raise ValueError("Probe size must be an integer of at least 256 pixels")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Probe seed must be a nonnegative integer")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, (array, description) in _probe_arrays(size, seed).items():
        filename = f"{name}.png"
        path = destination / filename
        Image.fromarray(array).save(path, format="PNG")
        entries.append(
            {
                "name": name,
                "file": filename,
                "description": description,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "width": size,
                "height": size,
            }
        )
    manifest = {
        "schema_version": 1,
        "size": size,
        "seed": seed,
        "pixel_format": "8-bit RGB PNG, no ICC profile or EXIF orientation",
        "probes": entries,
        "evaluation": {
            "required_variants": ["fattal_1", "fattal_2"],
            "full_effect_fade": 0,
            "instructions": [
                "Retain original inputs and their SHA-256 fingerprints.",
                "Use one explicit configuration per experiment.",
                "Keep encoded-file and decoded-pixel comparisons separate.",
                "Compare images at matching geometry without alignment or resampling.",
                "Record environment versions and repeat selected cases to check determinism.",
            ],
        },
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
