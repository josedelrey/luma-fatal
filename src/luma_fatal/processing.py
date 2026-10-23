"""Image-area budgets, working resolution and deterministic resampling."""

from __future__ import annotations

from dataclasses import replace
from .configuration import ProcessingConfig, NumericsConfig

import numpy as np


def input_size(size: tuple[int, int], max_pixels: int | None) -> tuple[int, int]:
    """Fit an image to a pixel-area budget using float64 and truncation."""
    width, height = size
    if max_pixels is None or width * height <= max_pixels:
        return size
    scaled_height = np.sqrt(max_pixels * float(height) / float(width))
    return max(1, int(width * scaled_height / height)), max(1, int(scaled_height))


def limit_input_pixels(array: np.ndarray, max_pixels: int | None) -> np.ndarray:
    """Reduce opaque decoded pixels through staged bilinear area fitting."""
    source_size = (array.shape[1], array.shape[0])
    destination = input_size(source_size, max_pixels)
    if destination == source_size:
        return array
    if array.shape[2] == 4 and np.any(array[..., 3] != 255):
        raise ValueError(
            "Input pixel limiting requires opaque pixels. Disable limit_input_pixels for straight alpha"
        )
    rgb = array[..., :3]
    width, height = source_size
    target_area = destination[0] * destination[1]
    while float(np.float32(width * height) * np.float32(0.25)) >= target_area:
        width, height = max(1, width // 2), max(1, height // 2)
        rgb = resize_bilinear_fixed(rgb, (width, height))
    rgb = resize_bilinear_fixed(rgb, destination)
    if array.shape[2] == 3:
        return rgb
    return np.concatenate((rgb, np.full((*rgb.shape[:2], 1), 255, np.uint8)), axis=2)


def working_size(size: tuple[int, int], max_edge: int | None = 1536) -> tuple[int, int]:
    """Fit the longest edge using float32 scaling, rounding positive half ties up."""
    width, height = size
    if max_edge is None or max(size) <= max_edge:
        return size
    scale = min(
        np.float32(max_edge) / np.float32(width),
        np.float32(max_edge) / np.float32(height),
    )
    # Widen before +0.5 so addition does not round a near-tie in float32.
    return tuple(
        max(1, int(np.floor(float(np.float32(dimension) * scale) + 0.5)))
        for dimension in (width, height)
    )


def nearest_coordinates(source_size: tuple[int, int], size: tuple[int, int]):
    """Select half-pixel source coordinates with a fixed tie bias.

    Uses float32 horizontally and float64 vertically, subtracting 2**-15
    before flooring.
    """
    width, height = size
    x = (np.arange(width, dtype=np.float32) + np.float32(0.5)) * np.float32(
        source_size[0] / width
    )
    y = (np.arange(height, dtype=np.float64) + 0.5) * (source_size[1] / height)
    ix = np.floor(x - np.float32(2**-15)).astype(np.int64)
    iy = np.floor(y - 2**-15).astype(np.int64)
    return np.clip(ix, 0, source_size[0] - 1), np.clip(iy, 0, source_size[1] - 1)


def resize_nearest(rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    x, y = nearest_coordinates((rgb.shape[1], rgb.shape[0]), size)
    return np.ascontiguousarray(rgb[y[:, None], x[None, :]])


def resize_linear_floor(rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Half-pixel bilinear RGB8 interpolation with final truncation.

    Intermediate interpolation uses float64 and the final RGB8 conversion
    truncates fractional values.
    """
    width, height = size
    if (rgb.shape[1], rgb.shape[0]) == size:
        return rgb.copy()
    x = np.clip(
        (np.arange(width, dtype=np.float64) + 0.5) * (rgb.shape[1] / width) - 0.5,
        0,
        rgb.shape[1] - 1,
    )
    y = np.clip(
        (np.arange(height, dtype=np.float64) + 0.5) * (rgb.shape[0] / height) - 0.5,
        0,
        rgb.shape[0] - 1,
    )
    ix, iy = x.astype(np.int64), y.astype(np.int64)
    nx, ny = np.minimum(ix + 1, rgb.shape[1] - 1), np.minimum(iy + 1, rgb.shape[0] - 1)
    wx, wy = x - ix, y - iy
    result = np.empty((height, width, 3), np.uint8)
    for start in range(0, height, 64):
        end = min(height, start + 64)
        a = rgb[iy[start:end, None], ix[None, :]].astype(np.float64)
        b = rgb[iy[start:end, None], nx[None, :]].astype(np.float64)
        c = rgb[ny[start:end, None], ix[None, :]].astype(np.float64)
        d = rgb[ny[start:end, None], nx[None, :]].astype(np.float64)
        top = a * (1 - wx[None, :, None]) + b * wx[None, :, None]
        bottom = c * (1 - wx[None, :, None]) + d * wx[None, :, None]
        value = (
            top * (1 - wy[start:end, None, None]) + bottom * wy[start:end, None, None]
        )
        result[start:end] = np.clip(value, 0, 255).astype(np.uint8)
    return result


def resize_working(
    rgb: np.ndarray, size: tuple[int, int], method: str = "nearest"
) -> np.ndarray:
    """Select working pixels without an antialias prefilter.

    Bilinear uses half-pixel coordinates and truncates to RGB8. Bicubic uses
    the scalar-field kernel (a = -0.75), clamps overshoot and truncates to RGB8.
    Nearest uses half-pixel coordinates with a fixed tie bias.
    """
    if method == "nearest":
        return resize_nearest(rgb, size)
    if method == "bilinear":
        return resize_linear_floor(rgb, size)
    if method == "bicubic":
        from .core import _resize_cubic

        shape = (size[1], size[0])
        channels = [
            _resize_cubic(rgb[..., channel].astype(np.float32), shape)
            for channel in range(3)
        ]
        return np.clip(np.stack(channels, axis=-1), 0, 255).astype(np.uint8)
    raise ValueError("working_resampler must be nearest, bilinear or bicubic")


def _filtered_axis(source: int, destination: int, *, horizontal: bool):
    """Compute clamped coordinates with float32 scaling and 32.32 stepping."""
    inverse = np.float32(1 / np.float32(destination / source))
    index = np.arange(destination, dtype=np.int64)
    if horizontal:
        # A 128-word coordinate buffer reserves one word for the vertical pair.
        start = index // 127 * 127
        initial = ((start.astype(np.float32) + np.float32(0.5)) * inverse).astype(
            np.float64
        )
        fixed = (initial * 2**32).astype(np.int64) + (index - start) * int(
            float(inverse) * 2**32
        )
    else:
        centers = (index.astype(np.float32) + np.float32(0.5)) * inverse
        fixed = (centers.astype(np.float64) * 2**32).astype(np.int64)
    fixed -= 2**31
    lower = fixed >> 32
    weight = ((fixed >> 28) & 15).astype(np.int32)
    return np.clip(lower, 0, source - 1), np.clip(lower + 1, 0, source - 1), weight


def resize_bilinear_fixed(rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resample RGB8 with four-bit bilinear weights and integer quantization.

    Coordinates use float32 inverse scaling and 32.32 horizontal stepping
    in blocks of 127 pixels, with clamping at image edges.
    """
    width, height = size
    if (rgb.shape[1], rgb.shape[0]) == size:
        return rgb.copy()
    ix, nx, wx = _filtered_axis(rgb.shape[1], width, horizontal=True)
    iy, ny, wy = _filtered_axis(rgb.shape[0], height, horizontal=False)
    result = np.empty((height, width, 3), np.uint8)
    for start in range(0, height, 64):
        end = min(height, start + 64)
        a = rgb[iy[start:end, None], ix[None, :]].astype(np.int32)
        b = rgb[iy[start:end, None], nx[None, :]].astype(np.int32)
        c = rgb[ny[start:end, None], ix[None, :]].astype(np.int32)
        d = rgb[ny[start:end, None], nx[None, :]].astype(np.int32)
        top = a * (16 - wx[None, :, None]) + b * wx[None, :, None]
        bottom = c * (16 - wx[None, :, None]) + d * wx[None, :, None]
        result[start:end] = (
            (
                top * (16 - wy[start:end, None, None])
                + bottom * wy[start:end, None, None]
            )
            >> 8
        ).astype(np.uint8)
    return result


def enlargement_sizes(source_size: tuple[int, int], size: tuple[int, int]):
    """Progressive enlargement doubles dimensions before the final scale."""
    width, height = source_size
    result = []
    while width * 2 <= size[0] and height * 2 <= size[1]:
        width, height = width * 2, height * 2
        result.append((width, height))
    if (width, height) != size:
        result.append(size)
    return result


def enlarge_progressive(rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    result = rgb
    for intermediate in enlargement_sizes((rgb.shape[1], rgb.shape[0]), size):
        result = resize_bilinear_fixed(result, intermediate)
    return result


def process_pixels(
    array,
    variant,
    params,
    processing: ProcessingConfig | None = None,
    numerics: NumericsConfig | None = None,
    *, source_cleanup=None, solver=None,
):
    """Apply area fitting, working-resolution filtering and progressive enlargement."""
    from .core import apply_core

    processing = processing or ProcessingConfig()
    numerics = numerics or NumericsConfig()
    if params.fade == 100:
        return array.copy()
    array = limit_input_pixels(array, processing.effective_max_input_pixels)
    size = (array.shape[1], array.shape[0])
    work_size = working_size(size, processing.effective_max_edge)
    cleanup_options = {} if source_cleanup is None else {"source_cleanup": source_cleanup}
    if solver is not None:
        cleanup_options["solver"] = solver
    if size == work_size:
        return apply_core(array, variant, params, numerics, **cleanup_options)
    # The effect operates at working resolution before output enlargement.
    rgb = array[..., :3]
    work = resize_working(rgb, work_size, processing.working_resampler)
    filtered = apply_core(work, variant, replace(params, fade=0), numerics, **cleanup_options)
    enlarged = enlarge_progressive(filtered, size)
    amount = np.float32(params.fade) / np.float32(100)
    mixed = (
        enlarged.astype(np.float32) * (np.float32(1) - amount)
        + rgb.astype(np.float32) * amount
    )
    result = array.copy()
    result[..., :3] = np.clip(np.ceil(mixed - np.float32(0.5)), 0, 255).astype(np.uint8)
    return result
