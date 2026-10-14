"""Image preparation and a reusable stateless decoded-frame pipeline."""

from __future__ import annotations

import numpy as np
from PIL import Image

from .configuration import Config, FilterParams, resolve_config
from .core import apply_core
from .image_io import decode_input, prepare_image


def apply_filter(
    image: Image.Image | np.ndarray,
    variant: int | None = None,
    params: FilterParams | None = None,
    *,
    profile: str | None = None,
    config: Config | None = None,
) -> Image.Image | np.ndarray:
    """Apply Fattal variant 1 or 2 to 8-bit RGB/RGBA pixels or a Pillow image.

    The default route includes an 8 MP input-area budget and
    1536-pixel effect working limit. File sampling requires a lazy Pillow
    image with its compressed stream available. Arrays are decoded pixels.
    The core profile processes the original size and preserves straight RGBA
    as an explicit extension. Full alpha compatibility is outside the validated scope.
    """
    config = resolve_config(config, profile)
    params = config.parameters if params is None else params
    if not isinstance(params, FilterParams):
        raise ValueError("params must be a FilterParams")
    if variant is None:
        if config.filter == "both":
            raise ValueError(
                'apply_filter returns one image. Specify variant=1 or variant=2 when filter="both"'
            )
        variant = int(config.filter[-1])
    if isinstance(variant, bool) or variant not in (1, 2):
        raise ValueError("variant must be 1 or 2")
    pil_input = isinstance(image, Image.Image)
    if pil_input:
        image = decode_input(
            image,
            config.processing.effective_max_input_pixels,
            config.image.decode_sampling,
        )
        image = prepare_image(image, config.image)
        array = np.asarray(image).copy()
    else:
        # Arrays have no ICC or EXIF metadata and are already interpreted RGB.
        array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError("image must be uint8 with shape (height, width, 3 or 4)")
    if min(array.shape[:2]) < 1:
        raise ValueError("image must have nonzero dimensions")
    if (
        params.fade < 100
        and config.processing.transparency == "reject"
        and array.shape[2] == 4
        and np.any(array[..., 3] != 255)
    ):
        raise ValueError(
            'opaque pixels required, set processing.transparency="straight" for the unvalidated RGBA extension'
        )
    if config.processing.resize:
        from .processing import process_pixels

        result = process_pixels(
            array, variant, params, config.processing, config.numerics
        )
        if pil_input:
            output = Image.fromarray(result)
            if config.image.color_space == "srgb":
                output.info["icc_profile"] = image.info["icc_profile"]
            if "_input_decode" in image.info:
                output.info["_input_decode"] = image.info["_input_decode"]
            if "_color_conversion" in image.info:
                output.info["_color_conversion"] = image.info["_color_conversion"]
            return output
        return result
    result = apply_core(array, variant, params, config.numerics)
    if pil_input:
        output = Image.fromarray(result)
        if config.image.color_space == "srgb":
            output.info["icc_profile"] = image.info["icc_profile"]
        if "_input_decode" in image.info:
            output.info["_input_decode"] = image.info["_input_decode"]
        if "_color_conversion" in image.info:
            output.info["_color_conversion"] = image.info["_color_conversion"]
        return output
    return result


def process_frames(frames, *, config: Config):
    """Yield processed decoded RGB8 frames lazily, in input order.

    The caller supplies RGB arrays after decoding and color interpretation.
    This adapter holds no temporal state. Container decoding, timestamps,
    audio and encoding belong to a future video adapter.
    """
    if not isinstance(config, Config):
        raise ValueError("config must be a Config")
    if config.filter == "both":
        raise ValueError("process_frames requires one filter per stream")
    for frame in frames:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frames must be decoded RGB arrays")
        yield apply_filter(frame, config=config)
