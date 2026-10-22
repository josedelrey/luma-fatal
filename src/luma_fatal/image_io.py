"""Photo interpretation and encoding policies, independent of the filter math."""

from __future__ import annotations

from io import BytesIO
import hashlib

import numpy as np
from PIL import Image, ImageCms, ImageOps

from .configuration import ImageConfig, OutputConfig
from .color import (
    cached_profile,
    convert_rgb,
    MalformedProfileError,
    UnsupportedProfileError,
)


def decode_input(
    image: Image.Image, max_pixels: int | None, enabled: bool
) -> Image.Image:
    """Sample original file inputs before color/orientation and final area fit.

    PNG uses center selection and floor dimensions. JPEG uses a power-of-two
    DCT reduction followed by residual center selection when necessary.
    Fixed decoder references cover sampling factors 1 through 4. Higher
    factors remain extrapolated. Loaded pixels retain their decoded interpretation.
    """
    from .processing import input_size

    width, height = image.size
    target = input_size(image.size, max_pixels)
    sample = max(1, min(width // target[0], height // target[1]))
    if not enabled or sample == 1 or image.format not in ("PNG", "JPEG"):
        return image
    if image.mode not in ("RGB", "RGBA", "L"):
        # Palette, CMYK and high-depth modes use the full decode extension.
        return image
    if image.format == "JPEG":
        stream = getattr(image, "fp", None)
        if stream is None or not getattr(image, "tile", None):
            return image
        # Copy the encoded stream without changing the caller's decode state.
        position = stream.tell()
        try:
            stream.seek(0)
            payload = stream.read()
        finally:
            stream.seek(position)
        result = Image.open(BytesIO(payload))
        dct_sample = min(sample & -sample, 8)
        result.draft("RGB", (max(1, width // dct_sample), max(1, height // dct_sample)))
        result.load()
        residual = sample // dct_sample
        if residual > 1:
            array = np.asarray(result)
            sx, sy = max(1, result.width // residual), max(1, result.height // residual)
            start = residual // 2
            reduced = array[
                start : start + sy * residual : residual,
                start : start + sx * residual : residual,
            ]
            info = result.info.copy()
            result = Image.fromarray(np.ascontiguousarray(reduced))
            result.info.update(info)
    else:
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] not in (3, 4):
            return image
        sx, sy = max(1, width // sample), max(1, height // sample)
        start = sample // 2
        reduced = array[
            start : start + sy * sample : sample, start : start + sx * sample : sample
        ]
        result = Image.fromarray(np.ascontiguousarray(reduced))
        result.info.update(image.info)
    result.info["_input_decode"] = dict(
        source_dimensions=[width, height],
        sample=sample,
        decoded_dimensions=list(result.size),
        codec=image.format,
    )
    return result


def prepare_image(image: Image.Image, policy: ImageConfig) -> Image.Image:
    original_profile = image.info.get("icc_profile")
    source = (
        ImageOps.exif_transpose(image) if policy.exif_orientation == "apply" else image
    )
    mode = "RGBA" if source.getbands()[-1] == "A" else "RGB"
    if policy.color_space == "srgb":
        srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
        record = dict(
            requested_backend=policy.color_backend,
            backend="none",
            source_icc_sha256=hashlib.sha256(original_profile).hexdigest()
            if original_profile
            else None,
        )
        if original_profile:
            try:
                fallback = None
                if policy.color_backend == "matrix_trc":
                    if source.mode not in ("RGB", "RGBA"):
                        fallback = f"{source.mode} source uses the LCMS extension"
                    elif source.mode == "RGBA" and np.any(
                        np.asarray(source)[..., 3] != 255
                    ):
                        fallback = "Transparent pixels use the LCMS extension"
                    else:
                        try:
                            parsed = cached_profile(original_profile)
                            pixels = np.asarray(source)
                            converted = convert_rgb(pixels[..., :3], parsed)
                            if mode == "RGBA":
                                converted = np.concatenate(
                                    (converted, pixels[..., 3:4]), axis=2
                                )
                            result = Image.fromarray(converted)
                            record["backend"] = "matrix_trc"
                        except UnsupportedProfileError as error:
                            fallback = str(error)
                        except MalformedProfileError as error:
                            raise ValueError(
                                f"Unable to convert embedded ICC profile to sRGB: {error}"
                            ) from error
                if policy.color_backend == "lcms" or fallback:
                    profile = ImageCms.ImageCmsProfile(BytesIO(original_profile))
                    result = ImageCms.profileToProfile(
                        source, profile, srgb, outputMode=mode
                    )
                    record["backend"] = "lcms"
                    if fallback:
                        record["fallback_reason"] = fallback
            except (ImageCms.PyCMSError, ValueError, OSError) as error:
                raise ValueError(
                    f"Unable to convert embedded ICC profile to sRGB: {error}"
                ) from error
        else:
            result = source.convert(mode)
            record["reason"] = "No embedded ICC profile, RGB is interpreted as sRGB"
        result.info["icc_profile"] = srgb.tobytes()
        result.info["_color_conversion"] = record
        if "_input_decode" in image.info:
            result.info["_input_decode"] = image.info["_input_decode"]
        return result
    result = source.convert(mode)
    result.info["_color_conversion"] = dict(
        requested_backend=policy.color_backend,
        backend="none",
        reason="Encoded color policy bypasses conversion",
    )
    # An encoded-mode conversion from CMYK cannot retain a CMYK ICC on RGB data.
    if source.mode not in ("RGB", "RGBA"):
        result.info.pop("icc_profile", None)
    return result


def save_image(image: Image.Image, path, policy: OutputConfig, icc_profile=None):
    options = {"icc_profile": icc_profile if policy.embed_icc else None}
    if policy.format == "png":
        image.save(path, format="PNG", **options)
        return
    if image.mode == "RGBA":
        if policy.jpeg_background is None:
            if np.any(np.asarray(image)[..., 3] != 255):
                raise ValueError(
                    "JPEG cannot preserve transparency. Set output.jpeg_background or export PNG"
                )
            image = image.convert("RGB")
        else:
            background = Image.new("RGBA", image.size, (*policy.jpeg_background, 255))
            image = Image.alpha_composite(background, image).convert("RGB")
    image.save(
        path,
        format="JPEG",
        quality=policy.jpeg_quality,
        subsampling={"444": 0, "422": 1, "420": 2}[policy.jpeg_subsampling],
        **options,
    )
