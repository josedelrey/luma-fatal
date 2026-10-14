"""RGB matrix/TRC ICC conversion with explicit numerical approximations.

Numerical stages are adapted from Skia's BSD-licensed skcms at
bfe565db8273c20dea06ce7be1a8e994560d7f47. The project LICENSE retains the
complete third-party license. This NumPy backend covers opaque RGB8 and
supported matrix/TRC profiles. Multidimensional ICC LUTs and universal
cross-platform bit equality remain outside its scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import struct

import numpy as np


F = np.float32
SRGB_MATRIX = np.array(
    [[0x6FA2, 0x6299, 0x24A0], [0x38F5, 0xB785, 0x0F84], [0x0390, 0x18DA, 0xB6CF]],
    dtype=np.float32,
) * F(1 / 65536)
SRGB_TRC = np.array(
    [2.4, 1 / 1.055, 0.055 / 1.055, 1 / 12.92, 0.04045, 0, 0], dtype=np.float32
)


# Fixed RGB8 probes for the matrix/TRC profile-equivalence test.
_SRGB_EQUIVALENCE_SAMPLE = np.frombuffer(
    bytes.fromhex(
        "08b380ccfd2686b84466208a6327a9d7771a03df5fef3484724a51ea6174f4cd1e9aad0c"
        "339f7a993de2ecb2e537b5dcbfc2a07ea8528312b4f5a316f645ebfc396c0e0698f0ffab"
        "f214e3b1ee605510d346c8959b927f9164976d13a5d0c3a489feb6f840c92dd10593cfd2"
        "71a253e1091f0fe773253a351831c53878ac3015d6816f0b32bbc4223c6747902fcb4d50"
        "e88cdefacea6f78bf9dd486a1bc77536db8776284f29fb2e5dd45ce9941c793f7b9e693b"
        "1d2a8f17006bb05768b79cc1bd5abc41be11c607baa1017c4e7daa85aeda439d4b6559d9"
        "3e218de419235be604020d4956a7ed54f32cb942826e968ed8587024e088ca4c5e62afd5"
    ),
    dtype=np.uint8,
).reshape(84, 3)


@dataclass(frozen=True)
class TabulatedCurve:
    values: np.ndarray


Curve = np.ndarray | TabulatedCurve


class MalformedProfileError(ValueError):
    """The ICC carrier or supported tags are malformed."""


class UnsupportedProfileError(ValueError):
    """The ICC carrier requires a transform outside this backend."""


@dataclass(frozen=True)
class MatrixProfile:
    matrix: np.ndarray
    curves: tuple[Curve, Curve, Curve]


def approx_log2(x: np.ndarray) -> np.ndarray:
    """Approximate log2 using float32 exponent bits and a rational mantissa term."""
    x = np.asarray(x, dtype=np.float32)
    bits = x.view(np.int32)
    e = bits.astype(np.float32) * F(1 / (1 << 23))
    m = ((bits & 0x007FFFFF) | 0x3F000000).view(np.float32)
    return (
        e
        - F(124.225514990)
        - F(1.498030302) * m
        - F(1.725879990) / (F(0.3520887068) + m)
    )


def approx_exp2(x: np.ndarray) -> np.ndarray:
    """Approximate exp2 by encoding a float32 bit pattern with truncation."""
    x = np.asarray(x, dtype=np.float32)
    fract = x - np.floor(x)
    encoded = F(1 << 23) * (
        x
        + F(121.274057500)
        - F(1.490129070) * fract
        + F(27.728023300) / (F(4.84252568) - fract)
    )
    return encoded.astype(np.int32).view(np.float32)


def approx_pow(x: np.ndarray, g: np.float32) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    special = (x == F(0)) | (x == F(1))
    safe = np.where(special, F(1), x)
    return np.where(special, x, approx_exp2(approx_log2(safe) * F(g)))


def apply_curve(x: np.ndarray, tf: Curve) -> np.ndarray:
    if isinstance(tf, TabulatedCurve):
        x = np.asarray(x, dtype=np.float32)
        ix = np.clip(x, F(0), F(1)) * F(len(tf.values) - 1)
        lo = ix.astype(np.int32)
        hi_float = ix + F(1)
        hi = (hi_float.view(np.uint32) - np.uint32(1)).view(np.float32).astype(np.int32)
        t = ix - lo.astype(np.float32)
        l = tf.values[lo].astype(np.float32) * F(1 / 65535)
        h = tf.values[hi].astype(np.float32) * F(1 / 65535)
        return l + (h - l) * t
    g, a, b, c, d, e, f = np.asarray(tf, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    bits = x.view(np.uint32)
    sign = bits & np.uint32(0x80000000)
    magnitude = (bits ^ sign).view(np.float32)
    with np.errstate(invalid="ignore"):
        value = np.where(
            magnitude < d,
            c * magnitude + f,
            approx_pow(a * magnitude + b, g) + e,
        )
    return (sign | value.view(np.uint32)).view(np.float32)


def is_identity_curve(tf: Curve) -> bool:
    return not isinstance(tf, TabulatedCurve) and np.array_equal(
        tf, np.array([1, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    )


def invert_curve(tf: np.ndarray) -> np.ndarray:
    """Invert a finite seven-coefficient piecewise transfer function."""
    g, a, b, c, d, e, f = np.asarray(tf, dtype=np.float32)
    inv = np.zeros(7, dtype=np.float32)
    dl = c * d + f
    dr = approx_pow(a * d + b, g) + e
    if abs(dl - dr) > F(1 / 512):
        raise ValueError("Discontinuous transfer function")
    inv[4] = dl
    if dl > 0:
        inv[3] = F(1) / c
        inv[6] = -f / c
    k = approx_pow(a, -g)
    inv[0] = F(1) / g
    inv[1] = k
    inv[2] = -k * e
    inv[5] = -b / a
    if inv[1] * inv[4] + inv[2] < 0:
        inv[2] = -inv[1] * inv[4]
    s = apply_curve(np.asarray(F(1)), tf)
    sign = F(-1) if s < 0 else F(1)
    s = s * sign
    if s < inv[4]:
        inv[6] = F(1) - sign * inv[3] * s
    else:
        inv[5] = F(1) - sign * approx_pow(inv[1] * s + inv[2], inv[0])
    return inv


def invert_matrix(src: np.ndarray) -> np.ndarray:
    """Invert a 3x3 matrix with a float64 adjugate before float32 conversion."""
    a00, a01, a02 = map(float, src[:, 0])
    a10, a11, a12 = map(float, src[:, 1])
    a20, a21, a22 = map(float, src[:, 2])
    b0 = a00 * a11 - a01 * a10
    b1 = a00 * a12 - a02 * a10
    b2 = a01 * a12 - a02 * a11
    b3, b4, b5 = a20, a21, a22
    determinant = b0 * b5 - b1 * b4 + b2 * b3
    if determinant == 0:
        raise ValueError("Singular ICC matrix")
    invdet = 1.0 / determinant
    b0, b1, b2, b3, b4, b5 = [v * invdet for v in (b0, b1, b2, b3, b4, b5)]
    return np.array(
        [
            [a11 * b5 - a12 * b4, a12 * b3 - a10 * b5, a10 * b4 - a11 * b3],
            [a02 * b4 - a01 * b5, a00 * b5 - a02 * b3, a01 * b3 - a00 * b4],
            [b2, -b1, b0],
        ],
        dtype=np.float32,
    )


def concat_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.empty((3, 3), dtype=np.float32)
    for r in range(3):
        for c in range(3):
            out[r, c] = a[r, 0] * b[0, c] + a[r, 1] * b[1, c] + a[r, 2] * b[2, c]
    return out


def parse_profile(data: bytes) -> MatrixProfile:
    """Separate corrupt carriers from unsupported transform representations."""
    try:
        return _parse_profile(data)
    except (KeyError, struct.error, IndexError) as exc:
        raise MalformedProfileError("Missing or truncated ICC data") from exc


def _parse_profile(data: bytes) -> MatrixProfile:
    """Parse RGB/XYZ profiles with parametric, gamma or tabulated TRCs."""
    if len(data) < 132 or data[36:40] != b"acsp":
        raise MalformedProfileError("Invalid ICC signature")
    declared_size = struct.unpack_from(">I", data)[0]
    if declared_size > len(data) or data[8] > 4:
        raise MalformedProfileError("Invalid ICC header")
    count = struct.unpack_from(">I", data, 128)[0]
    if 132 + count * 12 > declared_size:
        raise MalformedProfileError("Invalid ICC tag table")
    tags = {}
    for i in range(count):
        name, offset, length = struct.unpack_from(">4sII", data, 132 + i * 12)
        if length < 4 or offset + length > declared_size:
            raise MalformedProfileError("Invalid ICC tag extent")
        if name in tags:
            raise MalformedProfileError("Duplicate ICC tag signature")
        tags[name] = data[offset : offset + length]
    if any(tag in tags for tag in (b"A2B0", b"A2B1", b"A2B2")):
        raise UnsupportedProfileError("LUT ICC profiles require a different transform")
    if data[16:24] != b"RGB XYZ ":
        raise UnsupportedProfileError(
            "Only ICC RGB matrix profiles with XYZ PCS are supported"
        )
    white = np.array(struct.unpack_from(">3i", data, 68)) / 65536
    if np.any(np.abs(white - np.array([0.9642, 1.0, 0.8249])) > 0.01):
        raise MalformedProfileError("ICC PCS illuminant is not D50")
    matrix = np.empty((3, 3), dtype=np.float32)
    curves = []
    for channel, prefix in enumerate((b"r", b"g", b"b")):
        xyz = tags[prefix + b"XYZ"]
        if xyz[:4] != b"XYZ " or len(xyz) < 20:
            raise MalformedProfileError("Invalid ICC XYZ tag")
        matrix[:, channel] = np.array(
            struct.unpack_from(">3i", xyz, 8), dtype=np.float32
        ) * F(1 / 65536)
        curve = tags[prefix + b"TRC"]
        tf = np.array([1, 1, 0, 0, 0, 0, 0], dtype=np.float32)
        if curve[:4] == b"para":
            kind = struct.unpack_from(">H", curve, 8)[0]
            counts = {0: 1, 1: 3, 2: 4, 3: 5, 4: 7}
            if kind not in counts or len(curve) < 12 + counts[kind] * 4:
                raise MalformedProfileError("Invalid ICC parametric curve")
            values = np.array(
                struct.unpack_from(">" + "i" * counts[kind], curve, 12),
                dtype=np.float32,
            ) * F(1 / 65536)
            tf[0] = values[0]
            if kind in (1, 2):
                tf[1:3] = values[1:3]
                if tf[1] == 0:
                    raise MalformedProfileError("Invalid ICC curve divisor")
                tf[4] = -tf[2] / tf[1]
                if kind == 2:
                    tf[5:7] = values[3]
            elif kind == 3:
                tf[:5] = values
            elif kind == 4:
                tf[:] = values
        elif curve[:4] == b"curv":
            entries = struct.unpack_from(">I", curve, 8)[0]
            if len(curve) < 12 + entries * 2:
                raise MalformedProfileError("Truncated ICC curve table")
            if entries > 1:
                curves.append(
                    TabulatedCurve(
                        np.frombuffer(
                            curve, dtype=">u2", offset=12, count=entries
                        ).astype(np.uint16)
                    )
                )
                continue
            if entries == 1:
                tf[0] = F(struct.unpack_from(">H", curve, 12)[0]) * F(1 / 256)
        else:
            raise UnsupportedProfileError("Unsupported ICC curve type")
        if (
            not np.isfinite(tf).all()
            or np.any(tf[[0, 1, 3, 4]] < 0)
            or tf[1] * tf[4] + tf[2] < 0
        ):
            raise MalformedProfileError("Invalid transfer function")
        curves.append(tf)
    return MatrixProfile(matrix, tuple(curves))


def convert_rgb(
    rgb: np.ndarray, profile: MatrixProfile, *, codec_bypass: bool = True
) -> np.ndarray:
    if rgb.dtype != np.uint8 or rgb.ndim < 2 or rgb.shape[-1] != 3:
        raise ValueError("Expected RGB uint8 pixels")
    if codec_bypass and approximate_srgb(profile):
        return rgb.copy()
    skip_matrix = np.array_equal(SRGB_MATRIX, profile.matrix)
    matrix = concat_matrix(invert_matrix(SRGB_MATRIX), profile.matrix)
    inverse = invert_curve(SRGB_TRC)
    # An 8-bit source curve has just 256 possible input values.
    normalized = np.arange(256, dtype=np.float32) * F(1 / 255)
    luts = [
        normalized if is_identity_curve(tf) else apply_curve(normalized, tf)
        for tf in profile.curves
    ]
    output = np.empty(rgb.shape, dtype=np.uint8)
    source = rgb.reshape(-1, 3)
    destination = output.reshape(-1, 3)
    # Keep temporary float arrays bounded for large camera photographs.
    for start in range(0, len(source), 262144):
        end = min(start + 262144, len(source))
        linear = [lut[source[start:end, c]] for c, lut in enumerate(luts)]
        for c in range(3):
            transformed = (
                linear[c]
                if skip_matrix
                else (
                    matrix[c, 0] * linear[0]
                    + matrix[c, 1] * linear[1]
                    + matrix[c, 2] * linear[2]
                )
            )
            encoded = np.clip(apply_curve(transformed, inverse), F(0), F(1))
            destination[start:end, c] = (encoded * F(255) + F(0.5)).astype(np.uint8)
    return output


@lru_cache(maxsize=16)
def cached_profile(data: bytes) -> MatrixProfile:
    """Cache immutable parsed profiles, keyed by their complete ICC bytes."""
    profile = parse_profile(data)
    profile.matrix.flags.writeable = False
    for curve in profile.curves:
        values = curve.values if isinstance(curve, TabulatedCurve) else curve
        values.flags.writeable = False
    return profile


def approximate_srgb(profile: MatrixProfile) -> bool:
    """Check matrix/TRC proximity to sRGB on a fixed RGB8 sample."""
    sample = _SRGB_EQUIVALENCE_SAMPLE

    def xyz_bytes(p: MatrixProfile) -> np.ndarray:
        normalized = sample.astype(np.float32) * F(1 / 255)
        linear = [
            normalized[:, c]
            if is_identity_curve(tf)
            else apply_curve(normalized[:, c], tf)
            for c, tf in enumerate(p.curves)
        ]
        output = np.empty_like(sample)
        for c in range(3):
            xyz = (
                p.matrix[c, 0] * linear[0]
                + p.matrix[c, 1] * linear[1]
                + p.matrix[c, 2] * linear[2]
            )
            output[:, c] = (np.clip(xyz, F(0), F(1)) * F(255) + F(0.5)).astype(np.uint8)
        return output

    standard = MatrixProfile(SRGB_MATRIX, (SRGB_TRC, SRGB_TRC, SRGB_TRC))
    return bool(
        np.all(
            np.abs(
                xyz_bytes(profile).astype(np.int16)
                - xyz_bytes(standard).astype(np.int16)
            )
            <= 1
        )
    )
