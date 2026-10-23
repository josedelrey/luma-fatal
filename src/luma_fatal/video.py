"""Bounded-memory, silent CFR video processing through external FFmpeg tools."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, replace
from fractions import Fraction
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
from tqdm import tqdm

from . import __version__
from .configuration import Config, VideoConfig, VIDEO_COLOR_PROFILES, resolve_config
from .cleanup import MultiscaleSourceCleanup, SourceCleanup
from .pipeline import process_frames
from .prenoise import SourceNoise
from .denoise import SourceDenoise
from .processing import input_size


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def _process(command, **kwargs):
    # Keep diagnostics off pipes so a verbose failure cannot deadlock processing.
    with tempfile.TemporaryFile() as log:
        try:
            process = subprocess.Popen(
                command,
                stderr=log,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                **kwargs,
            )
        except FileNotFoundError as error:
            raise ValueError(
                f"Required executable not found: {command[0]}. Install FFmpeg and ffprobe"
            ) from error
        try:
            yield process, log
        finally:
            # Also runs after cancellation, failed filtering or a broken encoder pipe.
            if process.poll() is None:
                if os.name == "nt":
                    # Chocolatey shims launch a child that owns the files and pipes.
                    # Stop the tree before the launcher can leave that child orphaned.
                    try:
                        subprocess.run(
                            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            creationflags=subprocess.CREATE_NO_WINDOW,
                            timeout=5,
                            check=False,
                        )
                    except (OSError, subprocess.TimeoutExpired):
                        process.terminate()
                else:
                    process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            for pipe in (process.stdin, process.stdout):
                if pipe is not None:
                    try:
                        pipe.close()
                    except OSError:
                        pass


def _check(process, log, operation):
    code = process.wait()
    log.seek(0, os.SEEK_END)
    length = log.tell()
    log.seek(max(0, length - 8192))
    details = log.read().decode("utf-8", errors="replace").strip()
    if code or details:
        raise ValueError(f"{operation} failed (exit {code}): {details}")


def _capture(command):
    with _process(command, stdout=subprocess.PIPE, stdin=subprocess.DEVNULL) as (p, log):
        output = p.stdout.read()
        _check(p, log, Path(command[0]).name)
    return output.decode("utf-8")


def _fraction(value, name):
    try:
        result = Fraction(value)
    except (ValueError, TypeError, ZeroDivisionError) as error:
        raise ValueError(f"Missing or invalid video {name}") from error
    if result <= 0:
        raise ValueError(f"Video {name} must be positive")
    return result


def _known(value):
    return value not in (None, "unknown", "unspecified", "reserved", "N/A")


def _rgb_format(pixel_format):
    return any(name in pixel_format for name in ("rgb", "bgr", "gbr")) or pixel_format == "pal8"


def _resolve_color(info, settings):
    """Resolve missing tags without replacing known metadata in automatic mode."""
    rgb = _rgb_format(info.get("pix_fmt", ""))
    assumptions = {}
    keys = ("color_space", "color_transfer", "color_primaries")
    if settings.input_color != "auto":
        values = dict(zip(keys, VIDEO_COLOR_PROFILES[settings.input_color], strict=True))
    else:
        if info.get("color_transfer") in ("smpte2084", "arib-std-b67"):
            profile = "bt2020-pq" if info["color_transfer"] == "smpte2084" else "bt2020-hlg"
        elif rgb:
            profile = "srgb"
        elif "bt470bg" in (info.get("color_space"), info.get("color_primaries")):
            profile = "bt601-pal"
        elif "smpte170m" in (info.get("color_space"), info.get("color_primaries")):
            profile = "bt601-ntsc"
        elif "bt709" in (info.get("color_space"), info.get("color_primaries")):
            profile = "bt709"
        elif info["height"] <= 576 and info["width"] <= 1024:
            profile = "bt601-pal" if info["height"] > 480 else "bt601-ntsc"
        else:
            profile = "bt709"
        defaults = dict(zip(keys, VIDEO_COLOR_PROFILES[profile], strict=True))
        if (info.get("color_space") in ("bt2020nc", "bt2020c")
                or info.get("color_primaries") == "bt2020"):
            defaults["color_space"] = "bt2020nc"
            defaults["color_primaries"] = "bt2020"
            if info.get("color_transfer") not in ("smpte2084", "arib-std-b67"):
                defaults["color_transfer"] = "bt2020-10"
        values = {}
        for key in keys:
            values[key] = info[key] if _known(info.get(key)) else defaults[key]
            if not _known(info.get(key)):
                assumptions[key] = values[key]
    # RGB storage always has an identity matrix, even with a YUV color override.
    if rgb:
        values["color_space"] = "gbr"
        if "color_space" in assumptions:
            assumptions["color_space"] = "gbr"
    if settings.input_range != "auto":
        values["color_range"] = "tv" if settings.input_range == "limited" else "pc"
    elif info.get("color_range") in ("tv", "pc"):
        values["color_range"] = info["color_range"]
    else:
        values["color_range"] = "pc" if rgb or info.get("pix_fmt", "").startswith(
            ("yuvj", "gray")
        ) else "tv"
        assumptions["color_range"] = values["color_range"]
    return values, assumptions


def _probe(path, executable):
    report = json.loads(
        _capture([
            str(executable), "-v", "error", "-show_streams", "-of", "json", str(path),
        ])
    )
    streams = report.get("streams", [])
    videos = [s for s in streams if s.get("codec_type") == "video"
              and not s.get("disposition", {}).get("attached_pic")]
    if len(videos) != 1:
        raise ValueError("Input must contain exactly one video stream, excluding cover art")
    stream = videos[0]
    if min(stream.get("width", 0), stream.get("height", 0)) < 2:
        raise ValueError("Video dimensions must be at least 2 by 2")
    stream["audio_streams"] = sum(s.get("codec_type") == "audio" for s in streams)
    return stream


def _sample_aspect_ratio(value):
    if value in (None, "N/A", "0:1"):
        return Fraction(1)
    return _fraction(value.replace(":", "/"), "sample aspect ratio")


def _geometry(stream):
    sar = _sample_aspect_ratio(stream.get("sample_aspect_ratio"))
    width, height = max(2, round(stream["width"] * sar)), stream["height"]
    rotation = stream.get("tags", {}).get("rotate", 0)
    for data in stream.get("side_data_list", []):
        if data.get("side_data_type") == "Display Matrix":
            rotation = data.get("rotation", 0)
            # Perspective, skew and reflection need more than a rotation angle.
            rows = data.get("displaymatrix", "").strip().splitlines()
            if rows:
                matrix = [int(v) for row in rows for v in row.split(":", 1)[1].split()]
                rotations = {
                    (65536, 0, 0, 65536), (0, -65536, 65536, 0),
                    (-65536, 0, 0, -65536), (0, 65536, -65536, 0),
                }
                if (len(matrix) != 9
                        or (matrix[0], matrix[1], matrix[3], matrix[4]) not in rotations
                        or matrix[2] or matrix[5] or matrix[6] or matrix[7]
                        or matrix[8] != 1073741824):
                    raise ValueError("Unsupported video display matrix: requires a pure rotation")
    try:
        angle = float(rotation)
        quarter = round(angle / 90)
        if not np.isfinite(angle) or abs(angle - quarter * 90) > 0.001:
            raise ValueError
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError("Video display rotation must be a multiple of 90 degrees") from error
    rotation = quarter * 90 % 360
    return (height, width) if rotation in (90, 270) else (width, height), rotation, sar


def _source_rate(stream, settings):
    for key in ("avg_frame_rate", "r_frame_rate"):
        try:
            return _fraction(stream.get(key), "frame rate")
        except ValueError:
            pass
    if settings.frame_rate != "auto":
        return Fraction(settings.frame_rate)
    raise ValueError("Missing video frame rate: specify --frame-rate or video.frame_rate")


def _scan_timing(path, stream, executable, settings):
    """Scan source timing and reject changing geometry or color interpretation."""
    rate = _source_rate(stream, settings)
    tick = _fraction(stream.get("time_base"), "time base")
    color, _ = _resolve_color(stream, settings)
    sar = _sample_aspect_ratio(stream.get("sample_aspect_ratio"))
    entries = (
        "pts,duration,pkt_duration,width,height,pix_fmt,color_range,color_space,"
        "color_transfer,color_primaries,interlaced_frame,sample_aspect_ratio"
    )
    count, first, previous = 0, None, None
    variable = tick >= 1 / (2 * rate)
    interlaced = stream.get("field_order") not in (None, "unknown", "progressive")
    command = [
        str(executable), "-v", "error", "-select_streams", str(stream["index"]),
        "-show_frames", "-show_entries", f"frame={entries}",
        "-of", "compact=p=0", str(path),
    ]
    with _process(command, stdout=subprocess.PIPE, stdin=subprocess.DEVNULL) as (p, log):
        for raw in p.stdout:
            fields = dict(
                part.split("=", 1)
                for part in raw.decode("utf-8").strip().split("|")
                if "=" in part
            )
            if "width" not in fields:
                continue
            try:
                pts = int(fields["pts"]) * tick
            except (KeyError, ValueError) as error:
                raise ValueError("Every video frame must have a presentation timestamp") from error
            if previous is not None and pts <= previous:
                raise ValueError("Video presentation timestamps must increase strictly")
            if first is None:
                first = pts
            variable |= abs(pts - first - count / rate) > tick
            duration = fields.get("duration", fields.get("pkt_duration"))
            if duration not in (None, "N/A", "0"):
                variable |= abs(int(duration) * tick - 1 / rate) > tick
            if any(int(fields.get(key, 0)) != stream[key] for key in ("width", "height")):
                raise ValueError("Changing video dimensions are not supported")
            info = {**stream, **fields, "width": stream["width"], "height": stream["height"]}
            if _resolve_color(info, settings)[0] != color or info.get("pix_fmt") != stream.get("pix_fmt"):
                raise ValueError("Changing video color interpretation or pixel format is not supported")
            if _sample_aspect_ratio(info.get("sample_aspect_ratio")) != sar:
                raise ValueError("Changing video sample aspect ratio is not supported")
            interlaced |= fields.get("interlaced_frame") == "1"
            previous = pts
            count += 1
        _check(p, log, "Video timing scan")
    if not count:
        raise ValueError("Input contains no decoded video frames")
    if variable and settings.frame_rate == "auto":
        # Prefer the nominal cadence over an average distorted by pauses or gaps.
        try:
            rate = _fraction(stream.get("r_frame_rate"), "nominal frame rate")
        except ValueError:
            pass
    return rate, count, first, bool(variable), bool(interlaced)


def _decode_command(executable, path, stream, deblock="off", deband="off", *,
                    color=None, dimensions=None, rotation=0, interlaced=False, rate=None):
    color = color or {key: stream[key] for key in (
        "color_space", "color_transfer", "color_primaries", "color_range",
    )}
    source_range = "limited" if color["color_range"] == "tv" else "full"
    source = (
        f"matrixin={color['color_space']}:transferin={color['color_transfer']}:"
        f"primariesin={color['color_primaries']}:rangein={source_range}"
    )
    filters = []
    if interlaced:
        filters.append("bwdif=mode=send_frame:parity=auto:deint=interlaced")
    if rate is not None:
        filters += ["setpts=PTS-STARTPTS", f"fps=fps={rate}:start_time=0:round=near"]
    rgb = _rgb_format(stream.get("pix_fmt", ""))
    if rgb:
        filters.append("format=gbrp16le")
    if deblock != "off" or deband != "off":
        # Source cleanup operates on planar YUV. Preserve precision until RGB8.
        if rgb:
            filters += [
                f"zscale={source}:matrix=709:transfer={color['color_transfer']}:"
                f"primaries={color['color_primaries']}:range=full",
                "format=yuv444p16le",
            ]
            source = (f"matrixin=709:transferin={color['color_transfer']}:"
                      f"primariesin={color['color_primaries']}:rangein=full")
        if deblock != "off":
            block = {"weak4": 4, "weak8": 8}[deblock]
            filters.append(f"deblock=filter=weak:block={block}")
        if deband == "gentle":
            filters.append("deband=1thr=0.008:2thr=0.008:3thr=0.008:range=8")
    if color["color_transfer"] in ("smpte2084", "arib-std-b67"):
        filters += [
            f"zscale={source}:matrix=gbr:transfer=linear:primaries=709:range=full:npl=100",
            "format=gbrpf32le",
            # tonemap reads this tag for luma coefficients during desaturation.
            "setparams=colorspace=bt709", "tonemap=tonemap=hable:desat=2:peak=10",
            "zscale=matrixin=gbr:transferin=linear:primariesin=709:rangein=full:"
            "matrix=gbr:transfer=iec61966-2-1:primaries=709:range=full",
            "format=gbrp", "format=rgb24",
        ]
    else:
        filters += [
            f"zscale={source}:matrix=gbr:transfer=iec61966-2-1:primaries=709:range=full",
            "format=gbrp", "format=rgb24",
        ]
    if dimensions is not None:
        unrotated = dimensions[::-1] if rotation in (90, 270) else dimensions
        if unrotated != (stream["width"], stream["height"]):
            filters.append(f"scale={unrotated[0]}:{unrotated[1]}:flags=bicubic")
        filters.append("setsar=1")
    if rotation == 90:
        filters.append("transpose=cclock")
    elif rotation == 180:
        filters += ["hflip", "vflip"]
    elif rotation == 270:
        filters.append("transpose=clock")
    return [
        str(executable), "-v", "error", "-xerror", "-nostdin", "-noautorotate",
        "-i", str(path), "-map", f"0:{stream['index']}", "-an", "-sn", "-dn",
        "-vf", ",".join(filters),
        "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]


def _decoded_frames(pipe, width, height):
    size = width * height * 3
    while True:
        data = bytearray()
        while len(data) < size:
            block = pipe.read(size - len(data))
            if not block:
                break
            data.extend(block)
        if not data:
            return
        if len(data) != size:
            raise ValueError("Decoder returned an incomplete RGB frame")
        yield np.frombuffer(data, np.uint8).reshape(height, width, 3)


def _encoder_options(settings):
    if settings.encoder == "h264_nvenc":
        return [
            "-c:v", "h264_nvenc", "-preset", settings.nvenc_preset,
            "-rc", "vbr", "-cq", str(settings.nvenc_cq), "-b:v", "0",
        ]
    return ["-c:v", "libx264", "-crf", str(settings.crf), "-preset", settings.preset]


def _validate_nvenc(executable, width, height, rate, settings):
    """Test the requested hardware encoder before processing effect frames."""
    command = [
        str(executable), "-v", "error", "-xerror", "-nostdin",
        "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s:v", f"{width}x{height}",
        "-r", str(rate), "-i", "pipe:0", "-frames:v", "1",
        *_encoder_options(settings), "-f", "null", "-",
    ]
    try:
        with _process(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL) as (p, log):
            try:
                p.stdin.write(bytes(width * height * 3 // 2))
                p.stdin.close()
            except OSError:
                # An encoder that cannot initialize may close stdin immediately.
                _check(p, log, "NVENC preflight")
                raise
            _check(p, log, "NVENC preflight")
    except (ValueError, OSError) as error:
        raise ValueError(
            "NVIDIA H.264 encoding is unavailable for these settings. "
            "Check that FFmpeg includes h264_nvenc and that a compatible NVIDIA GPU "
            "and driver are available, or select --encoder libx264. "
            f"{error}"
        ) from error


def _encode_command(executable, path, width, height, rate, settings):
    return [
        str(executable), "-v", "error", "-xerror", "-nostdin", "-n",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s:v", f"{width}x{height}",
        "-r", str(rate), "-i", "pipe:0", "-an", "-sn", "-dn", "-map_metadata", "-1",
        "-vf", (
            "format=gbrp,zscale=matrixin=gbr:transferin=iec61966-2-1:"
            "primariesin=709:rangein=full:matrix=709:transfer=709:"
            "primaries=709:range=limited,format=yuv420p,setsar=1"
        ),
        *_encoder_options(settings),
        "-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc", "bt709",
        "-colorspace", "bt709", "-color_range", "tv", "-fps_mode", "passthrough",
        "-movflags", "+faststart", "-f", "mp4", str(path),
    ]


def process_video(
    input_path: str | Path,
    output_path: str | Path,
    *,
    config: Config | None = None,
    config_path: str | Path | None = None,
    video_config: VideoConfig | None = None,
    progress: bool = False,
    ffmpeg: str | Path = "ffmpeg",
    ffprobe: str | Path = "ffprobe",
) -> dict:
    """Prepare and filter video into silent, square-pixel SDR H.264 MP4.

    FFmpeg must provide zscale and libx264, plus h264_nvenc for GPU encoding,
    bwdif for interlaced input, tonemap for HDR, and optional source filters. A streaming
    preflight validates timestamps and stable source geometry/color. Missing color
    tags use recorded assumptions. HDR uses Hable tone mapping with a 1000-nit peak
    and 100-nit white.
    Variable timing is normalized to the nominal rate, unless frame_rate overrides
    it. Progressive CFR input retains its cadence by default.
    Optional weak deblocking and gentle debanding run on decoded YUV before
    RGB conversion.
    Image preparation/export settings do not apply to video. Frames enter the
    filter as full-range sRGB RGB8. Odd output edges are padded by replication.
    Unavailable NVENC encoding falls back to libx264 before effect processing.
    Existing destinations are refused. Failed runs publish no output or sidecar.
    """
    start = time.perf_counter()
    source, destination = Path(input_path).resolve(), Path(output_path).resolve()
    sidecar = destination.with_suffix(".meta.json")
    if source in (destination, sidecar):
        raise ValueError("Video input, output and sidecar paths must differ")
    if not source.is_file():
        raise ValueError(f"Video input is not a local file: {source}")
    if destination.suffix.lower() != ".mp4":
        raise ValueError("Video output must use the .mp4 extension")
    if destination.exists() or sidecar.exists():
        raise ValueError("Video output or metadata sidecar already exists")
    config = (
        Config.load(config_path)
        if config is None and config_path is not None
        else resolve_config(config)
    )
    if config.filter == "both":
        raise ValueError("process_video requires one filter per output")
    settings = config.video if video_config is None else video_config
    if not isinstance(settings, VideoConfig):
        raise ValueError("video_config must be a VideoConfig")
    stream = _probe(source, ffprobe)
    normalized, rotation, sar = _geometry(stream)
    color, assumptions = _resolve_color(stream, settings)
    source_rate, source_count, first, variable, interlaced = _scan_timing(
        source, stream, ffprobe, settings
    )
    rate = source_rate if settings.frame_rate == "auto" else Fraction(settings.frame_rate)
    resample = variable or rate != source_rate
    expected = None if resample else source_count
    prepared = input_size(
        normalized,
        config.processing.effective_max_input_pixels if config.parameters.fade < 100 else None,
    )
    width, height = (edge + edge % 2 for edge in prepared)
    encoding_settings, fallback_reason = settings, None
    if settings.encoder == "h264_nvenc":
        try:
            _validate_nvenc(ffmpeg, width, height, rate, settings)
        except ValueError as error:
            fallback_reason = str(error)
            encoding_settings = replace(settings, encoder="libx264")
            print(
                "NVIDIA encoding unavailable. Falling back to CPU encoding (libx264).",
                file=sys.stderr,
            )
    tool_versions = {
        name: _capture([str(executable), "-version"]).splitlines()[0]
        for name, executable in (("ffmpeg", ffmpeg), ("ffprobe", ffprobe))
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    count, processed_hash = 0, hashlib.sha256()
    cleanup = None
    if settings.flat_cleanup:
        if settings.flat_cleanup_mode == "multiscale":
            cleanup = MultiscaleSourceCleanup(
                fine_strength=settings.flat_cleanup,
                coarse_strength=settings.flat_cleanup_coarse_strength,
                sigma=settings.flat_cleanup_sigma,
            )
        else:
            cleanup = SourceCleanup(settings.flat_cleanup)
    noise = None
    if settings.pre_noise_rms:
        noise = SourceNoise(
            rms=settings.pre_noise_rms, kind=settings.pre_noise_kind,
            masked=settings.pre_noise_masked, seed=settings.pre_noise_seed,
            source_cleanup=cleanup,
        )
    source_treatment = noise if noise is not None else cleanup
    denoiser = None
    if settings.denoise:
        denoiser = SourceDenoise(settings.denoise, source_cleanup=source_treatment)
        source_treatment = denoiser
    cleanup_options = ({} if source_treatment is None
                       else {"source_cleanup": source_treatment})
    filter_solver = None
    if settings.filter_backend == "auto":
        from .gpu import AdaptiveSolver

        filter_solver = AdaptiveSolver(expected_frames=expected)
        cleanup_options["solver"] = filter_solver
    with tempfile.TemporaryDirectory(prefix=".luma-fatal-", dir=destination.parent) as directory:
        temporary = Path(directory) / "video.mp4"
        temporary_meta = Path(directory) / "video.meta.json"
        decode = _decode_command(
            ffmpeg, source, stream, settings.deblock, settings.deband,
            color=color, dimensions=normalized, rotation=rotation, interlaced=interlaced,
            rate=rate if resample else None,
        )
        encode = _encode_command(ffmpeg, temporary, width, height, rate, encoding_settings)
        with _process(
            decode, stdout=subprocess.PIPE, stdin=subprocess.DEVNULL
        ) as (decoder, decode_log):
            with _process(
                encode, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL
            ) as (encoder, encode_log):
                frames = _decoded_frames(decoder.stdout, *normalized)
                with tqdm(
                    total=expected, desc="Filtering frames", unit="frame",
                    file=sys.stderr, dynamic_ncols=True, disable=not progress,
                ) as bar:
                    try:
                        for frame in process_frames(frames, config=config, **cleanup_options):
                            if (frame.shape[1], frame.shape[0]) != prepared:
                                raise ValueError("Processed video frame dimensions changed")
                            frame = np.pad(
                                frame,
                                ((0, height - prepared[1]), (0, width - prepared[0]), (0, 0)),
                                mode="edge",
                            )
                            data = frame.tobytes()
                            encoder.stdin.write(data)
                            processed_hash.update(data)
                            count += 1
                            bar.update(1)
                        encoder.stdin.close()
                    except OSError as error:
                        try:
                            encoder.stdin.close()
                        except OSError:
                            pass
                        _check(encoder, encode_log, "Video encoding")
                        raise ValueError(
                            "Video encoder closed its input unexpectedly"
                        ) from error
                _check(decoder, decode_log, "Video decoding")
                _check(encoder, encode_log, "Video encoding")
        if not count:
            raise ValueError("Video preparation produced no frames: try a higher frame rate")
        if expected is not None and count != expected:
            raise ValueError(f"Decoded frame count changed: expected {expected}, got {count}")
        record = dict(
            schema_version=1,
            implementation_version=__version__,
            source=str(source),
            source_sha256=_digest(source),
            output=str(destination),
            output_sha256=_digest(temporary),
            filter=config.filter,
            configuration=config.to_dict(),
            video_configuration=asdict(settings),
            encoding=dict(
                requested_encoder=settings.encoder,
                encoder=encoding_settings.encoder,
                fallback_reason=fallback_reason,
            ),
            filtering=(
                dict(filter_solver.record) if filter_solver is not None else
                dict(requested_backend="cpu", backend="cpu", reason="explicit CPU filtering")
            ),
            source_dimensions=[stream["width"], stream["height"]],
            normalized_dimensions=list(normalized),
            prepared_dimensions=list(prepared),
            dimensions=[width, height],
            padding=dict(
                right=width - prepared[0], bottom=height - prepared[1], mode="edge"
            ),
            frame_count=count,
            frame_rate=str(rate),
            duration_seconds=float(count / rate),
            timing=dict(
                source_first_pts_seconds=str(first),
                source_frame_count=source_count,
                source_variable_frame_rate=variable,
                output_start_seconds=0,
                policy=("resampled to CFR, rebased to zero" if resample
                        else "validated CFR, rebased to zero"),
            ),
            color_policy=dict(
                input={key: stream.get(key) for key in (
                    "color_space", "color_transfer", "color_primaries",
                    "color_range", "pix_fmt",
                )},
                resolved_input=color,
                assumptions=assumptions,
                processing="full-range sRGB RGB8",
                output="limited-range BT.709 YUV420P",
                converter="FFmpeg zscale",
                transfer_interpretation="FFmpeg zscale display-referred transfer",
            ),
            input_preparation=dict(
                sample_aspect_ratio=str(sar),
                rotation_degrees=rotation,
                deinterlaced=interlaced,
                frame_rate_resampled=resample,
                hdr_tone_mapping=(
                    dict(method="hable", peak_nits=1000, reference_white_nits=100, desat=2)
                    if color["color_transfer"] in ("smpte2084", "arib-std-b67") else None
                ),
                filtergraph=decode[decode.index("-vf") + 1],
            ),
            audio_policy="discard",
            discarded_audio_streams=stream["audio_streams"],
            temporal_policy=(
                "independent effect frames, source cleanup mask history with motion/cut resets"
                if cleanup is not None or noise is not None
                else "independent frames, no stabilization"
            ),
            processed_rgb_sha256=processed_hash.hexdigest(),
            runtime=dict(
                python=platform.python_version(),
                platform=platform.system(),
                tools=tool_versions,
                dependencies={
                    name: version(name) for name in ("numpy", "scipy", "numba")
                },
            ),
            implementation_sources_sha256={
                name: _digest(Path(__file__).with_name(name)) for name in (
                    "video.py", "pipeline.py", "configuration.py", "cleanup.py",
                    "processing.py", "core.py", "numerics.py", "prenoise.py", "denoise.py", "gpu.py",
                )
            },
            seconds=time.perf_counter() - start,
        )
        if cleanup is not None:
            record["source_cleanup"] = dict(
                frames=cleanup.frames, mask_resets=cleanup.resets,
                seconds=cleanup.seconds, guide_max_edge=384,
                mode=settings.flat_cleanup_mode,
                policy="soft source mask, guided floating RGB/luminance, no temporal RGB averaging",
            )
        if denoiser is not None:
            record["source_denoise"] = dict(
                strength=settings.denoise, frames=denoiser.frames,
                seconds=denoiser.seconds, method="adaptive_multiscale",
                policy="floating RGB/luminance before cleanup and noise, independent frames",
            )
        if noise is not None:
            record["source_noise"] = dict(
                rms=settings.pre_noise_rms, kind=settings.pre_noise_kind,
                masked=settings.pre_noise_masked, seed=settings.pre_noise_seed,
                frames=noise.frames, mask_resets=noise.resets, seconds=noise.seconds,
                timing_scope="source cleanup, mask and noise combined",
                policy="floating RGB/luminance, fixed working-frame field, no motion compensation",
            )
        if config_path is not None:
            record["config_source"] = dict(
                path=str(config_path), sha256=_digest(config_path)
            )
            inherited = Config.source_paths(config_path)[1:]
            if inherited:
                record["config_source"]["inherits"] = [
                    dict(path=str(path), sha256=_digest(path)) for path in inherited
                ]
        temporary_meta.write_text(
            json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        # Exclusive creation also works on filesystems without hard-link support.
        created = []
        try:
            for existing, target in ((temporary, destination), (temporary_meta, sidecar)):
                with target.open("xb") as output:
                    created.append(target)
                    with existing.open("rb") as complete:
                        shutil.copyfileobj(complete, output, length=1024 * 1024)
        except BaseException:
            for target in reversed(created):
                target.unlink()
            raise
    return record
