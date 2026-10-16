"""Bounded-memory, silent CFR video processing through external FFmpeg tools."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
from fractions import Fraction
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import time

import numpy as np

from . import __version__
from .configuration import Config, VideoConfig, resolve_config
from .pipeline import process_frames
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


def _validate_color(info):
    if any(
        info.get(key) != "bt709"
        for key in ("color_space", "color_transfer", "color_primaries")
    ):
        raise ValueError(
            "Video requires explicit BT.709 matrix, transfer and primaries tags"
        )
    if info.get("color_range") not in ("tv", "pc"):
        raise ValueError("Video requires an explicit limited or full color range")
    if info.get("pix_fmt") not in (
        "yuv420p", "yuv422p", "yuv444p", "yuvj420p", "yuvj422p", "yuvj444p",
    ):
        raise ValueError("Video requires opaque 8-bit planar YUV SDR frames")


def _probe(path, executable):
    report = json.loads(
        _capture([
            str(executable), "-v", "error", "-show_streams", "-of", "json", str(path),
        ])
    )
    streams = report.get("streams", [])
    videos = [s for s in streams if s.get("codec_type") == "video"]
    if len(videos) != 1 or videos[0].get("disposition", {}).get("attached_pic"):
        raise ValueError("Input must contain exactly one video stream without cover art")
    stream = videos[0]
    _validate_color(stream)
    if min(stream.get("width", 0), stream.get("height", 0)) < 2:
        raise ValueError("Video dimensions must be at least 2 by 2")
    if stream.get("sample_aspect_ratio") != "1:1":
        raise ValueError("Only square-pixel video is supported")
    if stream.get("field_order") not in (None, "unknown", "progressive"):
        raise ValueError("Interlaced video is not supported")
    if stream.get("tags", {}).get("rotate") or any(
        s.get("side_data_type") == "Display Matrix"
        for s in stream.get("side_data_list", [])
    ):
        raise ValueError("Video display rotation/matrices are not supported")
    stream["audio_streams"] = sum(s.get("codec_type") == "audio" for s in streams)
    return stream


def _scan_timing(path, stream, executable):
    """Validate every decoded frame before output, with at most one tick of rounding."""
    rate = _fraction(stream.get("avg_frame_rate"), "frame rate")
    tick = _fraction(stream.get("time_base"), "time base")
    if tick >= 1 / (2 * rate):
        raise ValueError("Video time base is too coarse to validate constant frame rate")
    entries = (
        "pts,duration,pkt_duration,width,height,pix_fmt,color_range,color_space,"
        "color_transfer,color_primaries,interlaced_frame,sample_aspect_ratio"
    )
    count, first, previous = 0, None, None
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
            if "pts" not in fields:
                if "width" in fields:
                    raise ValueError(
                        "Every video frame must have a presentation timestamp"
                    )
                continue
            try:
                pts = int(fields["pts"]) * tick
            except ValueError as error:
                raise ValueError(
                    "Every video frame must have a presentation timestamp"
                ) from error
            if first is None:
                first = pts
            if (
                previous is not None and pts <= previous
            ) or abs(pts - first - count / rate) > tick:
                raise ValueError(
                    "Variable frame rate or discontinuous timestamps are not supported"
                )
            duration = fields.get("duration", fields.get("pkt_duration"))
            if duration not in (None, "N/A", "0") and (
                abs(int(duration) * tick - 1 / rate) > tick
            ):
                raise ValueError("Variable frame durations are not supported")
            if any(int(fields.get(key, 0)) != stream[key] for key in ("width", "height")):
                raise ValueError("Changing video dimensions are not supported")
            _validate_color(fields)
            if (
                fields.get("interlaced_frame") != "0"
                or fields.get("sample_aspect_ratio") != "1:1"
            ):
                raise ValueError("Only progressive square-pixel frames are supported")
            if fields["color_range"] != stream["color_range"]:
                raise ValueError("Changing video color range is not supported")
            previous = pts
            count += 1
        _check(p, log, "Video timing scan")
    if not count:
        raise ValueError("Input contains no decoded video frames")
    return rate, count, first


def _decode_command(executable, path, stream, deblock="off"):
    source_range = "limited" if stream["color_range"] == "tv" else "full"
    conversion = (
        "zscale=matrixin=709:transferin=709:primariesin=709:"
        f"rangein={source_range}:matrix=gbr:transfer=iec61966-2-1:"
        "primaries=709:range=full,format=gbrp,format=rgb24"
    )
    if deblock != "off":
        block = {"weak4": 4, "weak8": 8}[deblock]
        conversion = f"deblock=filter=weak:block={block}," + conversion
    return [
        str(executable), "-v", "error", "-xerror", "-nostdin", "-noautorotate",
        "-i", str(path), "-map", f"0:{stream['index']}", "-an", "-sn", "-dn",
        "-vf", conversion,
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
        "-c:v", "libx264", "-crf", str(settings.crf), "-preset", settings.preset,
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
    ffmpeg: str | Path = "ffmpeg",
    ffprobe: str | Path = "ffprobe",
) -> dict:
    """Process tagged BT.709, progressive, square-pixel CFR video into silent MP4.

    FFmpeg must provide zscale and libx264, plus deblock when enabled. Input
    timestamps are validated in a separate streaming pass, then rebased to zero
    without changing cadence.
    Optional weak deblocking runs on decoded YUV before RGB conversion.
    Image preparation/export settings do not apply to video. Frames enter the
    filter as full-range sRGB RGB8. Odd output edges are padded by replication.
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
    rate, expected, first = _scan_timing(source, stream, ffprobe)
    prepared = input_size(
        (stream["width"], stream["height"]),
        config.processing.effective_max_input_pixels,
    )
    width, height = (edge + edge % 2 for edge in prepared)
    tool_versions = {
        name: _capture([str(executable), "-version"]).splitlines()[0]
        for name, executable in (("ffmpeg", ffmpeg), ("ffprobe", ffprobe))
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    count, processed_hash = 0, hashlib.sha256()
    with tempfile.TemporaryDirectory(prefix=".luma-fatal-", dir=destination.parent) as directory:
        temporary = Path(directory) / "video.mp4"
        temporary_meta = Path(directory) / "video.meta.json"
        decode = _decode_command(ffmpeg, source, stream, settings.deblock)
        encode = _encode_command(ffmpeg, temporary, width, height, rate, settings)
        with _process(
            decode, stdout=subprocess.PIPE, stdin=subprocess.DEVNULL
        ) as (decoder, decode_log):
            with _process(
                encode, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL
            ) as (encoder, encode_log):
                frames = _decoded_frames(decoder.stdout, stream["width"], stream["height"])
                try:
                    for frame in process_frames(frames, config=config):
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
        if count != expected:
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
            source_dimensions=[stream["width"], stream["height"]],
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
                output_start_seconds=0,
                policy="validated CFR, rebased to zero",
            ),
            color_policy=dict(
                input={key: stream[key] for key in (
                    "color_space", "color_transfer", "color_primaries",
                    "color_range", "pix_fmt",
                )},
                processing="full-range sRGB RGB8",
                output="limited-range BT.709 YUV420P",
                converter="FFmpeg zscale",
                transfer_interpretation="display-referred BT.709 / BT.1886",
            ),
            audio_policy="discard",
            discarded_audio_streams=stream["audio_streams"],
            temporal_policy="independent frames, no stabilization",
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
                    "video.py", "pipeline.py", "configuration.py",
                    "processing.py", "core.py", "numerics.py",
                )
            },
            seconds=time.perf_counter() - start,
        )
        if config_path is not None:
            record["config_source"] = dict(
                path=str(config_path), sha256=_digest(config_path)
            )
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
