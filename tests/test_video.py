"""Video contracts use deterministic generated media, without external inputs."""

from fractions import Fraction
import hashlib
import io
import json
import os
import shutil
import signal
import subprocess
import sys

import numpy as np
import pytest

from luma_fatal import Config, FilterParams, NumericsConfig, ProcessingConfig, apply_filter
from luma_fatal.cli import main
from luma_fatal.video import VideoConfig, _decoded_frames, process_video


@pytest.fixture(scope="module")
def tools():
    paths = [shutil.which(name) for name in ("ffmpeg", "ffprobe")]
    if not all(paths):
        message = "Video integration tests require FFmpeg with zscale/libx264 and ffprobe"
        if os.environ.get("LUMA_FATAL_REQUIRE_VIDEO_TESTS") == "1":
            pytest.fail(message)
        pytest.skip(message)
    filters = run([paths[0], "-v", "error", "-filters"])
    encoders = run([paths[0], "-v", "error", "-encoders"])
    assert b"zscale" in filters, "FFmpeg needs the zscale filter"
    assert b"libx264" in encoders, "FFmpeg needs the libx264 encoder"
    return paths


def run(command, data=None):
    return subprocess.run(command, input=data, capture_output=True, check=True, timeout=30).stdout


def make_clip(path, tools, *, width=47, height=35, count=5, rate="25", vf=None,
              transfer="bt709", color_range="tv", audio=False, offset=0, neutral=False):
    """Encode lossless YUV planes, including a moving square and a cut."""
    frames = []
    for i in range(count):
        yy, xx = np.indices((height, width))
        y = (32 + (xx * 3 + yy * 2) % 160).astype(np.uint8)
        y[4:15, 3 + i:14 + i] = 210
        if i >= count // 2:
            y = 235 - y + 16
        u = np.full_like(y, 110 + i * 3)
        v = np.full_like(y, 145 - i * 2)
        if neutral:
            y = np.full_like(y, [16, 64, 128, 192, 235][i])
            u = v = np.full_like(y, 128)
        frames.append(np.stack([y, u, v]).tobytes())
    command = [
        tools[0], "-v", "error", "-nostdin", "-f", "rawvideo", "-pixel_format", "yuv444p",
        "-video_size", f"{width}x{height}", "-framerate", rate, "-i", "pipe:0",
    ]
    if audio:
        duration = float(count / Fraction(rate))
        command += [
            "-f", "lavfi", "-i",
            f"sine=frequency=440:sample_rate=8000:duration={duration}",
        ]
    command += ["-map", "0:v:0"]
    if audio:
        command += ["-map", "1:a:0", "-c:a", "pcm_s16le"]
    command += [
        "-vf", "setsar=1" + (f",{vf}" if vf else "") + (
            f",setparams=color_primaries=bt709:color_trc={transfer}:colorspace=bt709:"
            f"range={'limited' if color_range == 'tv' else 'full'}"
        ), "-c:v", "libx264",
        "-qp", "0", "-preset", "ultrafast", "-color_primaries", "bt709", "-color_trc", transfer,
        "-colorspace", "bt709", "-color_range", color_range, "-fps_mode", "passthrough",
        "-output_ts_offset", str(offset), "-f", "matroska", str(path),
    ]
    run(command, b"".join(frames))
    return path


def inspect(path, tools):
    return json.loads(run([
        tools[1], "-v", "error", "-show_streams", "-show_frames", "-of", "json",
        str(path),
    ]))


def reference_rgb(path, tools, width, height, color_range="limited"):
    # This is the documented video-to-filter boundary, before any Luma Fatal effects.
    data = run([
        tools[0], "-v", "error", "-i", str(path), "-map", "0:v:0", "-an",
        "-vf", "zscale=matrixin=709:transferin=709:primariesin=709:"
        f"rangein={color_range}:matrix=gbr:transfer=iec61966-2-1:"
        "primaries=709:range=full,format=gbrp,format=rgb24",
        "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ])
    return np.frombuffer(data, np.uint8).reshape(-1, height, width, 3)


@pytest.mark.parametrize("kwargs", [
    {"crf": True}, {"crf": -1}, {"crf": 52}, {"crf": 1.5}, {"preset": "invalid"},
])
def test_video_settings_reject_invalid_values(kwargs):
    with pytest.raises(ValueError):
        VideoConfig(**kwargs)


def test_decoder_handles_partial_reads_and_truncated_frames():
    class ShortReads(io.BytesIO):
        def read(self, size):
            return super().read(min(7, size))

    data = np.arange(72, dtype=np.uint8).tobytes()
    actual = list(_decoded_frames(ShortReads(data), 4, 3))
    np.testing.assert_array_equal(actual, np.arange(72, dtype=np.uint8).reshape(2, 3, 4, 3))
    with pytest.raises(ValueError, match="incomplete"):
        list(_decoded_frames(ShortReads(data[:-1]), 4, 3))


@pytest.mark.parametrize("filter_name", ["fattal1", "fattal2"])
@pytest.mark.parametrize("shadow_softness", [0, 6])
def test_video_matches_image_processing_and_preserves_timing(
    tmp_path, tools, filter_name, shadow_softness,
):
    source = make_clip(tmp_path / "synthetic.mkv", tools, audio=True, offset=2)
    original = source.read_bytes()
    destination = tmp_path / "nested" / "filtered.mp4"
    config = Config(
        filter=filter_name, parameters=FilterParams(fade=0),
        processing=ProcessingConfig(max_edge=24),
        numerics=NumericsConfig(iterations=11, shadow_softness=shadow_softness),
    )
    expected = hashlib.sha256()
    for frame in reference_rgb(source, tools, 47, 35):
        filtered = apply_filter(frame, config=config)
        expected.update(np.pad(filtered, ((0, 1), (0, 1), (0, 0)), mode="edge").tobytes())
    record = process_video(source, destination, config=config, ffmpeg=tools[0], ffprobe=tools[1])
    assert record["processed_rgb_sha256"] == expected.hexdigest()
    assert record["frame_count"] == 5
    assert Fraction(record["frame_rate"]) == 25
    assert record["duration_seconds"] == 0.2
    assert record["dimensions"] == [48, 36]
    assert record["prepared_dimensions"] == [47, 35]
    assert record["padding"] == {"right": 1, "bottom": 1, "mode": "edge"}
    assert Fraction(record["timing"]["source_first_pts_seconds"]) == 2
    assert record["discarded_audio_streams"] == 1
    assert source.read_bytes() == original
    assert record["source_sha256"] == hashlib.sha256(original).hexdigest()
    assert record["output_sha256"] == hashlib.sha256(destination.read_bytes()).hexdigest()
    assert json.loads(destination.with_suffix(".meta.json").read_text()) == record
    output = inspect(destination, tools)
    stream, = output["streams"]
    assert stream["codec_name"] == "h264"
    assert stream["pix_fmt"] == "yuv420p"
    assert stream["color_range"] == "tv"
    assert all(stream[key] == "bt709" for key in (
        "color_transfer", "color_primaries", "color_space",
    ))
    assert (stream["width"], stream["height"]) == (48, 36)
    frames = output["frames"]
    assert len(frames) == 5
    times = [int(frame["pts"]) * Fraction(stream["time_base"]) for frame in frames]
    assert times == [Fraction(i, 25) for i in range(5)]
    assert Fraction(stream["duration"]) == Fraction(1, 5)
    assert not list(destination.parent.glob(".luma-fatal-*"))


def test_fractional_rate_full_range_and_input_budget(tmp_path, tools):
    source = make_clip(tmp_path / "fractional.mkv", tools, rate="30000/1001", color_range="pc")
    config = Config(
        parameters=FilterParams(fade=100),
        processing=ProcessingConfig(max_input_pixels=400),
    )
    destination = tmp_path / "fractional.mp4"
    record = process_video(source, destination, config=config, ffmpeg=tools[0], ffprobe=tools[1])
    assert Fraction(record["frame_rate"]) == Fraction(30000, 1001)
    assert record["prepared_dimensions"] == [23, 17]
    assert record["dimensions"] == [24, 18]
    expected = hashlib.sha256()
    for frame in reference_rgb(source, tools, 47, 35, color_range="full"):
        processed = apply_filter(frame, config=config)
        expected.update(np.pad(processed, ((0, 1), (0, 1), (0, 0)), mode="edge").tobytes())
    assert record["processed_rgb_sha256"] == expected.hexdigest()
    output = inspect(destination, tools)
    stream, = output["streams"]
    assert Fraction(stream["avg_frame_rate"]) == Fraction(30000, 1001)
    assert len(output["frames"]) == 5


def test_cli_loads_config_and_applies_overrides(tmp_path, tools, capsys):
    source = make_clip(tmp_path / "cli.mkv", tools, width=48, height=36)
    config = tmp_path / "filter.yaml"
    config.write_text("filter: fattal1\nparameters:\n  fade: 0\n", encoding="utf-8")
    output = tmp_path / "cli.mp4"
    main([
        "video", str(source), str(output), "--config", str(config), "--filter", "fattal2",
        "--fade", "100", "--crf", "23", "--preset", "ultrafast",
        "--shadow-softness", "6",
        "--ffmpeg", tools[0], "--ffprobe", tools[1],
    ])
    record = json.loads(capsys.readouterr().out)
    assert record["filter"] == "fattal2"
    assert record["configuration"]["parameters"]["fade"] == 100
    assert record["configuration"]["numerics"]["shadow_softness"] == 6
    assert record["video_configuration"] == {"crf": 23, "preset": "ultrafast"}
    assert record["config_source"]["sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()
    assert record["padding"]["right"] == record["padding"]["bottom"] == 0


def test_color_transfer_and_encoded_roundtrip_use_known_neutral_values(
    tmp_path, tools, monkeypatch,
):
    import luma_fatal.video as video

    source = make_clip(tmp_path / "gray.mkv", tools, width=48, height=36, neutral=True)
    observed = []
    original = video.process_frames

    def observe(frames, *, config):
        for frame in original(frames, config=config):
            observed.append(frame.copy())
            yield frame

    monkeypatch.setattr(video, "process_frames", observe)
    destination = tmp_path / "gray.mp4"
    process_video(
        source, destination, config=Config(parameters=FilterParams(fade=100)),
        ffmpeg=tools[0], ffprobe=tools[1],
    )
    encoded = (np.array([16, 64, 128, 192, 235]) - 16) / 219
    # zscale interprets display-referred BT.709 through BT.1886 with ideal black.
    linear = encoded ** 2.4
    srgb = np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055)
    expected = np.rint(srgb * 255)
    for frame, gray in zip(observed, expected, strict=True):
        assert np.max(np.abs(frame.astype(float) - gray)) <= 1
    decoded = reference_rgb(destination, tools, 48, 36)
    for frame, gray in zip(decoded, expected, strict=True):
        assert np.max(np.abs(frame.astype(float) - gray)) <= 3


@pytest.mark.parametrize("case,match", [
    ("vfr", "Variable"), ("hdr", "BT.709"), ("untagged", "BT.709"),
    ("interlaced", "nterlaced|progressive"), ("anamorphic", "square-pixel"),
])
def test_unsupported_inputs_fail_without_outputs(tmp_path, tools, case, match):
    options = {
        "vfr": {"vf": "setpts='if(lt(N,2),N,N+2)/(25*TB)'"},
        "hdr": {"transfer": "smpte2084"},
        "untagged": {"transfer": "unknown"},
        "interlaced": {"vf": "setfield=tff"},
        "anamorphic": {"vf": "setsar=2"},
    }[case]
    source = make_clip(tmp_path / f"{case}.mkv", tools, **options)
    destination = tmp_path / "out.mp4"
    with pytest.raises(ValueError, match=match):
        process_video(source, destination, ffmpeg=tools[0], ffprobe=tools[1])
    assert not destination.exists()
    assert not destination.with_suffix(".meta.json").exists()
    assert not list(tmp_path.glob(".luma-fatal-*"))


@pytest.mark.parametrize("existing", ["output", "sidecar"])
def test_existing_files_are_preserved(tmp_path, existing):
    source = tmp_path / "input.mkv"
    source.write_bytes(b"source must not change")
    destination = tmp_path / "output.mp4"
    protected = destination if existing == "output" else destination.with_suffix(".meta.json")
    protected.write_bytes(b"keep this")
    with pytest.raises(ValueError, match="already exists"):
        process_video(source, destination)
    assert source.read_bytes() == b"source must not change"
    assert protected.read_bytes() == b"keep this"


def test_invalid_arguments_fail_before_starting_tools(tmp_path):
    source = tmp_path / "input.mp4"
    source.write_bytes(b"unchanged")
    for output, kwargs, match in [
        (source, {}, "paths must differ"),
        (tmp_path / "out.avi", {}, "extension"),
        (tmp_path / "out.mp4", {"config": Config(filter="both")}, "one filter"),
        (tmp_path / "out.mp4", {"video_config": {}}, "VideoConfig"),
    ]:
        with pytest.raises(ValueError, match=match):
            process_video(source, output, **kwargs)
    assert source.read_bytes() == b"unchanged"


def test_missing_tool_is_actionable(tmp_path):
    source = tmp_path / "input.mkv"
    source.write_bytes(b"placeholder")
    with pytest.raises(ValueError, match="Required executable not found"):
        process_video(source, tmp_path / "out.mp4", ffprobe=str(tmp_path / "missing-ffprobe"))


def test_filter_failure_cleans_up_processes_and_outputs(tmp_path, tools, monkeypatch):
    import luma_fatal.video as video

    source = make_clip(tmp_path / "failure.mkv", tools)

    def fail_after_one(frames, *, config):
        yield next(frames)
        raise RuntimeError("synthetic processing failure")

    monkeypatch.setattr(video, "process_frames", fail_after_one)
    destination = tmp_path / "failure.mp4"
    with pytest.raises(RuntimeError, match="synthetic processing failure"):
        process_video(source, destination, ffmpeg=tools[0], ffprobe=tools[1])
    assert not destination.exists()
    assert not destination.with_suffix(".meta.json").exists()
    assert not list(tmp_path.glob(".luma-fatal-*"))


@pytest.mark.skipif(os.name != "nt", reason="Windows launcher and file-lock regression")
def test_process_failure_stops_windows_launcher_children(tmp_path):
    from luma_fatal.video import _process

    locked = tmp_path / "child-output.mp4"
    child = (
        "import os, sys, time\n"
        "with open(sys.argv[1], 'wb') as output:\n"
        "    print(os.getpid(), flush=True)\n"
        "    time.sleep(30)\n"
    )
    launcher = (
        "import subprocess, sys\n"
        "subprocess.run([sys.executable, '-c', sys.argv[1], sys.argv[2]], check=True)\n"
    )
    child_pid = None
    try:
        with pytest.raises(RuntimeError, match="synthetic processing failure"):
            with _process(
                [sys.executable, "-c", launcher, child, str(locked)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            ) as (process, log):
                # The child announces readiness only after acquiring the file lock.
                child_pid = int(process.stdout.readline())
                raise RuntimeError("synthetic processing failure")
        assert process.poll() is not None
        assert process.stdin.closed
        assert process.stdout.closed
        # Windows refuses deletion while the launcher child still owns the file.
        locked.unlink()
    except BaseException:
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGTERM)
            except OSError:
                pass
        raise


def test_encoder_failure_reports_diagnostics_and_cleans_up(tmp_path, tools, monkeypatch):
    import luma_fatal.video as video

    source = make_clip(tmp_path / "failure.mkv", tools)
    original = video._encode_command

    def unavailable_codec(*args):
        command = original(*args)
        command[command.index("libx264")] = "luma-fatal-nonexistent-codec"
        return command

    monkeypatch.setattr(video, "_encode_command", unavailable_codec)
    destination = tmp_path / "failure.mp4"
    with pytest.raises(ValueError, match="Video encoding failed"):
        process_video(source, destination, ffmpeg=tools[0], ffprobe=tools[1])
    assert not destination.exists()
    assert not destination.with_suffix(".meta.json").exists()
    assert not list(tmp_path.glob(".luma-fatal-*"))


@pytest.mark.parametrize("case,match", [
    ("multiple", "exactly one"), ("rotation", "rotation"), ("ten-bit", "8-bit"),
])
def test_stream_layout_rotation_and_bit_depth_are_rejected(tmp_path, tools, case, match):
    source = make_clip(
        tmp_path / "source.mkv", tools,
        vf="format=yuv444p10le" if case == "ten-bit" else None,
    )
    if case != "ten-bit":
        remuxed = tmp_path / "remuxed.mp4"
        options = ["-map", "0:v", "-map", "0:v"] if case == "multiple" else []
        input_options = ["-display_rotation:v:0", "90"] if case == "rotation" else []
        run([
            tools[0], "-v", "error", *input_options, "-i", str(source), *options,
            "-c", "copy", str(remuxed),
        ])
        source = remuxed
    destination = tmp_path / "rejected.mp4"
    with pytest.raises(ValueError, match=match):
        process_video(source, destination, ffmpeg=tools[0], ffprobe=tools[1])
    assert not destination.exists()
    assert not destination.with_suffix(".meta.json").exists()


def test_sidecar_race_preserves_other_writer_and_rolls_back_video(tmp_path, tools, monkeypatch):
    from pathlib import Path

    source = make_clip(tmp_path / "source.mkv", tools)
    destination = tmp_path / "racing.mp4"
    sidecar = destination.with_suffix(".meta.json")
    original = Path.open

    def competing_writer(path, mode="r", *args, **kwargs):
        if path == sidecar and mode == "xb":
            with original(path, "wb") as other:
                other.write(b"other writer")
        return original(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", competing_writer)
    with pytest.raises(FileExistsError):
        process_video(source, destination, ffmpeg=tools[0], ffprobe=tools[1])
    assert not destination.exists()
    assert sidecar.read_bytes() == b"other writer"
    assert not list(tmp_path.glob(".luma-fatal-*"))
