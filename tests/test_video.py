"""Video contracts use deterministic generated media, without external inputs."""

from fractions import Fraction
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

import numpy as np
import pytest

from luma_fatal import Config, FilterParams, NumericsConfig, ProcessingConfig, apply_filter, process_frames
from luma_fatal.cleanup import SourceCleanup
from luma_fatal.denoise import SourceDenoise
from luma_fatal.cli import main
from luma_fatal.video import VideoConfig, _decoded_frames, _prefetched_frames, process_video


@pytest.mark.parametrize("cq,preset", [(0, "p1"), (18, "p4"), (51, "p7")])
def test_nvenc_uses_its_own_rate_control_and_preserves_output_contract(cq, preset):
    from luma_fatal.video import _encode_command

    command = _encode_command("ffmpeg", "output.mp4", 1920, 1080, Fraction(25), VideoConfig(
        encoder="h264_nvenc", nvenc_cq=cq, nvenc_preset=preset,
        crf=0, preset="ultrafast",
    ))
    assert command[command.index("-c:v") + 1] == "h264_nvenc"
    assert command[command.index("-preset") + 1] == preset
    assert command[command.index("-cq") + 1] == str(cq)
    assert command[command.index("-rc") + 1] == "vbr"
    assert command[command.index("-b:v") + 1] == "0"
    assert "-crf" not in command
    assert "ultrafast" not in command
    assert command[command.index("-color_range") + 1] == "tv"
    assert command[command.index("-color_trc") + 1] == "bt709"
    assert "format=yuv420p" in command[command.index("-vf") + 1]
    assert command[-3:] == ["-f", "mp4", "output.mp4"]


def test_default_encoder_retains_x264_quality_controls():
    from luma_fatal.video import _encode_command

    command = _encode_command("ffmpeg", "output.mp4", 1920, 1080, Fraction(25), VideoConfig(
        crf=0, preset="ultrafast", nvenc_cq=51, nvenc_preset="p7",
    ))
    assert command[command.index("-c:v") + 1] == "libx264"
    assert command[command.index("-crf") + 1] == "0"
    assert command[command.index("-preset") + 1] == "ultrafast"
    assert "-cq" not in command
    assert "-rc" not in command
    assert "-b:v" not in command


@pytest.mark.parametrize("suffix", ["yaml", "toml", "json"])
def test_nvenc_configuration_round_trip(tmp_path, suffix):
    config = Config(video=VideoConfig(
        encoder="h264_nvenc", nvenc_cq=23, nvenc_preset="p6", filter_backend="auto",
    ))
    content = {
        "yaml": config.to_yaml(), "toml": config.to_toml(),
        "json": json.dumps(config.to_dict()),
    }[suffix]
    path = tmp_path / f"nvenc.{suffix}"
    path.write_text(content)
    assert Config.load(path) == config


def test_cli_overrides_nvenc_configuration(tmp_path, monkeypatch, capsys):
    import luma_fatal.cli as cli

    path = tmp_path / "nvenc.yaml"
    path.write_text(Config(video=VideoConfig(
        encoder="libx264", nvenc_cq=28, nvenc_preset="p7",
    )).to_yaml())

    def observe(source, destination, *, config, **kwargs):
        return {"video_configuration": config.to_dict()["video"]}

    monkeypatch.setattr(cli, "process_video", observe)
    main([
        "video", "source.mp4", "output.mp4", "--config", str(path),
        "--encoder", "h264_nvenc", "--nvenc-cq", "21", "--nvenc-preset", "p3",
        "--filter-backend", "auto", "--json",
    ])
    settings = json.loads(capsys.readouterr().out)["video_configuration"]
    assert settings["encoder"] == "h264_nvenc"
    assert settings["nvenc_cq"] == 21
    assert settings["nvenc_preset"] == "p3"
    assert settings["crf"] == 18
    assert settings["preset"] == "medium"
    assert settings["filter_backend"] == "auto"


@pytest.mark.parametrize("diagnostic", [
    "Unknown encoder 'h264_nvenc'", "Cannot load libcuda", "No capable devices found",
])
@pytest.mark.parametrize("broken_pipe", [False, True])
def test_nvenc_preflight_failure_falls_back_to_cpu(
    tmp_path, tools, monkeypatch, capsys, diagnostic, broken_pipe,
):
    from contextlib import contextmanager
    import luma_fatal.video as video

    source = make_clip(tmp_path / "source.mkv", tools)
    original = video._process
    commands = []

    @contextmanager
    def unavailable(command, **kwargs):
        if "-c:v" in command:
            commands.append(command)
        if "h264_nvenc" not in command:
            with original(command, **kwargs) as result:
                yield result
            return

        class FailedInput(io.BytesIO):
            def write(self, data):
                if broken_pipe:
                    raise BrokenPipeError("Encoder failed during initialization")
                return super().write(data)

        class FailedEncoder:
            stdin = FailedInput()

            def wait(self):
                return 1

        yield FailedEncoder(), io.BytesIO(diagnostic.encode())

    monkeypatch.setattr(video, "_process", unavailable)
    config = Config(
        parameters=FilterParams(fade=100),
        video=VideoConfig(encoder="h264_nvenc", crf=23, preset="ultrafast"),
    )
    destination = tmp_path / "out" / "result.mp4"
    record = process_video(source, destination, config=config,
                           ffmpeg=tools[0], ffprobe=tools[1])
    assert "Falling back to CPU encoding (libx264)" in capsys.readouterr().err
    assert record["encoding"]["requested_encoder"] == "h264_nvenc"
    assert record["encoding"]["encoder"] == "libx264"
    assert diagnostic in record["encoding"]["fallback_reason"]
    assert record["configuration"]["video"]["encoder"] == "h264_nvenc"
    assert record["video_configuration"]["encoder"] == "h264_nvenc"
    assert config.video.encoder == "h264_nvenc"
    assert len(commands) == 2
    assert commands[0][commands[0].index("-s:v") + 1] == "48x36"
    cpu_command = commands[1]
    assert cpu_command[cpu_command.index("-c:v") + 1] == "libx264"
    assert cpu_command[cpu_command.index("-crf") + 1] == "23"
    assert cpu_command[cpu_command.index("-preset") + 1] == "ultrafast"
    expected = hashlib.sha256()
    for frame in reference_rgb(source, tools, 47, 35):
        expected.update(np.pad(frame, ((0, 1), (0, 1), (0, 0)), mode="edge").tobytes())
    assert record["processed_rgb_sha256"] == expected.hexdigest()
    assert record["frame_count"] == 5
    assert len(inspect(destination, tools)["frames"]) == 5
    assert json.loads(destination.with_suffix(".meta.json").read_text()) == record
    assert not list(destination.parent.glob(".luma-fatal-*"))


@pytest.mark.parametrize("force_cpu", [False, True])
def test_cli_yaml_encoder_fallback_and_cpu_override(tmp_path, tools, monkeypatch, capsys, force_cpu):
    import luma_fatal.video as video

    source = make_clip(tmp_path / "source.mkv", tools)
    preset = Path(__file__).resolve().parents[1] / "configs" / "fattal-1.yaml"
    config = tmp_path / "inherited.yaml"
    config.write_text(f"extends: {preset.as_posix()}\nparameters:\n  fade: 100\n")

    def unavailable(*args):
        if force_cpu:
            pytest.fail("Explicit CPU encoding must not probe NVIDIA hardware")
        raise ValueError("synthetic unavailable GPU")

    monkeypatch.setattr(video, "_validate_nvenc", unavailable)
    destination = tmp_path / "result.mp4"
    arguments = [
        "video", str(source), str(destination), "--config", str(config),
        "--ffmpeg", tools[0], "--ffprobe", tools[1], "--json",
    ]
    if force_cpu:
        arguments += ["--encoder", "libx264"]
    main(arguments)
    captured = capsys.readouterr()
    record = json.loads(captured.out)
    assert record["encoding"] == {
        "requested_encoder": "libx264" if force_cpu else "h264_nvenc",
        "encoder": "libx264",
        "fallback_reason": None if force_cpu else "synthetic unavailable GPU",
    }
    assert ("Falling back" in captured.err) is (not force_cpu)
    assert json.loads(destination.with_suffix(".meta.json").read_text()) == record


def test_cpu_encoding_failure_after_nvenc_fallback_publishes_no_output(tmp_path, tools, monkeypatch):
    import luma_fatal.video as video

    source = make_clip(tmp_path / "source.mkv", tools)
    original = video._encode_command

    def unavailable(*args):
        raise ValueError("synthetic unavailable GPU")

    def unavailable_cpu(*args):
        command = original(*args)
        command[command.index("libx264")] = "luma-fatal-nonexistent-codec"
        return command

    monkeypatch.setattr(video, "_validate_nvenc", unavailable)
    monkeypatch.setattr(video, "_encode_command", unavailable_cpu)
    destination = tmp_path / "result.mp4"
    with pytest.raises(ValueError, match="Video encoding failed"):
        process_video(source, destination, video_config=VideoConfig(encoder="h264_nvenc"),
                      ffmpeg=tools[0], ffprobe=tools[1])
    assert not destination.exists()
    assert not destination.with_suffix(".meta.json").exists()
    assert not list(tmp_path.glob(".luma-fatal-*"))


def test_nvenc_hardware_output_matches_cpu_filtering(tmp_path, tools):
    from luma_fatal.video import _validate_nvenc

    settings = VideoConfig(encoder="h264_nvenc", nvenc_cq=23, nvenc_preset="p2")
    try:
        _validate_nvenc(tools[0], 160, 128, Fraction(25), settings)
    except ValueError as error:
        if os.environ.get("LUMA_FATAL_REQUIRE_NVENC_TESTS") == "1":
            pytest.fail(str(error))
        pytest.skip(str(error))
    source = make_clip(tmp_path / "source.mkv", tools, width=159, height=127, audio=True)
    config = Config(processing=ProcessingConfig(max_edge=24), numerics=NumericsConfig(iterations=3))
    cpu = process_video(source, tmp_path / "cpu.mp4", config=config,
                        ffmpeg=tools[0], ffprobe=tools[1])
    destination = tmp_path / "gpu.mp4"
    gpu = process_video(source, destination, config=config, video_config=settings,
                        ffmpeg=tools[0], ffprobe=tools[1])
    assert gpu["processed_rgb_sha256"] == cpu["processed_rgb_sha256"]
    assert gpu["video_configuration"]["encoder"] == "h264_nvenc"
    assert gpu["video_configuration"]["nvenc_cq"] == 23
    assert gpu["video_configuration"]["nvenc_preset"] == "p2"
    assert gpu["encoding"] == {
        "requested_encoder": "h264_nvenc", "encoder": "h264_nvenc", "fallback_reason": None,
    }
    assert json.loads(destination.with_suffix(".meta.json").read_text()) == gpu
    output = inspect(destination, tools)
    stream, = output["streams"]
    assert stream["codec_name"] == "h264"
    assert stream["pix_fmt"] == "yuv420p"
    assert stream["color_range"] == "tv"
    assert all(stream[key] == "bt709" for key in (
        "color_transfer", "color_primaries", "color_space",
    ))
    assert (stream["width"], stream["height"]) == (160, 128)
    assert len(output["frames"]) == 5
    times = [int(frame["pts"]) * Fraction(stream["time_base"]) for frame in output["frames"]]
    assert times == [Fraction(i, 25) for i in range(5)]
    assert not list(tmp_path.glob(".luma-fatal-*"))


def test_gpu_filtering_fallback_preserves_source_treatments_and_records_selection(
    tmp_path, tools, monkeypatch, capsys,
):
    from dataclasses import replace
    from luma_fatal import gpu

    attempts = []

    def unavailable():
        attempts.append(1)
        raise gpu.GPUUnavailable("controlled missing CUDA runtime")

    monkeypatch.setattr(gpu, "CudaSolver", unavailable)
    source = make_clip(tmp_path / "source.mkv", tools, width=320, height=240, count=8)
    config = Config(
        processing=ProcessingConfig(resize=False),
        video=VideoConfig(
            filter_backend="auto", flat_cleanup=0.25, pre_noise_rms=0.2, denoise=0.5,
        ),
    )
    automatic = process_video(source, tmp_path / "auto.mp4", config=config,
                              ffmpeg=tools[0], ffprobe=tools[1])
    cpu_config = replace(config, video=replace(config.video, filter_backend="cpu"))
    cpu = process_video(source, tmp_path / "cpu.mp4", config=cpu_config,
                        ffmpeg=tools[0], ffprobe=tools[1])
    assert automatic["processed_rgb_sha256"] == cpu["processed_rgb_sha256"]
    assert automatic["filtering"]["requested_backend"] == "auto"
    assert automatic["filtering"]["backend"] == "cpu"
    assert automatic["filtering"]["cpu_frames"] == 8
    assert automatic["filtering"]["cuda_frames"] == 0
    assert automatic["filtering"]["reason"] == "controlled missing CUDA runtime"
    assert "gpu.py" in automatic["implementation_sources_sha256"]
    assert attempts == [1]
    assert "Using CPU filtering" in capsys.readouterr().err
    assert json.loads((tmp_path / "auto.meta.json").read_text()) == automatic


@pytest.fixture(scope="module")
def tools():
    paths = [shutil.which(name) for name in ("ffmpeg", "ffprobe")]
    if not all(paths):
        message = "Video integration tests require FFmpeg with zscale/deblock/deband/libx264 and ffprobe"
        if os.environ.get("LUMA_FATAL_REQUIRE_VIDEO_TESTS") == "1":
            pytest.fail(message)
        pytest.skip(message)
    filters = run([paths[0], "-v", "error", "-filters"])
    encoders = run([paths[0], "-v", "error", "-encoders"])
    assert b"zscale" in filters, "FFmpeg needs the zscale filter"
    assert b"deblock" in filters, "FFmpeg needs the deblock filter"
    assert b"deband" in filters, "FFmpeg needs the deband filter"
    assert b"libx264" in encoders, "FFmpeg needs the libx264 encoder"
    return paths


def run(command, data=None):
    return subprocess.run(command, input=data, capture_output=True, check=True, timeout=30).stdout


def make_clip(path, tools, *, width=47, height=35, count=5, rate="25", vf=None,
              transfer="bt709", color_range="tv", audio=False, offset=0, neutral=False,
              space="bt709", primaries="bt709"):
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
            f",setparams=color_primaries={primaries}:color_trc={transfer}:colorspace={space}:"
            f"range={'limited' if color_range == 'tv' else 'full'}"
        ), "-c:v", "libx264",
        "-qp", "0", "-preset", "ultrafast", "-color_primaries", primaries, "-color_trc", transfer,
        "-colorspace", space, "-color_range", color_range, "-fps_mode", "passthrough",
        "-output_ts_offset", str(offset), "-f", "matroska", str(path),
    ]
    run(command, b"".join(frames))
    return path


def inspect(path, tools):
    return json.loads(run([
        tools[1], "-v", "error", "-show_streams", "-show_frames", "-of", "json",
        str(path),
    ]))


def reference_rgb(path, tools, width, height, color_range="limited", deblock="off",
                  deband="off"):
    # This is the documented video-to-filter boundary, before any Luma Fatal effects.
    conversion = (
        "zscale=matrixin=709:transferin=709:primariesin=709:"
        f"rangein={color_range}:matrix=gbr:transfer=iec61966-2-1:"
        "primaries=709:range=full,format=gbrp,format=rgb24"
    )
    prefilters = []
    if deblock != "off":
        prefilters.append(f"deblock=filter=weak:block={int(deblock[-1])}")
    if deband == "gentle":
        prefilters.append(
            "deband=1thr=0.008:2thr=0.008:3thr=0.008:range=8"
        )
    if prefilters:
        conversion = ",".join(prefilters) + "," + conversion
    data = run([
        tools[0], "-v", "error", "-i", str(path), "-map", "0:v:0", "-an",
        "-vf", conversion,
        "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ])
    return np.frombuffer(data, np.uint8).reshape(-1, height, width, 3)


@pytest.mark.parametrize("kwargs", [
    {"crf": True}, {"crf": -1}, {"crf": 52}, {"crf": 1.5}, {"preset": "invalid"},
    {"deblock": "strong"}, {"deblock": True},
    {"deband": "strong"}, {"deband": True},
    {"flat_cleanup": True}, {"flat_cleanup": -0.1},
    {"flat_cleanup": 1.1}, {"flat_cleanup": float("nan")},
    {"frame_rate": 30}, {"frame_rate": True}, {"frame_rate": "0"},
    {"frame_rate": "-25"}, {"frame_rate": "1/0"}, {"frame_rate": "nan"},
    {"frame_rate": "25,fps=30"}, {"input_color": "unknown"},
    {"input_range": "tv"},
    {"encoder": "auto"}, {"encoder": True},
    {"nvenc_cq": True}, {"nvenc_cq": -1}, {"nvenc_cq": 52}, {"nvenc_cq": 18.5},
    {"nvenc_preset": "medium"}, {"nvenc_preset": "p8"}, {"nvenc_preset": True},
    {"filter_backend": "cuda"}, {"filter_backend": True},
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


def test_pinned_decoder_respects_exact_frame_boundaries(monkeypatch):
    import sys
    from types import SimpleNamespace

    class ShortReads(io.BytesIO):
        def readinto(self, target):
            return super().readinto(target[:7])

    fake = SimpleNamespace(cuda=SimpleNamespace(
        alloc_pinned_memory=lambda size: bytearray(size + 16)))
    monkeypatch.setitem(sys.modules, "cupy", fake)
    source = np.arange(72, dtype=np.uint8).tobytes()
    frames = list(_prefetched_frames(
        _decoded_frames(ShortReads(source), 4, 3, pinned=True), 2))
    np.testing.assert_array_equal(np.stack(frames),
                                  np.arange(72, dtype=np.uint8).reshape(2, 3, 4, 3))
    with pytest.raises(ValueError, match="incomplete"):
        list(_prefetched_frames(
            _decoded_frames(ShortReads(source[:-1]), 4, 3, pinned=True), 2))


@pytest.mark.parametrize("field,value", [
    ("calibration_cache", 1), ("gpu_iterations", 0),
    ("gpu_iterations", True), ("lighting_scale", 0.2),
    ("lighting_scale", True), ("decode_prefetch", 17),
    ("decode_prefetch", -1), ("pinned_frames", 1),
])
def test_fast_video_options_reject_invalid_values(field, value):
    with pytest.raises(ValueError):
        VideoConfig(**{field: value})


@pytest.mark.parametrize("filter_name", ["fattal1", "fattal2"])
@pytest.mark.parametrize(
    "shadow_softness,luminance_mode",
    [(0, "rounded"), (6, "rounded"), (6, "fractional")],
)
def test_video_matches_image_processing_and_preserves_timing(
    tmp_path, tools, filter_name, shadow_softness, luminance_mode,
):
    source = make_clip(tmp_path / "synthetic.mkv", tools, audio=True, offset=2)
    original = source.read_bytes()
    destination = tmp_path / "nested" / "filtered.mp4"
    config = Config(
        filter=filter_name, parameters=FilterParams(fade=0),
        processing=ProcessingConfig(max_edge=24),
        numerics=NumericsConfig(
            iterations=11, shadow_softness=shadow_softness,
            luminance_mode=luminance_mode,
        ),
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


@pytest.mark.parametrize("filter_name", ["fattal1", "fattal2"])
@pytest.mark.parametrize("deblock", ["weak4", "weak8"])
def test_video_deblocking_precedes_filter_and_records_setting(
    tmp_path, tools, filter_name, deblock,
):
    source = make_clip(tmp_path / "blocks.mkv", tools)
    config = Config(
        filter=filter_name,
        parameters=FilterParams(fade=0),
        processing=ProcessingConfig(max_edge=24),
        numerics=NumericsConfig(iterations=11, shadow_softness=6),
    )
    expected = hashlib.sha256()
    decoded = reference_rgb(source, tools, 47, 35, deblock=deblock)
    assert np.any(decoded != reference_rgb(source, tools, 47, 35))
    for frame in decoded:
        filtered = apply_filter(frame, config=config)
        expected.update(np.pad(filtered, ((0, 1), (0, 1), (0, 0)), mode="edge").tobytes())
    destination = tmp_path / "filtered.mp4"
    record = process_video(
        source, destination, config=config, video_config=VideoConfig(deblock=deblock),
        ffmpeg=tools[0], ffprobe=tools[1],
    )
    assert record["processed_rgb_sha256"] == expected.hexdigest()
    assert record["video_configuration"]["deblock"] == deblock
    assert record["frame_count"] == len(decoded)


@pytest.mark.parametrize("variant", [1, 2])
@pytest.mark.parametrize("deband", ["gentle", "off"])
@pytest.mark.parametrize("flat_cleanup", [0.75, 0])
def test_compressed_preset_video_defaults_and_cli_overrides(
    tmp_path, tools, capsys, variant, deband, flat_cleanup,
):
    source = make_clip(tmp_path / "compressed.mkv", tools)
    preset = (
        Path(__file__).resolve().parents[1]
        / "configs" / f"fattal-{variant}-compressed.yaml"
    )
    config = Config.load(preset)
    assert np.any(
        reference_rgb(source, tools, 47, 35, deband="gentle")
        != reference_rgb(source, tools, 47, 35)
    )
    destination = tmp_path / "filtered.mp4"
    command = [
        "video", str(source), str(destination), "--config", str(preset),
        "--ffmpeg", tools[0], "--ffprobe", tools[1], "--json",
    ]
    if deband == "off":
        command += ["--deband", "off"]
    if flat_cleanup == 0:
        command += ["--flat-cleanup", "0"]
    main(command)
    record = json.loads(capsys.readouterr().out)
    expected = hashlib.sha256()
    decoded = reference_rgb(source, tools, 47, 35, deband=deband)
    cleanup = SourceCleanup(flat_cleanup) if flat_cleanup else None
    cleanup = SourceDenoise(config.video.denoise, source_cleanup=cleanup)
    for frame in process_frames(decoded, config=config, source_cleanup=cleanup):
        expected.update(
            np.pad(frame, ((0, 1), (0, 1), (0, 0)), mode="edge").tobytes()
        )
    assert record["processed_rgb_sha256"] == expected.hexdigest()
    assert record["video_configuration"]["deblock"] == "off"
    assert record["video_configuration"]["deband"] == deband
    assert record["configuration"]["video"]["deband"] == deband
    assert record["video_configuration"]["flat_cleanup"] == flat_cleanup
    assert record["configuration"]["video"]["flat_cleanup"] == flat_cleanup
    assert record["configuration"]["numerics"]["shadow_softness"] == 6
    if flat_cleanup:
        assert record["source_cleanup"]["frames"] == len(decoded)
    else:
        assert "source_cleanup" not in record


def test_progress_reports_frames_without_changing_video(tmp_path, tools, capsys):
    source = make_clip(tmp_path / "progress.mkv", tools)
    config = Config(processing=ProcessingConfig(max_edge=24))
    plain = process_video(
        source, tmp_path / "plain.mp4", config=config,
        ffmpeg=tools[0], ffprobe=tools[1],
    )
    shown = process_video(
        source, tmp_path / "shown.mp4", config=config, progress=True,
        ffmpeg=tools[0], ffprobe=tools[1],
    )
    assert plain["processed_rgb_sha256"] == shown["processed_rgb_sha256"]
    assert plain["output_sha256"] == shown["output_sha256"]
    assert "Filtering frames" in capsys.readouterr().err


@pytest.mark.parametrize("variant", [1, 2])
def test_cli_cleanup_matches_working_frame_pipeline_and_records_state(
    tmp_path, tools, capsys, variant,
):
    source = make_clip(tmp_path / "smooth.mkv", tools, width=48, height=36, vf="boxblur=5:1")
    config = Config(
        filter=f"fattal{variant}", parameters=FilterParams(fade=0),
        processing=ProcessingConfig(max_edge=24),
        numerics=NumericsConfig(iterations=11, shadow_softness=6),
        video=VideoConfig(deband="gentle"),
    )
    path = tmp_path / "cleanup.toml"
    path.write_text(config.to_toml())
    expected = hashlib.sha256()
    decoded = reference_rgb(source, tools, 48, 36, deband="gentle")
    for frame in process_frames(decoded, config=config, source_cleanup=SourceCleanup(0.25)):
        expected.update(frame.tobytes())
    destination = tmp_path / "cleaned.mp4"
    main([
        "video", str(source), str(destination), "--config", str(path),
        "--flat-cleanup", "0.25", "--ffmpeg", tools[0], "--ffprobe", tools[1],
        "--json",
    ])
    record = json.loads(capsys.readouterr().out)
    assert record["processed_rgb_sha256"] == expected.hexdigest()
    assert record["configuration"]["video"]["flat_cleanup"] == 0.25
    assert record["video_configuration"]["flat_cleanup"] == 0.25
    assert record["source_cleanup"]["frames"] == 5
    assert record["source_cleanup"]["guide_max_edge"] == 384
    assert record["source_cleanup"]["seconds"] > 0
    assert "mask history" in record["temporal_policy"]
    assert len(inspect(destination, tools)["frames"]) == 5


@pytest.mark.parametrize("fade,prepared,dimensions", [
    (0, [23, 17], [24, 18]), (100, [47, 35], [48, 36]),
])
def test_fractional_rate_full_range_and_input_budget(tmp_path, tools, fade, prepared, dimensions):
    source = make_clip(tmp_path / "fractional.mkv", tools, rate="30000/1001", color_range="pc")
    config = Config(
        parameters=FilterParams(fade=fade),
        processing=ProcessingConfig(max_input_pixels=400),
    )
    destination = tmp_path / "fractional.mp4"
    record = process_video(source, destination, config=config, ffmpeg=tools[0], ffprobe=tools[1])
    assert Fraction(record["frame_rate"]) == Fraction(30000, 1001)
    assert record["prepared_dimensions"] == prepared
    assert record["dimensions"] == dimensions
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
        "--deblock", "weak8",
        "--progress",
        "--ffmpeg", tools[0], "--ffprobe", tools[1],
    ])
    captured = capsys.readouterr()
    assert "Filtering frames" in captured.err
    assert "Processing video:" in captured.out
    assert "Processed video with Fattal 2" in captured.out
    assert "5 frames" in captured.out
    assert str(output.with_suffix(".meta.json")) in captured.out
    assert "implementation_sources_sha256" not in captured.out
    record = json.loads(output.with_suffix(".meta.json").read_text())
    assert record["filter"] == "fattal2"
    assert record["configuration"]["parameters"]["fade"] == 100
    assert record["configuration"]["numerics"]["shadow_softness"] == 6
    assert record["video_configuration"] == {
        "calibration_cache": False, "gpu_iterations": None,
        "lighting_scale": 1.0,
        "decode_prefetch": 0, "pinned_frames": False,
        "filter_backend": "cpu",
        "encoder": "libx264", "nvenc_cq": 18, "nvenc_preset": "p4",
        "crf": 23, "preset": "ultrafast", "deblock": "weak8", "deband": "off",
        "denoise": 0.0,
        "flat_cleanup": 0.0,
        "flat_cleanup_mode": "standard", "flat_cleanup_coarse_strength": 1.0,
        "flat_cleanup_sigma": 1.2,
        "pre_noise_rms": 0.0, "pre_noise_kind": "highpass",
        "pre_noise_masked": True, "pre_noise_seed": 20261017,
        "frame_rate": "auto", "input_color": "auto", "input_range": "auto",
    }
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


def test_multiple_video_streams_are_rejected(tmp_path, tools):
    original = make_clip(tmp_path / "source.mkv", tools)
    source = tmp_path / "multiple.mp4"
    run([
        tools[0], "-v", "error", "-i", str(original), "-map", "0:v", "-map", "0:v",
        "-c", "copy", str(source),
    ])
    destination = tmp_path / "rejected.mp4"
    with pytest.raises(ValueError, match="exactly one"):
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


def test_variable_rate_repeats_frames_at_nominal_cadence(tmp_path, tools):
    source = make_clip(
        tmp_path / "vfr.mkv", tools, offset=2,
        vf="setpts='if(lt(N,2),N,N+2)/(25*TB)'",
    )
    config = Config(parameters=FilterParams(fade=100))
    decoded = reference_rgb(source, tools, 47, 35)
    expected = hashlib.sha256()
    for index in (0, 1, 1, 1, 2, 3, 4):
        expected.update(np.pad(decoded[index], ((0, 1), (0, 1), (0, 0)), mode="edge").tobytes())
    destination = tmp_path / "vfr.mp4"
    record = process_video(source, destination, config=config, ffmpeg=tools[0], ffprobe=tools[1])
    assert record["frame_rate"] == "25"
    assert record["frame_count"] == 7
    assert record["processed_rgb_sha256"] == expected.hexdigest()
    assert record["timing"]["source_frame_count"] == 5
    assert record["timing"]["source_variable_frame_rate"]
    output = inspect(destination, tools)
    assert [Fraction(f["pts"]) * Fraction(output["streams"][0]["time_base"])
            for f in output["frames"]] == [Fraction(i, 25) for i in range(7)]


@pytest.mark.parametrize("frame_rate,count", [("50", 10), ("30000/1001", 6), ("10", 2)])
def test_cli_frame_rate_overrides_config(tmp_path, tools, frame_rate, count):
    source = make_clip(tmp_path / "source.mkv", tools, width=48, height=36)
    config = Config(video=VideoConfig(frame_rate="5"), parameters=FilterParams(fade=100))
    path = tmp_path / "video.yaml"
    path.write_text(config.to_yaml())
    destination = tmp_path / "rate.mp4"
    main([
        "video", str(source), str(destination), "--config", str(path),
        "--frame-rate", frame_rate, "--ffmpeg", tools[0], "--ffprobe", tools[1],
    ])
    record = json.loads(destination.with_suffix(".meta.json").read_text())
    assert record["frame_rate"] == frame_rate
    assert record["configuration"]["video"]["frame_rate"] == frame_rate
    assert record["frame_count"] == count
    output = inspect(destination, tools)
    assert Fraction(output["streams"][0]["avg_frame_rate"]) == Fraction(frame_rate)
    assert len(output["frames"]) == count
    assert [Fraction(f["pts"]) * Fraction(output["streams"][0]["time_base"])
            for f in output["frames"]] == [i / Fraction(frame_rate) for i in range(count)]


@pytest.mark.parametrize("rotation", [90, 180, 270])
def test_rotation_matches_array_rotation(tmp_path, tools, rotation):
    original = make_clip(tmp_path / "source.mkv", tools)
    source = tmp_path / "rotated.mp4"
    run([
        tools[0], "-v", "error", "-display_rotation:v:0", str(rotation),
        "-i", str(original), "-c", "copy", str(source),
    ])
    expected = hashlib.sha256()
    for frame in reference_rgb(original, tools, 47, 35):
        rotated = np.rot90(frame, rotation // 90)
        expected.update(np.pad(rotated, ((0, 1), (0, 1), (0, 0)), mode="edge").tobytes())
    record = process_video(
        source, tmp_path / "result.mp4", config=Config(parameters=FilterParams(fade=100)),
        ffmpeg=tools[0], ffprobe=tools[1],
    )
    assert record["processed_rgb_sha256"] == expected.hexdigest()
    assert record["normalized_dimensions"] == ([35, 47] if rotation != 180 else [47, 35])
    assert record["input_preparation"]["rotation_degrees"] == rotation


def test_anamorphic_video_becomes_square_pixel_without_changing_display_aspect(tmp_path, tools):
    source = make_clip(tmp_path / "anamorphic.mkv", tools, vf="setsar=2")
    destination = tmp_path / "square.mp4"
    record = process_video(source, destination, ffmpeg=tools[0], ffprobe=tools[1])
    assert record["source_dimensions"] == [47, 35]
    assert record["normalized_dimensions"] == [94, 35]
    assert record["dimensions"] == [94, 36]
    assert record["input_preparation"]["sample_aspect_ratio"] == "2"
    stream, = inspect(destination, tools)["streams"]
    assert stream["sample_aspect_ratio"] == "1:1"


def test_interlaced_video_is_deinterlaced_before_filtering(tmp_path, tools):
    source = make_clip(tmp_path / "interlaced.mkv", tools, vf="setfield=tff")
    observed = run([
        tools[0], "-v", "error", "-i", str(source), "-vf",
        "bwdif=mode=send_frame:parity=auto:deint=interlaced,"
        "zscale=matrixin=709:transferin=709:primariesin=709:rangein=limited:"
        "matrix=gbr:transfer=iec61966-2-1:primaries=709:range=full,format=gbrp,format=rgb24",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ])
    expected = hashlib.sha256()
    for frame in np.frombuffer(observed, np.uint8).reshape(-1, 35, 47, 3):
        expected.update(np.pad(frame, ((0, 1), (0, 1), (0, 0)), mode="edge").tobytes())
    destination = tmp_path / "progressive.mp4"
    record = process_video(
        source, destination, config=Config(parameters=FilterParams(fade=100)),
        ffmpeg=tools[0], ffprobe=tools[1],
    )
    assert record["processed_rgb_sha256"] == expected.hexdigest()
    assert record["input_preparation"]["deinterlaced"]
    assert record["frame_count"] == 5
    assert all(f["interlaced_frame"] == 0 for f in inspect(destination, tools)["frames"])


@pytest.mark.parametrize("pixel_format", ["yuv444p10le", "yuv444p12le"])
def test_high_bit_depth_sdr_converts_before_filtering(tmp_path, tools, pixel_format):
    # FFV1 retains 12-bit data, which the local H.264 encoder cannot encode.
    original = make_clip(tmp_path / "original.mkv", tools)
    source = tmp_path / "deep.mkv"
    run([
        tools[0], "-v", "error", "-i", str(original), "-vf", f"format={pixel_format}",
        "-c:v", "ffv1", str(source),
    ])
    decoded = reference_rgb(source, tools, 47, 35)
    expected = hashlib.sha256()
    for frame in decoded:
        expected.update(np.pad(frame, ((0, 1), (0, 1), (0, 0)), mode="edge").tobytes())
    record = process_video(
        source, tmp_path / "deep.mp4", config=Config(parameters=FilterParams(fade=100)),
        ffmpeg=tools[0], ffprobe=tools[1],
    )
    assert record["processed_rgb_sha256"] == expected.hexdigest()
    assert record["color_policy"]["input"]["pix_fmt"] == pixel_format


@pytest.mark.parametrize("transfer", ["smpte2084", "arib-std-b67"])
def test_hdr_tone_mapping_produces_sdr_and_records_policy(tmp_path, tools, monkeypatch, transfer):
    import luma_fatal.video as video

    source = make_clip(
        tmp_path / "hdr.mkv", tools, width=48, height=36, neutral=True,
        vf="format=yuv444p10le", transfer=transfer, space="bt2020nc", primaries="bt2020",
    )
    observed = []

    def observe(frames, *, config):
        for frame in frames:
            observed.append(frame.copy())
            yield frame

    monkeypatch.setattr(video, "process_frames", observe)
    destination = tmp_path / "sdr.mp4"
    record = process_video(source, destination, ffmpeg=tools[0], ffprobe=tools[1])
    grays = np.array([float(frame.mean()) for frame in observed])
    assert grays[0] < 2
    assert np.all(np.diff(grays) >= 0)
    assert len(set(np.rint(grays))) >= 4
    assert record["input_preparation"]["hdr_tone_mapping"] == {
        "method": "hable", "peak_nits": 1000, "reference_white_nits": 100, "desat": 2,
    }
    output = inspect(destination, tools)["streams"][0]
    assert output["color_transfer"] == output["color_space"] == output["color_primaries"] == "bt709"
    if transfer == "smpte2084":
        # ST 2084 EOTF, then Hable normalized to the documented 1000-nit peak.
        encoded = (np.array([16, 64, 128, 192, 235]) - 16) / 219
        power = encoded ** (1 / (2523 / 32))
        linear = (np.maximum(power - 3424 / 4096, 0) /
                  (2413 / 128 - 2392 / 128 * power)) ** (1 / (2610 / 16384)) * 100

        def hable(x):
            return (x * (0.15 * x + 0.05) + 0.004) / (x * (0.15 * x + 0.5) + 0.06) - 0.02 / 0.3

        mapped = np.clip(hable(linear) / hable(10), 0, 1)
        expected = np.where(mapped <= 0.0031308, mapped * 12.92,
                            1.055 * mapped ** (1 / 2.4) - 0.055) * 255
        np.testing.assert_allclose(grays, expected, atol=3)


@pytest.mark.parametrize("deband", ["off", "gentle"])
def test_untagged_rgb_assumes_srgb_and_preserves_values(tmp_path, tools, monkeypatch, deband):
    import luma_fatal.video as video

    values = np.array([[20, 80, 160], [200, 150, 60], [100, 100, 100]], np.uint8)
    frames = np.repeat(np.repeat(values[:, None, None, :], 36, axis=1), 48, axis=2)
    source = tmp_path / "rgb.mkv"
    run([
        tools[0], "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "48x36",
        "-r", "25", "-i", "pipe:0", "-c:v", "ffv1", "-pix_fmt", "bgr0", str(source),
    ], frames.tobytes())
    observed = []

    def observe(decoded, *, config):
        for frame in decoded:
            observed.append(frame.copy())
            yield frame

    monkeypatch.setattr(video, "process_frames", observe)
    record = process_video(
        source, tmp_path / "rgb.mp4", video_config=VideoConfig(deband=deband),
        ffmpeg=tools[0], ffprobe=tools[1],
    )
    np.testing.assert_allclose(np.array(observed), frames, atol=1)
    assert record["color_policy"]["resolved_input"] == {
        "color_space": "gbr", "color_transfer": "iec61966-2-1",
        "color_primaries": "bt709", "color_range": "pc",
    }
    assert record["color_policy"]["assumptions"]["color_transfer"] == "iec61966-2-1"


@pytest.mark.parametrize("height,profile", [(480, "smpte170m"), (576, "bt470bg"), (720, "bt709")])
def test_untagged_yuv_uses_documented_geometry_defaults(height, profile):
    from luma_fatal.video import _resolve_color

    # Resolve HD/SD assumptions without encoding large clips for each height.
    color, assumptions = _resolve_color(
        dict(width=720, height=height, pix_fmt="yuv420p"), VideoConfig(),
    )
    assert color["color_space"] == profile
    assert color["color_range"] == "tv"
    assert assumptions == color


def test_cli_color_overrides_replace_incorrect_tags(tmp_path, tools):
    source = make_clip(tmp_path / "wrong-tags.mkv", tools)
    config = Config(video=VideoConfig(input_color="bt601-ntsc", input_range="full"))
    path = tmp_path / "settings.toml"
    path.write_text(config.to_toml())
    destination = tmp_path / "corrected.mp4"
    main([
        "video", str(source), str(destination), "--config", str(path),
        "--input-color", "bt709", "--input-range", "limited",
        "--ffmpeg", tools[0], "--ffprobe", tools[1],
    ])
    record = json.loads(destination.with_suffix(".meta.json").read_text())
    assert record["color_policy"]["resolved_input"] == {
        "color_space": "bt709", "color_transfer": "bt709",
        "color_primaries": "bt709", "color_range": "tv",
    }
    assert record["video_configuration"]["input_color"] == "bt709"
    assert record["video_configuration"]["input_range"] == "limited"
    assert record["color_policy"]["assumptions"] == {}


def test_preparation_failure_reports_diagnostics_and_leaves_no_outputs(tmp_path, tools, monkeypatch):
    import luma_fatal.video as video

    source = make_clip(tmp_path / "source.mkv", tools, vf="format=yuv444p10le")
    original = video._decode_command

    def fail(*args, **kwargs):
        command = original(*args, **kwargs)
        command[command.index("-vf") + 1] += ",luma_fatal_missing_filter"
        return command

    monkeypatch.setattr(video, "_decode_command", fail)
    destination = tmp_path / "failure.mp4"
    with pytest.raises(ValueError, match="Video decoding failed"):
        process_video(source, destination, ffmpeg=tools[0], ffprobe=tools[1])
    assert not destination.exists()
    assert not destination.with_suffix(".meta.json").exists()
    assert not list(tmp_path.glob(".luma-fatal-*"))


@pytest.mark.parametrize("space,transfer,primaries", [
    ("unknown", "unknown", "unknown"),
    ("bt470bg", "bt709", "bt470bg"),
    ("smpte170m", "smpte170m", "smpte170m"),
    ("bt2020nc", "bt2020-10", "bt2020"),
])
def test_yuv_color_profiles_and_missing_tags(tmp_path, tools, space, transfer, primaries):
    source = make_clip(
        tmp_path / "source.mkv", tools, space=space, transfer=transfer, primaries=primaries,
    )
    record = process_video(source, tmp_path / "result.mp4", ffmpeg=tools[0], ffprobe=tools[1])
    resolved = record["color_policy"]["resolved_input"]
    if space == "unknown":
        assert resolved["color_space"] == "smpte170m"
        assert record["color_policy"]["assumptions"] == {
            "color_space": "smpte170m", "color_transfer": "smpte170m",
            "color_primaries": "smpte170m",
        }
    else:
        assert resolved["color_space"] == space
        assert resolved["color_transfer"] == transfer
        assert resolved["color_primaries"] == primaries
        assert record["color_policy"]["assumptions"] == {}
    assert record["frame_count"] == 5


@pytest.mark.parametrize("pixel_format,codec,container", [
    ("nv12", "rawvideo", "nut"), ("gray", "ffv1", "matroska"),
    ("gray16le", "ffv1", "matroska"), ("bgra", "ffv1", "matroska"),
])
def test_other_common_pixel_formats(tmp_path, tools, pixel_format, codec, container):
    original = make_clip(tmp_path / "source.mkv", tools, width=48, height=36)
    source = tmp_path / "converted.bin"
    run([
        tools[0], "-v", "error", "-i", str(original), "-pix_fmt", pixel_format,
        "-c:v", codec, "-f", container, str(source),
    ])
    record = process_video(source, tmp_path / "result.mp4", ffmpeg=tools[0], ffprobe=tools[1])
    assert record["frame_count"] == 5
    assert record["dimensions"] == [48, 36]


@pytest.mark.parametrize("angle", [45, -30])
def test_non_right_angle_rotation_fails_without_outputs(tmp_path, tools, angle):
    original = make_clip(tmp_path / "source.mkv", tools)
    source = tmp_path / "tilted.mp4"
    run([
        tools[0], "-v", "error", "-display_rotation:v:0", str(angle),
        "-i", str(original), "-c", "copy", str(source),
    ])
    destination = tmp_path / "rejected.mp4"
    with pytest.raises(ValueError, match="rotation"):
        process_video(source, destination, ffmpeg=tools[0], ffprobe=tools[1])
    assert not destination.exists()
    assert not destination.with_suffix(".meta.json").exists()


@pytest.mark.parametrize("pts,changes,match", [
    ([0, 0], {}, "increase strictly"),
    ([0, None], {}, "presentation timestamp"),
    ([0, 40], {"width": "49"}, "Changing video dimensions"),
    ([0, 40], {"color_range": "pc"}, "Changing video color"),
    ([0, 40], {"sample_aspect_ratio": "2:1"}, "Changing video sample aspect"),
])
def test_preflight_still_rejects_ambiguous_or_changing_sources(monkeypatch, pts, changes, match):
    from contextlib import contextmanager
    import luma_fatal.video as video

    stream = dict(
        index=0, width=48, height=36, pix_fmt="yuv420p", color_space="bt709",
        color_transfer="bt709", color_primaries="bt709", color_range="tv",
        time_base="1/1000", avg_frame_rate="25", sample_aspect_ratio="1:1",
    )
    frames = []
    for i, timestamp in enumerate(pts):
        info = {**stream, "pts": timestamp, "interlaced_frame": 0}
        if i:
            info.update(changes)
        frames.append("|".join(f"{k}={v}" for k, v in info.items()))

    class Process:
        stdout = io.BytesIO("\n".join(frames).encode())

    @contextmanager
    def process(*args, **kwargs):
        yield Process(), io.BytesIO()

    monkeypatch.setattr(video, "_process", process)
    with pytest.raises(ValueError, match=match):
        video._scan_timing(Path("source"), stream, "ffprobe", VideoConfig())


def test_cover_art_is_ignored_when_selecting_video(tmp_path, tools):
    from PIL import Image

    original = make_clip(tmp_path / "source.mkv", tools, width=48, height=36)
    cover = tmp_path / "cover.png"
    Image.new("RGB", (16, 16), (40, 80, 120)).save(cover)
    source = tmp_path / "with-cover.mp4"
    run([
        tools[0], "-v", "error", "-i", str(original), "-i", str(cover),
        "-map", "0:v", "-map", "1:v", "-c", "copy", "-disposition:v:1", "attached_pic",
        str(source),
    ])
    record = process_video(source, tmp_path / "result.mp4", ffmpeg=tools[0], ffprobe=tools[1])
    assert record["source_dimensions"] == [48, 36]
    assert record["frame_count"] == 5
