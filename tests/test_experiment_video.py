"""Experiment excerpts share production color defaults and encoder fallback."""

from dataclasses import replace
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from luma_fatal import Config
from luma_fatal.video import process_video
from test_video import inspect, run, tools


@pytest.mark.parametrize("runner_name", ["investigate_cleanup", "investigate_prenoise"])
@pytest.mark.parametrize("encoder", ["h264_nvenc", "libx264"])
def test_video_experiments_prepare_untagged_rgb_and_encode_without_gpu(
    tmp_path, tools, monkeypatch, capsys, runner_name, encoder,
):
    import luma_fatal.video as video

    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "experiments"))
    study = importlib.import_module("investigate_cleanup")
    runner = importlib.import_module(runner_name)
    # The noise runner replaces the shared case registry and help text.
    monkeypatch.setattr(study, "CASES", study.CASES.copy())
    monkeypatch.setattr(study, "__doc__", study.__doc__)
    values = np.array([[20, 80, 160], [200, 150, 60], [100, 100, 100]], np.uint8)
    frames = np.broadcast_to(values[:, None, None, :], (3, 36, 48, 3)).copy()
    source = tmp_path / "untagged.mkv"
    run([
        tools[0], "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "48x36",
        "-r", "25", "-i", "pipe:0", "-c:v", "ffv1", "-pix_fmt", "bgr0", str(source),
    ], frames.tobytes())
    config = Config.load(root / "configs/fattal-1-compressed.yaml")
    config = replace(config, video=replace(config.video, encoder=encoder, crf=19, preset="fast"))
    config_path = tmp_path / "settings.yaml"
    config_path.write_text(config.to_yaml())
    probes = []

    def unavailable(executable, width, height, rate, settings):
        probes.append((width, height, str(rate)))
        raise ValueError("controlled unavailable NVIDIA device")

    monkeypatch.setattr(video, "_validate_nvenc", unavailable)
    output = tmp_path / "experiment"
    cases = (["current075", "configured"] if runner_name == "investigate_prenoise"
             else ["off", "current075"])
    monkeypatch.setattr(sys, "argv", [
        runner_name, "video", "--input", str(source), "--config", str(config_path),
        "--frames", "3", "--cases", *cases, "--output", str(output),
    ])
    runner.main()
    report = json.loads((output / "report.json").read_text())
    measured = report["measurements"]
    assert measured["frames"] == 3
    assert measured["color_policy"]["resolved_input"] == {
        "color_space": "gbr", "color_transfer": "iec61966-2-1",
        "color_primaries": "bt709", "color_range": "pc",
    }
    assert measured["color_policy"]["assumptions"]["color_transfer"] == "iec61966-2-1"
    assert measured["encoding"]["encoder"] == "libx264"
    assert measured["encoding"]["requested_encoder"] == encoder
    if encoder == "h264_nvenc":
        assert probes == [(48, 36, "25")]
        assert "controlled unavailable" in measured["encoding"]["fallback_reason"]
        assert "Falling back to CPU" in capsys.readouterr().err
    else:
        assert not probes
        assert measured["encoding"]["fallback_reason"] is None
    for case in cases:
        result = inspect(output / f"{case}.mp4", tools)
        assert len(result["frames"]) == 3
        assert result["streams"][0]["codec_name"] == "h264"
        assert (output / f"{case}-000.png").is_file()
    cpu = replace(config, video=replace(config.video, encoder="libx264", filter_backend="cpu"))
    reference = process_video(source, tmp_path / "reference.mp4", config=cpu,
                              ffmpeg=tools[0], ffprobe=tools[1])
    assert measured["cases"]["current075"]["processed_rgb_sha256"] == reference["processed_rgb_sha256"]
