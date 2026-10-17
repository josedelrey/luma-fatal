"""Reproduce cleanup experiments with explicit local inputs and output paths.

Run from an installed project environment, for example:
    python experiments/investigate_cleanup.py still --input FRAME.png \
        --config configs/fattal-1-compressed.yaml --output outputs/NEW-STUDY
    python experiments/investigate_cleanup.py synthetic \
        --config configs/fattal-1-compressed.yaml --output outputs/NEW-SYNTHETIC
    python experiments/investigate_cleanup.py video --input CLIP.mp4 \
        --start 18 --frames 48 --config configs/fattal-1-compressed.yaml \
        --output outputs/NEW-VIDEO

Still inputs are decoded sRGB RGB8 images. Video input follows the production
SDR decoder and preprocessing. All effects run at working resolution, with no
final enlargement. Video excerpts start with fresh mask and decoder state.
No result measures subjective artifact removal or proves absence of flicker.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter

from cleanup_multiscale import ScaleSeparatedCleanup
from luma_fatal.cleanup import SourceCleanup
from luma_fatal.configuration import Config
from luma_fatal.core import apply_core
from luma_fatal.numerics import luminance_fractional, luminance_u8
from luma_fatal.processing import limit_input_pixels, resize_working, working_size
from luma_fatal.video import (
    _check, _decode_command, _decoded_frames, _encode_command, _probe, _process,
)


CASES = {
    "off": lambda: None,
    "current075": lambda: SourceCleanup(0.75),
    "current085": lambda: SourceCleanup(0.85),
    "current090": lambda: SourceCleanup(0.9),
    "current1": lambda: SourceCleanup(1),
    "split075-s07": lambda: ScaleSeparatedCleanup(0.75, 1, 0.7),
    "split075-s12": lambda: ScaleSeparatedCleanup(0.75, 1, 1.2),
    "split050-s07": lambda: ScaleSeparatedCleanup(0.5, 1, 0.7),
    "split050-s12": lambda: ScaleSeparatedCleanup(0.5, 1, 1.2),
    "split075-s20": lambda: ScaleSeparatedCleanup(0.75, 1, 2),
}


def prepare(rgb, config):
    if not config.processing.resize:
        return rgb
    rgb = limit_input_pixels(rgb, config.processing.effective_max_input_pixels)
    return resize_working(
        rgb, working_size((rgb.shape[1], rgb.shape[0]), config.processing.effective_max_edge),
        config.processing.working_resampler,
    )


def reference(rgb, config):
    if config.numerics.luminance_mode == "fractional":
        return luminance_fractional(rgb)
    return luminance_u8(rgb).astype(np.float32) * np.float32(1 / 255)


def filtered(rgb, config, callback):
    return apply_core(rgb, int(config.filter[-1]), config.parameters,
                      config.numerics, source_cleanup=callback)


def contact_sheet(images, output, crop=None):
    width, height = 480, 320
    columns = min(len(images), 3)
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * width, rows * height), "#202020")
    font = ImageFont.load_default(size=18)
    for index, (name, array) in enumerate(images.items()):
        panel = Image.fromarray(array)
        if crop:
            panel = panel.crop(crop)
        panel.thumbnail((width - 16, height - 42))
        x, y = index % columns * width, index // columns * height
        ImageDraw.Draw(sheet).text((x + 8, y + 8), name, fill="white", font=font)
        sheet.paste(panel, (x + 8, y + 34))
    sheet.save(output)


def band_rms(array, sigma=1.2):
    value = np.asarray(array, np.float32)
    residual = value - gaussian_filter(value, sigma, mode="reflect")
    return float(np.sqrt(np.mean(residual * residual)))


def run_still(args, config):
    rgb = prepare(np.array(Image.open(args.input).convert("RGB")), config)
    images = {"source": rgb}
    timings = {}
    for name in args.cases:
        # Warm compilation on the actual shape before measuring a new instance.
        filtered(rgb, config, CASES[name]())
        start = time.perf_counter()
        images[name] = filtered(rgb, config, CASES[name]())
        timings[name] = time.perf_counter() - start
        Image.fromarray(images[name]).save(args.output / f"{name}.png")
    contact_sheet(images, args.output / "comparison.png")
    if args.crop:
        contact_sheet(images, args.output / "crop.png", args.crop)
    return {"seconds_per_warm_frame": timings, "working_shape": list(rgb.shape)}


def synthetic_pair(kind, seed=20261016):
    height, width = 256, 512
    y, x = np.mgrid[:height, :width]
    smooth = 80 + 0.035 * x + 0.02 * y
    rng = np.random.default_rng(seed)
    # Identical grain in the clean and damaged input isolates added shading error.
    grain = rng.normal(0, 0.8, (height, width))
    if kind == "steps":
        damaged = np.round(smooth / 3) * 3
    elif kind == "blocks":
        offsets = rng.normal(0, 0.8, (height // 16, width // 16))
        damaged = smooth + offsets.repeat(16, 0).repeat(16, 1)
    elif kind == "real-edge":
        smooth = smooth + 2 * (x >= width // 2)
        damaged = smooth.copy()
    else:
        raise ValueError(kind)
    def rgb(values):
        return np.repeat(np.clip(np.round(values + grain), 0, 255)
                         .astype(np.uint8)[..., None], 3, axis=2)
    return rgb(smooth), rgb(damaged)


def run_synthetic(args, config):
    results = {}
    interior = np.s_[24:-24, 24:-24]
    for kind in ("steps", "blocks", "real-edge"):
        truth, damaged = synthetic_pair(kind)
        target = reference(truth, config) * 255
        raw = reference(damaged, config)
        truth_output = filtered(truth, config, None)
        target_high = target - gaussian_filter(target, 1.2)
        images = {"clean reference": truth_output}
        rows = {}
        for name in args.cases:
            callback = CASES[name]()
            if callback is None:
                restored = raw * 255
            else:
                _, restored = callback(damaged, raw)
                restored = restored * 255
            images[name] = filtered(damaged, config, CASES[name]())
            error = restored - target
            high = restored - gaussian_filter(restored, 1.2)
            output_y = luminance_fractional(images[name]) * 255
            truth_y = luminance_fractional(truth_output) * 255
            rows[name] = {
                "source_low_band_error_rms": float(np.sqrt(np.mean(
                    gaussian_filter(error, 3)[interior] ** 2))),
                "source_fine_band_error_rms": band_rms(error[interior]),
                "source_fine_band_gain": float(np.sum(high[interior] * target_high[interior])
                                                / np.sum(target_high[interior] ** 2)),
                "filtered_mean_abs_error": float(np.mean(np.abs(
                    output_y[interior] - truth_y[interior]))),
                "filtered_fine_band_rms": band_rms(output_y[interior]),
            }
            if kind == "real-edge":
                # The clean pair has a real two-level edge and no added artifact.
                rows[name]["real_edge_error_rms"] = float(np.sqrt(np.mean(
                    error[24:-24, 250:262] ** 2)))
        contact_sheet(images, args.output / f"{kind}.png")
        results[kind] = rows
    return results


def run_benchmark(args, config):
    rgb = prepare(np.array(Image.open(args.input).convert("RGB")), config)
    luminance = reference(rgb, config)
    names = [name for name in args.cases if name != "off"]
    for name in names:
        CASES[name]()(rgb, luminance)
    durations = {name: [] for name in names}
    for trial in range(12):
        # Reverse order on alternating passes to reduce order bias.
        for name in names if trial % 2 == 0 else names[::-1]:
            callback = CASES[name]()
            start = time.perf_counter()
            callback(rgb, luminance)
            durations[name].append(time.perf_counter() - start)
    return {
        "working_shape": list(rgb.shape), "trials": 12,
        "scope": "Fresh cleanup callback only, warm compiled code, no Fattal or encoding",
        "median_seconds": {name: float(np.median(values))
                           for name, values in durations.items()},
    }


def run_video(args, config):
    stream = _probe(args.input, "ffprobe")
    command = _decode_command("ffmpeg", args.input, stream,
                              config.video.deblock, config.video.deband)
    command[command.index("-i"):command.index("-i")] = ["-ss", str(args.start)]
    command[-1:-1] = ["-frames:v", str(args.frames)]
    callbacks = {name: CASES[name]() for name in args.cases}
    previous = {}
    deltas = {name: [] for name in args.cases}
    times = {name: [] for name in args.cases}
    hashes = {name: hashlib.sha256() for name in args.cases}
    sample_indices = {0, args.frames // 2, args.frames - 1}
    count = 0
    with ExitStack() as stack:
        decoder, decoder_log = stack.enter_context(_process(
            command, stdout=subprocess.PIPE, stdin=subprocess.DEVNULL,
        ))
        encoders = {}
        for frame in _decoded_frames(decoder.stdout, stream["width"], stream["height"]):
            rgb = prepare(frame, config)
            height, width = rgb.shape[:2]
            samples = {"source": rgb}
            for name, callback in callbacks.items():
                if name not in encoders:
                    cmd = _encode_command(
                        "ffmpeg", args.output / f"{name}.mp4", width + width % 2,
                        height + height % 2, stream["avg_frame_rate"], config.video,
                    )
                    encoders[name] = stack.enter_context(_process(
                        cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                    ))
                start = time.perf_counter()
                result = filtered(rgb, config, callback)
                times[name].append(time.perf_counter() - start)
                if name in previous:
                    deltas[name].append(float(np.mean(np.abs(
                        result.astype(np.float32) - previous[name]))))
                previous[name] = result.astype(np.float32)
                hashes[name].update(result.tobytes())
                padded = np.pad(result, ((0, height % 2), (0, width % 2), (0, 0)), mode="edge")
                encoders[name][0].stdin.write(padded.tobytes())
                if count in sample_indices:
                    samples[name] = result
                    Image.fromarray(result).save(args.output / f"{name}-{count:03d}.png")
            if count in sample_indices:
                contact_sheet(samples, args.output / f"frame-{count:03d}.png")
            count += 1
            if count % 12 == 0:
                print(f"Compared {count}/{args.frames} frames", flush=True)
        for encoder, log in encoders.values():
            encoder.stdin.close()
            _check(encoder, log, "Encoding")
        _check(decoder, decoder_log, "Decoding")
    if count != args.frames:
        raise ValueError(f"Requested {args.frames} frames but decoded {count}")
    return {
        "frames": count, "frame_rate": stream["avg_frame_rate"],
        "start_seconds": args.start, "fresh_decoder_and_mask_history": True,
        "cases": {name: {
            "mean_adjacent_frame_abs_change": float(np.mean(deltas[name])),
            "median_processing_seconds_excluding_first": float(np.median(times[name][1:])),
            "processed_rgb_sha256": hashes[name].hexdigest(),
            "mask_resets": getattr(getattr(callback, "cleanup", callback), "resets", 0),
        } for name, callback in callbacks.items()},
        "warning": "Adjacent-frame change includes scene motion and is not a flicker score",
    }


def main(argv=None, extra_record=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("still", "synthetic", "video", "benchmark"))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--crop", nargs=4, type=int)
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--frames", type=int, default=48)
    args = parser.parse_args(argv)
    if args.mode != "synthetic" and args.input is None:
        parser.error("--input is required outside synthetic mode")
    if args.start < 0 or args.frames < 2:
        parser.error("--start must be nonnegative and --frames at least 2")
    config = Config.load(args.config)
    if config.filter == "both":
        parser.error("Select one filter in the configuration")
    args.output.mkdir(parents=True, exist_ok=False)
    result = {"configuration": config.to_dict(), "cases": args.cases,
              "input": str(args.input), "mode": args.mode}
    if extra_record:
        result.update(extra_record)
    result["measurements"] = {
        "still": run_still, "synthetic": run_synthetic, "video": run_video,
        "benchmark": run_benchmark,
    }[args.mode](args, config)
    import luma_fatal
    runtime = Path(luma_fatal.__file__).parent
    files = list(Path(__file__).parent.glob("*.py")) + list(runtime.glob("*.py"))
    result["source_sha256"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files
    }
    if args.input:
        with args.input.open("rb") as source:
            result["input_sha256"] = hashlib.file_digest(source, "sha256").hexdigest()
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {args.output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
