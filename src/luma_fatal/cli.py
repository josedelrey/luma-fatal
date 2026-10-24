"""Command-line image/video processing, probes and strict comparison."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .pipeline import _prepare_pillow_input
from .processing import limit_input_pixels
from .configuration import Config, FilterParams, VIDEO_COLOR_PROFILES, resolve_config
from .images import process_image, STATUS, _write_json, _digest
from .probes import generate_probes
from .validation import compare_images, write_difference_panel
from .video import process_video


def _settings(args: argparse.Namespace) -> Config:
    if args.config is not None and args.profile is not None:
        raise ValueError("Use --config or --profile, not both")
    config = (
        Config.load(args.config)
        if args.config
        else resolve_config(profile=args.profile)
    )
    parameters = {
        key: getattr(args, key)
        for key in asdict(config.parameters)
        if getattr(args, key, None) is not None
    }
    processing = {
        key: getattr(args, key)
        for key in (
            "device_class",
            "max_edge",
            "max_input_pixels",
            "limit_input_pixels",
            "resize",
            "transparency",
            "blend_mode",
            "working_resampler",
        )
        if getattr(args, key, None) is not None
    }
    if args.device_class is not None and args.max_edge is None:
        processing["max_edge"] = None
    if args.resize is False and args.max_edge is None:
        processing["max_edge"] = None
    if (
        args.device_class is not None
        or args.resize is False
        or args.limit_input_pixels is False
    ) and args.max_input_pixels is None:
        processing["max_input_pixels"] = None
    image = {
        key: getattr(args, key)
        for key in (
            "color_space",
            "color_backend",
            "exif_orientation",
            "decode_sampling",
        )
        if getattr(args, key, None) is not None
    }
    output = {
        key: getattr(args, key)
        for key in ("jpeg_quality", "jpeg_subsampling", "embed_icc", "jpeg_background")
        if getattr(args, key, None) is not None
    }
    if getattr(args, "output_format", None) is not None:
        output["format"] = args.output_format
    numerics = {
        name: getattr(args, name)
        for name in asdict(config.numerics)
        if getattr(args, name) is not None
    }
    video = {
        name: getattr(args, name)
        for name in (
            "crf", "preset", "deblock", "deband", "denoise", "flat_cleanup",
            "flat_cleanup_mode", "flat_cleanup_coarse_strength", "flat_cleanup_sigma",
            "pre_noise_rms", "pre_noise_kind", "pre_noise_masked", "pre_noise_seed",
            "frame_rate", "input_color", "input_range",
            "encoder", "nvenc_cq", "nvenc_preset",
            "filter_backend", "calibration_cache", "gpu_iterations", "lighting_scale",
            "decode_prefetch", "pinned_frames", "gpu_buffer_reuse", "gpu_cleanup_graph",
            "solver_halo", "pipeline_depth", "validation_mode",
        )
        if getattr(args, name, None) is not None
    }
    return replace(
        config,
        filter=args.filter or config.filter,
        parameters=replace(config.parameters, **parameters),
        processing=replace(config.processing, **processing),
        image=replace(config.image, **image),
        output=replace(config.output, **output),
        numerics=replace(config.numerics, **numerics),
        video=replace(config.video, **video),
    )


def _comparison(
    reference: Path,
    candidate: Path,
    original: Path | Image.Image | None,
    output: Path,
    gain: float,
) -> dict:
    record = compare_images(reference, candidate)
    record.update(
        reference=str(reference),
        reference_sha256=_digest(reference),
        candidate=str(candidate),
        candidate_sha256=_digest(candidate),
    )
    _write_json(output / "metrics.json", record)
    if original:
        write_difference_panel(
            original, reference, candidate, output / "differences.png", gain=gain
        )
    return record


def _evaluate(args: argparse.Namespace) -> dict:
    config = _settings(args)
    if args.filter is None and args.config is None:
        config = replace(config, filter="both")
    params = config.parameters
    rows = []
    inputs = sorted(args.inputs.glob("*.png"))
    if not inputs:
        raise ValueError(f"No PNG inputs found in {args.inputs}")
    variants = (1, 2) if config.filter == "both" else (int(config.filter[-1]),)
    for variant in variants:
        label = f"fattal_{variant}"
        for original in inputs:
            destination = args.output / label / (original.stem + config.output.suffix)
            run = process_image(
                original,
                destination,
                config=replace(config, filter=f"fattal{variant}"),
                config_path=args.config,
            )
            options = [
                args.references / label / (original.stem + suffix)
                for suffix in (".png", ".jpg", ".jpeg")
            ]
            existing = [path for path in options if path.exists()]
            if len(existing) > 1:
                raise ValueError(
                    f"Ambiguous reference for {label}/{original.stem}: {existing}"
                )
            reference = existing[0] if existing else options[0]
            row = dict(filter=label, input=original.name, run=run)
            if reference.exists():
                output = args.output / "comparisons" / label / original.stem
                with Image.open(original) as source_image:
                    if params.fade < 100:
                        interpreted = _prepare_pillow_input(source_image, config)
                        pixels = limit_input_pixels(
                            np.asarray(interpreted),
                            config.processing.effective_max_input_pixels,
                        )
                        interpreted = Image.fromarray(pixels)
                    else:
                        interpreted = source_image.copy()
                row.update(
                    status="compared",
                    metrics=_comparison(reference, destination, interpreted, output, 8),
                )
            else:
                row.update(
                    status="missing_reference", expected_reference=str(reference)
                )
            rows.append(row)
    compared = [row for row in rows if row["status"] == "compared"]
    summary = dict(
        fidelity_status=STATUS,
        parameters=asdict(params),
        processing_profile=config.processing.profile_name,
        configuration=config.to_dict(),
        records=rows,
        compared_count=len(compared),
        missing_count=len(rows) - len(compared),
    )
    if compared:
        # Equal image weight. This average is not a dataset-wide pixel-weighted error.
        summary["mean_image_mae"] = sum(
            row["metrics"]["mae"] for row in compared
        ) / len(compared)
    _write_json(args.output / "evaluation.json", summary)
    return summary


def _show_image(record: dict) -> None:
    output = Path(record["output"])
    width, height = record["dimensions"]
    print(f"Processed image with {record['filter']}: {output}")
    print(f"  Size: {width} × {height} | Time: {record['seconds']:.2f} s")
    print(f"  Details: {output.with_suffix('.meta.json')}")


def _show_video(record: dict) -> None:
    output = Path(record["output"])
    width, height = record["dimensions"]
    print(f"Processed video with Fattal {record['filter'][-1]}: {output}")
    print(
        f"  {record['frame_count']} frames at {record['frame_rate']} fps"
        f" | {width} × {height} | Run time: {record['seconds']:.2f} s"
    )
    print(f"  Details: {output.with_suffix('.meta.json')}")


def _show_comparison(record: dict, output: Path, original: Path | None) -> None:
    print(f"Compared: {record['reference']} and {record['candidate']}")
    print(
        f"  Mean absolute error: {record['mae']:.3f} / 255"
        f" | SSIM: {record['ssim']:.4f}"
        f" | Exact pixels: {record['exact_pixel_fraction']:.1%}"
    )
    print(f"  Details: {output / 'metrics.json'}")
    if original is not None:
        print(f"  Difference panel: {output / 'differences.png'}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Luma Fatal gradient-domain image and video processing. " + STATUS
    )
    commands = parser.add_subparsers(dest="command", required=True)
    apply = commands.add_parser(
        "apply", help="Process an image and write it plus provenance JSON"
    )
    apply.add_argument("input", type=Path)
    apply.add_argument("output", type=Path)
    apply.add_argument("--filter", choices=["fattal1", "fattal2", "both"], default=None)
    video = commands.add_parser(
        "video",
        help="Prepare and filter video into silent SDR H.264 MP4",
    )
    video.add_argument("input", type=Path)
    video.add_argument("output", type=Path)
    video.add_argument("--filter", choices=("fattal1", "fattal2"), default=None)
    video.add_argument(
        "--encoder", choices=("libx264", "h264_nvenc"), default=None,
        help="H.264 encoder. YAML presets select h264_nvenc with automatic CPU fallback",
    )
    video.add_argument("--crf", type=int, default=None, help="libx264 quality, 0 to 51")
    video.add_argument(
        "--filter-backend", choices=("cpu", "auto"), default=None,
        help="cpu uses reference filtering. auto benchmarks the optional CUDA solver before selecting it",
    )
    video.add_argument("--calibration-cache", action=argparse.BooleanOptionalAction,
                       default=None, help="Reuse compatible CUDA calibration across runs")
    video.add_argument("--gpu-iterations", type=int, default=None,
                       help="GPU Poisson iterations, leaving the CPU reference unchanged")
    video.add_argument("--lighting-scale", type=float, default=None,
                       help="Resolution fraction of the GPU lighting correction grid, 0.25 to 1")
    video.add_argument("--decode-prefetch", type=int, default=None,
                       help="Decoded frames to queue ahead, 0 to 16")
    video.add_argument("--pinned-frames", action=argparse.BooleanOptionalAction,
                       default=None, help="Decode into pinned host buffers for GPU transfer")
    video.add_argument("--gpu-buffer-reuse", action=argparse.BooleanOptionalAction,
                       default=None, help="Reuse GPU buffers and retain device cleanup history")
    video.add_argument("--gpu-cleanup-graph", action=argparse.BooleanOptionalAction,
                       default=None, help="Capture compatible cleanup operations in a CUDA graph")
    video.add_argument("--solver-halo", type=int, default=None,
                       help="GPU boundary dependency rows, 32 to 96")
    video.add_argument("--pipeline-depth", type=int, default=None,
                       help="Reusable decode/filter/encode slots, 0 disables overlap")
    video.add_argument("--validation-mode", choices=("scan", "stream"), default=None,
                       help="Validate with a separate scan or during compatible decoding")
    video.add_argument("--preset", default=None, help="libx264 speed preset")
    video.add_argument(
        "--nvenc-cq", type=int, default=None,
        help="NVENC target quality, 1 to 51, lower is better. 0 selects automatic quality",
    )
    video.add_argument(
        "--nvenc-preset", choices=("p1", "p2", "p3", "p4", "p5", "p6", "p7"),
        default=None, help="NVENC speed preset, p1 fastest to p7 slowest, default p4",
    )
    video.add_argument(
        "--frame-rate", default=None,
        help="Output fps, such as 30 or 30000/1001, or auto to use the source rate",
    )
    video.add_argument(
        "--input-color", choices=("auto", *VIDEO_COLOR_PROFILES), default=None,
        help="Override source color interpretation, default auto uses tags and recorded assumptions",
    )
    video.add_argument(
        "--input-range", choices=("auto", "limited", "full"), default=None,
        help="Override source signal range, default auto uses tags or a recorded assumption",
    )
    video.add_argument(
        "--deblock", choices=("off", "weak4", "weak8"), default=None,
        help="Optional weak source deblocking before the Fattal filter",
    )
    video.add_argument(
        "--deband", choices=("off", "gentle"), default=None,
        help="Optional gentle source debanding before the Fattal filter",
    )
    video.add_argument(
        "--denoise", type=float, default=None,
        help="Adaptive source denoising before Fattal, 0 to 1, default 0",
    )
    video.add_argument(
        "--flat-cleanup", type=float, default=None,
        help="Source-guided cleanup strength in smooth areas, 0 to 1, default 0",
    )
    video.add_argument(
        "--flat-cleanup-mode", choices=("standard", "multiscale"), default=None,
        help="Select standard or multiscale source-guided cleanup",
    )
    video.add_argument(
        "--flat-cleanup-coarse-strength", type=float, default=None,
        help="Broad correction strength for multiscale cleanup, 0 to 1",
    )
    video.add_argument(
        "--flat-cleanup-sigma", type=float, default=None,
        help="Broad correction radius in effect working pixels, 0.01 to 16",
    )
    video.add_argument(
        "--pre-noise-rms", type=float, default=None,
        help="Noise RMS before Fattal in source 0-255 units, 0 to 4, default 0",
    )
    video.add_argument(
        "--pre-noise-kind", choices=("highpass", "fbm"), default=None,
        help="Fine high-pass or fBm-like source noise",
    )
    video.add_argument(
        "--pre-noise-masked", action=argparse.BooleanOptionalAction, default=None,
        help="Restrict source noise to smooth regions, enabled by default",
    )
    video.add_argument(
        "--pre-noise-seed", type=int, default=None,
        help="Nonnegative seed for the fixed source noise field",
    )
    video.add_argument(
        "--ffmpeg", default="ffmpeg", help="FFmpeg executable with zscale and the selected encoder"
    )
    video.add_argument("--ffprobe", default="ffprobe", help="ffprobe executable")
    video.add_argument(
        "--progress", action="store_true",
        help="Show filtered frames and speed, with ETA when the frame count is known",
    )
    compare = commands.add_parser(
        "compare", help="Measure matching RGB pixels without alignment or resizing"
    )
    compare.add_argument("reference", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument("--original", type=Path)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--gain", type=float, default=8)
    probes = commands.add_parser(
        "probes", help="Generate deterministic RGB test patterns"
    )
    probes.add_argument("output", type=Path)
    probes.add_argument("--size", type=int, default=512)
    probes.add_argument("--seed", type=int, default=3095)
    evaluate = commands.add_parser(
        "evaluate", help="Process both variants and compare every available reference"
    )
    evaluate.add_argument("inputs", type=Path)
    evaluate.add_argument("references", type=Path)
    evaluate.add_argument("output", type=Path)
    configuration = commands.add_parser(
        "config", help="Show defaults or write a reusable configuration file"
    )
    configuration.add_argument("--output", type=Path)
    for command in (apply, video, compare, configuration):
        command.add_argument(
            "--json", action="store_true",
            help="Print the full machine-readable result to stdout",
        )
    evaluate.add_argument(
        "--filter", choices=["fattal1", "fattal2", "both"], default=None
    )
    for command in (apply, evaluate, video):
        command.add_argument(
            "--config",
            type=Path,
            help="YAML configuration, CLI values override it. TOML and JSON are also supported",
        )
        command.add_argument(
            "--profile",
            choices=("core", "bounded-1536"),
            default=None,
            help="Processing profile, default: bounded working resolution",
        )
        for name, default in asdict(FilterParams()).items():
            command.add_argument(
                "--" + name.replace("_", "-"),
                type=float,
                default=None,
                help={
                    "output_gamma": f"Positive output gamma, x^(1/gamma), default {default:g}",
                    "tone_strength": "Blend source and filtered brightness before color "
                    "reconstruction, 0 to 100, default 100",
                    "highlight_compression": "Compress bright RGB peaks before clipping, "
                    "0 to 100, default 0",
                    "black_level": "Output black level as a percentage, 0 to 100, default 0",
                    "white_level": "Output white level as a percentage, 0 to 100, default 100",
                    "vibrance": "Boost muted colors more than saturated colors, "
                    "0 to 100, default 0",
                }.get(name, f"Parameter scale 0..100, default {default:g}"),
            )
        for name, value_type, description in (
            ("iterations", int, "Poisson solver iterations, at least 1"),
            ("relaxation", float, "Poisson relaxation factor, greater than 0 and below 2"),
            ("pyramid_min_size", int, "Minimum pyramid dimension, at least 2"),
            ("normalization_low", float, "Lower normalization quantile, 0 to below 1"),
            ("normalization_high", float, "Upper normalization quantile, 0 to below 1"),
            ("normalization_floor", float, "Positive normalization floor, at most 1"),
            ("gradient_threshold", float, "Positive gradient threshold, at most 1"),
            ("log_offset", float, "Positive luminance offset before logarithms, at most 1"),
            ("shadow_threshold", int, "Shadow preservation cutoff, 0 to 255"),
        ):
            command.add_argument(
                "--" + name.replace("_", "-"),
                type=value_type,
                default=None,
                help=description,
            )
        command.add_argument(
            "--detail-level",
            type=int,
            default=None,
            help="Number of finest pyramid levels excluded from attenuation",
        )
        command.add_argument(
            "--power-mode",
            choices=("approximate", "standard"),
            default=None,
            help="Power function for attenuation and color recombination",
        )
        command.add_argument(
            "--shadow-softness",
            type=int,
            default=None,
            help="Shadow transition half-width in luminance levels, 0 keeps the reference cutoff",
        )
        command.add_argument(
            "--luminance-mode",
            choices=("rounded", "fractional"),
            default=None,
            help="Use rounded reference luminance or unrounded fractional luminance",
        )
        command.add_argument(
            "--working-resampler",
            choices=("nearest", "bilinear", "bicubic"),
            default=None,
            help="Resampler used to reduce to effect working resolution",
        )
        command.add_argument(
            "--device-class", choices=("low", "medium", "high"), default=None
        )
        command.add_argument(
            "--max-edge",
            type=int,
            default=None,
            help="Override the device-class working limit",
        )
        command.add_argument(
            "--max-input-pixels",
            type=int,
            default=None,
            help="Override the import pixel-area limit",
        )
        command.add_argument(
            "--limit-input-pixels", action=argparse.BooleanOptionalAction, default=None
        )
        command.add_argument(
            "--resize", action=argparse.BooleanOptionalAction, default=None
        )
        command.add_argument(
            "--transparency", choices=("reject", "straight"), default=None
        )
        command.add_argument(
            "--blend-mode", default=None, help="Only normal is supported"
        )
        if command is video:
            continue
        command.add_argument("--color-space", choices=("encoded", "srgb"), default=None)
        command.add_argument(
            "--color-backend",
            choices=("matrix_trc", "lcms"),
            default=None,
            help="sRGB conversion backend, default matrix_trc. Unsupported inputs use a recorded LCMS fallback",
        )
        command.add_argument(
            "--exif-orientation", choices=("ignore", "apply"), default=None
        )
        command.add_argument(
            "--decode-sampling", action=argparse.BooleanOptionalAction, default=None
        )
        command.add_argument("--output-format", choices=("png", "jpeg"), default=None)
        command.add_argument("--jpeg-quality", type=int, default=None)
        command.add_argument(
            "--jpeg-subsampling", choices=("444", "422", "420"), default=None
        )
        command.add_argument(
            "--embed-icc", action=argparse.BooleanOptionalAction, default=None
        )
        command.add_argument(
            "--jpeg-background",
            type=int,
            nargs=3,
            default=None,
            metavar=("R", "G", "B"),
        )
    args = parser.parse_args(argv)
    try:
        if args.command == "apply":
            config = _settings(args)
            if config.filter == "both":
                records = [
                    process_image(
                        args.input,
                        args.output / f"fattal{variant}{config.output.suffix}",
                        config=replace(config, filter=f"fattal{variant}"),
                        config_path=args.config,
                    )
                    for variant in (1, 2)
                ]
            else:
                records = [
                    process_image(
                        args.input, args.output, config=config, config_path=args.config
                    )
                ]
            if args.json:
                print(json.dumps(records, indent=2))
            else:
                for record in records:
                    _show_image(record)
        elif args.command == "video":
            config = _settings(args)
            if not args.json:
                print(f"Processing video: {args.input} -> {args.output}", flush=True)
            record = process_video(
                args.input,
                args.output,
                config=config,
                config_path=args.config,
                progress=args.progress,
                ffmpeg=args.ffmpeg,
                ffprobe=args.ffprobe,
            )
            if args.json:
                print(json.dumps(record, indent=2))
            else:
                _show_video(record)
        elif args.command == "config":
            config = Config()
            if args.output:
                if args.output.exists():
                    raise ValueError(
                        f"Configuration file already exists: {args.output}"
                    )
                if args.output.suffix.lower() not in (
                    ".yaml",
                    ".yml",
                    ".toml",
                    ".json",
                ):
                    raise ValueError(
                        "Configuration files must use .yaml, .yml, .toml or .json"
                    )
                args.output.parent.mkdir(parents=True, exist_ok=True)
                if args.output.suffix.lower() in (".yaml", ".yml"):
                    args.output.write_text(config.to_yaml(), encoding="utf-8")
                elif args.output.suffix.lower() == ".toml":
                    args.output.write_text(config.to_toml(), encoding="utf-8")
                else:
                    _write_json(args.output, config.to_dict())
            if args.json:
                print(json.dumps(config.to_dict(), indent=2))
            elif args.output:
                print(f"Default configuration written to {args.output}")
            else:
                print(config.to_yaml(), end="")
        elif args.command == "compare":
            record = _comparison(
                args.reference,
                args.candidate,
                args.original,
                args.output,
                args.gain,
            )
            if args.json:
                print(json.dumps(record, indent=2))
            else:
                _show_comparison(record, args.output, args.original)
        elif args.command == "probes":
            result = generate_probes(args.output, args.size, args.seed)
            print(f"{len(result['probes'])} probes written to {args.output}")
        else:
            result = _evaluate(args)
            print(
                f"{result['compared_count']} compared, {result['missing_count']} missing references"
            )
            print(f"  Details: {args.output / 'evaluation.json'}")
    except (ValueError, OSError) as error:
        parser.error(str(error))
