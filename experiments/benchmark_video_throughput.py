"""Compare complete video throughput and pre-encode RGB fingerprints."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
from statistics import median
from time import perf_counter

from luma_fatal import Config
from luma_fatal.gpu import AdaptiveFrameProcessor
from luma_fatal.video import process_video


@contextmanager
def frame_timings():
    original = AdaptiveFrameProcessor.__call__
    samples = []

    def measured(self, frame):
        start = perf_counter()
        result = original(self, frame)
        samples.append(perf_counter() - start)
        return result

    AdaptiveFrameProcessor.__call__ = measured
    try:
        yield samples
    finally:
        AdaptiveFrameProcessor.__call__ = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True, help="Reference YAML")
    parser.add_argument("--candidate", type=Path, required=True, help="Candidate YAML")
    parser.add_argument("--output", type=Path, required=True, help="New result directory")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--cpu-reference", action="store_true")
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    configs = {"reference": Config.load(args.config),
               "candidate": Config.load(args.candidate)}
    if args.cpu_reference:
        original = configs["reference"]
        configs["reference"] = replace(original, video=replace(original.video, filter_backend="cpu"))
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for index in range(args.repeat):
        for name, config in configs.items():
            destination = args.output / f"{name}-{index + 1}.mp4"
            with frame_timings() as samples:
                record = process_video(args.input, destination, config=config)
            warm = samples[8:]
            row = dict(
                name=name, run=index + 1, seconds=record["seconds"],
                frame_count=record["frame_count"],
                frames_per_second=record["frame_count"] / record["seconds"],
                median_filter_ms=median(warm) * 1000 if warm else None,
                processed_rgb_sha256=record["processed_rgb_sha256"],
                filtering=record["filtering"], output=str(destination.resolve()),
            )
            results.append(row)
            report = dict(input=str(args.input.resolve()), runs=results,
                          all_rgb_fingerprints_match=len({r["processed_rgb_sha256"] for r in results}) == 1)
            (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            print(f"{name} {index + 1}: {row['seconds']:.2f} s, "
                  f"{row['frames_per_second']:.2f} frames/s", flush=True)


if __name__ == "__main__":
    main()
