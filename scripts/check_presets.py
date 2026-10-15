#!/usr/bin/env python3
"""Check frozen image presets using only bundled fixtures.

Usage: uv run --locked python scripts/check_presets.py
Add --sample p3-noise to select a specific fixture.
Add --output DIRECTORY to retain generated images and a JSON report.
This checker never regenerates its expected baselines.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
from io import BytesIO
import hashlib
import json
from pathlib import Path

from PIL import Image, JpegImagePlugin
import yaml

from luma_fatal import Config, apply_filter
from luma_fatal.image_io import save_image

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "tests/fixtures/presets.json"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_contract(path: Path = CONTRACT):
    """Require complete explicit configuration and frozen semantic values."""
    contract = json.loads(path.read_text())
    if contract["schema_version"] != 2:
        raise ValueError("Unsupported preset regression schema")
    presets = []
    for entry in contract["presets"]:
        config_path = ROOT / entry["config"]
        config = Config.load(config_path)
        data = yaml.safe_load(config_path.read_text())
        if set(data) != {item.name for item in fields(Config)}:
            raise ValueError(f"Preset must specify every section: {entry['name']}")
        for section in ("parameters", "processing", "image", "output", "numerics"):
            if set(data[section]) != {
                item.name for item in fields(getattr(config, section))
            }:
                raise ValueError(
                    f"Preset must specify every {section} field: {entry['name']}"
                )
        if config.to_dict() != entry["configuration"]:
            raise ValueError(f"Frozen preset configuration changed: {entry['name']}")
        presets.append((entry, config))
    return contract, presets


def snapshot(source: Path, config: Config):
    """Fingerprint pixels separately from JPEG encoding and decoded output."""
    with Image.open(source) as original:
        result = apply_filter(original, config=config)
    if result.mode != "RGB":
        raise ValueError("Frozen regression inputs must produce opaque RGB")
    buffer = BytesIO()
    save_image(result, buffer, config.output, result.info.get("icc_profile"))
    encoded = buffer.getvalue()
    with Image.open(BytesIO(encoded)) as exported:
        exported.load()
        record = dict(
            dimensions=list(result.size),
            rgb_sha256=hashlib.sha256(result.tobytes()).hexdigest(),
            encoded_sha256=hashlib.sha256(encoded).hexdigest(),
            decoded_rgb_sha256=hashlib.sha256(exported.tobytes()).hexdigest(),
            codec=dict(
                format=exported.format,
                sampling=JpegImagePlugin.get_sampling(exported),
                all_quantization_values_one=all(
                    value == 1
                    for table in exported.quantization.values()
                    for value in table
                ),
                embedded_icc=bool(exported.info.get("icc_profile")),
            ),
            color_backend=result.info.get("_color_conversion", {}).get("backend"),
            input_decode=result.info.get("_input_decode"),
        )
    return record, encoded


def mismatches(expected, actual):
    return [
        key
        for key in sorted(set(expected) | set(actual))
        if expected.get(key) != actual.get(key)
    ]


def check(
    *,
    output: Path | None = None,
    samples: tuple[str, ...] | None = None,
):
    contract, presets = load_contract()
    entries = contract["samples"]
    if samples is not None:
        if not samples:
            raise ValueError("Select at least one sample")
        unknown = set(samples) - {entry["id"] for entry in entries}
        if unknown:
            raise ValueError(
                f"Unknown regression samples: {', '.join(sorted(unknown))}"
            )
        entries = [entry for entry in entries if entry["id"] in samples]
    cases = [(entry, ROOT / entry["source"]) for entry in entries]
    # Verify all inputs before producing images or creating an output directory.
    for entry, source in cases:
        if not source.is_file():
            raise ValueError(f"Missing regression input: {source}")
        if digest(source) != entry["source_sha256"]:
            raise ValueError(f"Regression input changed: {entry['id']}")
    if output is not None:
        output = output.expanduser().resolve()
        output.mkdir(parents=True, exist_ok=False)
    records = []
    for entry, source in cases:
        for preset, config in presets:
            actual, encoded = snapshot(source, config)
            failed = mismatches(entry["expected"][config.filter], actual)
            print(
                f"{'FAIL' if failed else 'PASS'}: {preset['name']} / {entry['id']}"
                + (f" ({', '.join(failed)})" if failed else ""),
                flush=True,
            )
            row = dict(
                preset=preset["name"],
                sample=entry["id"],
                matched=not failed,
                mismatched_fields=failed,
                actual=actual,
            )
            if output is not None:
                destination = output / entry["id"] / f"{config.filter}.jpg"
                destination.parent.mkdir(exist_ok=True)
                with destination.open("xb") as stream:
                    stream.write(encoded)
                row["output"] = str(destination)
            records.append(row)
    report = dict(
        schema_version=2,
        baseline_origin=contract["baseline_origin"],
        all_matched=all(row["matched"] for row in records),
        records=records,
        selected_samples=[entry["id"] for entry in entries],
        runtime_requirements="Local Python project dependencies only",
    )
    if output is not None:
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--sample", action="append", help="Select a sample ID, repeat to select several"
    )
    args = parser.parse_args(argv)
    try:
        report = check(
            output=args.output, samples=tuple(args.sample) if args.sample else None
        )
    except (ValueError, FileExistsError, OSError) as error:
        parser.exit(1, f"{error}\n")
    if not report["all_matched"]:
        parser.exit(
            1, "Preset regression detected. Expected baselines were preserved\n"
        )


if __name__ == "__main__":
    main()
