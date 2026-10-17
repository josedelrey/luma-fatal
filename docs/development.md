# Development and verification

[Overview](../README.md) · [Method](method.md) · [Experiments](experiments.md)

## Checks

From the repository root:

```bash
uv sync --locked --python 3.11
uv run --locked python -m pytest -q
uv run --locked python scripts/check_presets.py
uv build
```

[CI](../.github/workflows/ci.yml) runs the suite, preset checks, and distribution
builds on Ubuntu and Windows with Python 3.11. It installs FFmpeg and ffprobe and
requires video integration tests. Locally, those tests skip if the tools are absent.

Regression inputs are bundled in [tests/fixtures/](../tests/fixtures/) or generated
deterministically. Coverage includes scalar float32 bit patterns, filter outputs,
interpolation, geometry, decoding, and color conversion. Video tests generate
short clips and synthetic audio, checking frame counts, timing, geometry, color
tags, and agreement with the image API before lossy encoding.

## Frozen presets

The preset checker compares dimensions, RGB pixels, encoded JPEG bytes, decoded
pixels, and codec settings against [presets.json](../tests/fixtures/presets.json).
It fails on missing inputs or mismatches and never regenerates expectations.
Retain outputs and a JSON report with:

```bash
uv run --locked python scripts/check_presets.py --output outputs/preset-check
```

The output directory must be new. Frozen results establish behavior for the
covered cases. They do not guarantee equality for every environment or source.

## README comparison

Regenerate both filtered results from the bundled AI-restored input:

```bash
uv run --locked luma-fatal apply assets/lantern-garden-original.png outputs/readme-comparison-new \
  --config configs/fattal-1.yaml --filter both --output-format png
```

This uses both reference variants with PNG export as the only override. Outputs
are `fattal1.png`, `fattal2.png`, and their `.meta.json` run records. The source
and published results are 1122×1402 PNGs. The input is an AI reconstruction of a
centered crop from [Acerola’s FFXIV showcase at 24:03](https://www.youtube.com/watch?v=jJnwzkBre8Y&t=1443s),
with invented detail. Filtering is reproducible from the bundled input.
AI generation is outside this reproduction workflow.

`assets/` contains only the three README images. Experimental inputs belong in
ignored `local_inputs/`, and generated artifacts belong in ignored `outputs/`.
Source distributions include the docs, presets, tests, and comparison assets.
Wheels include the runtime and complete license notices.
