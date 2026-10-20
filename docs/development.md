# Development and verification

[Overview](../README.md) · [Method](method.md) · [Experiments](experiments.md)

## Checks

Run the development checks:

```bash
uv sync --locked --python 3.11
python -m pytest -q
python scripts/check_presets.py
uv build
```

[CI](../.github/workflows/ci.yml) runs the test suite, preset checks, and distribution
builds on Ubuntu and Windows with Python 3.11. FFmpeg and ffprobe are installed
there, and video integration tests must pass. In local runs, those tests are
skipped if the tools are unavailable.

Regression tests use bundled [fixtures](../tests/fixtures/) or deterministically
generated inputs. They cover scalar float32 bit patterns, filter outputs,
interpolation, geometry, decoding, and color conversion. Video tests use generated
clips and synthetic audio to check frame counts, timing, geometry, and color tags,
as well as agreement with the image API before lossy encoding.

## Frozen presets

The preset checker compares dimensions, RGB pixels, encoded JPEG bytes, decoded
pixels, and codec settings against [presets.json](../tests/fixtures/presets.json).
Missing inputs or mismatches cause the check to fail. The checker never
regenerates expected results. To retain outputs and a JSON report, run:

```bash
python scripts/check_presets.py --output outputs/preset-check
```

Use a new output directory for each run. Frozen results detect numerical and
encoding changes in the bundled preset cases.

## README images

Regenerate the opening comparison from its bundled input:

```bash
luma-fatal apply assets/lantern-garden-original.png outputs/readme-comparison-new \
  --config configs/fattal-1.yaml --filter both --shadow-softness 6 --output-format png
```

This command applies both reference variants with shadow softness 6 and saves
`fattal1.png`, `fattal2.png`, and their `.meta.json` run records. The source and
published results are 1122×1402 PNGs.

Regenerate the image in the CLI example:

```bash
luma-fatal apply assets/tea-by-the-water-original.png outputs/readme-tea-new.png \
  --config configs/fattal-1.yaml --filter fattal1 --shadow-softness 6 --output-format png
```

Its source and filtered result are 1448×1086 PNGs.

README image sources and filtered results are kept in `assets/`. Use the ignored
`local_inputs/` directory for experimental inputs and `outputs/` for generated artifacts.
Source distributions include the documentation, presets, tests, and comparison
assets, while wheels include the runtime and complete license notices.
