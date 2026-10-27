# Development and verification

[Overview](../README.md) · [Method](method.md) · [Experiments](experiments.md)

## Checks

Run the development checks:

```bash
uv sync --locked --python 3.11
source .venv/bin/activate
python -m pytest -q
python scripts/check_presets.py
```

[CI](../.github/workflows/ci.yml) runs the test suite and preset checks on Ubuntu
and Windows with Python 3.11 and 3.12. FFmpeg and ffprobe are installed
there, and video integration tests must pass. In local runs, those tests are
skipped if the tools are unavailable.

NVENC option validation, automatic CPU fallback, and failure cleanup are tested
without NVIDIA hardware.
The hardware integration test runs when NVENC can initialize, otherwise it is
skipped. On an NVIDIA machine, set `LUMA_FATAL_REQUIRE_NVENC_TESTS=1` to require
that test to pass, and run `python -m pytest -q tests/test_video.py -k nvenc`.

CUDA solver and frame agreement tests skip when the optional GPU dependencies or hardware
are unavailable. Automatic selection and fallback tests use controlled clocks
and fake devices on every platform. To require the real GPU tests:

```bash
uv sync --locked --extra gpu
LUMA_FATAL_REQUIRE_GPU_TESTS=1 uv run --extra gpu python -m pytest -q tests/test_gpu.py
```

Set the environment variable with your shell's syntax on Windows. GPU tests
cover odd/even geometry, reference boundary behavior, both filter variants,
resizing, source cleanup, effect controls, noise, and denoising against the CPU
reference. CUDA graphs and kernels initialize on the first eligible frame.
Controlled failures verify cleanup history restoration on fallback.

Regression tests use bundled [fixtures](../tests/fixtures/) or deterministically
generated inputs. They cover scalar float32 bit patterns, filter outputs,
interpolation, geometry, decoding, and color conversion. Video tests use generated
clips and synthetic audio to check frame counts, timing, geometry, and color tags,
as well as agreement with the image API before lossy encoding.

## Frozen presets

The preset checker compares dimensions, RGB pixels, encoded JPEG bytes, decoded
pixels, and codec settings against [presets.json](../tests/fixtures/presets.json).
It explicitly uses shadow softness `0` to preserve the frozen image baselines.
Bundled YAML presets default to `6`.
The checker never regenerates expected results. To retain outputs and a JSON
report, run:

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
published results are 880×1100 PNGs.

Regenerate the image in the CLI example:

```bash
luma-fatal apply assets/tea-by-the-water-original.png outputs/readme-tea-new.png \
  --config configs/fattal-1.yaml --filter fattal1 --shadow-softness 6 --output-format png
```

Its source and filtered result are 1080×810 PNGs.

README image sources and filtered results are kept in `assets/`. Use the ignored
`local_inputs/` directory for experimental inputs and `outputs/` for generated artifacts.
