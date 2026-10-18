# Luma Fatal

[![CI](https://github.com/josedelrey/luma-fatal/actions/workflows/ci.yml/badge.svg)](https://github.com/josedelrey/luma-fatal/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
[![License](https://img.shields.io/badge/License-MIT%20%2B%20BSD--3--Clause-22A559)](LICENSE)

Experiments in gradient-domain image and video stylization. Two Fattal-inspired
filters reshape local contrast and color, with configurable numerical controls,
a Python API, and a command-line interface. Processing runs on the CPU and saves
resolved settings and input/output fingerprints alongside each file output.

<p>
  <img src="assets/lantern-garden-original.png" alt="Final Fantasy XIV input" width="32%" align="middle">
  <img src="assets/lantern-garden-fattal-1.png" alt="Fattal 1 result" width="32%" align="middle">
  <img src="assets/lantern-garden-fattal-2.png" alt="Fattal 2 result" width="32%" align="middle">
</p>

Left to right: *Final Fantasy XIV* image, then the same image with
Fattal 1 and Fattal 2 applied.

## Method

Both filters attenuate log-luminance gradients across an image pyramid,
reconstruct brightness with a Poisson solver, and recombine it with color.
Fattal 1 accumulates attenuation across scales. Fattal 2 replaces it at each
finer scale. The names refer to this project's experimental variants, based on
[Fattal, Lischinski, and Werman (2002)](https://doi.org/10.1145/566570.566573).
See the [method notes](docs/method.md) for processing order and numerical choices.

## Install

Requires Git, Python 3.11 or later, and [uv](https://docs.astral.sh/uv/).
CI covers Linux and Windows with Python 3.11. Run commands from the repository root.

```bash
git clone https://github.com/josedelrey/luma-fatal.git
cd luma-fatal
uv sync --locked --python 3.11
```

Numba compiles numerical kernels on first use, so the first run takes longer.

## Process an image

Apply both filters to the bundled image:

```bash
uv run --locked luma-fatal apply assets/lantern-garden-original.png outputs/comparison \
  --config configs/fattal-1.yaml --filter both --output-format png
```

This writes `fattal1.png`, `fattal2.png`, and their `.meta.json` run records to
`outputs/comparison/`. Replace the input path with your own JPEG or PNG.
For one result, pass an output filename and select `--filter fattal1` or `fattal2`.

The [configuration guide](docs/configuration.md) covers presets, tone controls,
resolution, and export. Use `uv run --locked luma-fatal apply --help` for CLI options.

## Process a video

Install FFmpeg and ffprobe with `zscale` and `libx264` support, then run:

```bash
uv run --locked luma-fatal video local_inputs/input.mp4 outputs/stylized.mp4 \
  --config configs/fattal-1.yaml --progress
```

Video input must be progressive, constant-frame-rate, 8-bit YUV with explicit
BT.709 color tags. Output is silent H.264 MP4. Filtering is independent per frame,
so flicker is possible. The [video guide](docs/video.md) covers input requirements,
compression cleanup, noise experiments, and encoding.

## Documentation

- [Configuration](docs/configuration.md): presets, inheritance, and effect controls.
- [Python API](docs/python-api.md): images, arrays, and decoded frame streams.
- [Method](docs/method.md): numerical behavior, color handling, and scope.
- [Experiments](docs/experiments.md): source cleanup and pre-filter noise studies.
- [Development](docs/development.md): tests, regression checks, and comparison builds.

## Tests

```bash
uv run --locked python -m pytest -q
uv run --locked python scripts/check_presets.py
```

Tests use bundled synthetic fixtures and generated clips. Video integration tests
require FFmpeg and ffprobe. Frozen references cover the tested cases, with no
guarantee of identical results across all platforms or dependency versions.

## References and license

Raanan Fattal, Dani Lischinski, and Michael Werman.
[*Gradient Domain High Dynamic Range Compression*](https://doi.org/10.1145/566570.566573),
SIGGRAPH 2002.

Project-authored code is licensed under MIT. Color conversion includes skcms
adaptations under BSD-3-Clause. Image credits and complete notices are in
[LICENSE](LICENSE).
