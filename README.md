# Luma Fatal

Luma Fatal is a Python library and command-line tool for gradient-domain image
stylization. It provides two filters, Fattal 1 and Fattal 2, with configurable
attenuation, color recombination, resolution, and export settings.

The project emphasizes reproducible numerical behavior. Image preparation,
filtering, and encoding are configured separately, and file-based runs save
the resolved settings and implementation fingerprints alongside each output.

## Method

Both filters modify the gradients of log luminance, reconstruct a luminance
field with a Poisson solver, and recombine it with the input color. Fattal 1
accumulates attenuation across pyramid levels. Fattal 2 replaces the attenuation
field at each finer level. The numerical core uses explicit floating-point
operation order and fixed boundary conditions.

The methodological background is the gradient-domain compression approach of
[Fattal, Lischinski, and Werman (2002)](https://doi.org/10.1145/566570.566573).
The two filter names denote this project's variants. The current implementation
operates on 8-bit SDR images.

## Installation

Requires Git, Python 3.11 or later, and
[uv](https://docs.astral.sh/uv/getting-started/installation/). The commands below
select Python 3.11, matching the CI configuration for Linux and Windows.
Run subsequent commands from the repository root.

```bash
git clone https://github.com/josedelrey/luma-fatal.git
cd luma-fatal
uv sync --locked --python 3.11
```

Processing runs on the CPU. Numba compiles numerical kernels on first use, so
the first invocation includes compilation overhead.

## Process an image

Apply either reference preset to the bundled synthetic color fixture:

```bash
uv run --locked luma-fatal apply tests/fixtures/color/noise-display-p3.jpg outputs/fattal1.jpg --config configs/fattal-1.yaml
uv run --locked luma-fatal apply tests/fixtures/color/noise-display-p3.jpg outputs/fattal2.jpg --config configs/fattal-2.yaml
```

Replace the input path with your own image when needed. Each command writes the
processed image and a `.meta.json` sidecar containing configuration, dimensions,
color conversion details, dependency versions, and input/output hashes.
Output directories are created automatically. The input file cannot be used
as the output path.

For both variants in one invocation, set `--filter both` and pass an output
directory. For the complete command reference, run
`uv run --locked luma-fatal --help` or `uv run --locked luma-fatal apply --help`.

## Configuration

Settings can be loaded from YAML, TOML, or JSON. Explicit CLI options override
the configuration file. The bundled configurations are:

| Configuration | Purpose |
| --- | --- |
| [`configs/default.yaml`](configs/default.yaml) | General defaults, including fade 50 |
| [`configs/high.yaml`](configs/high.yaml) | General defaults with a 12 MP input cap and 2048-pixel working edge |
| [`configs/fattal-1.yaml`](configs/fattal-1.yaml) | Complete Fattal 1 reference preset with fade 0 |
| [`configs/fattal-2.yaml`](configs/fattal-2.yaml) | Complete Fattal 2 reference preset with fade 0 |

`alpha`, `beta`, `saturation`, `noise`, and `fade` use a 0–100 scale, with `alpha`
strictly positive. `alpha` and `beta` control gradient attenuation, `saturation`
controls color recombination, and `noise` adds an offset in the attenuation
calculation. `fade: 0` applies the
full effect, while `fade: 100` bypasses the effect. Image preparation still
applies when the effect is bypassed.

The general defaults and reference presets cap input area at 8 MP and evaluate
the effect at a maximum working edge of 1536 pixels. The filtered image is then
enlarged to the prepared input dimensions. The input-area cap can reduce output
dimensions, and changing either limit can change the result.

Reference presets apply EXIF orientation, convert supported ICC profiles to
sRGB, and export JPEG at quality 100 with 4:2:0 chroma subsampling and no embedded
profile. Override individual settings when exporting another format:

```bash
uv run --locked luma-fatal apply tests/fixtures/color/noise-display-p3.jpg outputs/preview.png --config configs/fattal-2.yaml --fade 25 --output-format png
```

Export a complete editable configuration with:

```bash
uv run --locked luma-fatal config --output outputs/settings.yaml
```

### Experimental controls

The following settings are available in configuration files, the Python API,
and CLI options. Both reference presets explicitly pin their neutral defaults.

| Setting | Default | Experiment |
| --- | --- | --- |
| `numerics.detail_level` | `0` | Exclude this many finest pyramid levels from attenuation |
| `numerics.power_mode` | `approximate` | Select the current approximation or `standard` NumPy exponentiation |
| `processing.working_resampler` | `nearest` | Select `nearest`, `bilinear`, or `bicubic` working-resolution sampling |
| `parameters.output_gamma` | `1.0` | Apply the curve `x^(1/gamma)` to clipped, normalized filtered RGB |

`detail_level` is a nonnegative integer. It excludes fine scales while
`pyramid_min_size` controls the coarsest scale. Retained attenuation fields are
interpolated through the excluded levels. Excluding every available level gives
unit attenuation, while reconstruction and normalization still apply.

`standard` powers are evaluated in float64, then narrowed to float32 for
attenuation. Color recombination retains float64. Output gamma is positive and
finite, with a finite reciprocal. Values above 1 brighten the filtered result,
and values below 1 darken it. The curve precedes RGB8 quantization, shadow
preservation, and fade. Protected shadows and effect bypass skip the curve.
In the bounded route, gamma is applied at working resolution before enlargement.

The working resampler applies only when reducing to effect working resolution.
Input-area fitting and output enlargement retain their existing methods. These
resamplers have no antialias prefilter. Bilinear uses half-pixel coordinates and
truncates to RGB8. Bicubic uses the scalar-field kernel with `a = -0.75`, clamps
overshoot, and truncates to RGB8.

For example, run all four controls on a bundled fixture:

```bash
uv run --locked luma-fatal apply tests/fixtures/color/noise-display-p3.jpg outputs/experiment.png --config configs/fattal-2.yaml --output-format png --detail-level 1 --power-mode standard --working-resampler bicubic --max-edge 128 --output-gamma 1.2
```

## Python API

Apply a configuration to a Pillow image in memory:

```python
from PIL import Image

from luma_fatal import Config, apply_filter

config = Config.load("configs/fattal-1.yaml")
with Image.open("tests/fixtures/color/noise-display-p3.jpg") as image:
    result = apply_filter(image, config=config)
```

`apply_filter` also accepts NumPy `uint8` arrays with shape `(height, width, 3)`
and returns an array for array inputs. Arrays must already have the intended RGB
color interpretation. The convenience functions `fattal1` and `fattal2` accept
the same image types and effect controls.

For file output with the configured encoding and metadata sidecar, use the
file adapter:

```python
from luma_fatal.images import process_image

record = process_image(
    "tests/fixtures/color/noise-display-p3.jpg",
    "outputs/api.jpg",
    config=config,
)
```

## Tests and reproducibility

Run the test suite and complete preset regression check with:

```bash
uv run --locked python -m pytest -q
uv run --locked python scripts/check_presets.py
```

All regression inputs are bundled in [`tests/fixtures/`](tests/fixtures/) or
generated deterministically. References cover scalar float32 bit patterns,
filter outputs, interpolation, geometry, decoding, and color conversion. Preset
checks compare dimensions, RGB pixels, encoded JPEG bytes, decoded pixels, and
codec settings against frozen values in
[`presets.json`](tests/fixtures/presets.json).

The checker fails on missing inputs or mismatches and never regenerates its
expected values. Add `--output outputs/preset-check` to retain generated images
and a JSON report. The output directory must be new.

The [CI workflow](.github/workflows/ci.yml) runs these checks and builds the
source distribution and wheel on Ubuntu and Windows with Python 3.11. Frozen
references establish behavior for the tested cases. Exact equality across all
platforms, dependency versions, and input images is not guaranteed.

## Scope and limitations

The validated image workflow covers opaque 8-bit SDR JPEG and PNG. The default
color converter handles RGB matrix/TRC ICC profiles. Unsupported transforms
use a recorded Little CMS fallback, whose numerical results can differ.
Transparency, HEIC, high-bit-depth images, and arbitrary ICC LUT transforms
are outside the validated scope.

`process_frames` processes decoded RGB8 arrays lazily and in order. It has no
temporal state. Video decoding, encoding, audio, timestamps, and temporal
stabilization are not implemented.

## References and license

Raanan Fattal, Dani Lischinski, and Michael Werman. 2002.
[*Gradient Domain High Dynamic Range Compression*](https://doi.org/10.1145/566570.566573).
SIGGRAPH '02, pp. 249–256.

Project-authored code is licensed under MIT. The color conversion routines
include adaptations from skcms under BSD-3-Clause. The complete licenses and
upstream attribution are included in [`LICENSE`](LICENSE).
