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
The terminal prints the output path, size, run time, and metadata path. Add
`--json` to print the full run record for scripts.
Output directories are created automatically. The input file cannot be used
as the output path.

For both variants in one invocation, set `--filter both` and pass an output
directory. For the complete command reference, run
`uv run --locked luma-fatal --help` or `uv run --locked luma-fatal apply --help`.

## Process a video

Video processing also requires `ffmpeg` and `ffprobe` on `PATH`. FFmpeg must
include the `zscale` filter and `libx264` encoder. The optional source
treatments also require the corresponding `deblock` or `deband` filter. No
additional Python dependencies are needed.

```bash
uv run --locked luma-fatal video local_inputs/input.mp4 outputs/stylized.mp4 --config configs/fattal-1.yaml
```

For compressed inputs, the paired presets enable smooth shadows and gentle
debanding with standard smooth-area cleanup at strength 0.75 while retaining
the original Fattal parameters:

```bash
uv run --locked luma-fatal video local_inputs/videos/four-people.mp4 outputs/four-people-compressed.mp4 --config configs/fattal-1-compressed.yaml --progress
```

Use `configs/fattal-2-compressed.yaml` for Fattal 2. The original preset files
pin the previous shadow and video preprocessing settings. Gentle debanding
reduces the coarse patches visible in the Tears of Steel sample, where weak
deblocking made little visible difference. The compressed presets retain rounded
luminance because the fractional-luminance experiment gave little visible
improvement to the block patches.

The multiscale presets inherit these settings and strengthen the broad part of
the cleanup correction while retaining more fine source variation:

```bash
uv run --locked luma-fatal video local_inputs/videos/four-people.mp4 outputs/four-people-multiscale.mp4 --config configs/fattal-1-compressed-multiscale.yaml --progress
```

Use `configs/fattal-2-compressed-multiscale.yaml` for Fattal 2. Both preset
files use `extends` to load their compressed parent from the same directory.

Four noise presets reproduce the pre-filter treatments from the noise study.
`fattal-1-compressed-noise.yaml` and `fattal-2-compressed-noise.yaml` combine
standard cleanup 0.75 with fine masked noise at 0.35 RMS.
`fattal-1-compressed-fbm.yaml` and `fattal-2-compressed-fbm.yaml` use masked
fBm-like noise at 0.75 RMS with cleanup disabled.

```bash
uv run --locked luma-fatal video local_inputs/videos/four-people.mp4 outputs/four-people-noise.mp4 --config configs/fattal-1-compressed-noise.yaml --progress
```
The resolved settings and parent file fingerprint are recorded in video metadata.
Multiscale cleanup is slower and can retain some jagged contour fragments along
with desirable texture. Compare moving footage before choosing a preset.

To process the complete Tears of Steel clip with Fattal 1 and see frames per
second and an estimated time remaining:

```bash
uv run --locked luma-fatal video local_inputs/videos/tears-of-steel.mp4 outputs/tears-of-steel-fattal-1-compressed.mp4 --config configs/fattal-1-compressed.yaml --progress
```

The progress bar reports filtered frames on stderr. The terminal reports the
output path, frame count, size, run time, and metadata path when processing
finishes. Add `--json` to print the full run record for scripts. Use a new output
filename for each run, since existing outputs are refused.

The first video adapter supports one progressive, square-pixel, constant-frame-rate
video stream with explicit BT.709 matrix, transfer, primaries and range tags.
Inputs must use opaque 8-bit planar YUV 4:2:0, 4:2:2 or 4:4:4. Untagged color,
HDR, interlacing, display rotation and variable frame rate are rejected.

Frames are converted to full-range sRGB RGB8 before filtering and converted back
to limited-range BT.709 for H.264 MP4 export. The conversion uses zscale's
display-referred BT.709/BT.1886 interpretation. Processing is sequential with
bounded frame buffering. A preliminary decoding pass validates every frame's
timing, geometry and color tags. Output timestamps start at zero and retain the
source's rational frame rate.

Output is **silent**. Audio, subtitles and other streams are discarded. Filtering
is independent per frame, so temporal flicker is possible. The image processing
limits still apply, while image ICC, EXIF and JPEG/PNG export settings do not.
Odd processed dimensions are padded at the right and bottom by repeating edge
pixels to make the H.264 YUV420P output dimensions even.

Use `--crf` (default 18) and `--preset` (default `medium`) to control H.264 export.
For blocky sources, `--deblock weak8` applies FFmpeg's weak deblocking to
8-pixel boundaries before color conversion and filtering. `weak4` uses 4-pixel
boundaries. This is video-only and defaults to `off`. Deblocking can soften
detail, and its fixed grid may miss blocks from variable-size codecs.
`--deband gentle` applies FFmpeg debanding with an 8-pixel range and low
threshold to decoded YUV before color conversion. It can reduce coarse
quantization patches, though fine texture and frame-to-frame variation may
change. It is video-only and defaults to `off` in the original presets. Both
source filters can be combined.

Override smooth-area cleanup strength with `--flat-cleanup`:

```bash
uv run --locked luma-fatal video local_inputs/videos/tears-of-steel.mp4 outputs/tears-of-steel-cleanup.mp4 --config configs/fattal-1-compressed.yaml --flat-cleanup 0.5 --progress
```

The matching configuration setting is `video.flat_cleanup`, a strength from
0 to 1. It is 0 in the reference and general presets and 0.75 in the
compressed presets. Set it to 0 to disable cleanup in any video preset.

`--flat-cleanup-mode multiscale` selects the scale-separated correction without
changing the other settings. In this mode, `flat_cleanup` controls fine-detail
strength. `--flat-cleanup-coarse-strength` controls broad correction strength
from 0 to 1, and `--flat-cleanup-sigma` sets its Gaussian radius in effect
working pixels from 0.01 to 16. Their configuration keys are
`video.flat_cleanup_coarse_strength` and `video.flat_cleanup_sigma`. The
multiscale presets use 0.75 fine strength, 1.0 broad strength, and sigma 1.2.
The standard mode ignores the broad-strength and sigma values. The feature
changes luminance and RGB before the Fattal effect, and does not add grain.

Video source noise is controlled by `--pre-noise-rms` from 0 to 4 in input
0-255 units and `--pre-noise-kind highpass|fbm`. Their configuration keys are
`video.pre_noise_rms` and `video.pre_noise_kind`. Noise enters floating RGB and
luminance after any cleanup and before the Fattal effect. It is restricted to
the original source's smooth-region mask by default. `--no-pre-noise-masked`
disables this restriction, and `--pre-noise-seed` selects its deterministic
field. The corresponding keys are `video.pre_noise_masked` and
`video.pre_noise_seed`. Set `--pre-noise-rms 0` to disable noise independently
of cleanup. These settings affect video processing only.

Noise is fixed in working-frame coordinates, without motion compensation.
Fine noise disguises some contours with grain, while fBm-like noise creates
broader cloudy variation. The effect can strongly amplify either treatment.
See the [noise study](experiments/prenoise.md) for comparisons and limitations.

Cleanup detects smooth regions from decoded source RGB after any FFmpeg
deblocking/debanding and at the effect's working resolution. It combines
texture at two scales, abrupt-edge protection, and variation inside reduced
pixels after subtracting estimated smooth shading. Gradual brightness ramps
can qualify for cleanup while fine texture survives detection downsampling.
The edge guard measures changes in the local slope to protect sudden transitions
in brightness or color. A soft mask controls
[guided filtering](https://people.csail.mit.edu/kaiming/eccv10/index.html).
Local cleanup strength is the requested strength multiplied by the continuous
mask weight. Smooth areas receive more treatment, texture receives less, and
intermediate areas receive partial treatment. Smooth threshold curves, guided
mask refinement and bilinear mask upsampling soften transitions between them.
RGB and luminance corrections are blended without rounding intermediate values.
The guide's longest edge is capped at 384 pixels, and a compiled pass upsamples
the coefficients and applies bounded corrections. Corrected RGB and luminance
remain floating point through the effect to avoid adding byte-quantization bands.
Fade and shadow protection still use the original source pixels.

Only the small mask has temporal history. Local source changes release that
history, strong edges override it immediately, and scene cuts reset it. RGB
frames are never temporally averaged. This is an experimental structure mask,
with no semantic face recognition. Smooth skin and desired grain can still
change. The sidecar records cleanup time and mask resets. First use also incurs
Numba compilation or cache-loading time. The disabled setting bypasses cleanup.

Existing output files or metadata sidecars are refused. Each successful run writes
a `.meta.json` sidecar with resolved settings, color/timing policies, dimensions,
frame count, hashes, dependency/tool versions and implementation fingerprints.
Encoded video bytes can differ between FFmpeg versions and platforms.

The Python API reads video settings from the configuration. Passing a
`VideoConfig` to `process_video` explicitly overrides those settings:

```python
from luma_fatal import Config
from luma_fatal.video import process_video

record = process_video(
    "local_inputs/input.mp4",
    "outputs/stylized.mp4",
    config=Config.load("configs/fattal-1.yaml"),
)
```

## Configuration

Settings can be loaded from YAML, TOML, or JSON. Explicit CLI options override
the configuration file. The bundled configurations are:

| Configuration | Purpose |
| --- | --- |
| [`configs/default.yaml`](configs/default.yaml) | General defaults, including fade 50 |
| [`configs/high.yaml`](configs/high.yaml) | General defaults with a 12 MP input cap and 2048-pixel working edge |
| [`configs/fattal-1.yaml`](configs/fattal-1.yaml) | Complete Fattal 1 reference preset with fade 0 |
| [`configs/fattal-2.yaml`](configs/fattal-2.yaml) | Complete Fattal 2 reference preset with fade 0 |
| [`configs/fattal-1-compressed.yaml`](configs/fattal-1-compressed.yaml) | Fattal 1 with smooth shadows and gentle video debanding |
| [`configs/fattal-2-compressed.yaml`](configs/fattal-2-compressed.yaml) | Fattal 2 with smooth shadows and gentle video debanding |
| [`configs/fattal-1-compressed-multiscale.yaml`](configs/fattal-1-compressed-multiscale.yaml) | Fattal 1 compressed preset with multiscale cleanup |
| [`configs/fattal-2-compressed-multiscale.yaml`](configs/fattal-2-compressed-multiscale.yaml) | Fattal 2 compressed preset with multiscale cleanup |
| [`configs/fattal-1-compressed-noise.yaml`](configs/fattal-1-compressed-noise.yaml) | Fattal 1 with cleanup 0.75 and fine masked noise 0.35 |
| [`configs/fattal-2-compressed-noise.yaml`](configs/fattal-2-compressed-noise.yaml) | Fattal 2 with cleanup 0.75 and fine masked noise 0.35 |
| [`configs/fattal-1-compressed-fbm.yaml`](configs/fattal-1-compressed-fbm.yaml) | Fattal 1 with masked fBm-like noise 0.75 |
| [`configs/fattal-2-compressed-fbm.yaml`](configs/fattal-2-compressed-fbm.yaml) | Fattal 2 with masked fBm-like noise 0.75 |

`alpha`, `beta`, `saturation`, `noise`, `fade`, `tone_strength`, and
`highlight_compression` use a 0–100 scale, with `alpha` strictly positive.
`alpha` and `beta` control gradient attenuation, `saturation`
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
New controls are opt-in and retain the original preset output when disabled.

| Setting | Default | Experiment |
| --- | --- | --- |
| `numerics.detail_level` | `0` | Exclude this many finest pyramid levels from attenuation |
| `numerics.power_mode` | `approximate` | Select the current approximation or `standard` NumPy exponentiation |
| `numerics.shadow_softness` | `0` | Smooth the original shadow cutoff with a transition half-width of 1–255 luminance levels |
| `numerics.luminance_mode` | `rounded` | Use `fractional` to retain sub-byte luminance through the filter |
| `processing.working_resampler` | `nearest` | Select `nearest`, `bilinear`, or `bicubic` working-resolution sampling |
| `parameters.output_gamma` | `1.0` | Apply the curve `x^(1/gamma)` to clipped, normalized filtered RGB |
| `parameters.tone_strength` | `100` | Blend original and filtered brightness before color reconstruction |
| `parameters.highlight_compression` | `0` | Roll off bright RGB peaks together before clipping, retaining their color ratios |

`tone_strength` controls how strongly the filter changes brightness in images
and video. `100` retains the full effect. `0` uses original source brightness,
with saturation, source cleanup or noise, output gamma, shadow protection and
Fade still taking effect. Intermediate values blend source and filtered
luminance before color reconstruction, RGB clipping and quantization.
This can reduce the dominance of deep black and bright white areas while
retaining more original shading. It does not guarantee that highlights avoid
clipping. Try `--tone-strength 60` or set `parameters.tone_strength: 60`:

```bash
uv run --locked luma-fatal video local_inputs/input.mp4 outputs/gentler-tones.mp4 --config configs/fattal-1-compressed-noise.yaml --tone-strength 60
```

Tone strength is applied at working resolution before enlargement. The source
brightness uses the selected rounded or fractional reference luminance from
before cleanup or noise. Fade instead blends already filtered RGB with original
RGB at the output resolution. A tone strength of 100 skips the new blend and
preserves the reference output. At intermediate strengths, nonfinite mapped
brightness uses white for NaN or positive infinity and black for negative
infinity. A strength of 0 ignores mapped brightness entirely.

`highlight_compression` controls a smooth highlight roll-off for both images
and video. `0` disables it and preserves the reference output. `100` applies the
full roll-off, with intermediate values blending toward it. Try
`--highlight-compression 60` or set `parameters.highlight_compression: 60`:

```bash
uv run --locked luma-fatal video local_inputs/input.mp4 outputs/softer-highlights.mp4 --config configs/fattal-1-compressed-noise.yaml --tone-strength 60 --highlight-compression 60
```

Compression acts on reconstructed floating RGB after tone strength and
saturation, before output gamma, clipping, shadow protection and Fade. Pixels
whose brightest channel is at or below 75 percent of white are unchanged.
Above that knee, a rational curve gradually approaches white. All channels
receive the same scale, preserving their ratios during compression. Output
gamma and quantization can subsequently change those ratios.

At full strength, finite nonnegative RGB peaks stay within the output range.
Partial strengths can still clip very bright values. The control preserves
color from the filter's floating output and cannot restore highlights already
clipped in the input. Nonfinite pixels retain the existing quantization policy.
The curve is fixed across frames and adds no temporal state. Like output gamma,
it runs at working resolution before enlargement.

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

`shadow_softness: 0` retains the original hard shadow cutoff byte-for-byte.
A positive integer blends original RGB into filtered RGB with a smoothstep curve
between `max(0, shadow_threshold - shadow_softness)` and
`min(255, shadow_threshold + shadow_softness)`. Below that interval the original
pixels are preserved, and above it the existing filtered output is unchanged.
The mask uses fractional luminance with the existing RGB weights. The blend
uses clipped filtered RGB after output gamma and before quantization and fade.
It applies at working resolution in the bounded route, including video frames.
This control changes shadow handling while retaining the existing texture
calculation. It does not denoise frames or suppress codec blocks.

`luminance_mode: fractional` keeps the same RGB weights but omits the final
rounding to integer luminance before the logarithm. The decoded RGB input is
still 8-bit, so this option cannot recover detail lost in the source video.
It is an experiment for measuring the filter's own quantization effects.
The `rounded` mode retains the reference behavior. For a direct video
comparison with the smooth shadow transition, use
`--luminance-mode fractional --shadow-softness 6` with a reference preset.

For example, try a half-width of 6 on a video with either reference preset:

```bash
uv run --locked luma-fatal video local_inputs/input.mp4 outputs/smooth-shadows.mp4 --config configs/fattal-1.yaml --shadow-softness 6
```

The preset remains unchanged. The CLI override and resolved settings are recorded
in the output metadata. Brightening deep shadows may reveal source noise, so
compare the transition on your own footage before increasing its width.

For example, run the other four controls on a bundled fixture:

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

Video tests generate short deterministic clips and synthetic audio locally with
FFmpeg. They require no downloaded media or private inputs. Video integration
tests are skipped when FFmpeg/ffprobe are absent locally. CI installs these tools
and requires the video tests to run. Known grayscale values check color transfer,
and frame hashes check exact agreement with the image API before lossy encoding.
Container checks cover frame counts, timestamps, geometry and color tags.

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
temporal state. The video adapter provides decoding, encoding and validated CFR
timing. Audio preservation and temporal stabilization are not implemented.

## References and license

Raanan Fattal, Dani Lischinski, and Michael Werman. 2002.
[*Gradient Domain High Dynamic Range Compression*](https://doi.org/10.1145/566570.566573).
SIGGRAPH '02, pp. 249–256.

Project-authored code is licensed under MIT. The color conversion routines
include adaptations from skcms under BSD-3-Clause. The complete licenses and
upstream attribution are included in [`LICENSE`](LICENSE).
