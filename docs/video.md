# Video processing

[Overview](../README.md) · [Presets](configuration.md#presets) · [Experiments](experiments.md)

## Requirements and input

Install `ffmpeg` and `ffprobe` so they are available on `PATH`. FFmpeg must include
`zscale` and `libx264`, plus `bwdif` for interlaced input, `tonemap` for HDR, and
`deblock` or `deband` if you enable those source treatments. No additional Python
packages are required.

Inputs are decoded and prepared through FFmpeg without an intermediate video
file. Common RGB and YUV formats, including higher bit depths, are converted to
full-range sRGB RGB8 for filtering. Input must contain one video stream, excluding
cover art. Its dimensions, pixel format, color interpretation, and pixel shape
must remain stable, with strictly increasing presentation timestamps.

```bash
luma-fatal video local_inputs/input.mp4 outputs/stylized.mp4 \
  --config configs/fattal-1.yaml --progress
```

A preliminary decoding pass checks timing, geometry, and color throughout the
stream. Filtered frames are converted back to limited-range BT.709 for H.264 MP4
output, using zscale's display-referred BT.709/BT.1886 interpretation. Processing
is sequential, with bounded buffering. The output starts at timestamp zero.

## Input preparation

- Tagged color spaces and signal ranges are converted using their metadata.
  Missing tags use sRGB for RGB input, BT.601 for SD YUV input up to 1024×576
  (PAL above 480 lines, NTSC otherwise), and BT.709 for larger YUV input. Known
  matrix, primaries, or HDR transfer tags take precedence over size assumptions.
  Missing range defaults to limited for YUV and full for RGB, grayscale, or
  full-range JPEG YUV. Assumptions are recorded in the metadata sidecar.
- PQ and HLG HDR are tone-mapped to SDR using Hable, a fixed 1000-nit signal peak,
  and 100-nit reference white. HDR output and dynamic HDR metadata are not retained.
- Interlaced frames are deinterlaced with `bwdif`, keeping one frame per input
  frame. Non-square pixels are resized to square pixels, with unspecified pixel
  shape treated as square. Display rotations in multiples of 90 degrees are
  applied. Reflected, skewed, and perspective display matrices are rejected.
- Constant-frame-rate input keeps its cadence by default. Variable timing is
  converted to the source's nominal rate, falling back to its average rate.
  Frames are duplicated or dropped as needed. Use `--frame-rate` to select a rate
  explicitly, such as `30` or `30000/1001`.

Use `--input-color` and `--input-range` to override missing or incorrect metadata.
Color profiles are `bt709`, `bt601-pal`, `bt601-ntsc`, `srgb`, `bt2020-pq`, and
`bt2020-hlg`. The `srgb` profile uses a BT.709 matrix for YUV storage. RGB storage
always uses an identity matrix. These options describe the source. For example:

```bash
luma-fatal video local_inputs/input.mp4 outputs/stylized.mp4 \
  --config configs/fattal-1.yaml --input-color bt709 --input-range limited \
  --frame-rate 30000/1001 --progress
```

The corresponding configuration keys are `video.input_color`,
`video.input_range`, and `video.frame_rate`. All default to the string `auto`.
Explicit frame rates must also be strings in configuration files and `VideoConfig`.
The sidecar records resolved color, geometry, timing, and preparation filters.

## Output

The output contains only silent, opaque video. Audio, subtitles, alpha, and other
streams are discarded. If an output dimension is odd, edge pixels are repeated
at the right or bottom to accommodate YUV420P. Image ICC, EXIF, and JPEG/PNG export
options do not apply to video.

Use `--progress` to report filtered frames and speed on stderr, with estimated
remaining time when the frame count is known before filtering. Rate conversion
uses an open-ended frame count. Use `--json` to print the complete run record.
Choose a new output filename for each run, since existing output files and
metadata sidecars are never overwritten.

The `.meta.json` sidecar records resolved settings, dimensions, frame count,
color and timing policies, hashes, tool versions, and implementation fingerprints.
Encoded bytes may differ across FFmpeg versions and platforms.

| CLI option | Config key | Default | Purpose |
| --- | --- | --- | --- |
| `--crf` | `video.crf` | `18` | H.264 quality, range 0–51 |
| `--preset` | `video.preset` | `medium` | H.264 encoder speed preset |
| `--deblock` | `video.deblock` | `off` | Weak deblocking on 4- or 8-pixel boundaries with `weak4` or `weak8` |
| `--deband` | `video.deband` | `off` | Gentle decoded-YUV debanding with `gentle` |

Deblocking and debanding run on planar YUV before the final sRGB conversion and
filtering. RGB sources are converted to YUV first when these treatments are
enabled. Deblocking can soften detail and may miss artifacts with variable codec
block sizes, while debanding can alter fine texture and variation between frames.

## Compressed sources

The compressed presets enable gentle debanding, smooth shadows, and standard
smooth-area cleanup at strength 0.75:

```bash
luma-fatal video local_inputs/input.mp4 outputs/cleaned.mp4 \
  --config configs/fattal-1-compressed.yaml --progress
```

Cleanup detects smooth regions at working resolution and uses a continuous mask
to protect texture and abrupt edges. Guided filtering then corrects RGB and
luminance in floating point before the Fattal effect. Fade and shadow protection
continue to use the original source pixels.

Only the mask carries temporal history: local changes release it, strong edges
override it, and scene cuts reset it. RGB frames are never averaged over time.
Cleanup can still change smooth skin and grain you may wish to preserve. The
sidecar records cleanup time and mask resets.

| CLI option | Config key | Default | Meaning |
| --- | --- | --- | --- |
| `--flat-cleanup` | `video.flat_cleanup` | `0` | Cleanup strength, 0–1, with 0 disabling it |
| `--flat-cleanup-mode` | `video.flat_cleanup_mode` | `standard` | `standard` or `multiscale` correction |
| `--flat-cleanup-coarse-strength` | `video.flat_cleanup_coarse_strength` | `1.0` | Broad correction strength, 0–1, used only in multiscale mode |
| `--flat-cleanup-sigma` | `video.flat_cleanup_sigma` | `1.2` | Multiscale Gaussian radius, 0.01–16 working pixels |

In multiscale mode, `flat_cleanup` controls the strength of fine correction.
The bundled multiscale presets use fine strength 0.75, broad strength 1.0, and
sigma 1.2. They take longer to process and can preserve unwanted contour fragments
along with fine texture.

## Source noise

Noise is added to floating-point RGB and luminance after cleanup and before
filtering. By default, the original source's smooth-region mask limits where
it is applied.

| CLI option | Config key | Default | Meaning |
| --- | --- | --- | --- |
| `--pre-noise-rms` | `video.pre_noise_rms` | `0` | RMS in input 0–255 units, range 0–4, with 0 disabling noise |
| `--pre-noise-kind` | `video.pre_noise_kind` | `highpass` | Fine `highpass` or correlated `fbm` noise |
| `--pre-noise-seed` | `video.pre_noise_seed` | `20261017` | Seed for the deterministic field |
| `--no-pre-noise-masked` | `video.pre_noise_masked` | `true` | Flag disables masking |

The noise presets combine standard cleanup at 0.75 with high-pass noise at
0.35 RMS. The fBm presets use correlated noise at 0.75 RMS with cleanup disabled.
The noise field stays fixed in working-frame coordinates, without motion
compensation. It can conceal contours with grain, but may also introduce cloudy
patterns that the filter strongly amplifies.

Because filtering has no temporal stabilization, assess cleanup and noise on
moving footage. The [experiment notes](experiments.md) describe the observed
tradeoffs.
