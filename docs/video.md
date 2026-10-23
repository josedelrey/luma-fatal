# Video processing

[Overview](../README.md) · [Configuration](configuration.md#video-settings) · [Experiments](experiments.md)

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
is sequential, with bounded buffering.

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

For missing or incorrect metadata, specify the source interpretation and cadence:

```bash
luma-fatal video local_inputs/input.mp4 outputs/stylized.mp4 \
  --config configs/fattal-1.yaml --input-color bt709 --input-range limited \
  --frame-rate 30000/1001 --progress
```

See [encoding and source preparation](configuration.md#encoding-and-source-preparation)
for configuration keys, defaults, and accepted values. The sidecar records resolved
color, geometry, timing, and preparation filters.

## Output

The output contains only silent, opaque video. Audio, subtitles, alpha, and other
streams are discarded. If an output dimension is odd, edge pixels are repeated
at the right or bottom to accommodate YUV420P. Image ICC, EXIF, and JPEG/PNG export
options do not apply to video.

Use `--progress` to report filtered frames and speed on stderr, with estimated
remaining time when the frame count is known before filtering. Rate conversion
uses an open-ended frame count. Use `--json` to print the complete run record.

The `.meta.json` sidecar records resolved settings, dimensions, frame count,
color and timing policies, hashes, tool versions, and implementation fingerprints.
Encoded bytes may differ across FFmpeg versions and platforms.

Encoder controls are listed in the
[configuration reference](configuration.md#encoding-and-source-preparation).

## Compressed sources

Use a [compressed-source preset](configuration.md#presets):

```bash
luma-fatal video local_inputs/input.mp4 outputs/cleaned.mp4 \
  --config configs/fattal-1-compressed.yaml --progress
```

Deblocking and debanding operate on planar YUV. RGB sources are converted to YUV
first when these treatments are enabled.

The cleanup mask carries temporal history: local changes release it, strong edges
override it, and scene cuts reset it. RGB frames are never averaged over time.
The sidecar records cleanup time and mask resets.

See [smooth-area cleanup](configuration.md#smooth-area-cleanup) and
[source noise](configuration.md#source-noise) for controls, defaults, and ranges.
