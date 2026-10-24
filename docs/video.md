# Video processing

[Overview](../README.md) · [Configuration](configuration.md#video-settings) · [Experiments](experiments.md)

## Requirements and input

Install `ffmpeg` and `ffprobe` so they are available on `PATH`. FFmpeg must include
`zscale` and `libx264` for CPU encoding and fallback, plus `bwdif` for
interlaced input, `tonemap` for HDR, and `deblock` or `deband` if you enable those
source treatments. No additional Python packages are required.

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

## NVIDIA hardware encoding

The bundled YAML configurations request NVIDIA NVENC encoding by default.
Presets that extend another YAML inherit that selection. Using NVENC requires
an NVENC-capable NVIDIA GPU, a working NVIDIA driver, and an FFmpeg build that
includes `h264_nvenc`. No additional Python packages are required.

```bash
luma-fatal video local_inputs/input.mp4 outputs/stylized.mp4 \
  --config configs/fattal-1.yaml --progress
```

NVENC accelerates output compression. The [optional CUDA backend](#gpu-filtering)
can separately accelerate filtering, source treatments, and resizing. Input
validation, decoding, FFmpeg color conversion, and optional deblocking/debanding
remain on the CPU. Total speed depends on the input codec and processing settings.

NVENC defaults to target quality `18` and preset `p4`. Use `--nvenc-cq` to select
quality from `1` to `51`, with lower values giving higher quality. `0` selects
automatic quality and does not request lossless encoding. Use `--nvenc-preset`
from `p1` (fastest) to `p7` (slowest, best compression quality). `--crf` and
`--preset` control `libx264`, including automatic CPU fallback. CQ and CRF values
do not imply equivalent quality or file size across encoders. See
[FFmpeg's NVENC documentation](https://www.ffmpeg.org/ffmpeg-codecs.html#NVENC-encoders).

The same settings are available as `video.encoder`, `video.nvenc_cq`, and
`video.nvenc_preset` in YAML, TOML, or JSON. A one-frame hardware encoding check
runs before filtering. If it fails, the run reports the switch on stderr and
automatically uses `libx264` with `video.crf` and `video.preset`.
The metadata sidecar retains the requested `video_configuration`. Its `encoding`
section records `requested_encoder`, the actual `encoder`, and `fallback_reason`
with the preflight diagnostics, or `null` when no fallback occurred.

Select `--encoder libx264` to use CPU encoding directly without probing the GPU.
Runs without a configuration also default to CPU encoding. Failures after
encoding starts still abort the run and publish no video or sidecar.

## GPU filtering

The experimental CUDA backend keeps luminance conversion, source cleanup,
attenuation pyramids, the Poisson solver, normalization, color reconstruction,
and resizing on the GPU. Optional denoising, multiscale cleanup, and source noise
also run there. Seeded noise is generated once on the CPU to retain its pattern.
Only RGB8 frames and small cleanup history cross the host/device boundary during
standard cleanup. The solver retains checkerboard sweeps and both right-edge
updates. Its parallel edge calculation uses a 96-row dependency halo, with the serial implementation
retained for columns that exceed shared-memory capacity. Small numerical
differences are permitted, while image commands retain the CPU reference.

Install the optional GPU dependencies and run a video:

```bash
uv sync --locked --extra gpu
uv run --extra gpu luma-fatal video local_inputs/input.mp4 outputs/stylized.mp4 \
  --config configs/fattal-1-compressed.yaml --progress
```

The extra supplies CuPy and matched CUDA 12.9 runtime/compiler libraries for
Linux or Windows with a compatible NVIDIA driver. A separate CUDA Toolkit
installation is unnecessary. See [CuPy installation](https://docs.cupy.dev/en/stable/install.html).
CPU processing and NVENC encoding do not require this extra.

All bundled YAML presets use `video.filter_backend: auto`. Automatic mode warms
both complete frame paths and compares two CPU and GPU timings, including
transfers and source treatments. It selects CUDA with at least a 20 percent
measured improvement and enough remaining frames to recover setup cost with a
25 percent margin. Known clips shorter than eight frames and working images
below 65,536 pixels skip initialization. RGB agreement on the calibration frame
must meet all of these limits, measured on channel values from 0 to 255:

- Mean absolute error at most 0.25 and root mean square error at most 0.75.
- 99th-percentile absolute error at most 2 and maximum error at most 12.

Calibration has a one-time startup cost, including first-use kernel compilation.
It cannot guarantee a speedup for every clip or changing machine load. A GPU
failure or three consecutive calls slower than calibrated CPU frame processing
switches to the CPU. Calibration and failed calls restore source-mask history
before retrying, so each source frame advances that history once.

Use `--filter-backend cpu` for reference processing or `--filter-backend auto`
to enable selection without a preset. Missing optional dependencies or hardware
also use the CPU. Selection messages go to stderr. The sidecar's `filtering`
section records `scope: frame_pipeline`, the actual backend, GPU, selection
reason, synchronized frame timings, calibration time, calibration RGB error
metrics, and CPU/CUDA frame counts. Counts cover every output frame. GPU filtering
and NVENC encoding can be selected independently.

## Compressed sources

Use a [compressed-source preset](configuration.md#presets):

```bash
luma-fatal video local_inputs/input.mp4 outputs/cleaned.mp4 \
  --config configs/fattal-1-compressed.yaml --progress
```

All compressed presets disable resizing, so the effect runs at full frame
resolution without the 8 MP input cap or 1536-pixel working limit. Full-resolution
processing takes more time and memory.

Deblocking and debanding operate on planar YUV. RGB sources are converted to YUV
first when these treatments are enabled.

All presets leave adaptive denoising disabled by default, including the compressed
multiscale, noise, and fBm variations. Enable it with `--denoise 0.5` for a lighter
blend or `--denoise 1` for full strength. Denoising suppresses fine and correlated
grain in floating-point RGB/luminance before cleanup and any intentional source
noise. Stronger source detail receives less smoothing. Very fine texture may
soften. Use `--denoise 0` to bypass it. Each frame is denoised independently,
without temporal color trails. The sidecar records strength, processed frames,
method, and denoising time separately from cleanup.

The cleanup mask carries temporal history: local changes release it, strong edges
override it, and scene cuts reset it. RGB frames are never averaged over time.
The sidecar records cleanup time and mask resets.

See [source denoising](configuration.md#source-denoising),
[smooth-area cleanup](configuration.md#smooth-area-cleanup), and
[source noise](configuration.md#source-noise) for controls, defaults, and ranges.
