# Video processing

[Overview](../README.md) · [Presets](configuration.md#presets) · [Experiments](experiments.md)

## Requirements and input

Install `ffmpeg` and `ffprobe` so they are available on `PATH`. FFmpeg must include
`zscale` and `libx264`, as well as `deblock` or `deband` if you enable those source
treatments. No additional Python packages are required.

Input must contain one progressive, square-pixel video stream at a constant frame
rate, with explicit BT.709 matrix, transfer, and primaries tags and an explicit
range tag. Pixels must be opaque 8-bit planar YUV 4:2:0, 4:2:2, or 4:4:4.
Untagged color, HDR, interlacing, display rotation, and variable frame rate are
rejected.

```bash
luma-fatal video local_inputs/input.mp4 outputs/stylized.mp4 \
  --config configs/fattal-1.yaml --progress
```

A preliminary decoding pass checks timing, geometry, and color tags throughout
the stream. Frames are then converted to full-range sRGB RGB8 for filtering and
back to limited-range BT.709 for H.264 MP4 output, using zscale's display-referred
BT.709/BT.1886 interpretation. Processing is sequential, with bounded buffering.
The output starts at timestamp zero and retains the source's rational frame rate.

## Output

The output contains only silent video. Audio, subtitles, and other streams are
discarded. If an output dimension is odd, edge pixels are repeated at the right
or bottom to accommodate YUV420P. Image ICC, EXIF, and JPEG/PNG export options do
not apply to video.

Use `--progress` to report filtered frames, speed, and estimated remaining time on
stderr, and `--json` to print the complete run record. Choose a new output filename
for each run, since existing output files and metadata sidecars are never overwritten.

The `.meta.json` sidecar records resolved settings, dimensions, frame count,
color and timing policies, hashes, tool versions, and implementation fingerprints.
Encoded bytes may differ across FFmpeg versions and platforms.

| CLI option | Config key | Default | Purpose |
| --- | --- | --- | --- |
| `--crf` | `video.crf` | `18` | H.264 quality, range 0–51 |
| `--preset` | `video.preset` | `medium` | H.264 encoder speed preset |
| `--deblock` | `video.deblock` | `off` | Weak deblocking on 4- or 8-pixel boundaries with `weak4` or `weak8` |
| `--deband` | `video.deband` | `off` | Gentle decoded-YUV debanding with `gentle` |

Deblocking and debanding run before color conversion and filtering. Deblocking
can soften detail and may miss artifacts with variable codec block sizes, while
debanding can alter fine texture and variation between frames.

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
tradeoffs and limitations.
