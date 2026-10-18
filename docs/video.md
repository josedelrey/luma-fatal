# Video processing

[Overview](../README.md) · [Presets](configuration.md#presets) · [Experiments](experiments.md)

After [installation](../README.md#install), run commands from the repository root
and activate the virtual environment once per shell session:

```bash
source .venv/bin/activate
```

## Requirements and input

Install `ffmpeg` and `ffprobe` on `PATH`. FFmpeg needs `zscale` and `libx264`,
plus `deblock` or `deband` when those source treatments are enabled.
No additional Python packages are required.

Accepted input has one progressive, square-pixel, constant-frame-rate video stream
with explicit BT.709 matrix, transfer, primaries, and range tags. Pixels must be
opaque 8-bit planar YUV 4:2:0, 4:2:2, or 4:4:4. Untagged color, HDR, interlacing,
display rotation, and variable frame rate are rejected.

```bash
luma-fatal video local_inputs/input.mp4 outputs/stylized.mp4 \
  --config configs/fattal-1.yaml --progress
```

A preliminary decoding pass validates timing, geometry, and color tags across the
whole stream. Frames then convert to full-range sRGB RGB8 for filtering and back
to limited-range BT.709 for H.264 MP4 output. Conversion uses zscale's
display-referred BT.709/BT.1886 interpretation. Frames are processed sequentially
with bounded buffering. Output starts at timestamp zero and retains the source's
rational frame rate.

## Output

Output is silent. Audio, subtitles, and other streams are discarded. Odd output
dimensions are padded at the right or bottom by repeating edge pixels for YUV420P.
Image ICC, EXIF, and JPEG/PNG export options do not apply to video.

`--progress` reports filtered frames, speed, and an estimated remaining time on
stderr. `--json` prints the complete run record. Existing output files or metadata
sidecars are refused, so use a new output filename for each run.

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
can soften detail and miss variable codec block sizes. Debanding can alter fine
texture and frame-to-frame variation.

## Compressed sources

The compressed presets enable gentle debanding, smooth shadows, and standard
smooth-area cleanup at strength 0.75:

```bash
luma-fatal video local_inputs/input.mp4 outputs/cleaned.mp4 \
  --config configs/fattal-1-compressed.yaml --progress
```

Cleanup detects smooth regions at working resolution, protecting texture and
abrupt edges with a continuous mask. Guided filtering corrects RGB and luminance
in floating point before the Fattal effect. Fade and shadow protection retain
the original source pixels.

Only the mask has temporal history. Local changes release it, strong edges
override it, and scene cuts reset it. RGB frames are never temporally averaged.
Smooth skin and desired grain can still change. Cleanup time and mask resets
are recorded in the sidecar.

| CLI option | Config key | Default | Meaning |
| --- | --- | --- | --- |
| `--flat-cleanup` | `video.flat_cleanup` | `0` | Cleanup strength, 0–1, with 0 disabling it |
| `--flat-cleanup-mode` | `video.flat_cleanup_mode` | `standard` | `standard` or `multiscale` correction |
| `--flat-cleanup-coarse-strength` | `video.flat_cleanup_coarse_strength` | `1.0` | Broad correction strength, 0–1, used only in multiscale mode |
| `--flat-cleanup-sigma` | `video.flat_cleanup_sigma` | `1.2` | Multiscale Gaussian radius, 0.01–16 working pixels |

In multiscale mode, `flat_cleanup` controls fine correction strength. The bundled
multiscale presets use fine strength 0.75, broad strength 1.0, and sigma 1.2.
They are slower and can retain contour fragments alongside fine texture.

## Source noise

Noise enters floating RGB and luminance after cleanup and before filtering.
By default, the original source's smooth-region mask restricts it.

| CLI option | Config key | Default | Meaning |
| --- | --- | --- | --- |
| `--pre-noise-rms` | `video.pre_noise_rms` | `0` | RMS in input 0–255 units, range 0–4, with 0 disabling noise |
| `--pre-noise-kind` | `video.pre_noise_kind` | `highpass` | Fine `highpass` or correlated `fbm` noise |
| `--pre-noise-seed` | `video.pre_noise_seed` | `20261017` | Seed for the deterministic field |
| `--no-pre-noise-masked` | `video.pre_noise_masked` | `true` | Flag disables masking |

The noise presets combine standard cleanup 0.75 with high-pass noise 0.35 RMS.
The fBm presets use correlated noise 0.75 RMS with cleanup disabled.
Noise stays fixed in working-frame coordinates without motion compensation.
It can conceal contours with grain or introduce strongly amplified cloudy patterns.

Filtering has no temporal stabilization. Judge cleanup and noise on moving footage.
The [experiment notes](experiments.md) report the measured tradeoffs and limits.
