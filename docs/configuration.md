# Configuration

[Overview](../README.md) · [Video settings](video.md) · [Method](method.md)

After [installation](../README.md#install), run all documentation commands from
the repository root. Activate the virtual environment once per shell session:

```bash
source .venv/bin/activate
```

On Windows, use `.venv\Scripts\Activate.ps1` in PowerShell or
`source .venv/Scripts/activate` in Git Bash.

Use `--config` to load a YAML, TOML, or JSON file. Explicit CLI options take
precedence over file settings. Without `--config`, the application uses built-in
defaults. It does not load `configs/default.yaml` automatically.

To export all built-in defaults, run:

```bash
luma-fatal config --output outputs/settings.yaml
```

The file extension selects the format. Existing files are never overwritten.
Run `luma-fatal config` to print the defaults without writing a file.

## File structure and validation

The tables below list every supported key. Section tables use full paths such as
`parameters.fade`, written in YAML as:

```yaml
parameters:
  fade: 20
```

Omitted settings use built-in defaults. Defaults in these tables apply before
preset values and CLI overrides. Numeric values must be finite, and integer
settings require integers. Booleans must be `true` or `false`. YAML words such as
`on`, `off`, `yes`, and `no` are strings. Quote numeric strings such as
`jpeg_subsampling: "420"` and `frame_rate: "30000/1001"`.

Unknown keys, duplicate keys, invalid values, and inheritance cycles are rejected.
The inherited configuration is validated before CLI overrides, so an override
cannot repair an invalid file. TOML has no `null` value, so omit optional settings
to use their defaults.

| Key | Default | Accepted values and behavior |
| --- | --- | --- |
| `schema_version` | `1` | Integer `1` is the only supported schema. |
| `filter` | `fattal1` | `fattal1`, `fattal2`, or `both`. Fattal 1 accumulates gradient attenuation across scales. Fattal 2 replaces it at finer scales. The image CLI can export both variants. Video and single-result Python calls require one variant. |
| `extends` | Omitted | Path to one parent configuration file, resolved relative to this file. Child values override parent values within each section. |

## Presets

The [configs/](../configs/) directory contains all presets, with a matching
`fattal-2` version for each `fattal-1` family.
All bundled YAML presets use `numerics.shadow_softness: 6`, equivalent to
`--shadow-softness 6`. API and CLI runs without a config keep the default of `0`.
Each preset has a `-fast.yaml` counterpart. The fast copies keep the original
CPU recipe and enable GPU-specific solver acceleration, reusable calibration,
and pinned input frames. They require the optional GPU dependencies to realize
those speedups. CPU fallback produces the original preset result.

| Preset | Purpose |
| --- | --- |
| [default.yaml](../configs/default.yaml) | General defaults, 8 MP input cap, 1536-pixel working edge |
| [high.yaml](../configs/high.yaml) | General defaults with a 12 MP input cap and 2048-pixel working edge |
| [fattal-1.yaml](../configs/fattal-1.yaml), [fattal-2.yaml](../configs/fattal-2.yaml) | Full reference effect with smooth shadows |
| [fattal-1-compressed.yaml](../configs/fattal-1-compressed.yaml) | Smooth shadows, gentle video debanding, standard cleanup 0.75 |
| [fattal-1-compressed-multiscale.yaml](../configs/fattal-1-compressed-multiscale.yaml) | Compressed preset with scale-separated cleanup |
| [fattal-1-compressed-noise.yaml](../configs/fattal-1-compressed-noise.yaml) | Compressed preset with masked high-pass noise at 0.35 RMS |
| [fattal-1-compressed-fbm.yaml](../configs/fattal-1-compressed-fbm.yaml) | Masked fBm-like noise at 0.75 RMS, flat cleanup disabled |

All `compressed` presets process frames at full resolution with
`processing.resize: false`. They disable both the input-area cap and
working-resolution reduction.

Use `extends` to inherit from a parent configuration and override selected settings.
Parent paths are relative to the child file. Named sections merge by setting,
and the child can add sections absent from its parent. For example, save this
file beside the presets:

```yaml
extends: fattal-1.yaml
parameters:
  tone_strength: 60
  highlight_compression: 60
  vibrance: 50
output:
  format: png
```

The run record includes the resolved settings and fingerprints of inherited files.

## Effect controls

| Key | Default | Accepted values and behavior |
| --- | --- | --- |
| `parameters.alpha` | `1` | Greater than `0`, at most `100`. Scales the reference gradient magnitude used for attenuation. Must remain positive in float32 after division by 100. |
| `parameters.beta` | `80` | `0–100`. Controls how attenuation varies with gradient strength. At `100`, attenuation is neutral, but reconstruction and normalization still run. Lower values increase the difference in treatment between weak and strong gradients. |
| `parameters.saturation` | `100` | `0–100`. Strength of source channel-to-brightness ratios in color reconstruction. At `0`, the filtered component is grayscale. |
| `parameters.noise` | `0` | `0–100`. Offset added to gradient magnitudes in the attenuation calculation. It changes low-gradient behavior without generating noise pixels. |
| `parameters.fade` | `0` | `0–100`. Blend toward the original after filtering and enlargement. `0` applies the full result. `100` bypasses the effect, image orientation/color preparation, and resizing. Export encoding and video input preparation still apply. |
| `parameters.output_gamma` | `1.0` | Positive number with a finite reciprocal. Applies `x^(1/gamma)` to clipped, normalized RGB. Above `1` brightens, below `1` darkens. |
| `parameters.tone_strength` | `100` | `0–100`. Blend original and reconstructed brightness before color reconstruction. At `0`, source brightness is retained while other color and source treatments can still apply. |
| `parameters.highlight_compression` | `0` | `0–100`. Roll off RGB peaks above 75% of white while preserving channel ratios. `0` disables compression. At `100`, finite nonnegative peaks are bounded. Partial strength can still leave clipping. |
| `parameters.black_level` | `0` | `0–100`, strictly below `white_level`. Output black endpoint as a percentage of white. Raising it lifts blacks. |
| `parameters.white_level` | `100` | `0–100`, strictly above `black_level`. Output white endpoint. Lowering it dims highlights. |
| `parameters.vibrance` | `0` | `0–100`. Boost muted colors with a gain limited to keep channels in gamut. Grays and fully saturated colors remain unchanged at this stage. |

For gradients above `numerics.gradient_threshold`, attenuation at each scale is
`((gradient + noise / 100) / ((alpha / 100) * mean_gradient))^(beta / 100 - 1)`.
Alpha, beta, and noise interact, so their visual effects depend on the input.

For a softer result:

```bash
luma-fatal apply assets/lantern-garden-original.png outputs/softer.png \
  --config configs/fattal-1.yaml --output-format png \
  --tone-strength 60 --highlight-compression 60 --black-level 5 --white-level 95
```

Output levels apply after shadow protection, including to protected pixels.
Raising the black level can reduce saturation. Fade applies afterward and can
move pixels outside the selected output range.
See [processing order](method.md#processing-order) for interactions between controls.

## Resolution and export

### Processing

| Key | Default | Accepted values and behavior |
| --- | --- | --- |
| `processing.device_class` | `medium` | `low`, `medium`, or `high`. Selects the size budgets below. This is a fixed profile, with no hardware detection. |
| `processing.max_edge` | `null` | `null` uses the device-class working edge. An integer ≥ `2` overrides it. Limits the longest edge at which the effect runs. The result is enlarged to the prepared input dimensions afterward. |
| `processing.max_input_pixels` | `null` | `null` uses the device-class input budget. An integer ≥ `4` overrides it. Caps input area before working-resolution filtering and can reduce final output dimensions. |
| `processing.limit_input_pixels` | `true` | Boolean. Disable to remove the input-area cap while retaining the working-edge limit. |
| `processing.resize` | `true` | Boolean. Disable to run at original dimensions without input-area fitting or working-resolution reduction. |
| `processing.transparency` | `reject` | `reject` refuses nonopaque pixels. `straight` permits experimental straight-alpha RGBA processing, retaining alpha. Video output is always opaque. |
| `processing.blend_mode` | `normal` | Only `normal` is supported. Controls blending of the filtered result with the original. |
| `processing.working_resampler` | `nearest` | `nearest`, `bilinear`, or `bicubic` reduction to working resolution. Bilinear and bicubic use no antialias prefilter and truncate to RGB8. Bicubic clamps overshoot. Input-area fitting and enlargement use their own fixed resamplers. |

| Device class | Input-area budget | Working longest edge |
| --- | ---: | ---: |
| `low` | 4,000,000 pixels | 1024 pixels |
| `medium` | 8,000,000 pixels | 1536 pixels |
| `high` | 12,000,000 pixels | 2048 pixels |

`null` selects a device-class budget. To disable a limit, use the corresponding
boolean. With `resize: false`, both explicit limits must be `null`. With
`limit_input_pixels: false`, `max_input_pixels` must be `null`. Input-area
reduction requires opaque pixels, so disable input pixel limiting when processing
transparent RGBA that would exceed the budget.

### Image input

These settings apply to file and Pillow image inputs. NumPy arrays must already
contain interpreted RGB pixels. Video uses [its own input preparation](video.md#input-preparation).

| Key | Default | Accepted values and behavior |
| --- | --- | --- |
| `image.color_space` | `srgb` | `srgb` converts embedded ICC profiles to sRGB, assuming sRGB when no profile exists. `encoded` bypasses ICC conversion and processes encoded channel values. |
| `image.color_backend` | `matrix_trc` | `matrix_trc` uses the built-in converter for supported RGB matrix/TRC profiles and falls back to Little CMS for unsupported transforms or input modes. `lcms` requests Little CMS directly. Used when converting to sRGB. |
| `image.exif_orientation` | `apply` | `apply` rotates or mirrors according to EXIF orientation. `ignore` keeps the stored pixel orientation. |
| `image.decode_sampling` | `true` | Boolean. Allows early sampling of supported JPEG/PNG file inputs according to the input-area budget, before color conversion, orientation, and final area fitting. Disable for full decoding. Can change pixels. Already decoded arrays are unaffected. |

### Image output

These settings control image file encoding. They do not affect video encoding
or in-memory array outputs.

| Key | Default | Accepted values and behavior |
| --- | --- | --- |
| `output.format` | `jpeg` | `jpeg` or `png`. The destination must have a matching `.jpg`/`.jpeg` or `.png` extension. |
| `output.jpeg_quality` | `100` | Integer `1–100`. Higher values generally increase JPEG quality and file size. |
| `output.jpeg_subsampling` | `"420"` | `"444"` preserves full chroma resolution. `"422"` halves it horizontally. `"420"` halves it horizontally and vertically. JPEG only. |
| `output.embed_icc` | `false` | Boolean. Embed the available output ICC profile in PNG or JPEG. |
| `output.jpeg_background` | `null` | `null` or three RGB integers from `0–255`, such as `[255, 255, 255]`. Sets the background for compositing RGBA into JPEG. Without a background, JPEG export rejects nonopaque pixels. PNG retains alpha. |

## Numerical controls

| Key | Default | Accepted values and behavior |
| --- | --- | --- |
| `numerics.iterations` | `80` | Integer ≥ `1`. Poisson solver iteration count. More iterations increase computation and change convergence. |
| `numerics.relaxation` | `1.9` | Greater than `0`, below `2`. Solver update factor. Float32 weights `relaxation` and `1 - relaxation` must remain strictly inside `(0, 2)` and `(-1, 1)`, respectively. Values too close to the endpoints are rejected. |
| `numerics.pyramid_min_size` | `32` | Integer ≥ `2`. Controls the coarsest pyramid scale. Smaller values permit more coarse scales. |
| `numerics.normalization_low` | `0.001` | `0 ≤ low < high < 1`. Lower quantile of nonzero reconstructed brightness used for normalization. Default selects approximately the 0.1st percentile. |
| `numerics.normalization_high` | `0.995` | `low < high < 1`. Upper normalization quantile. Default selects approximately the 99.5th percentile. |
| `numerics.normalization_floor` | `0.0001` | Greater than `0`, at most `1`, positive in float32. Fallback for nonpositive normalized brightness and all-zero normalization input. |
| `numerics.gradient_threshold` | `0.0001` | Greater than `0`, at most `1`, positive in float32. Gradients at or below this magnitude retain neutral attenuation at that scale. |
| `numerics.log_offset` | `0.01` | Greater than `0`, at most `1`, positive in float32. Offset added before the luminance logarithm and subtracted after exponentiation. Avoids log-of-zero and changes dark-region behavior. |
| `numerics.shadow_threshold` | `6` | Integer `0–255`. Original-source luminance cutoff for preserving source RGB before output-level mapping. |
| `numerics.detail_level` | `0` | Integer ≥ `0`. Number of finest attenuation scales to omit. If all are omitted, attenuation is neutral, but reconstruction and normalization still run. |
| `numerics.power_mode` | `approximate` | `approximate` uses reference power approximations. `standard` uses NumPy float64 powers for attenuation and color reconstruction, narrowing attenuation to float32. |
| `numerics.shadow_softness` | `0` | Integer `0–255`. Half-width of the smooth transition around `shadow_threshold`, in source luminance levels. `0` keeps a hard cutoff. Transition endpoints are clamped to `0–255`. |
| `numerics.luminance_mode` | `rounded` | `rounded` rounds source luminance to integer levels before the logarithm. `fractional` retains fractional luminance. Neither recovers detail lost in an 8-bit source. |

## Video settings

These settings apply to the video adapter. See [video processing](video.md) for
requirements, input preparation, and examples. Output is silent SDR H.264 MP4.

### Encoding and source preparation

| Key | Default | Accepted values and behavior |
| --- | --- | --- |
| `video.encoder` | `libx264` | `libx264` for CPU encoding or `h264_nvenc` for NVIDIA hardware encoding. Bundled YAML presets select `h264_nvenc`. A failed NVENC preflight automatically falls back to `libx264` using `video.crf` and `video.preset`. The actual encoder and fallback diagnostics are recorded in the sidecar's `encoding` section. |
| `video.filter_backend` | `cpu` | `cpu` retains reference processing. `auto` benchmarks the optional complete CUDA frame pipeline, including transfers and setup amortization, and uses it only when sufficiently faster and visually close. Missing GPU support, short or small workloads, insufficient speedup, and runtime GPU failures use the CPU. Bundled YAML presets select `auto`. Independent of `video.encoder`. |
| `video.calibration_cache` | `false` | Reuse compatible CUDA calibration across runs. A cache hit still compares the first GPU frame with the CPU reference. The cache is stored under `XDG_CACHE_HOME/luma-fatal` or the user's default cache directory. |
| `video.gpu_iterations` | `null` | `null` uses `numerics.iterations` on both backends. A positive integer overrides only the GPU solver, leaving CPU fallback unchanged. Fast presets use `32`. |
| `video.lighting_scale` | `1.0` | Fraction `0.25–1`. Below `1`, the GPU estimates the remaining lighting correction on a smaller grid and applies it at full resolution. Fast presets use `0.5`. CPU processing retains the reference correction. |
| `video.decode_prefetch` | `0` | Integer `0–16`. Queue that many decoded frames in a background reader. Workload dependent and disabled in the fast presets because it slowed the tested clip when combined with pinned frames. |
| `video.pinned_frames` | `false` | Decode RGB frames into pinned host memory for GPU upload. Falls back to ordinary buffers if pinning is unavailable. Fast presets enable it. |
| `video.crf` | `18` | Integer `0–51`, used only by `libx264`. Lower values generally give higher quality and larger files. `0` requests lossless encoding, but input preparation and pixel-format conversions still apply. |
| `video.preset` | `medium` | Used only by `libx264`: `ultrafast`, `superfast`, `veryfast`, `faster`, `fast`, `medium`, `slow`, `slower`, or `veryslow`. Slower presets spend more time on compression. |
| `video.nvenc_cq` | `18` | Integer `0–51`, used only by `h264_nvenc` with quality-targeted variable bitrate and no fixed bitrate target. Lower positive values give higher quality. `0` selects automatic quality, rather than lossless encoding. CQ and CRF values are not equivalent across encoders. |
| `video.nvenc_preset` | `p4` | Used only by `h264_nvenc`: `p1` through `p7`, from fastest to slowest with increasing compression quality. |
| `video.deblock` | `off` | `off`, `weak4`, or `weak8`. Weak source deblocking on 4- or 8-pixel boundaries before sRGB conversion and filtering. Can soften detail and miss artifacts with other block sizes. |
| `video.deband` | `off` | `off` or `gentle`. Source debanding before sRGB conversion and filtering. Can reduce tonal steps and alter fine texture or variation between frames. |
| `video.frame_rate` | `"auto"` | `"auto"` or a positive rate string such as `"30"` or `"30000/1001"`. Selects constant output cadence, duplicating or dropping frames as needed. Automatic rate selection is described under [input preparation](video.md#input-preparation). |
| `video.input_color` | `auto` | `auto`, `bt709`, `bt601-pal`, `bt601-ntsc`, `srgb`, `bt2020-pq`, or `bt2020-hlg`. Overrides source color interpretation. `auto` uses metadata and recorded assumptions. The `srgb` profile uses a BT.709 matrix for YUV storage. RGB storage always uses an identity matrix. |
| `video.input_range` | `auto` | `auto`, `limited`, or `full`. Overrides the source signal range. `auto` uses metadata and the [input preparation rules](video.md#input-preparation). |

### Source denoising

Adaptive denoising runs at working resolution before smooth-area cleanup,
generated source noise, and the Fattal effect. It estimates noise in RGB detail
bands at three scales, including correlated grain, and shrinks detail according
to local signal energy. Floating-point RGB and luminance avoid a new RGB8
quantization step. Frames are independent, with no temporal color averaging.
Original pixels still control shadow protection and Fade.

All presets and API defaults leave denoising disabled at `0.0`, including the
compressed multiscale, noise, and fBm variations. Enable it explicitly with
`video.denoise` or `--denoise`, for example `--denoise 0.5`. Very fine texture
near the estimated noise floor may soften. Reduce the strength to retain more
grain and detail. The existing noise/fBm variations retain their intentional
noise, which is added after denoising when enabled.

| Key | Default | Accepted values and behavior |
| --- | --- | --- |
| `video.denoise` | `0.0` | `0–1`. Adaptive spatial denoising strength. `0` bypasses denoising in all presets by default. |

### Smooth-area cleanup

Cleanup runs at working resolution before the Fattal effect. Its mask protects
texture and abrupt edges. Original pixels still control shadow protection and
Fade. Cleanup can change smooth skin and grain.

| Key | Default | Accepted values and behavior |
| --- | --- | --- |
| `video.flat_cleanup` | `0.0` | `0–1`. Cleanup strength. `0` disables cleanup in either mode, including broad correction. In multiscale mode, positive values control fine correction. |
| `video.flat_cleanup_mode` | `standard` | `standard` applies guided RGB/luminance correction. `multiscale` splits that correction into fine and broad components. |
| `video.flat_cleanup_coarse_strength` | `1.0` | `0–1`. Broad correction strength, used only when cleanup is enabled in multiscale mode. |
| `video.flat_cleanup_sigma` | `1.2` | `0.01–16`. Gaussian standard deviation in working pixels for separating broad and fine correction. Used only in multiscale mode. |

### Source noise

Generated noise enters floating-point RGB and luminance after denoising and
cleanup and before the effect. The same change enters every RGB channel.
Original pixels still control shadow protection and Fade.

| Key | Default | Accepted values and behavior |
| --- | --- | --- |
| `video.pre_noise_rms` | `0.0` | `0–4`, in source 0–255 units. Noise RMS before masking and range limits. `0` disables noise. |
| `video.pre_noise_kind` | `highpass` | `highpass` produces fine high-pass noise. `fbm` produces correlated noise from multiple spatial scales. |
| `video.pre_noise_masked` | `true` | Boolean. Restrict noise to the source's smooth-region mask, measured after denoising when enabled. Disable to apply it across the frame, subject to range limits. |
| `video.pre_noise_seed` | `20261017` | Integer ≥ `0`. Seed for the deterministic noise field. The field stays fixed in working-frame coordinates without motion compensation. |

## CLI overrides

Settings use CLI flags with the section name removed and underscores replaced by
hyphens, such as `--tone-strength 60`, `--max-edge 2048`, and `--pre-noise-rms 0.35`.
`output.format` uses `--output-format`. Boolean settings also accept a `--no-`
form, such as `--no-resize`, `--no-embed-icc`, and `--no-pre-noise-masked`.
`--jpeg-background` takes three integers, such as `--jpeg-background 255 255 255`.
`schema_version` and `extends` are file-only settings.

Image input/output flags apply to `apply` and `evaluate`. Video flags apply to
`video`. Effect, processing, and numerical flags apply to all three commands.
Run `luma-fatal COMMAND --help` for command-specific options.

Changing `--device-class` clears inherited explicit size limits unless they are
also supplied on the CLI. `--no-resize` clears inherited limits, and
`--no-limit-input-pixels` clears the inherited input-area limit. Explicit CLI
limits must still satisfy the processing constraints.

`--profile bounded-1536` selects built-in defaults. `--profile core` starts from
those defaults with resizing disabled and straight RGBA allowed. `--profile`
cannot be combined with `--config`. Profiles are CLI/API conveniences, with no
configuration-file key.
