# Configuration

[Overview](../README.md) · [Video settings](video.md) · [Method](method.md)

After [installation](../README.md#install), run the following commands from the
repository root. Activate the virtual environment once per shell session:

```bash
source .venv/bin/activate
```

Use `--config` to load a YAML, TOML, or JSON file. Explicit CLI options take
precedence over file settings, and unknown settings are rejected. To export a
complete configuration, including defaults, run:

```bash
luma-fatal config --output outputs/settings.yaml
```

The general defaults use Fattal 1 with `fade: 50`. Reference presets set `fade: 0`
to apply the full effect.

## Presets

The [configs/](../configs/) directory contains all presets, with a matching
`fattal-2` version for each `fattal-1` family.

| Preset | Purpose |
| --- | --- |
| [default.yaml](../configs/default.yaml) | General defaults, fade 50, 8 MP input cap, 1536-pixel working edge |
| [high.yaml](../configs/high.yaml) | General defaults with a 12 MP input cap and 2048-pixel working edge |
| [fattal-1.yaml](../configs/fattal-1.yaml), [fattal-2.yaml](../configs/fattal-2.yaml) | Full reference effect with neutral experimental controls |
| [fattal-1-compressed.yaml](../configs/fattal-1-compressed.yaml) | Smooth shadows, gentle video debanding, standard cleanup 0.75 |
| [fattal-1-compressed-multiscale.yaml](../configs/fattal-1-compressed-multiscale.yaml) | Compressed preset with scale-separated cleanup |
| [fattal-1-compressed-noise.yaml](../configs/fattal-1-compressed-noise.yaml) | Compressed preset with masked high-pass noise at 0.35 RMS |
| [fattal-1-compressed-fbm.yaml](../configs/fattal-1-compressed-fbm.yaml) | Masked fBm-like noise at 0.75 RMS, cleanup disabled |

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

The settings below belong to `parameters` and use a 0–100 scale, except for
`output_gamma`. Their CLI flags use hyphens in place of underscores, as in
`--tone-strength 60`.

| Setting | Reference value | Effect |
| --- | ---: | --- |
| `alpha` | 1 | Gradient attenuation, must remain positive in float32 after division by 100 |
| `beta` | 80 | Gradient attenuation |
| `saturation` | 100 | Color recombination strength |
| `noise` | 0 | Offset in the attenuation calculation, separate from source noise |
| `fade` | 0 | Blend toward the original, with 100 bypassing the effect |
| `tone_strength` | 100 | Blend source and filtered brightness before color reconstruction |
| `highlight_compression` | 0 | Roll off bright RGB peaks while preserving their channel ratios |
| `output_gamma` | 1.0 | Positive finite gamma, above 1 brightens and below 1 darkens |
| `vibrance` | 0 | Boost muted colors with brightness and gamut protection |
| `black_level` | 0 | Filtered output black endpoint, as a percentage of white |
| `white_level` | 100 | Filtered output white endpoint, strictly above black |

Even if an alpha value passes validation, an extremely small value can cause
division by zero or overflow for some image gradients. Increase alpha if filtering
reports either condition.

At tone strength 0, the filter retains source brightness while still applying
other color and source treatments. Fade blends the completed RGB result with the
original. Input preparation applies even at fade 100.

For a softer result:

```bash
luma-fatal apply assets/lantern-garden-original.png outputs/softer.png \
  --config configs/fattal-1.yaml --output-format png \
  --tone-strength 60 --highlight-compression 60 --black-level 5 --white-level 95
```

Clipped input highlights cannot be recovered. Raising the black level can reduce
saturation, and Fade can move pixels outside the selected output range.
See [processing order](method.md#processing-order) for interactions between controls.

## Resolution and export

`processing.max_input_pixels` limits the prepared input area, so it can also reduce
the output size. `processing.max_edge` limits the longest edge at which the effect
runs. After filtering, the image is enlarged to the prepared input dimensions.
Either limit can affect the appearance. Use `--max-input-pixels` and `--max-edge`
to override them from the CLI.

Reference presets apply EXIF orientation, convert supported ICC profiles to sRGB,
and export JPEG at quality 100 with 4:2:0 subsampling and no embedded profile.
For lossless export, use `--output-format png` with a `.png` destination.
Output directories are created automatically, but existing output files and
metadata sidecars are never overwritten. Input, output, and sidecar paths must
refer to distinct files, even when accessed through hard links or symbolic links.
The image and metadata are staged before publication. If publication fails with
a handled error, files created by the run are removed.

## Numerical controls

| Key | Reference value | Behavior |
| --- | --- | --- |
| `numerics.iterations` | `80` | Poisson solver iteration count, at least 1 |
| `numerics.relaxation` | `1.9` | Poisson relaxation factor, strictly between 0 and 2 after float32 conversion |
| `numerics.pyramid_min_size` | `32` | Minimum pyramid dimension, at least 2 |
| `numerics.normalization_low` | `0.001` | Lower normalization quantile, below `normalization_high` |
| `numerics.normalization_high` | `0.995` | Upper normalization quantile, below 1 |
| `numerics.normalization_floor` | `0.0001` | Positive normalization floor, at most 1 |
| `numerics.gradient_threshold` | `0.0001` | Positive gradient threshold, at most 1 |
| `numerics.log_offset` | `0.01` | Positive luminance offset before logarithms, at most 1 |
| `numerics.shadow_threshold` | `6` | Shadow preservation cutoff, 0–255 |
| `numerics.detail_level` | `0` | Omit this many finest attenuation scales |
| `numerics.power_mode` | `approximate` | Select reference approximations or `standard` NumPy powers |
| `numerics.shadow_softness` | `0` | Smooth the shadow cutoff with a half-width of 1–255 luminance levels |
| `numerics.luminance_mode` | `rounded` | Select rounded or `fractional` luminance before the logarithm |
| `processing.working_resampler` | `nearest` | Select `nearest`, `bilinear`, or `bicubic` reduction to working resolution |

The solver also converts `1 - relaxation` to float32 and requires this weight to
remain strictly between -1 and 1. Values too close to either endpoint are rejected.

All numerical settings have matching CLI flags, such as `--iterations 40` and
`--normalization-low 0.01`, which take precedence over YAML settings. Neutral
values preserve reference behavior. See the [method notes](method.md) for the
numerical implications and
[configuration.py](../src/luma_fatal/configuration.py) for solver settings and
validation rules. Exported configurations include these settings.
