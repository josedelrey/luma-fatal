# Configuration

[Overview](../README.md) · [Video settings](video.md) · [Method](method.md)

Load YAML, TOML, or JSON with `--config`. Explicit CLI options override the file.
Unknown settings are rejected. Export a complete configuration with defaults:

```bash
uv run --locked luma-fatal config --output outputs/settings.yaml
```

Without a preset, the general defaults use Fattal 1 and `fade: 50`. Reference
presets use `fade: 0` for the full effect.

## Presets

All presets are in [configs/](../configs/). Each `fattal-1` family has a matching
`fattal-2` version.

| Preset | Purpose |
| --- | --- |
| [default.yaml](../configs/default.yaml) | General defaults, fade 50, 8 MP input cap, 1536-pixel working edge |
| [high.yaml](../configs/high.yaml) | General defaults with a 12 MP input cap and 2048-pixel working edge |
| [fattal-1.yaml](../configs/fattal-1.yaml), [fattal-2.yaml](../configs/fattal-2.yaml) | Full reference effect with neutral experimental controls |
| [fattal-1-compressed.yaml](../configs/fattal-1-compressed.yaml) | Smooth shadows, gentle video debanding, standard cleanup 0.75 |
| [fattal-1-compressed-multiscale.yaml](../configs/fattal-1-compressed-multiscale.yaml) | Compressed preset with scale-separated cleanup |
| [fattal-1-compressed-noise.yaml](../configs/fattal-1-compressed-noise.yaml) | Compressed preset with masked high-pass noise at 0.35 RMS |
| [fattal-1-compressed-fbm.yaml](../configs/fattal-1-compressed-fbm.yaml) | Masked fBm-like noise at 0.75 RMS, cleanup disabled |

Use `extends` to override a parent configuration. Parent paths are relative to
the child file, and named sections merge by setting. Save this beside the presets:

```yaml
extends: fattal-1.yaml
parameters:
  tone_strength: 60
  highlight_compression: 60
  vibrance: 50
output:
  format: png
```

The run record includes the resolved settings and inherited file fingerprints.

## Effect controls

The keys below belong to `parameters`. CLI flags replace underscores with hyphens,
such as `--tone-strength 60`. Values use a 0–100 scale except `output_gamma`.

| Setting | Reference value | Effect |
| --- | ---: | --- |
| `alpha` | 1 | Gradient attenuation, must be greater than 0 |
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

Tone strength 0 retains source brightness while other color and source treatments
still apply. Fade blends the completed RGB result with the original instead.
Input preparation still applies at fade 100.

For a softer result:

```bash
uv run --locked luma-fatal apply assets/lantern-garden-original.png outputs/softer.png \
  --config configs/fattal-1.yaml --output-format png \
  --tone-strength 60 --highlight-compression 60 --black-level 5 --white-level 95
```

Highlights already clipped in the input cannot be recovered. Raised black levels
can reduce saturation, and Fade can move pixels outside the selected output range.
See [processing order](method.md#processing-order) for interactions between controls.

## Resolution and export

`processing.max_input_pixels` caps the prepared input area and can reduce output
size. `processing.max_edge` caps the effect's working edge. The filtered image is
enlarged to the prepared input dimensions afterward. Changing either limit can
change the appearance. CLI overrides are `--max-input-pixels` and `--max-edge`.

Reference presets apply EXIF orientation, convert supported ICC profiles to sRGB,
and export JPEG at quality 100 with 4:2:0 subsampling and no embedded profile.
Use `--output-format png` with a `.png` destination for lossless export.
Image output directories are created automatically. Input and output paths must differ.

## Numerical controls

| Key | Reference value | Behavior |
| --- | --- | --- |
| `numerics.detail_level` | `0` | Omit this many finest attenuation scales |
| `numerics.power_mode` | `approximate` | Select reference approximations or `standard` NumPy powers |
| `numerics.shadow_softness` | `0` | Smooth the shadow cutoff with a half-width of 1–255 luminance levels |
| `numerics.luminance_mode` | `rounded` | Select rounded or `fractional` luminance before the logarithm |
| `processing.working_resampler` | `nearest` | Select `nearest`, `bilinear`, or `bicubic` reduction to working resolution |

These controls have matching CLI flags. Neutral values retain the reference
behavior. The [method notes](method.md) describe their numerical implications.
Solver settings and validation rules are defined in
[configuration.py](../src/luma_fatal/configuration.py) and included in exported configs.
