# Method and scope

[Overview](../README.md) · [Controls](configuration.md) · [Verification](development.md)

## Gradient-domain stylization

The filter builds a log-luminance pyramid, attenuates its gradients, and uses a
Poisson solver to reconstruct luminance. It then recombines the reconstructed
brightness with the source RGB. Fattal 1 accumulates attenuation across scales,
while Fattal 2 replaces the attenuation field at each finer scale. Both are
project variants of the approach described by
[Fattal, Lischinski, and Werman (2002)](#references).
The implementation operates on 8-bit SDR inputs.

## Processing order

1. Decode, orient, and interpret color. Apply the configured input-area budget.
2. Reduce to working resolution. Video can apply source cleanup and noise here.
3. Attenuate log-luminance gradients and reconstruct brightness.
4. Apply tone strength, recombine color, compress highlights, and apply output gamma.
5. Apply vibrance, shadow protection, output levels, and RGB8 quantization.
6. Enlarge to the prepared input dimensions, apply Fade, and encode.

Tone strength blends source and mapped luminance before color reconstruction,
using source brightness from before cleanup and noise. Highlight compression
scales channels together above a knee at 75 percent of white. At full strength,
it bounds finite, nonnegative RGB peaks. Partial strength may still leave bright
values that clip.

Gamma applies `x^(1/gamma)` to clipped, normalized RGB. Vibrance increases channel
differences around weighted brightness, with a requested gain of
`1 + (vibrance / 100) * (1 - saturation)^2`, where saturation is `(max - min) / max`.
The gain is limited to keep channels in gamut, leaving grays and fully saturated
colors unchanged at this stage.

Shadow protection uses the original source brightness. A positive
`shadow_softness` blends through a smoothstep interval around `shadow_threshold`,
while zero retains the hard cutoff. Output levels then remap channels, including
protected shadows, before Fade blends the result with the original RGB. At fade
100, the effect and its output controls are bypassed.

## Numerical choices

The reference path preserves the order of float32 operations, scalar power
approximations, and fixed boundary behavior. Selecting `power_mode: standard`
uses NumPy float64 powers, narrowing attenuation to float32 while retaining
float64 color recombination.

`detail_level` omits the finest attenuation scales and interpolates retained
fields through the excluded levels. Omitting every level gives unit attenuation,
although reconstruction and normalization still apply. `pyramid_min_size`
controls the coarsest scale. Fractional luminance skips integer rounding before
the logarithm, but cannot recover detail lost in an 8-bit source.

Bilinear and bicubic reduction to working resolution use no antialias prefilter
and truncate their results to RGB8. Bicubic sampling uses the scalar kernel with
`a = -0.75` and clamps overshoot. These choices leave input-area fitting and
enlargement unchanged.

Experimental controls skip their added operations when set to neutral values.
Frozen references check numerical and encoding behavior for the tested cases.

## Color handling

By default, image processing applies EXIF orientation and converts supported RGB
matrix/TRC ICC profiles to sRGB. Unsupported transforms fall back to Little CMS,
which can produce different numerical results. This fallback is recorded.
Reference JPEG export uses quality 100, 4:2:0 chroma subsampling, and no embedded
profile. In-memory processing retains color interpretation metadata for Pillow
inputs, while NumPy callers must supply already interpreted RGB pixels.

## Scope

The image workflow supports opaque 8-bit SDR JPEG and PNG. HEIC, high-bit-depth
images, and arbitrary ICC LUT transforms are outside its supported scope.
Straight RGBA is available through explicit configuration or the `core` profile,
as an experimental extension.

The [video adapter](video.md) prepares inputs through FFmpeg and produces silent
SDR H.264 output at a constant frame rate. It filters each frame independently,
with source cleanup retaining only mask history. Audio preservation and temporal
stabilization are not implemented.

## References

- Raanan Fattal, Dani Lischinski, and Michael Werman (2002).
  [Gradient Domain High Dynamic Range Compression](https://doi.org/10.1145/566570.566573).
  *SIGGRAPH 2002*, pp. 249–256.
