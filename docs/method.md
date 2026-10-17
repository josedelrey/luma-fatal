# Method and scope

[Overview](../README.md) · [Controls](configuration.md) · [Verification](development.md)

## Gradient-domain stylization

The filter builds a pyramid of log luminance, attenuates its gradients, and
reconstructs a luminance field with a Poisson solver. Color recombination uses
the source RGB and reconstructed brightness. Fattal 1 accumulates attenuation
across scales. Fattal 2 replaces the attenuation field at each finer scale.
These are project variants of the approach described by Fattal, Lischinski,
and Werman in
[*Gradient Domain High Dynamic Range Compression*](https://doi.org/10.1145/566570.566573),
SIGGRAPH 2002. The implementation operates on 8-bit SDR inputs.

## Processing order

1. Decode, orient, and interpret color. Apply the configured input-area budget.
2. Reduce to working resolution. Video can apply source cleanup and noise here.
3. Attenuate log-luminance gradients and reconstruct brightness.
4. Apply tone strength, recombine color, compress highlights, and apply output gamma.
5. Apply vibrance, shadow protection, output levels, and RGB8 quantization.
6. Enlarge to the prepared input dimensions, apply Fade, and encode.

Tone strength blends source and mapped luminance before color reconstruction.
Its source brightness precedes cleanup and noise. Highlight compression scales
channels together above a knee at 75 percent of white. Full strength bounds finite
nonnegative RGB peaks, while partial strength may still clip bright values.

Gamma applies `x^(1/gamma)` to clipped normalized RGB. Vibrance increases channel
differences around weighted brightness with requested gain
`1 + (vibrance / 100) * (1 - saturation)^2`, where saturation is `(max - min) / max`.
The gain is limited to keep channels in gamut. Grays and fully saturated colors
are unchanged at this stage.

Shadow protection uses original source brightness. Positive `shadow_softness`
blends through a smoothstep interval around `shadow_threshold`. Zero retains the
hard cutoff. Output levels remap channels after shadow protection and also affect
protected shadows. Fade blends original RGB afterward. At fade 100, the effect
and its output controls are bypassed.

## Numerical choices

The reference path preserves float32 operation order, scalar power approximations,
and fixed boundary behavior. `power_mode: standard` uses NumPy float64 powers,
narrowing attenuation to float32 while retaining float64 color recombination.

`detail_level` omits the finest attenuation scales. Retained fields interpolate
through excluded levels. Omitting all levels yields unit attenuation, with
reconstruction and normalization still applied. `pyramid_min_size` controls the
coarsest scale. Fractional luminance omits integer rounding before the logarithm
and cannot recover detail lost in an 8-bit source.

Working-resolution bilinear and bicubic sampling have no antialias prefilter
and truncate to RGB8. Bicubic uses the scalar kernel with `a = -0.75` and clamps
overshoot. Input-area fitting and enlargement keep their existing methods.

Neutral experimental settings skip their added operations. Frozen references
check the tested numerical and encoding behavior. Equality across arbitrary
platforms, versions, and inputs remains unproven.

## Color handling

The default image route applies EXIF orientation and converts supported RGB
matrix/TRC ICC profiles to sRGB. Unsupported transforms use a recorded Little CMS
fallback, whose numerical results can differ. Reference JPEG export uses quality
100, 4:2:0 chroma subsampling, and no embedded profile. Pillow inputs retain color
interpretation metadata through the in-memory route. NumPy callers supply already
interpreted RGB pixels.

## Scope

The validated image workflow covers opaque 8-bit SDR JPEG and PNG. HEIC,
high-bit-depth images, transparency, and arbitrary ICC LUT transforms are outside
that scope. Straight RGBA is available through explicit configuration or the
`core` profile, with limited compatibility coverage.

The [video adapter](video.md) supports tagged BT.709 CFR input and silent H.264
output. Filtering is independent per frame. Source cleanup retains only mask
history. Audio preservation and temporal stabilization are not implemented.
