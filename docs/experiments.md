# Source treatment experiments

[Overview](../README.md) · [Video controls](video.md) · [Experiment scripts](../experiments/)

These studies examine whether preprocessing can reduce compression artifacts
that the Fattal filters amplify. The treatments are optional. The findings below
come from local experiments on 2026-10-16 and 2026-10-17, with limited validation
on moving footage.

## Multiscale cleanup

The prototype separates the standard masked correction into broad and residual
components. For correction `D`, mask `m`, and Gaussian filter `G`, it computes:

```text
broad = m * G(D) / max(G(m), epsilon)
correction = fine_strength * D + (coarse_strength - fine_strength) * broad
```

Candidate A uses fine strength 0.75, coarse strength 1, and sigma 1.2 working
pixels. Candidate B reduces fine strength to 0.5 while keeping the same coarse
settings. Both retain floating-point RGB and luminance, bound corrections, and
leave pixels unchanged where the mask is zero. They reuse the existing mask
history without averaging RGB frames.

The controlled 512×256 inputs combine smooth ramps and shared fine noise with
either three-level shading steps or 16-pixel block offsets. A faint-edge control
tests whether real structure is preserved. The errors are synthetic and do not
come from codec encoding.

| Source measurement | Standard 0.75 | Candidate A | Candidate B |
| --- | ---: | ---: | ---: |
| Step case, broad error RMS | 0.696 | 0.679 | 0.679 |
| Block case, broad error RMS | 0.544 | 0.521 | 0.522 |
| Block case, fine signal gain | 0.239 | 0.230 | 0.435 |
| Faint-edge case, local error RMS | 0.731 | 0.798 | 0.635 |

Errors are measured in 0–255 luminance units. Fine signal gain is a projection
onto the known clean high-pass source and does not measure perceptual quality.
Candidate A modestly reduces broad source error, although it slightly increases
filtered error in the step case. Candidate B retains more variation but also
restores unwanted contour fragments. Because the scale components overlap, the
method cannot reliably separate texture from compression damage.

Motion checks used two 48-frame excerpts: a portrait clip at 30 fps and Tears of
Steel at 24 fps. Candidate A changed the mean RGB difference between adjacent
frames by about −1.0% and −3.2%, respectively. Candidate B changed it by +2.2% and
+2.3%. These measurements include motion, so they cannot establish perceptual
flicker. The mask reset 23 times in the portrait and never in the other excerpt.
This investigation did not visually evaluate Fattal 2 or long sequences.

An alternating 12-trial benchmark, run after warm-up on one 1536×864 portrait,
measured median cleanup times of 59 ms for standard cleanup at 0.75, 282 ms for A,
and 277 ms for B. These local timings cover cleanup alone. The multiscale presets
use Candidate A.

## Pre-filter noise

Noise is added to floating-point RGB and luminance immediately before the core
logarithm, after optional cleanup. The original source supplies the smooth-region
mask and continues to determine Fade and shadow protection.

- `highpass` subtracts a sigma-0.8 Gaussian from white noise.
- `fbm` combines normalized Gaussian octaves with sigmas 4, 2, 1, and 0.5 and
  weights 1, 0.5, 0.25, and 0.125.
- RMS is measured in input 0–255 units before masking and clipping. The same
  perturbation is added to all channels, with range protection and a bound of
  four code values times the mask.
- The seeded field stays fixed in working-frame coordinates without motion
  compensation. Changes in the mask can still produce temporal variation.

On the portrait frame, high-pass noise at 0.75 RMS made some contours less distinct
but added conspicuous grain. Combining cleanup at 0.75 with noise at 0.35 produced
a more restrained result. The filter amplified fBm-like noise into cloudy
patterns. Changes were less visible with Fattal 2.

In the synthetic block case, broad source RMS error remained at 0.623 with
high-pass noise at 0.75, matching the result without noise. fBm-like noise raised
it to 0.839. All tested noise variants increased absolute filtered error against
the clean reference, although this measure also penalizes intentionally added
grain. Blocks remained visible.

In one 48-frame portrait trial, adding cleanup and noise changed the mean RGB
difference between adjacent frames from 25.509 to 24.877, with 23 mask resets in
both cases. Median processing time rose from 0.346 to 0.454 seconds, excluding
the first frame of each case. This short comparison does not establish
reconstruction of damaged shading or reliable temporal improvement.

## Reproduction

Run these commands from the installed checkout, using a new output directory
each time. Synthetic comparisons require no external media:

```bash
python experiments/investigate_cleanup.py synthetic \
  --config configs/fattal-1-compressed.yaml --output outputs/cleanup-synthetic-new

python experiments/investigate_prenoise.py synthetic \
  --config configs/fattal-1-compressed.yaml \
  --cases off current075 highpass035 highpass075 fbm075 clean075-highpass035 \
  --output outputs/prenoise-synthetic-new
```

For still images, use decoded sRGB RGB8 input with the `still` subcommand.
The cleanup script also provides a `benchmark` subcommand. To compare treatments
on your own video clip, run:

```bash
python experiments/investigate_cleanup.py video \
  --input local_inputs/clip.mp4 --start 0 --frames 48 \
  --config configs/fattal-1-compressed.yaml \
  --cases current075 split075-s12 split050-s12 --output outputs/cleanup-motion-new

python experiments/investigate_prenoise.py video \
  --input local_inputs/clip.mp4 --start 0 --frames 48 \
  --config configs/fattal-1-compressed.yaml --cases current075 configured \
  --noise-rms 0.35 --noise-kind highpass --noise-cleanup 0.75 \
  --output outputs/prenoise-motion-new
```

`--noise-cleanup`, `--noise-unmasked`, and `--noise-seed` affect the `configured`
noise case. Named cases retain their predefined settings. Reports include
settings, input/source hashes, timings, and sampled lossless frames. For exact
frame comparisons, review the PNGs before video encoding.

Each excerpt starts with fresh decoder and mask state and runs at working
resolution, without final enlargement or the production adapter's full-stream
preflight. Video mode requires FFmpeg and ffprobe. The experiment scripts are
available in the Git checkout but are excluded from release distributions.

## Background

- [Guided Image Filtering](https://people.csail.mit.edu/kaiming/publications/pami12guidedfilter.pdf).
- [Edge-Preserving Decomposition for Multi-Scale Tone and Detail Manipulation](https://www.microsoft.com/en-us/research/video/edge-preserving-decomposition-for-multi-scale-tone-and-detail-manipulation/).
- [Physically Based Rendering: Noise](https://www.pbr-book.org/3ed-2018/Texture/Noise).

These references describe the filtering and noise constructions used here.
They do not validate the particular artifact treatments tested in these studies.
