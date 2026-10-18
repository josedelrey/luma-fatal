# Source treatment experiments

[Overview](../README.md) · [Video controls](video.md) · [Experiment scripts](../experiments/)

After [installation](../README.md#install), run commands from the repository root
and activate the virtual environment once per shell session:

```bash
source .venv/bin/activate
```

These studies test whether preprocessing reduces compression artifacts amplified
by the Fattal filters. They are optional visual treatments. Findings below are
local observations from 2026-10-16 and 2026-10-17, with limited motion validation.

## Multiscale cleanup

The prototype separates the standard masked correction into broad and residual
components. With correction `D`, mask `m`, and Gaussian filter `G`:

```text
broad = m * G(D) / max(G(m), epsilon)
correction = fine_strength * D + (coarse_strength - fine_strength) * broad
```

Candidate A uses fine strength 0.75, coarse strength 1, and sigma 1.2 working
pixels. Candidate B uses fine strength 0.5 with the same coarse settings. RGB and
luminance remain floating point, corrections are bounded, and zero-mask pixels
retain their input. The existing mask history is reused without RGB frame averaging.

Controlled 512×256 inputs combine smooth ramps, shared fine noise, and either
three-level shading steps or 16-pixel block offsets. A faint-edge control tests
preservation of real structure. These are synthetic errors rather than codec encodes.

| Source measurement | Standard 0.75 | Candidate A | Candidate B |
| --- | ---: | ---: | ---: |
| Step case, broad error RMS | 0.696 | 0.679 | 0.679 |
| Block case, broad error RMS | 0.544 | 0.521 | 0.522 |
| Block case, fine signal gain | 0.239 | 0.230 | 0.435 |
| Faint-edge case, local error RMS | 0.731 | 0.798 | 0.635 |

Errors use 0–255 luminance units. Fine signal gain is a projection onto the known
clean high-pass source, not a perceptual quality score. Candidate A modestly
reduces broad source error but slightly worsens filtered error in the step case.
Candidate B retains more variation and restores unwanted contour fragments.
The scale components overlap, so the method cannot reliably separate texture
from compression damage.

Two 48-frame excerpts used a portrait clip at 30 fps and Tears of Steel at 24 fps.
Candidate A changed mean adjacent-frame RGB difference by about −1.0% and −3.2%.
Candidate B changed it by +2.2% and +2.3%. These measurements include motion and
do not establish perceptual flicker. The mask reset 23 times in the portrait and
zero times in the other excerpt. Fattal 2 and long sequences were not visually
evaluated in this investigation.

A warm, alternating 12-trial benchmark on one 1536×864 portrait measured median
cleanup times of 59 ms for standard 0.75, 282 ms for A, and 277 ms for B.
These are local cleanup-only timings. The multiscale presets expose Candidate A.

## Pre-filter noise

Noise enters floating RGB and luminance immediately before the core logarithm,
after optional cleanup. The original source supplies the smooth-region mask and
continues to control Fade and shadow protection.

- `highpass` subtracts a sigma-0.8 Gaussian from white noise.
- `fbm` combines normalized Gaussian octaves with sigmas 4, 2, 1, and 0.5 and
  weights 1, 0.5, 0.25, and 0.125.
- RMS uses input 0–255 units before masking and clipping. A common perturbation
  enters all channels, with range protection and a bound of four code values times the mask.
- The seeded field stays fixed in working-frame coordinates. It is not motion
  compensated, and changing masks can still produce temporal variation.

On the portrait frame, high-pass noise at 0.75 RMS made some contours less distinct
but added conspicuous grain. Cleanup 0.75 with noise 0.35 was more restrained.
fBm-like noise produced amplified cloudy patterns. Fattal 2 changed less visibly.

In the synthetic block case, broad source RMS error was 0.623 with noise off and
0.623 with high-pass noise 0.75. fBm-like noise raised it to 0.839. All tested
noisy variants increased absolute filtered error against the clean reference,
which also penalizes intentionally added grain. Blocks remained visible.

One 48-frame portrait trial changed mean adjacent-frame RGB difference from
25.509 to 24.877 with cleanup plus noise, with 23 mask resets in both cases.
Median processing time, excluding each first frame, rose from 0.346 to 0.454
seconds. This short multi-case trial demonstrates neither reconstruction of
damaged shading nor reliable temporal improvement.

## Reproduction

Run from the installed checkout with a new output directory each time.
Synthetic comparisons need no external media:

```bash
python experiments/investigate_cleanup.py synthetic \
  --config configs/fattal-1-compressed.yaml --output outputs/cleanup-synthetic-new

python experiments/investigate_prenoise.py synthetic \
  --config configs/fattal-1-compressed.yaml \
  --cases off current075 highpass035 highpass075 fbm075 clean075-highpass035 \
  --output outputs/prenoise-synthetic-new
```

For stills, use decoded sRGB RGB8 input and the `still` subcommand. Cleanup also
provides a `benchmark` subcommand. To compare motion with your own clip:

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
noise case. Named cases retain their encoded settings. Reports record settings,
input/source hashes, timings, and sampled lossless frames. Review PNGs before
video encoding for exact frame comparisons.

Excerpts start with fresh decoder and mask state, use working resolution without
final enlargement, and omit the production adapter's full-stream preflight.
Video mode requires FFmpeg and ffprobe. Experiment scripts are available in the
Git checkout and excluded from release distributions.

## Background

- [Guided Image Filtering](https://people.csail.mit.edu/kaiming/publications/pami12guidedfilter.pdf).
- [Edge-Preserving Decomposition for Multi-Scale Tone and Detail Manipulation](https://www.microsoft.com/en-us/research/video/edge-preserving-decomposition-for-multi-scale-tone-and-detail-manipulation/).
- [Physically Based Rendering: Noise](https://www.pbr-book.org/3ed-2018/Texture/Noise).

These references describe the underlying filtering and noise constructions.
They do not validate these particular artifact treatments.
