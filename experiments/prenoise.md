# Noise before the filter

This opt-in experiment tests whether small source perturbations make amplified
block and shading boundaries less conspicuous. It does not modify production
presets, configuration, or the video CLI. The existing experiment runner provides
still, synthetic, short video and timing comparisons.

## Method

Noise enters floating RGB and luminance at working resolution immediately before
the Fattal core's logarithm. Adding it after resizing avoids losing it in the
RGB8 resize path. Original pixels still control Fade and shadow protection.

- `highpass` subtracts a Gaussian with sigma 0.8 pixels from white noise. This
  suppresses broad variation. It is not an optimized blue-noise distribution.
- `fbm` is an fBm-like sum of independently normalized Gaussian noise octaves
  with sigmas 4, 2, 1 and 0.5 pixels and weights 1, 0.5, 0.25 and 0.125.
  It tests correlated noise with broader variation. It is not Perlin noise.
- RMS is measured in source 0-255 code values before masking and clipping. A
  value of 0.35 is a small fractional perturbation, not 35 percent strength.
- The original source supplies the existing continuous smooth-region mask.
  Noise never feeds back into mask detection. The same perturbation enters all
  three color channels and luminance. A common range limit avoids colored
  clipping, and perturbations are bounded to four code values times the mask.
- Optional standard cleanup runs before noise. The named cases explicitly
  replace `video.flat_cleanup` from the supplied configuration. Decoder debanding
  and deblocking still follow that configuration for video input.
- A seeded field is fixed in working-frame coordinates across frames. It is
  not motion compensated. The mask keeps its existing motion/cut history.
  Fixed noise can look attached to the screen, and changing masks or nonlinear
  filtering can still produce temporal variation.

The general octave construction is described in
[Physically Based Rendering, Noise](https://www.pbr-book.org/3ed-2018/Texture/Noise).
That reference does not validate this particular artifact treatment.

## Findings on 2026-10-17

On the portrait frame near 19 seconds, Fattal 1 with masked high-pass noise at
0.75 RMS makes some door contours less distinct, with conspicuous added fine
grain. Standard cleanup 0.75 followed by noise at 0.35 RMS gives a more restrained
candidate. The fBm-like variant creates new cloudy patterns that the filter
strongly amplifies. Fattal 2 changes much less visibly on this same frame.

Controlled synthetic inputs use the earlier runner's shared clean and damaged
grain, stepped ramps, 16-pixel block offsets and a genuine faint-edge control.
High-pass noise alone leaves broad source error essentially unchanged. For the
block case, broad RMS error is 0.623 with noise off and 0.623 at noise 0.75.
The fBm-like case raises it to 0.839. All tested noisy variants increase mean
absolute filtered error against the clean filtered reference. That metric also
penalizes intentionally added grain, so it is not a perceptual preference score.
Visible blocks persist in these synthetic comparisons.

The evidence supports a cosmetic boundary-disguising experiment, without a
demonstrated reconstruction of damaged shading. The initial local motion trial
contains only 48 portrait frames starting at 18 seconds. It cannot establish
long-sequence temporal quality. No production defaults are changed.

The cleanup-plus-noise case changes mean adjacent-frame RGB difference from
25.509 to 24.877, about -2.5 percent. Both masks reset 23 times. These figures
do not establish flicker suppression. Median per-frame processing time in this
multi-case run rises from 0.346 to 0.454 seconds, excluding each first frame.
This prototype includes full-resolution mask interpolation and noise arithmetic.
The measurements are local observations, not production throughput guarantees.
All four case videos and the labeled comparison contain the expected 48 frames.

## Reproduction

Use a new output directory each time. Still inputs must already be decoded sRGB
RGB8. Video uses the production decoder, but this experiment does not perform
the production adapter's full-stream preflight. Excerpts use fresh decoder and
mask state. All outputs use working resolution without final enlargement.

```bash
uv run python experiments/investigate_prenoise.py video \
  --input local_inputs/clip.mp4 --start 0 --frames 48 \
  --config configs/fattal-1-compressed.yaml \
  --cases current075 configured \
  --noise-rms 0.35 --noise-kind highpass --noise-cleanup 0.75 \
  --output outputs/prenoise-custom

uv run python experiments/investigate_prenoise.py synthetic \
  --config configs/fattal-1-compressed.yaml \
  --cases off current075 highpass035 highpass075 fbm075 clean075-highpass035 \
  --output outputs/prenoise-synthetic-new

uv run python -m pytest -q experiments/test_prenoise.py \
  tests/test_cleanup.py tests/test_multiscale_cleanup.py
```

Use `--noise-cleanup 0` to test noise alone. `--noise-unmasked` disables noise
masking, and `--noise-seed` selects a deterministic field. These options control
only the `configured` case. Named cases retain the values encoded in their
names, all masked except `unmasked075`. Reports save the configured settings,
input and source hashes, timings and sampled pre-encoding PNGs. Adjacent-frame
differences include motion and are not flicker scores.

Local outputs are under ignored `outputs/prenoise/`: `portrait-still`,
`portrait-fattal2`, `synthetic` and `portrait-motion`. The experiment and its
tests are excluded from release distributions. The 26 noise tests and 60
existing cleanup tests pass, covering range bounds, mask protection, unchanged
zero-noise behavior, stable repeated frames, noise spectra, Fade and shadows.
