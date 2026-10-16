# Source cleanup investigation

Investigated on 2026-10-16. This prototype is separate from the runtime and does
not add a public CLI setting. Its purpose is to test stronger correction of
broad shading artifacts while retaining fine source variation. It generates no
new grain. The existing `--flat-cleanup 0.75` remains the comparison baseline.

## Method

`cleanup_multiscale.py` wraps `SourceCleanup(1)` and reuses its continuous mask.
Let `D` be its full masked correction, `m` its mask, and `G` a Gaussian filter:

```text
broad = m * G(D) / max(G(m), epsilon)
correction = fine_strength * D + (coarse_strength - fine_strength) * broad
```

The experiment uses `coarse_strength=1`. Candidate A uses fine strength 0.75
and Gaussian sigma 1.2 working pixels. Candidate B uses fine strength 0.5 and
the same sigma. Additional strengths and radii provide comparisons.

RGB and luminance remain floating point. Per-pixel corrections stay bounded,
and pixels with a zero cleanup mask retain their exact input. Gaussian filtering
acts on whole-frame fields, without independently filtered tiles. The original
reduced-resolution mask still has its existing limitations. Its history is
reused without temporal averaging of RGB frames.

This is a separation by scale, not a classifier for compression errors. The
bands overlap. Quantized contours have fine edges that can return with the
retained residual. Fattal's nonlinear response also means preserving a source
frequency component does not preserve its exact output appearance.

Related background: [Guided Image Filtering](https://people.csail.mit.edu/kaiming/publications/pami12guidedfilter.pdf)
and [Edge-Preserving Decomposition for Multi-Scale Tone and Detail Manipulation](https://www.microsoft.com/en-us/research/video/edge-preserving-decomposition-for-multi-scale-tone-and-detail-manipulation/).
These describe general filtering and decomposition ideas. They do not validate
this particular correction or its use with this filter.

## Findings

The local portrait frame near 19 seconds shows a useful but incomplete tradeoff.
Candidate A weakens some coherent shading patterns relative to cleanup 0.75,
with changes in grain. Candidate B retains more fine variation but visibly
restores contour fragments on the door. Neither demonstrates reliable
separation of desirable grit from compression damage. Uniform strengths 0.85,
0.9 and 1 are useful controls when judging whether a difference is worthwhile.

Controlled 512x256 inputs contain a smooth ramp, identical fine noise in the
clean and damaged versions, and either three-level shading steps or 16-pixel
block offsets. A separate clean case has a genuine faint edge. The seed is
fixed. These are synthetic shading errors, not actual codec encodes.

| Source measurement | Current 0.75 | Candidate A | Candidate B |
| --- | ---: | ---: | ---: |
| Step case, broad error RMS | 0.696 | 0.679 | 0.679 |
| Block case, broad error RMS | 0.544 | 0.521 | 0.522 |
| Block case, fine signal gain | 0.239 | 0.230 | 0.435 |
| Real faint edge case, local error RMS | 0.731 | 0.798 | 0.635 |

Errors are measured in 0-255 luminance units. Fine signal gain is the projection
of the restored high-pass source onto the known clean high-pass source. It is
not a percentage of perceptually preserved grit. The numbers show a tradeoff,
with only a modest reduction in broad source error. Against the filtered clean
reference, Candidate A slightly worsens mean error on the step case and barely
changes it on the block case. Source statistics alone cannot select a winner.

Short video trials used 48 frames from the portrait at 18 seconds (30 fps) and
48 frames from Tears of Steel at 8 seconds (24 fps). Candidate A changed the
mean adjacent-frame RGB difference by about -1.0% and -3.2%, respectively.
Candidate B increased it by about 2.2% and 2.3%. This measurement includes
motion and scene content and does not establish perceptual flicker. The existing
mask reset 23 times in the portrait excerpt and zero times in the other clip.
The prototype adds no temporal state beyond that existing mask history.

An isolated, warm, alternating 12-trial benchmark on the 1536x864 portrait
measured median cleanup-only times of approximately 59 ms for current 0.75,
282 ms for Candidate A and 277 ms for Candidate B. The prototype allocates and
filters multiple full-frame arrays. It requires optimization before adoption.
Concurrent video runs are unsuitable for a precise end-to-end speed claim.

Recommendation: retain current cleanup 0.75 as the production baseline.
Candidate A is an optional research comparison. The present decomposition is
not a demonstrated replacement, and restoring more fine residual can undo the
desired contour suppression. Longer motion review, stronger edge checks and
performance work would be needed before integration.

## Reproduction

Run from the repository root with its installed environment. Explicit inputs
are local and unversioned. Use a new output directory for every command.

```bash
uv run python experiments/investigate_cleanup.py synthetic \
  --config configs/fattal-1-compressed.yaml \
  --output outputs/cleanup-synthetic-new

uv run python experiments/investigate_cleanup.py still \
  --input local_inputs/frame.png \
  --config configs/fattal-1-compressed.yaml \
  --cases current075 current085 split075-s12 split050-s12 \
  --output outputs/cleanup-still-new

uv run python experiments/investigate_cleanup.py video \
  --input local_inputs/clip.mp4 --start 0 --frames 48 \
  --config configs/fattal-1-compressed.yaml \
  --cases current075 split075-s12 split050-s12 \
  --output outputs/cleanup-motion-new

uv run python experiments/investigate_cleanup.py benchmark \
  --input local_inputs/frame.png \
  --config configs/fattal-1-compressed.yaml \
  --cases current075 split075-s12 split050-s12 \
  --output outputs/cleanup-benchmark-new

uv run python -m pytest -q experiments/test_cleanup_multiscale.py tests/test_cleanup.py
```

The runner records settings, source fingerprints, timings and input hashes.
PNG comparisons are saved before video encoding. Clips use working resolution
without final enlargement and start with fresh decoder and mask state. Video
mode requires FFmpeg and ffprobe and assumes a CFR source. It does not perform
the production adapter's full-stream timing validation.

Local investigation artifacts are under ignored `outputs/cleanup-multiscale/`.
The `portrait` folder contains still sweeps, `synthetic` contains controlled
comparisons, and the `*-motion` and `*-selected` folders contain short videos
and sampled lossless frames. Candidate A is `split075-s12` and Candidate B is
`split050-s12`. The conservative comparison videos place current cleanup on
the left and Candidate A on the right. They are resized and re-encoded review
previews. Use the sampled PNGs for exact frame inspection.

Verification: 15 experiment tests and 33 existing cleanup tests passed. Checks
cover bounds, finite output, zero-mask identity, equal-strength agreement with
existing cleanup, constant images, odd sizes and repeated-frame stability.
Each generated comparison clip contains the expected 48 frames. Fattal 2 and
long sequences have not been visually evaluated in this investigation.
