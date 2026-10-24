# Experiments

Source treatment studies and video throughput benchmarks.
The [experiment notes](../docs/experiments.md) document methods, findings,
and reproduction commands. Runtime options are in the
[video guide](../docs/video.md).

- `investigate_cleanup.py`: synthetic, still, video, and timing comparisons.
- `investigate_prenoise.py`: source-noise comparisons using the runtime implementation.
- `benchmark_video_throughput.py`: complete video timings and filtered RGB hash comparisons.

Keep inputs in ignored `local_inputs/` and generated artifacts in ignored `outputs/`.
