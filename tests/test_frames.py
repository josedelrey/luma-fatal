"""The decoded-frame adapter is lazy, stateless and shares image processing."""

from dataclasses import replace

import numpy as np
import pytest

from luma_fatal import (
    Config,
    FilterParams,
    NumericsConfig,
    ProcessingConfig,
    apply_filter,
    process_frames,
)


@pytest.mark.parametrize("filter_name", ["fattal1", "fattal2"])
def test_frame_stream_is_lazy_ordered_and_matches_image_api(filter_name):
    config = Config(
        filter=filter_name,
        parameters=FilterParams(fade=0),
        processing=ProcessingConfig(max_edge=24),
        numerics=NumericsConfig(iterations=11),
    )
    frames = [
        np.random.default_rng(seed).integers(0, 256, (35, 47, 3), np.uint8)
        for seed in (3, 4, 3)
    ]
    snapshots = [frame.copy() for frame in frames]
    consumed = []

    def source():
        for index, frame in enumerate(frames):
            consumed.append(index)
            yield frame

    stream = process_frames(source(), config=config)
    assert consumed == []
    first = next(stream)
    assert consumed == [0]
    remaining = list(stream)
    assert consumed == [0, 1, 2]
    for original, snapshot, actual in zip(frames, snapshots, [first, *remaining]):
        np.testing.assert_array_equal(actual, apply_filter(snapshot, config=config))
        np.testing.assert_array_equal(original, snapshot)
    np.testing.assert_array_equal(first, remaining[-1])


def test_interleaved_streams_keep_independent_configurations():
    frame = np.random.default_rng(5).integers(0, 256, (35, 47, 3), np.uint8)
    config = Config(parameters=FilterParams(fade=0))
    identity = replace(config, parameters=FilterParams(fade=100))
    effect_stream = process_frames(iter([frame, frame]), config=config)
    identity_stream = process_frames(iter([frame, frame]), config=identity)
    first = next(effect_stream)
    np.testing.assert_array_equal(next(identity_stream), frame)
    np.testing.assert_array_equal(next(effect_stream), first)
    np.testing.assert_array_equal(next(identity_stream), frame)


@pytest.mark.parametrize(
    "frame", [np.zeros((3, 4, 4), np.uint8), np.zeros((3, 4, 3), float), [[1, 2, 3]]]
)
def test_frames_require_decoded_rgb8_arrays(frame):
    with pytest.raises(ValueError):
        list(process_frames([frame], config=Config()))


def test_frame_stream_rejects_ambiguous_filter_before_consuming_inputs():
    def source():
        raise AssertionError("Invalid configuration must fail before iteration")
        yield

    with pytest.raises(ValueError, match="one filter"):
        list(process_frames(source(), config=Config(filter="both")))
