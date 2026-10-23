"""GPU agreement and automatic selection use controlled fields and clocks."""

import os

import numpy as np
import pytest

from luma_fatal import Config, FilterParams, NumericsConfig, ProcessingConfig, process_frames
from luma_fatal import core, gpu
from luma_fatal.cleanup import SourceCleanup
from luma_fatal.denoise import SourceDenoise
from luma_fatal.prenoise import SourceNoise


@pytest.fixture(scope="module")
def cuda_solver():
    try:
        return gpu.CudaSolver()
    except gpu.GPUUnavailable as error:
        if os.environ.get("LUMA_FATAL_REQUIRE_GPU_TESTS") == "1":
            pytest.fail(str(error))
        pytest.skip(str(error))


@pytest.fixture(scope="module")
def cuda_composer(cuda_solver):
    return gpu.CudaComposer()


@pytest.mark.parametrize("shape,factor", [((1, 1), 1), ((17, 23), 1), ((37, 53), 3), ((385, 769), 3)])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("strength", [0.0, 0.25, 1.0])
def test_cuda_cleanup_composition_matches_cpu(cuda_composer, shape, factor, dtype, strength):
    from luma_fatal.cleanup import _compose

    rng = np.random.default_rng(21)
    rgb = rng.integers(0, 256, (*shape, 3), np.uint8).astype(dtype)
    if dtype == np.float32:
        rgb = np.clip(rgb + rng.uniform(-0.3, 0.3, rgb.shape).astype(np.float32), 0, 255)
    reference = rng.uniform(0, 1, shape).astype(np.float32)
    fields = rng.normal(0, 0.1, ((*np.ceil(np.array(shape) / factor).astype(int), 7))).astype(np.float32)
    fields[..., 0] = rng.uniform(-0.1, 1.1, fields.shape[:2])
    original = rgb.copy(), reference.copy(), fields.copy()
    expected = _compose(rgb, reference, fields, factor, strength)
    actual = cuda_composer(rgb, reference, fields, factor, strength)
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=2e-5)
    for a, b in zip((rgb, reference, fields), original):
        np.testing.assert_array_equal(a, b)
    assert np.max(np.abs(actual[0] - rgb)) <= 4.00002


def test_cuda_cleanup_device_arrays_avoid_host_round_trip(cuda_composer):
    from luma_fatal.cleanup import _compose

    cp = cuda_composer.cp
    rgb = np.random.default_rng(17).integers(0, 256, (37, 53, 3), np.uint8)
    reference = rgb[..., 0].astype(np.float32) / np.float32(255)
    fields = np.ones((13, 18, 7), np.float32)
    fields[..., 0] = 0
    expected = _compose(rgb, reference, fields, 3, 1.0)
    with cuda_composer.stream:
        result = cuda_composer.compose_device(cp.asarray(rgb), cp.asarray(reference), fields, 3, 1.0)
        assert all(isinstance(value, cp.ndarray) for value in result)
        actual = [value.get(stream=cuda_composer.stream) for value in result]
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("shape", [(2, 2), (3, 2), (2, 9), (17, 24), (24, 17), (257, 259)])
@pytest.mark.parametrize("iterations,relaxation", [(1, 0.5), (17, 1.0), (80, 1.9)])
def test_cuda_solver_preserves_sweeps_and_boundary_conditions(cuda_solver, shape, iterations, relaxation):
    rhs = np.random.default_rng(7).normal(0, 0.1, shape).astype(np.float32)
    rhs[0] = 0
    original = rhs.copy()
    expected = core._sor(rhs, iterations, relaxation)
    actual = cuda_solver(rhs, iterations, relaxation)
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    np.testing.assert_array_equal(actual[0], 0)
    np.testing.assert_array_equal(rhs, original)
    np.testing.assert_array_equal(cuda_solver(rhs, iterations, relaxation), actual)


@pytest.mark.parametrize("variant", [1, 2])
@pytest.mark.parametrize("resize", [False, True])
@pytest.mark.parametrize("treatment", ["none", "cleanup", "noise", "denoise"])
def test_cuda_frames_remain_close_with_effect_controls_and_source_treatments(
    cuda_solver, variant, resize, treatment,
):
    yy, xx = np.indices((137, 193))
    rgb = np.stack(((xx * 3) % 256, (yy * 4) % 256, (xx + yy) % 256), axis=-1).astype(np.uint8)
    rgb[20:60, 80:110] = 2
    original = rgb.copy()
    config = Config(
        filter=f"fattal{variant}",
        processing=ProcessingConfig(resize=resize, max_edge=128 if resize else None),
        parameters=FilterParams(
            fade=23, output_gamma=1.2, tone_strength=70, highlight_compression=30,
            black_level=3, white_level=97, vibrance=25,
        ),
        numerics=NumericsConfig(
            shadow_softness=6, luminance_mode="fractional", power_mode="standard",
        ),
    )

    def callback():
        if treatment == "cleanup":
            return SourceCleanup(0.75)
        if treatment == "noise":
            return SourceNoise(rms=0.35, seed=9)
        if treatment == "denoise":
            return SourceDenoise(0.5)
        return None

    expected = list(process_frames([rgb], config=config, source_cleanup=callback()))[0]
    actual = list(process_frames([rgb], config=config, source_cleanup=callback(), solver=cuda_solver))[0]
    error = np.abs(expected.astype(np.int16) - actual.astype(np.int16))
    assert error.mean() <= 0.1
    assert np.quantile(error, 0.99) <= 1
    assert error.max() <= 3
    assert actual.shape == rgb.shape
    assert actual.dtype == np.uint8
    np.testing.assert_array_equal(rgb, original)


@pytest.mark.parametrize("resize,fade", [(False, 0), (True, 0), (True, 37), (True, 100)])
def test_solver_callback_is_forwarded_without_changing_cpu_frames(resize, fade):
    rgb = np.random.default_rng(13).integers(0, 256, (81, 97, 3), np.uint8)
    config = Config(
        parameters=FilterParams(fade=fade),
        processing=ProcessingConfig(resize=resize, max_edge=48 if resize else None),
    )
    expected = list(process_frames([rgb], config=config))[0]
    calls = []

    def solver(rhs, iterations, relaxation):
        calls.append((rhs.shape, iterations, relaxation))
        return core._sor(rhs, iterations, relaxation)

    actual = list(process_frames([rgb], config=config, solver=solver))[0]
    np.testing.assert_array_equal(actual, expected)
    assert len(calls) == (0 if fade == 100 else 1)
    if calls:
        assert calls[0][0] == ((40, 48) if resize else (81, 97))


def fake_hardware(monkeypatch, *, cuda_time=0.01, different=False, fail_after=None):
    clock = [0.0]
    instances = []

    def cpu(rhs, iterations=80, relaxation=1.9):
        clock[0] += 0.1
        return rhs.copy()

    class FakeCuda:
        name, version = "controlled GPU", "test"

        def __init__(self):
            self.calls = 0
            self.seconds = cuda_time
            instances.append(self)

        def __call__(self, rhs, iterations=80, relaxation=1.9):
            self.calls += 1
            if fail_after is not None and self.calls > fail_after:
                raise gpu.GPUUnavailable("controlled device failure")
            clock[0] += self.seconds
            return rhs + (1 if different else 0)

    monkeypatch.setattr(core, "_sor", cpu)
    monkeypatch.setattr(gpu, "CudaSolver", FakeCuda)
    monkeypatch.setattr(gpu.time, "perf_counter", lambda: clock[0])
    return instances


@pytest.mark.parametrize("cuda_time,expected", [(0.01, "cuda"), (0.09, "cpu"), (0.2, "cpu")])
def test_auto_selects_only_a_faster_gpu(monkeypatch, cuda_time, expected):
    instances = fake_hardware(monkeypatch, cuda_time=cuda_time)
    rhs = np.ones((256, 256), np.float32)
    solver = gpu.AdaptiveSolver(expected_frames=64)
    for _ in range(3):
        np.testing.assert_array_equal(solver(rhs), rhs)
    assert solver.record["backend"] == expected
    assert solver.record[f"{expected}_frames"] == 3
    assert len(instances) == 1
    assert solver.record["calibration_seconds"] > 0


def test_auto_rejects_a_visually_different_solver(monkeypatch):
    fake_hardware(monkeypatch, different=True)
    rhs = np.ones((256, 256), np.float32)
    solver = gpu.AdaptiveSolver(expected_frames=64)
    np.testing.assert_array_equal(solver(rhs), rhs)
    assert solver.record["backend"] == "cpu"
    assert "tolerance" in solver.record["reason"]


@pytest.mark.parametrize("frames,shape", [(2, (256, 256)), (64, (16, 16))])
def test_auto_skips_gpu_setup_for_short_or_small_workloads(monkeypatch, frames, shape):
    instances = fake_hardware(monkeypatch)
    rhs = np.ones(shape, np.float32)
    solver = gpu.AdaptiveSolver(expected_frames=frames)
    np.testing.assert_array_equal(solver(rhs), rhs)
    assert solver.record["backend"] == "cpu"
    assert solver.record["calibration_seconds"] == 0
    assert not instances


def test_auto_accounts_for_setup_before_selecting_gpu(monkeypatch):
    instances = fake_hardware(monkeypatch)
    rhs = np.ones((256, 256), np.float32)
    solver = gpu.AdaptiveSolver(expected_frames=8)
    # Additional first-use CUDA setup must be recovered by the remaining frames.
    original = gpu.CudaSolver

    def expensive_setup():
        start = gpu.time.perf_counter()
        instance = original()
        original_cpu = core._sor
        for _ in range(20):
            original_cpu(rhs)
        assert gpu.time.perf_counter() > start
        return instance

    monkeypatch.setattr(gpu, "CudaSolver", expensive_setup)
    np.testing.assert_array_equal(solver(rhs), rhs)
    assert instances
    assert solver.record["backend"] == "cpu"
    assert "setup cost" in solver.record["reason"]


def test_missing_cuda_dependencies_fall_back_once(monkeypatch, capsys):
    fake_hardware(monkeypatch)
    attempts = []

    def unavailable():
        attempts.append(1)
        raise gpu.GPUUnavailable("controlled missing dependency")

    monkeypatch.setattr(gpu, "CudaSolver", unavailable)
    rhs = np.ones((256, 256), np.float32)
    solver = gpu.AdaptiveSolver(expected_frames=64)
    for _ in range(2):
        np.testing.assert_array_equal(solver(rhs), rhs)
    assert attempts == [1]
    assert solver.record["cpu_frames"] == 2
    assert solver.record["backend"] == "cpu"
    assert "missing dependency" in solver.record["reason"]
    assert capsys.readouterr().err.count("Using CPU filtering") == 1


def test_cuda_failure_after_calibration_falls_back_for_remaining_frames(monkeypatch):
    instances = fake_hardware(monkeypatch, fail_after=3)
    rhs = np.ones((256, 256), np.float32)
    solver = gpu.AdaptiveSolver(expected_frames=64)
    solver(rhs)
    assert solver.record["backend"] == "cuda"
    for _ in range(2):
        np.testing.assert_array_equal(solver(rhs), rhs)
    assert instances[0].calls == 4
    assert solver.record["backend"] == "cpu"
    assert solver.record["cuda_frames"] == 1
    assert solver.record["cpu_frames"] == 2
    assert "device failure" in solver.record["reason"]


def test_sustained_gpu_slowdown_switches_back_to_cpu(monkeypatch):
    instances = fake_hardware(monkeypatch)
    rhs = np.ones((256, 256), np.float32)
    solver = gpu.AdaptiveSolver(expected_frames=64)
    solver(rhs)
    instances[0].seconds = 0.2
    for _ in range(4):
        np.testing.assert_array_equal(solver(rhs), rhs)
    assert solver.record["backend"] == "cpu"
    assert solver.record["cuda_frames"] == 4
    assert solver.record["cpu_frames"] == 1
    assert "became slower" in solver.record["reason"]


@pytest.fixture(scope="module")
def cuda_core(cuda_solver):
    from luma_fatal.gpu_core import CudaCore

    return CudaCore()


@pytest.mark.parametrize("variant", [1, 2])
@pytest.mark.parametrize("resize", [False, True])
@pytest.mark.parametrize("treatment", ["none", "cleanup", "noise", "denoise", "multiscale"])
def test_cuda_frame_pipeline_preserves_appearance_and_controls(cuda_core, variant, resize, treatment):
    from luma_fatal.cleanup import MultiscaleSourceCleanup

    yy, xx = np.indices((137, 193))
    rgb = np.stack(((xx * 3) % 256, (yy * 4) % 256, (xx + yy) % 256), axis=-1).astype(np.uint8)
    rgb[20:60, 80:110] = 2
    original = rgb.copy()
    config = Config(
        filter=f"fattal{variant}",
        processing=ProcessingConfig(resize=resize, max_edge=128 if resize else None),
        parameters=FilterParams(fade=23, output_gamma=1.2, tone_strength=70, highlight_compression=30,
                                black_level=3, white_level=97, vibrance=25),
        numerics=NumericsConfig(shadow_softness=6, pyramid_min_size=8),
    )

    def source():
        if treatment == "cleanup":
            return SourceCleanup(0.75)
        if treatment == "noise":
            return SourceNoise(rms=0.5, cleanup=0.75)
        if treatment == "denoise":
            return SourceDenoise(0.5, source_cleanup=SourceCleanup(0.75))
        if treatment == "multiscale":
            return MultiscaleSourceCleanup()

    cpu_treatment, cuda_treatment = source(), source()
    for frame in [rgb, rgb, 255 - rgb]:
        expected = next(process_frames([frame], config=config, source_cleanup=cpu_treatment))
        actual = cuda_core(frame, config, cuda_treatment)
        assert actual.shape == frame.shape and actual.dtype == np.uint8
        difference = np.abs(actual.astype(np.int16) - expected)
        assert difference.mean() < 0.1
        assert difference.max() <= 3
    np.testing.assert_array_equal(rgb, original)
    if treatment != "none":
        assert cuda_treatment.frames == cpu_treatment.frames == 3
        if treatment == "denoise":
            assert cuda_treatment.cleanup.resets == cpu_treatment.cleanup.resets
        else:
            assert cuda_treatment.resets == cpu_treatment.resets


@pytest.mark.parametrize("variant", [1, 2])
@pytest.mark.parametrize("kind", ["random", "flat", "black", "one_pixel"])
@pytest.mark.parametrize("power_mode,luminance_mode,detail", [("approximate", "rounded", 0),
                                                              ("standard", "fractional", 2),
                                                              ("approximate", "fractional", 20)])
def test_cuda_frame_degenerate_fields_and_numerical_modes(cuda_core, variant, kind, power_mode, luminance_mode, detail):
    rng = np.random.default_rng(611)
    rgb = rng.integers(0, 256, (47, 71, 3), np.uint8)
    if kind == "flat":
        rgb[:] = 70
    elif kind == "black":
        rgb[:] = 0
    elif kind == "one_pixel":
        rgb = rgb[:1]
    config = Config(filter=f"fattal{variant}", processing=ProcessingConfig(resize=False),
                    numerics=NumericsConfig(power_mode=power_mode, luminance_mode=luminance_mode, detail_level=detail))
    if kind == "one_pixel":
        with pytest.warns(RuntimeWarning):
            expected = next(process_frames([rgb], config=config))
        with pytest.warns(RuntimeWarning):
            actual = cuda_core(rgb, config)
    else:
        expected = next(process_frames([rgb], config=config))
        actual = cuda_core(rgb, config)
    difference = np.abs(actual.astype(np.int16) - expected)
    assert difference.mean() < 0.1
    assert difference.max() <= 2


def fake_frame_hardware(monkeypatch, *, cuda_time=0.01, difference=0, fail_after=None, setup_time=0):
    from luma_fatal import gpu_core

    clock = [0.0]
    instances = []

    def cpu(self, frame):
        clock[0] += 0.1
        if self.source_cleanup is not None:
            self.source_cleanup.frames += 1
        return frame.copy()

    class FakeCuda:
        name, version = "controlled GPU", "test"

        def __init__(self):
            self.calls, self.seconds = 0, cuda_time
            instances.append(self)
            clock[0] += setup_time

        def __call__(self, frame, config, treatment=None):
            self.calls += 1
            if treatment is not None:
                treatment.frames += 1
            if fail_after is not None and self.calls > fail_after:
                raise gpu.GPUUnavailable("controlled device failure")
            clock[0] += self.seconds
            return frame + difference

    monkeypatch.setattr(gpu.AdaptiveFrameProcessor, "cpu", cpu)
    monkeypatch.setattr(gpu_core, "CudaCore", FakeCuda)
    monkeypatch.setattr(gpu.time, "perf_counter", lambda: clock[0])
    return instances


@pytest.mark.parametrize("seconds,expected", [(0.01, "cuda"), (0.09, "cpu"), (0.2, "cpu")])
def test_frame_selection_measures_complete_processing_and_advances_history_once(monkeypatch, seconds, expected):
    fake_frame_hardware(monkeypatch, cuda_time=seconds)
    treatment = SourceCleanup(0.75)
    processor = gpu.AdaptiveFrameProcessor(Config(processing=ProcessingConfig(resize=False)), treatment, expected_frames=64)
    rgb = np.full((256, 256, 3), 70, np.uint8)
    for _ in range(3):
        np.testing.assert_array_equal(processor(rgb), rgb)
    assert treatment.frames == 3
    assert processor.record["backend"] == expected
    assert processor.record[f"{expected}_frames"] == 3


@pytest.mark.parametrize("difference,setup,frames,reason", [(5, 0, 64, "tolerance"), (0, 2, 8, "setup cost")])
def test_frame_selection_rejects_quality_loss_or_unrecoverable_setup(monkeypatch, difference, setup, frames, reason):
    fake_frame_hardware(monkeypatch, difference=difference, setup_time=setup)
    processor = gpu.AdaptiveFrameProcessor(Config(processing=ProcessingConfig(resize=False)), expected_frames=frames)
    rgb = np.full((256, 256, 3), 70, np.uint8)
    np.testing.assert_array_equal(processor(rgb), rgb)
    assert processor.record["backend"] == "cpu"
    assert reason in processor.record["reason"]


def test_frame_failure_restores_history_before_cpu_retry(monkeypatch):
    instances = fake_frame_hardware(monkeypatch, fail_after=3)
    treatment = SourceCleanup(0.75)
    processor = gpu.AdaptiveFrameProcessor(Config(processing=ProcessingConfig(resize=False)), treatment, expected_frames=64)
    rgb = np.full((256, 256, 3), 70, np.uint8)
    for _ in range(3):
        np.testing.assert_array_equal(processor(rgb), rgb)
    assert treatment.frames == 3
    assert instances[0].calls == 4
    assert processor.record["cuda_frames"] == 1
    assert processor.record["cpu_frames"] == 2
    assert "device failure" in processor.record["reason"]


def test_frame_sustained_slowdown_switches_to_cpu(monkeypatch):
    instances = fake_frame_hardware(monkeypatch)
    processor = gpu.AdaptiveFrameProcessor(Config(processing=ProcessingConfig(resize=False)), expected_frames=64)
    rgb = np.full((256, 256, 3), 70, np.uint8)
    processor(rgb)
    instances[0].seconds = 0.2
    for _ in range(4):
        np.testing.assert_array_equal(processor(rgb), rgb)
    assert processor.record["backend"] == "cpu"
    assert processor.record["cuda_frames"] == 4
    assert processor.record["cpu_frames"] == 1
