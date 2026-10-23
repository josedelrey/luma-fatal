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
