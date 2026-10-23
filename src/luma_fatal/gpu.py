"""Optional CUDA acceleration for the reference solver's paired sweeps."""

from __future__ import annotations

import sys
import time

import numpy as np


_SOURCE = r'''
extern "C" __global__ void sweep(
    const float* rhs, float* u, int h, int w, int color, float ow, float nw
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= h * w) return;
    int y = i / w, x = i % w;
    if (y == 0 || x == w - 1 || ((x + y) & 1) == color) return;
    float v = rhs[i] - u[i - w];
    v = v - u[i + 1];
    if (y < h - 1) v = v - u[i + w];
    if (x > 0) v = v - u[i - 1];
    if (x > 0 && y < h - 1) v = v * -0.25f;
    else if (x == 0 && y == h - 1) v = v * -0.5f;
    else v = v / -3.0f;
    u[i] = u[i] * ow + v * nw;
}

extern "C" __global__ void right_edge(
    const float* rhs, float* u, int h, int w, float ow, float nw
) {
    __shared__ float r[256], old[256], below[256], left[256];
    int t = threadIdx.x;
    float above = 0.0f;
    for (int base = 1; base < h; base += 256) {
        int y = base + t;
        if (y < h) {
            int i = y * w + w - 1;
            r[t] = rhs[i];
            old[t] = u[i];
            below[t] = y < h - 1 ? u[i + w] : 0.0f;
            left[t] = u[i - 1];
        }
        __syncthreads();
        if (t == 0) {
            int count = h - base < 256 ? h - base : 256;
            for (int j = 0; j < count; ++j) {
                float v = r[j] - above;
                if (base + j < h - 1) {
                    v = v - below[j];
                    v = v - left[j];
                    v = v / -3.0f;
                } else {
                    v = v - left[j];
                    v = v * -0.5f;
                }
                above = old[j] * ow + v * nw;
                old[j] = above;
            }
        }
        __syncthreads();
        if (y < h) u[y * w + w - 1] = old[t];
        __syncthreads();
    }
}
'''


class GPUUnavailable(RuntimeError):
    """The optional CUDA backend cannot process the requested field."""


class CudaSolver:
    """Reuse GPU buffers and a CUDA graph for one stream's solver geometry.

    Interior checkerboard updates are independent within each half-sweep.
    The right column advances serially after them, preserving the reference's
    double boundary updates. Transfers are included in every call.
    """

    def __init__(self):
        try:
            import cupy as cp

            self.cp = cp
            self.device = cp.cuda.runtime.getDevice()
            properties = cp.cuda.runtime.getDeviceProperties(self.device)
            name = properties["name"]
            self.name = name.decode() if isinstance(name, bytes) else name
            self.version = cp.__version__
            module = cp.RawModule(code=_SOURCE, options=("--fmad=false",))
            self.sweep = module.get_function("sweep")
            self.edge = module.get_function("right_edge")
            self.stream = cp.cuda.Stream(non_blocking=True)
            self.key = None
            self.graph = None
        except Exception as error:
            raise GPUUnavailable(
                "CUDA filtering needs the optional gpu dependencies and a working NVIDIA GPU. "
                f"Install them with uv sync --extra gpu. {error}"
            ) from error

    def __call__(self, rhs, iterations=80, relaxation=1.9):
        cp = self.cp
        try:
            with cp.cuda.Device(self.device), self.stream:
                key = (rhs.shape, iterations, relaxation)
                if self.key != key:
                    self._prepare(key)
                self.rhs.set(np.ascontiguousarray(rhs, dtype=np.float32), stream=self.stream)
                self.graph.launch(self.stream)
                return self.solution.get(stream=self.stream)
        except Exception as error:
            raise GPUUnavailable(f"CUDA solver failed: {error}") from error

    def _prepare(self, key):
        cp = self.cp
        shape, iterations, relaxation = key
        height, width = shape
        if min(shape) < 2 or height * width > np.iinfo(np.int32).max:
            raise GPUUnavailable("CUDA solver requires a field of at least 2 by 2 within int32 indexing")
        self.rhs = cp.zeros(shape, dtype=cp.float32)
        self.solution = cp.zeros(shape, dtype=cp.float32)
        arguments = (self.rhs, self.solution, np.int32(height), np.int32(width))
        weights = (np.float32(1 - relaxation), np.float32(relaxation))
        blocks = ((height * width + 255) // 256,)
        # Compile all kernels and initialize fill before stream capture.
        self.sweep(blocks, (256,), (*arguments, np.int32(1), *weights))
        self.edge((1,), (256,), (*arguments, *weights))
        self.stream.synchronize()
        self.stream.begin_capture()
        self.solution.fill(0)
        for _ in range(iterations):
            for color in (1, 0):
                self.sweep(blocks, (256,), (*arguments, np.int32(color), *weights))
                self.edge((1,), (256,), (*arguments, *weights))
        self.graph = self.stream.end_capture()
        self.key = key


class AdaptiveSolver:
    """Select CUDA only when synchronized timings justify its setup cost.

    Each stream owns its calibration, buffers and counters. CPU evaluation
    stays authoritative, including when optional dependencies are missing.
    """

    def __init__(self, *, expected_frames=None):
        self.expected_frames = expected_frames
        self.cuda = None
        self.disabled = False
        self.key = None
        self.backend = "cpu"
        self.slow_calls = 0
        self.record = dict(
            requested_backend="auto", backend="cpu", device=None, cupy_version=None,
            reason="solver not used", cpu_frames=0, cuda_frames=0,
            cpu_seconds=None, cuda_seconds=None, calibration_seconds=0.0,
        )

    def __call__(self, rhs, iterations=80, relaxation=1.9):
        from .core import _sor

        key = (rhs.shape, iterations, relaxation)
        if self.key != key and not self.disabled:
            self.key = key
            return self._calibrate(rhs, iterations, relaxation)
        if self.backend == "cuda":
            try:
                begin = time.perf_counter()
                output = self.cuda(rhs, iterations, relaxation)
                elapsed = time.perf_counter() - begin
                self.record["cuda_frames"] += 1
                self.slow_calls = self.slow_calls + 1 if elapsed > self.record["cpu_seconds"] else 0
                if self.slow_calls >= 3:
                    self._disable("CUDA became slower than the calibrated CPU solver")
                return output
            except GPUUnavailable as error:
                self._disable(str(error))
        self.record["cpu_frames"] += 1
        return _sor(rhs, iterations, relaxation)

    def _calibrate(self, rhs, iterations, relaxation):
        from .core import _sor

        self.backend = "cpu"
        self.record["backend"] = "cpu"
        if rhs.size < 65536 or iterations < 4:
            self.record["reason"] = "solver workload too small to amortize GPU overhead"
            self.record["cpu_frames"] += 1
            return _sor(rhs, iterations, relaxation)
        if self.expected_frames is not None and self.expected_frames < 8:
            self.record["reason"] = "clip too short to amortize GPU setup"
            self.record["cpu_frames"] += 1
            return _sor(rhs, iterations, relaxation)
        start = time.perf_counter()
        try:
            if self.cuda is None:
                self.cuda = CudaSolver()
                self.record.update(device=self.cuda.name, cupy_version=self.cuda.version)
            # Warm JIT code separately so compilation cannot bias the CPU timing.
            _sor(np.zeros((2, 2), np.float32), 1, relaxation)
            self.cuda(rhs, iterations, relaxation)
            cpu_samples, cuda_samples = [], []
            for _ in range(2):
                begin = time.perf_counter()
                cpu_output = _sor(rhs, iterations, relaxation)
                cpu_samples.append(time.perf_counter() - begin)
                begin = time.perf_counter()
                cuda_output = self.cuda(rhs, iterations, relaxation)
                cuda_samples.append(time.perf_counter() - begin)
            cpu_seconds, cuda_seconds = min(cpu_samples), max(cuda_samples)
            elapsed = time.perf_counter() - start
            self.record.update(
                cpu_seconds=cpu_seconds, cuda_seconds=cuda_seconds,
                calibration_seconds=self.record["calibration_seconds"] + elapsed,
            )
            remaining = None if self.expected_frames is None else max(0, self.expected_frames - 1)
            setup_extra = max(0, elapsed - cpu_seconds)
            profitable = remaining is None or remaining * (cpu_seconds - cuda_seconds) > setup_extra * 1.25
            close = np.allclose(cuda_output, cpu_output, rtol=2e-5, atol=2e-5)
            if cuda_seconds < cpu_seconds * 0.8 and profitable and close:
                self.backend = "cuda"
                self.record.update(backend="cuda", reason="CUDA is faster including transfers and amortized setup")
                self.record["cuda_frames"] += 1
                print(
                    f"GPU filtering selected on {self.cuda.name} "
                    f"({cpu_seconds / max(cuda_seconds, 1e-12):.1f}x solver speedup).",
                    file=sys.stderr,
                )
                return cuda_output
            if not close:
                self.record["reason"] = "CUDA result exceeded the solver agreement tolerance"
            elif not profitable:
                self.record["reason"] = "clip too short to recover measured GPU setup cost"
            else:
                self.record["reason"] = "CUDA did not beat CPU solver time by at least 20 percent"
            self.record["cpu_frames"] += 1
            return cpu_output
        except GPUUnavailable as error:
            self.record["calibration_seconds"] += time.perf_counter() - start
            self._disable(str(error))
            self.record["cpu_frames"] += 1
            return _sor(rhs, iterations, relaxation)

    def _disable(self, reason):
        self.disabled = True
        self.backend = "cpu"
        self.record.update(backend="cpu", reason=reason)
        print(f"Using CPU filtering. {reason}", file=sys.stderr)
