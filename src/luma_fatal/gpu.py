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


_COMPOSE_SOURCE = r'''
template<typename T> __device__ void compose_pixel(
    const T* rgb, const float* reference, const float* fields, float* output,
    int h, int w, int fh, int fw, int factor, double strength
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= h * w) return;
    int y = i / w, x = i % w;
    double fy = fmin(fmax((y + 0.5) / factor - 0.5, 0.0), fh - 1.0);
    double fx = fmin(fmax((x + 0.5) / factor - 0.5, 0.0), fw - 1.0);
    int y0 = (int)fy, x0 = (int)fx;
    int y1 = min(y0 + 1, fh - 1), x1 = min(x0 + 1, fw - 1);
    double wy = fy - y0, wx = fx - x0;
    double weights[4] = {(1 - wy) * (1 - wx), (1 - wy) * wx,
                         wy * (1 - wx), wy * wx};
    int indices[4] = {(y0 * fw + x0) * 7, (y0 * fw + x1) * 7,
                      (y1 * fw + x0) * 7, (y1 * fw + x1) * 7};
    double mask = fields[indices[0]] * weights[0] + fields[indices[1]] * weights[1]
                + fields[indices[2]] * weights[2] + fields[indices[3]] * weights[3];
    double amount = strength * fmin(fmax(mask, 0.0), 1.0);
    if (amount == 0) {
        for (int c = 0; c < 3; ++c) output[i * 3 + c] = rgb[i * 3 + c];
        output[h * w * 3 + i] = reference[i];
        return;
    }
    double guide = (4899.0 * rgb[i * 3] + 9617.0 * rgb[i * 3 + 1]
                  + 1868.0 * rgb[i * 3 + 2]) / 16384.0;
    double candidate_luminance = 0;
    for (int c = 0; c < 3; ++c) {
        double a = fields[indices[0] + c + 1] * weights[0]
                 + fields[indices[1] + c + 1] * weights[1]
                 + fields[indices[2] + c + 1] * weights[2]
                 + fields[indices[3] + c + 1] * weights[3];
        double b = fields[indices[0] + c + 4] * weights[0]
                 + fields[indices[1] + c + 4] * weights[1]
                 + fields[indices[2] + c + 4] * weights[2]
                 + fields[indices[3] + c + 4] * weights[3];
        double original = rgb[i * 3 + c];
        double candidate = fmin(fmax(a * guide + b, original - 4.0), original + 4.0);
        candidate = fmin(fmax(candidate, 0.0), 255.0);
        output[i * 3 + c] = original + amount * (candidate - original);
        candidate_luminance += candidate * (c == 0 ? 4899 : c == 1 ? 9617 : 1868);
    }
    double ref = reference[i];
    output[h * w * 3 + i] = ref + amount * (candidate_luminance / (16384.0 * 255.0) - ref);
}

extern "C" __global__ void compose_u8(
    const unsigned char* rgb, const float* reference, const float* fields, float* output,
    int h, int w, int fh, int fw, int factor, double strength
) { compose_pixel(rgb, reference, fields, output, h, w, fh, fw, factor, strength); }

extern "C" __global__ void compose_f32(
    const float* rgb, const float* reference, const float* fields, float* output,
    int h, int w, int fh, int fw, int factor, double strength
) { compose_pixel(rgb, reference, fields, output, h, w, fh, fw, factor, strength); }
'''


class CudaComposer:
    """Interpolate cleanup fields and compose RGB/luminance in one GPU pass."""

    def __init__(self):
        try:
            import cupy as cp

            self.cp = cp
            self.device = cp.cuda.runtime.getDevice()
            module = cp.RawModule(code=_COMPOSE_SOURCE, options=("--fmad=false",))
            self.kernels = {
                np.dtype(np.uint8): module.get_function("compose_u8"),
                np.dtype(np.float32): module.get_function("compose_f32"),
            }
            self.stream = cp.cuda.Stream(non_blocking=True)
            self.key = None
            self.input_key = None
        except Exception as error:
            raise GPUUnavailable(f"CUDA cleanup composition is unavailable: {error}") from error

    def __call__(self, rgb, reference_luminance, fields, factor, strength):
        cp = self.cp
        try:
            with cp.cuda.Device(self.device), self.stream:
                height, width = rgb.shape[:2]
                if rgb.dtype not in self.kernels or height * width * 4 > np.iinfo(np.int32).max:
                    raise GPUUnavailable("CUDA cleanup requires uint8 or float32 RGB within int32 indexing")
                key = (rgb.shape, rgb.dtype, fields.shape)
                if self.input_key != key:
                    self.rgb = cp.empty(rgb.shape, dtype=rgb.dtype)
                    self.reference = cp.empty((height, width), dtype=cp.float32)
                    self.fields = cp.empty(fields.shape, dtype=cp.float32)
                    self.input_key = key
                self.rgb.set(np.ascontiguousarray(rgb), stream=self.stream)
                self.reference.set(np.ascontiguousarray(reference_luminance, dtype=np.float32), stream=self.stream)
                self.fields.set(np.ascontiguousarray(fields, dtype=np.float32), stream=self.stream)
                self.compose_device(self.rgb, self.reference, self.fields, factor, strength)
                output = self.output.get(stream=self.stream)
                return output[:height * width * 3].reshape(rgb.shape), output[height * width * 3:].reshape(height, width)
        except Exception as error:
            raise GPUUnavailable(f"CUDA cleanup composition failed: {error}") from error

    def compose_device(self, rgb, reference, fields, factor, strength):
        """Return device views using the caller's current CUDA stream."""
        cp = self.cp
        height, width = rgb.shape[:2]
        if rgb.dtype not in self.kernels or height * width * 4 > np.iinfo(np.int32).max:
            raise GPUUnavailable("CUDA cleanup requires uint8 or float32 RGB within int32 indexing")
        key = (rgb.shape, rgb.dtype, fields.shape)
        if self.key != key:
            self.output = cp.empty(height * width * 4, dtype=cp.float32)
            self.key = key
        fields = cp.asarray(fields, dtype=cp.float32)
        self.kernels[rgb.dtype](
            ((height * width + 255) // 256,), (256,),
            (rgb, reference, fields, self.output,
             np.int32(height), np.int32(width), np.int32(fields.shape[0]),
             np.int32(fields.shape[1]), np.int32(factor), np.float64(strength)),
        )
        return self.output[:height * width * 3].reshape(rgb.shape), self.output[height * width * 3:].reshape(height, width)


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
