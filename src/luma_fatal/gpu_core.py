"""Device-resident experimental Fattal filtering with reference RGB controls."""

from __future__ import annotations

from copy import copy
from dataclasses import replace
import time

import numpy as np

from .gpu import CudaComposer, GPUUnavailable
from .numerics import gradient_scale


_SOURCE = r'''
__device__ double power(double v, double e, int standard) {
    if (standard) return pow(v, e);
    long long high = ((unsigned long long)__double_as_longlong(v)) >> 32;
    long long word = (long long)trunc(e * (high - 1072632447LL) + 1072632447LL);
    return __longlong_as_double((unsigned long long)word << 32);
}
extern "C" __global__ void luminance(const unsigned char* rgb, float* out, int n, int fractional) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int v = 4899 * rgb[i*3] + 9617 * rgb[i*3+1] + 1868 * rgb[i*3+2];
    out[i] = fractional ? (float)v / (float)(16384*255) : (float)((v + 8192) >> 14) * (float)(1.0/255.0);
}
extern "C" __global__ void log_source(const float* src, const float* maximum, float* out, int n, double offset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (float)log((double)(float)((double)(src[i] * 100.0f / maximum[0]) + offset));
}
extern "C" __global__ void gaussian(const float* src, float* out, int h, int w) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= h*w) return;
    int y = i/w, x = i%w;
    float v = 0.0f;
    for (int dy=-1; dy<=1; ++dy) {
        for (int dx=-1; dx<=1; ++dx) {
            float weight = (dy == 0 ? 2.0f : 1.0f) * (dx == 0 ? 2.0f : 1.0f) / 16.0f;
            v = v + weight * src[min(max(y+dy,0),h-1)*w + min(max(x+dx,0),w-1)];
        }
    }
    out[i] = v;
}
extern "C" __global__ void cubic(const float* src, float* out, const long long* indices,
    const float* weights, int h, int w, int oh, int ow, int horizontal) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= oh*ow) return;
    int y = i/ow, x = i%ow, a = horizontal ? x : y;
    float v = 0.0f;
    for (int k=0; k<4; ++k) {
        int j = horizontal ? y*w + indices[a*4+k] : indices[a*4+k]*w+x;
        v = v + src[j] * weights[a*4+k];
    }
    out[i] = v;
}
extern "C" __global__ void gradient(const float* src, float* out, int h, int w, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= h*w) return;
    int y=i/w, x=i%w;
    float dx = (src[y*w+max(x-1,0)] - src[y*w+min(x+1,w-1)]) / scale;
    float dy = (src[min(y+1,h-1)*w+x] - src[max(y-1,0)*w+x]) / scale;
    out[i] = sqrtf(dy*dy + dx*dx);
}
extern "C" __global__ void row_sums(const float* src, float* out, int h, int w) {
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= h) return;
    float v=0.0f;
    for (int x=0; x<w; ++x) v = v + src[y*w+x];
    out[y] = v;
}
extern "C" __global__ void average(const float* rows, float* out, int h, int n) {
    float v=0.0f;
    for (int y=0; y<h; ++y) v = v + rows[y];
    out[0] = v / (float)n;
}
extern "C" __global__ void attenuation(const float* grad, const float* avg, float* phi,
    int n, float alpha, float exponent, float noise, float threshold, int multiply, int standard) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v=1.0f;
    if (grad[i] > threshold && avg[0] > 0) {
        float denominator = alpha * avg[0];
        float numerator = grad[i] + noise;
        double ratio = denominator == 0 || (double)numerator / denominator > 3.4028234663852886e38
            ? (double)numerator / ((double)alpha * avg[0]) : (double)(numerator / denominator);
        v = (float)power(ratio, (double)exponent, standard);
    }
    phi[i] = multiply ? phi[i] * v : v;
}
extern "C" __global__ void divergence(const float* src, const float* phi, float* out, int h, int w) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= h*w) return;
    int y=i/w, x=i%w;
    if (y==0) { out[i]=0; return; }
    float gx = x<w-1 ? (src[i+1]-src[i])*phi[i] : 0.0f;
    float gy = y<h-1 ? (src[i+w]-src[i])*phi[i] : 0.0f;
    float v=gx+gy;
    if (x>0) v = v - (src[i]-src[i-1])*phi[i-1];
    v = v - (src[i]-src[i-w])*phi[i-w];
    out[i]=v;
}
extern "C" __global__ void exponential(const float* src, float* out, int n, float offset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (float)exp((double)src[i]) - offset;
}
extern "C" __global__ void normalize(const float* src, float* out, const float* sorted,
    int n, int count, float low_q, float high_q, float floor_value) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (count == 0) { out[i]=floor_value; return; }
    float low=sorted[min(count-1,(int)((float)count*low_q))];
    float high=sorted[min(count-1,(int)((float)count*high_q))];
    float v=(src[i]-low)/(high-low);
    out[i] = isnan(v) ? __int_as_float(0x7fc00000) : v > 0 ? v : floor_value;
}
__device__ double clip(double v) { return fmax(fmin(v,255.0),0.0); }
extern "C" __global__ void recombine(const unsigned char* rgb, const float* colors,
    const float* lum, const float* reference, const float* mapped, unsigned char* out,
    int n, float saturation, int standard, double gamma, double tone, double highlight,
    double black, double white, double vibrance, int threshold, int softness, float fade) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float m=mapped[i];
    if (tone < 100) {
        float target=isnan(m) || (isinf(m) && m>0) ? 1.0f : isinf(m) ? 0.0f : m;
        m = tone == 0 ? reference[i] : reference[i] + (float)((float)tone / 100.0f) * (target-reference[i]);
    }
    float scale=lum[i]*255.0f;
    float mscale=m*255.0f;
    double c[3];
    for (int k=0;k<3;++k) {
        float ratio=scale>0 ? colors[i*3+k]/scale : 0.0f;
        c[k]=power((double)ratio,(double)saturation,standard)*(double)mscale;
    }
    double peak=fmax(c[0],fmax(c[1],c[2]));
    if (highlight>0 && isfinite(c[0]) && isfinite(c[1]) && isfinite(c[2]) && peak>191.25) {
        double compressed=191.25+63.75/(1.0+63.75/(peak-191.25));
        double amount=highlight/100.0;
        double target=(1.0-amount)*peak+amount*compressed;
        for (int k=0;k<3;++k) c[k]=(c[k]/peak)*target;
    }
    if (gamma != 1) for (int k=0;k<3;++k) c[k]=pow(clip(c[k])/255.0,1.0/gamma)*255.0;
    if (vibrance>0) {
        for (int k=0;k<3;++k) c[k]=clip(c[k]);
        peak=fmax(c[0],fmax(c[1],c[2]));
        double minimum=fmin(c[0],fmin(c[1],c[2]));
        if (peak != minimum) {
            double saturation=(peak-minimum)/peak;
            double gain=1.0+(vibrance/100.0)*(1.0-saturation)*(1.0-saturation);
            double brightness=(4899.0*c[0]+9617.0*c[1]+1868.0*c[2])/16384.0;
            if (peak>brightness) gain=fmin(gain,(255.0-brightness)/(peak-brightness));
            if (minimum<brightness) gain=fmin(gain,brightness/(brightness-minimum));
            if (gain>1) for (int k=0;k<3;++k) c[k]=clip(brightness+gain*(c[k]-brightness));
        }
    }
    double shadow=softness ? (4899.0*rgb[i*3]+9617.0*rgb[i*3+1]+1868.0*rgb[i*3+2])/16384.0
        : (double)(reference[i]*255.0f);
    int lower=max(0,threshold-softness), upper=min(255,threshold+softness);
    double weight=1.0;
    if (shadow <= lower) weight=0;
    else if (shadow < upper) { double t=(shadow-lower)/(upper-lower); weight=t*t*(3-2*t); }
    for (int k=0;k<3;++k) {
        double v=c[k], original=rgb[i*3+k];
        if (black != 0 || white != 100) {
            v=original;
            if (weight>0) v += weight*(clip(c[k])-v);
            v=black*255.0/100.0+(white-black)/100.0*v+0.5;
        } else if (softness) {
            if (weight==0) v=original;
            else if (weight<1) v=original+weight*(clip(v)-original);
            v=clip(v+0.5);
        } else {
            v=shadow<=threshold ? original : clip(v+0.5);
        }
        float mixed=(float)((unsigned char)v)*(1.0f-fade)+(float)rgb[i*3+k]*fade;
        out[i*3+k]=(unsigned char)fmaxf(fminf(ceilf(mixed-0.5f),255.0f),0.0f);
    }
}
template<typename T> __device__ void reduce_pixel(const T* rgb, float* small, float* variance,
    int h, int w, int sh, int sw, int factor) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=sh*sw*3) return;
    int c=i%3, x=(i/3)%sw, y=i/(3*sw);
    int bottom=min((y+1)*factor,h), right=min((x+1)*factor,w);
    int count=(bottom-y*factor)*(right-x*factor);
    double total=0, squares=0;
    for (int yy=y*factor;yy<bottom;++yy) for (int xx=x*factor;xx<right;++xx) {
        double v=rgb[(yy*w+xx)*3+c]; total+=v; squares+=v*v;
    }
    small[i]=total/count;
    variance[i]=fmax(squares/count-(total/count)*(total/count),0.0);
}
extern "C" __global__ void reduce_u8(const unsigned char* rgb, float* small, float* variance,
    int h, int w, int sh, int sw, int factor) { reduce_pixel(rgb,small,variance,h,w,sh,sw,factor); }
extern "C" __global__ void reduce_f32(const float* rgb, float* small, float* variance,
    int h, int w, int sh, int sw, int factor) { reduce_pixel(rgb,small,variance,h,w,sh,sw,factor); }
extern "C" __global__ void activity(const float* small, const float* variance, float* out,
    int h, int w, int sh, int sw, int factor) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=sh*sw) return;
    int y=i/sw,x=i%sw,top=max(y-1,0),bottom=min(y+1,sh-1),left=max(x-1,0),right=min(x+1,sw-1);
    double tc=(top*factor+min((top+1)*factor,h)-1)*0.5, bc=(bottom*factor+min((bottom+1)*factor,h)-1)*0.5;
    double lc=(left*factor+min((left+1)*factor,w)-1)*0.5, rc=(right*factor+min((right+1)*factor,w)-1)*0.5;
    int th=min((y+1)*factor,h)-y*factor, tw=min((x+1)*factor,w)-x*factor;
    double sy=(th*th-1)/12.0, sx=(tw*tw-1)/12.0, residual=0;
    for (int c=0;c<3;++c) {
        double dx=left==right ? 0 : (double)(small[(y*sw+right)*3+c]-small[(y*sw+left)*3+c])/(rc-lc);
        double dy=top==bottom ? 0 : (double)(small[(bottom*sw+x)*3+c]-small[(top*sw+x)*3+c])/(bc-tc);
        residual=fmax(residual,variance[i*3+c]-(dx*dx*sx+dy*dy*sy));
    }
    out[i]=sqrt(fmax(residual,0.0));
}
extern "C" __global__ void resize_fixed(const unsigned char* src, unsigned char* out,
    const long long* ix, const long long* nx, const int* wx,
    const long long* iy, const long long* ny, const int* wy, int w, int oh, int ow) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=oh*ow*3) return;
    int c=i%3,x=(i/3)%ow,y=i/(3*ow);
    int top=src[(iy[y]*w+ix[x])*3+c]*(16-wx[x])+src[(iy[y]*w+nx[x])*3+c]*wx[x];
    int bottom=src[(ny[y]*w+ix[x])*3+c]*(16-wx[x])+src[(ny[y]*w+nx[x])*3+c]*wx[x];
    out[i]=(top*(16-wy[y])+bottom*wy[y])>>8;
}
extern "C" __global__ void resize_linear(const unsigned char* src, unsigned char* out, int h, int w, int oh, int ow) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=oh*ow*3) return;
    int c=i%3,x=(i/3)%ow,y=i/(3*ow);
    double fx=fmin(fmax((x+0.5)*((double)w/ow)-0.5,0.0),w-1.0);
    double fy=fmin(fmax((y+0.5)*((double)h/oh)-0.5,0.0),h-1.0);
    int ix=(int)fx,iy=(int)fy,nx=min(ix+1,w-1),ny=min(iy+1,h-1);
    double wx=fx-ix,wy=fy-iy;
    double top=src[(iy*w+ix)*3+c]*(1-wx)+src[(iy*w+nx)*3+c]*wx;
    double bottom=src[(ny*w+ix)*3+c]*(1-wx)+src[(ny*w+nx)*3+c]*wx;
    out[i]=(unsigned char)clip(top*(1-wy)+bottom*wy);
}
extern "C" __global__ void fade_rgb(const unsigned char* rgb, const unsigned char* effect,
    unsigned char* out, int n, float amount) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<n) out[i]=fminf(fmaxf(ceilf((float)effect[i]*(1.0f-amount)+(float)rgb[i]*amount-0.5f),0.0f),255.0f);
}
extern "C" __global__ void filter_axis(const float* src, float* out, const double* weights,
    int n, int length, int stride, int radius, int uniform) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=n) return;
    int coordinate=(i/stride)%length;
    double sum=0;
    for (int k=-radius;k<=radius;++k) {
        int index=(coordinate+k)%(2*length);
        if (index<0) index+=2*length;
        if (index>=length) index=2*length-1-index;
        double value=src[i+(index-coordinate)*stride];
        sum += uniform ? value : value*weights[k+radius];
    }
    out[i]=uniform ? sum/(2*radius+1) : sum;
}
'''


class CudaCore:
    """Keep cleanup composition, attenuation, solving and color on one device."""

    def __init__(self):
        from .gpu import CudaSolver

        self.solver = CudaSolver()
        self.solver.parallel_edge = True
        self.cp = cp = self.solver.cp
        self.stream = self.solver.stream
        self.name, self.version = self.solver.name, self.solver.version
        try:
            module = cp.RawModule(code=_SOURCE, options=("--fmad=false",))
            names = ("luminance", "log_source", "gaussian", "cubic", "gradient", "row_sums",
                     "average", "attenuation", "divergence", "exponential", "normalize", "recombine",
                     "reduce_u8", "reduce_f32", "activity", "resize_fixed", "resize_linear", "fade_rgb", "filter_axis")
            self.kernels = {name: module.get_function(name) for name in names}
            self.composer = CudaComposer()
            self.axes = {}
            self.resize_axes = {}
            self.mask_axes = {}
            self.noise_fields = {}
            self.filter_weights = {}
        except Exception as error:
            raise GPUUnavailable(f"CUDA frame kernels are unavailable: {error}") from error

    def launch(self, name, shape, *args):
        count = int(np.prod(shape))
        self.kernels[name](((count + 255) // 256,), (256,), args)

    def gaussian(self, source):
        out = self.cp.empty_like(source)
        h, w = source.shape
        self.launch("gaussian", source.shape, source, out, np.int32(h), np.int32(w))
        return out

    def cubic(self, source, shape):
        from .core import _cubic_axis

        cp = self.cp
        if source.shape == shape:
            return source.copy()
        key = source.shape, shape
        if key not in self.axes:
            yi, yw = _cubic_axis(source.shape[0], shape[0], False)
            xi, xw = _cubic_axis(source.shape[1], shape[1], True)
            self.axes[key] = tuple(cp.asarray(a) for a in (yi, yw, xi, xw))
        yi, yw, xi, xw = self.axes[key]
        h, w = source.shape
        horizontal = cp.empty((h, shape[1]), cp.float32)
        out = cp.empty(shape, cp.float32)
        self.launch("cubic", horizontal.shape, source, horizontal, xi, xw,
                    np.int32(h), np.int32(w), np.int32(h), np.int32(shape[1]), np.int32(1))
        self.launch("cubic", shape, horizontal, out, yi, yw, np.int32(h), np.int32(shape[1]),
                    np.int32(shape[0]), np.int32(shape[1]), np.int32(0))
        return out

    def phi(self, pyramid, variant, params, numerics):
        cp = self.cp
        detail = numerics.detail_level
        if detail >= len(pyramid):
            return cp.ones_like(pyramid[0])
        phi = cp.ones_like(pyramid[-1])
        for level in range(len(pyramid) - 1, -1, -1):
            source = pyramid[level]
            if level >= detail:
                h, w = source.shape
                grad, rows, avg = cp.empty_like(source), cp.empty(h, cp.float32), cp.empty(1, cp.float32)
                self.launch("gradient", source.shape, source, grad, np.int32(h), np.int32(w), gradient_scale(level + 1))
                self.launch("row_sums", (h,), grad, rows, np.int32(h), np.int32(w))
                self.kernels["average"]((1,), (1,), (rows, avg, np.int32(h), np.int32(source.size)))
                if variant != 1:
                    phi = cp.empty_like(source)
                self.launch("attenuation", source.shape, grad, avg, phi, np.int32(source.size),
                            np.float32(params.alpha) / np.float32(100),
                            np.float32(np.float32(params.beta) / np.float32(100) - np.float32(1)),
                            np.float32(params.noise) / np.float32(100), np.float32(numerics.gradient_threshold),
                            np.int32(variant == 1), np.int32(numerics.power_mode == "standard"))
            if level and (variant == 1 or level <= detail):
                phi = self.gaussian(self.cubic(phi, pyramid[level - 1].shape))
        return phi

    def normalize(self, source, numerics):
        cp = self.cp
        values = cp.sort(source[source != 0])
        output = cp.empty_like(source)
        self.launch("normalize", source.shape, source, output, values, np.int32(source.size), np.int32(values.size),
                    np.float32(numerics.normalization_low), np.float32(numerics.normalization_high),
                    np.float32(numerics.normalization_floor))
        return output

    def filter(self, source, sigma, *, uniform=False, mode="reflect"):
        """Use double accumulators and float32 axis outputs like the CPU filters."""
        cp = self.cp
        scales = (sigma,) * source.ndim if np.isscalar(sigma) else sigma
        result = source
        for axis, scale in enumerate(scales):
            radius = int(scale) // 2 if uniform else int(4 * scale + 0.5)
            if radius == 0:
                continue
            key = float(scale), uniform
            if key not in self.filter_weights:
                coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
                weights = np.ones(coordinates.shape) if uniform else np.exp(-0.5 / (scale * scale) * coordinates ** 2)
                self.filter_weights[key] = cp.asarray(weights / weights.sum())
            output = cp.empty_like(result)
            stride = int(np.prod(result.shape[axis + 1:]))
            self.launch("filter_axis", result.shape, result, output, self.filter_weights[key],
                        np.int32(result.size), np.int32(result.shape[axis]), np.int32(stride),
                        np.int32(radius), np.int32(uniform))
            result = output
        return result.copy() if result is source else result

    def uniform(self, source, size=3, *, mode="reflect"):
        return self.filter(source, size, uniform=True, mode=mode)

    def cleanup(self, rgb, reference, treatment):
        from cupyx.scipy.ndimage import maximum_filter

        cp = self.cp
        start = time.perf_counter()
        h, w = rgb.shape[:2]
        factor = max(1, (max(h, w) + 383) // 384)
        shape = ((h + factor - 1) // factor, (w + factor - 1) // factor)
        small, variance = cp.empty((*shape, 3), cp.float32), cp.empty((*shape, 3), cp.float32)
        activity = cp.empty(shape, cp.float32)
        dims = (np.int32(h), np.int32(w), np.int32(shape[0]), np.int32(shape[1]), np.int32(factor))
        self.launch("reduce_u8" if rgb.dtype == cp.uint8 else "reduce_f32", small.shape, rgb, small, variance, *dims)
        self.launch("activity", shape, small, variance, activity, *dims)
        device_state = copy(treatment)
        if treatment.guide is not None:
            device_state.guide = cp.asarray(treatment.guide)
            device_state.mask = cp.asarray(treatment.mask)
        fields = device_state.prepare_fields(small, activity, rgb.shape, backend=cp,
                                             filters=(self.filter, self.uniform, maximum_filter))
        # Keep only small history on the host so CPU fallback can resume any frame.
        treatment.guide = device_state.guide.get(stream=self.stream)
        treatment.mask = device_state.mask.get(stream=self.stream)
        treatment.shape, treatment.frames, treatment.resets = rgb.shape, device_state.frames, device_state.resets
        output = self.composer.compose_device(rgb, reference, fields, factor, treatment.strength)
        treatment.seconds += time.perf_counter() - start
        return output

    def mask(self, mask, shape):
        from cupyx.scipy.ndimage import map_coordinates

        cp = self.cp
        if mask.shape == shape:
            return cp.asarray(mask)
        factor = max(1, (max(shape) + 383) // 384)
        key = shape, factor
        if key not in self.mask_axes:
            y, x = cp.meshgrid((cp.arange(shape[0], dtype=cp.float32) + 0.5) / factor - 0.5,
                               (cp.arange(shape[1], dtype=cp.float32) + 0.5) / factor - 0.5, indexing="ij")
            self.mask_axes[key] = cp.stack((y, x))
        return cp.clip(map_coordinates(cp.asarray(mask), self.mask_axes[key], order=1,
                                        mode="nearest", prefilter=False), 0, 1)

    def treat(self, rgb, reference, treatment):
        from cupyx.scipy.ndimage import convolve1d, gaussian_filter, uniform_filter
        from .cleanup import MultiscaleSourceCleanup, SourceCleanup
        from .denoise import SourceDenoise, denoise_rgb, _LUMA
        from .prenoise import SourceNoise, noise_field

        cp = self.cp
        if type(treatment) is SourceCleanup:
            return self.cleanup(rgb, reference, treatment)
        start = time.perf_counter()
        if type(treatment) is SourceDenoise:
            color, luminance = rgb, reference
            if treatment.strength:
                candidate, sigma = denoise_rgb(rgb, backend=cp, filters=(convolve1d, uniform_filter))
                if bool(cp.any(sigma)):
                    amount = np.float32(treatment.strength)
                    color = rgb + amount * (candidate - rgb)
                    luminance = reference + amount * (cp.sum(candidate * cp.asarray(_LUMA), axis=2) - reference)
            treatment.frames += 1
            treatment.seconds += time.perf_counter() - start
            return self.treat(color, luminance, treatment.cleanup) if treatment.cleanup is not None else (color, luminance)
        if type(treatment) is MultiscaleSourceCleanup:
            color, luminance = self.treat(rgb, reference, treatment.cleanup)
            mask = self.mask(treatment.cleanup.mask, rgb.shape[:2])
            original = cp.concatenate((rgb.astype(cp.float32), (reference * 255)[..., None]), axis=2)
            cleaned = cp.concatenate((color, (luminance * 255)[..., None]), axis=2)
            correction = cleaned - original
            denominator = cp.maximum(gaussian_filter(mask, treatment.sigma, mode="reflect"), 1e-6)
            broad = gaussian_filter(correction, (treatment.sigma, treatment.sigma, 0), mode="reflect") * (mask / denominator)[..., None]
            correction = treatment.fine_strength * correction + (treatment.coarse_strength - treatment.fine_strength) * broad
            limit = mask[..., None] * cp.asarray([4, 4, 4, 4.5], cp.float32)
            result = cp.clip(original + cp.clip(correction, -limit, limit), 0, 255)
            protected = mask == 0
            color = cp.where(protected[..., None], rgb, result[..., :3])
            luminance = cp.where(protected, reference, result[..., 3] / np.float32(255))
            if treatment.fine_strength == treatment.coarse_strength == 0:
                color, luminance = rgb.astype(cp.float32), reference.copy()
            treatment.mask = mask.get(stream=self.stream)
            treatment.seconds += time.perf_counter() - start
            return cp.ascontiguousarray(color), luminance
        if type(treatment) is SourceNoise:
            color, luminance = self.treat(rgb, reference, treatment.cleanup)
            shape = rgb.shape[:2]
            mask = self.mask(treatment.cleanup.mask, shape) if treatment.masked else cp.ones(shape, cp.float32)
            treatment.mask = mask.get(stream=self.stream)
            if treatment.rms == 0:
                treatment.seconds += time.perf_counter() - start
                return color, luminance
            if treatment.field is None or treatment.field.shape != shape:
                treatment.field = noise_field(shape, treatment.kind, treatment.seed)
            key = shape, treatment.kind, treatment.seed
            if key not in self.noise_fields:
                self.noise_fields[key] = cp.asarray(treatment.field)
            delta = self.noise_fields[key] * np.float32(treatment.rms) * mask
            limit = np.float32(4) * mask
            lower = cp.maximum(-color.min(2), -luminance * np.float32(255))
            upper = cp.minimum(255 - color.max(2), (1 - luminance) * np.float32(255))
            delta = cp.clip(delta, cp.maximum(lower, -limit), cp.minimum(upper, limit))
            output = color + delta[..., None], luminance + delta / np.float32(255)
            treatment.seconds += time.perf_counter() - start
            return output
        # Preserve arbitrary third-party callbacks through the reference interface.
        color, luminance = treatment(rgb.get(), reference.get())
        return cp.asarray(color), cp.asarray(luminance)

    def resize(self, rgb, size, method="fixed"):
        from .processing import _filtered_axis, nearest_coordinates

        cp = self.cp
        h, w = rgb.shape[:2]
        ow, oh = size
        if (w, h) == size:
            return rgb
        key = (w, h), size, method
        if method == "nearest":
            if key not in self.resize_axes:
                self.resize_axes[key] = tuple(cp.asarray(a) for a in nearest_coordinates((w, h), size))
            x, y = self.resize_axes[key]
            return cp.ascontiguousarray(rgb[y[:, None], x[None, :]])
        if method == "bicubic":
            channels = [self.cubic(cp.ascontiguousarray(rgb[..., c], dtype=cp.float32), (oh, ow)) for c in range(3)]
            return cp.clip(cp.stack(channels, axis=-1), 0, 255).astype(cp.uint8)
        output = cp.empty((oh, ow, 3), cp.uint8)
        if method == "bilinear":
            self.launch("resize_linear", output.shape, rgb, output, np.int32(h), np.int32(w), np.int32(oh), np.int32(ow))
        else:
            if key not in self.resize_axes:
                axes = (*_filtered_axis(w, ow, horizontal=True), *_filtered_axis(h, oh, horizontal=False))
                self.resize_axes[key] = tuple(cp.asarray(a) for a in axes)
            self.launch("resize_fixed", output.shape, rgb, output, *self.resize_axes[key], np.int32(w), np.int32(oh), np.int32(ow))
        return output

    def core(self, rgb, variant, params, numerics, *, source_cleanup=None,
             lighting_scale=1.0):
        cp = self.cp
        if params.fade == 100 or min(rgb.shape[:2]) < 2:
            # The CPU identity extension also reports its one-pixel warning.
            from .core import apply_core

            return cp.asarray(apply_core(rgb.get(), variant, params, numerics))
        h, w = rgb.shape[:2]
        lum = cp.empty((h, w), cp.float32)
        self.launch("luminance", lum.shape, rgb, lum, np.int32(h*w), np.int32(numerics.luminance_mode == "fractional"))
        reference = lum
        colors = rgb.astype(cp.float32)
        if source_cleanup is not None:
            colors, lum = self.treat(rgb, reference, source_cleanup)
        maximum = cp.max(lum)
        if float(maximum) <= 0:
            from .core import apply_core

            return cp.asarray(apply_core(rgb.get(), variant, params, numerics))
        log_source = cp.empty_like(lum)
        self.launch("log_source", lum.shape, lum, maximum, log_source, np.int32(h*w), np.float64(numerics.log_offset))
        pyramid = [log_source]
        minimum, levels = min(h, w), 0
        while minimum >= numerics.pyramid_min_size:
            levels += 1
            minimum //= 2
        for _ in range(max(1, levels) - 1):
            previous = pyramid[-1]
            pyramid.append(self.cubic(self.gaussian(previous), (previous.shape[0] // 2, previous.shape[1] // 2)))
        phi = self.phi(pyramid, variant, params, numerics)
        rhs = cp.empty_like(lum)
        self.launch("divergence", lum.shape, log_source, phi, rhs, np.int32(h), np.int32(w))
        solution = self.solver.solve_device(rhs, numerics.iterations, numerics.relaxation)
        if lighting_scale < 1.0:
            low_shape = (max(2, round(h * lighting_scale)),
                         max(2, round(w * lighting_scale)))
            laplacian = cp.empty_like(rhs)
            self.launch("divergence", rhs.shape, solution, cp.ones_like(rhs),
                        laplacian, np.int32(h), np.int32(w))
            residual = rhs - laplacian
            coarse = self.cubic(self.gaussian(residual), low_shape)
            coarse *= np.float32(1.0 / (lighting_scale * lighting_scale))
            correction = self.solver.solve_device(
                coarse, max(1, round(numerics.iterations * lighting_scale)),
                numerics.relaxation)
            solution += self.cubic(correction, (h, w))
        mapped = cp.empty_like(lum)
        self.launch("exponential", lum.shape, solution, mapped, np.int32(h*w), np.float32(numerics.log_offset))
        mapped = self.normalize(mapped, numerics)
        output = cp.empty(rgb.shape, cp.uint8)
        self.launch("recombine", lum.shape, rgb, colors, lum, reference, mapped, output, np.int32(h*w),
                    np.float32(params.saturation) / np.float32(100), np.int32(numerics.power_mode == "standard"),
                    np.float64(params.output_gamma), np.float64(params.tone_strength), np.float64(params.highlight_compression),
                    np.float64(params.black_level), np.float64(params.white_level), np.float64(params.vibrance),
                    np.int32(numerics.shadow_threshold), np.int32(numerics.shadow_softness),
                    np.float32(params.fade) / np.float32(100))
        return output

    def __call__(self, array, config, source_cleanup=None):
        from .processing import enlargement_sizes, input_size, working_size

        cp = self.cp
        try:
            if config.parameters.fade == 100:
                return array.copy()
            if array.size > np.iinfo(np.int32).max:
                raise GPUUnavailable("CUDA frames must fit int32 indexing")
            with cp.cuda.Device(self.solver.device), self.stream:
                rgb = cp.asarray(array)
                size = (array.shape[1], array.shape[0])
                if config.processing.resize:
                    destination = input_size(size, config.processing.effective_max_input_pixels)
                    if destination != size:
                        w, h = size
                        while float(np.float32(w * h) * np.float32(0.25)) >= destination[0] * destination[1]:
                            w, h = max(1, w // 2), max(1, h // 2)
                            rgb = self.resize(rgb, (w, h))
                        rgb = self.resize(rgb, destination)
                        size = destination
                work_size = working_size(size, config.processing.effective_max_edge) if config.processing.resize else size
                work = self.resize(rgb, work_size, config.processing.working_resampler)
                params = config.parameters if size == work_size else replace(config.parameters, fade=0)
                numerics = (config.numerics if config.video.gpu_iterations is None else
                            replace(config.numerics, iterations=config.video.gpu_iterations))
                output = self.core(work, int(config.filter[-1]), params, numerics,
                                   source_cleanup=source_cleanup,
                                   lighting_scale=config.video.lighting_scale)
                if size != work_size:
                    for intermediate in enlargement_sizes(work_size, size):
                        output = self.resize(output, intermediate)
                    mixed = cp.empty_like(rgb)
                    self.launch("fade_rgb", rgb.shape, rgb, output, mixed, np.int32(rgb.size),
                                np.float32(config.parameters.fade) / np.float32(100))
                    output = mixed
                return output.get(stream=self.stream)
        except Exception as error:
            raise GPUUnavailable(f"CUDA frame filtering failed: {error}") from error
