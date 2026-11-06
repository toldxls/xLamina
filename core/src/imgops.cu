#include "stackcore/imgops.h"

#include <cuda_fp16.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace stackcore {

#define CK(x)                                                                                        \
    do {                                                                                             \
        cudaError_t e_ = (x);                                                                        \
        if (e_ != cudaSuccess) throw std::runtime_error(std::string(#x) + ": " + cudaGetErrorString(e_)); \
    } while (0)

namespace {

__constant__ float c_luma[3] = {0.2126f, 0.7152f, 0.0722f};

__device__ __forceinline__ int refl101(int p, int n) {   // BORDER_REFLECT_101
    if (n == 1) return 0;
    while ((unsigned)p >= (unsigned)n) p = p < 0 ? -p : 2 * n - 2 - p;
    return p;
}

inline unsigned blocks1(size_t n) { return (unsigned)((n + 255) / 256); }
inline dim3 grid2(int w, int h) { return dim3((w + 31) / 32, (h + 7) / 8); }
const dim3 BLK(32, 8);

__global__ void k_luma(const float* __restrict__ rgb, float* __restrict__ y, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* p = rgb + i * 3;
    y[i] = p[0] * c_luma[0] + p[1] * c_luma[1] + p[2] * c_luma[2];
}

// separable Gaussian: row filter left to right, column filter centre then symmetric pairs (cv::sepFilter2D)
__global__ void k_blur_row(const float* __restrict__ in, float* __restrict__ out, int w, int h, const float* __restrict__ k, int R) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const float* row = in + (size_t)y * w;
    float s = 0.f;
    for (int i = -R; i <= R; ++i) s += k[i + R] * row[refl101(x + i, w)];
    out[(size_t)y * w + x] = s;
}

__global__ void k_blur_col(const float* __restrict__ in, float* __restrict__ out, int w, int h, const float* __restrict__ k, int R) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float s = k[R] * in[(size_t)y * w + x];
    for (int i = 1; i <= R; ++i)
        s += k[R + i] * (in[(size_t)refl101(y + i, h) * w + x] + in[(size_t)refl101(y - i, h) * w + x]);
    out[(size_t)y * w + x] = s;
}

// box: plain sums (OpenCV's sliding sums differ by rounding only); normalised in double like ColumnSum
__global__ void k_box_row(const float* __restrict__ in, float* __restrict__ out, int w, int h, int R) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const float* row = in + (size_t)y * w;
    float s = 0.f;
    for (int i = -R; i <= R; ++i) s += row[refl101(x + i, w)];
    out[(size_t)y * w + x] = s;
}

__global__ void k_box_col(const float* __restrict__ in, float* __restrict__ out, int w, int h, int R, double scale) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float s = 0.f;
    for (int i = -R; i <= R; ++i) s += in[(size_t)refl101(y + i, h) * w + x];
    out[(size_t)y * w + x] = (float)((double)s * scale);
}

template <bool SQ>
__global__ void k_lap3(const float* __restrict__ in, float* __restrict__ out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int xm = refl101(x - 1, w), xp = refl101(x + 1, w), ym = refl101(y - 1, h), yp = refl101(y + 1, h);
    const float* r0 = in + (size_t)ym * w;
    const float* r1 = in + (size_t)y * w;
    const float* r2 = in + (size_t)yp * w;
    float l = 2.f * r0[xm] + 2.f * r0[xp] - 8.f * r1[x] + 2.f * r2[xm] + 2.f * r2[xp];
    out[(size_t)y * w + x] = SQ ? l * l : fabsf(l);
}

__global__ void k_pyrdown3(const float* __restrict__ in, int w, int h, float* __restrict__ out, int w2, int h2) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w2 || y >= h2) return;
    int xs[5], ys[5];
#pragma unroll
    for (int i = 0; i < 5; ++i) {
        xs[i] = refl101(2 * x + i - 2, w);
        ys[i] = refl101(2 * y + i - 2, h);
    }
#pragma unroll
    for (int c = 0; c < 3; ++c) {
        float r[5];
#pragma unroll
        for (int i = 0; i < 5; ++i) {
            const float* p = in + (size_t)ys[i] * w * 3 + c;
            r[i] = p[xs[0] * 3] + p[xs[4] * 3] + 4.f * (p[xs[1] * 3] + p[xs[3] * 3]) + 6.f * p[xs[2] * 3];
        }
        float s = r[2] * 6.f + (r[1] + r[3]) * 4.f + r[0] + r[4];
        out[((size_t)y * w2 + x) * 3 + c] = s * (1.f / 256.f);
    }
}

// cv::pyrUp value at (x, y) of the dw x dh output from the sw x sh source: reflect-101 at the
// top/left, replicate-style 7/8 weights at the right and bottom (FINDINGS 18)
__device__ __forceinline__ float pyrup_at(const float* __restrict__ gn, int sw, int sh, int x, int y, int c) {
    int xi = x >> 1, yi = y >> 1;
    bool xo = x & 1, yo = y & 1;
    bool right = xi == sw - 1, bottom = yi == sh - 1;
    int xa = refl101(xi - 1, sw), xb = refl101(xi + 1, sw), ya = refl101(yi - 1, sh), yb = refl101(yi + 1, sh);
    xi = refl101(xi, sw);
    yi = refl101(yi, sh);
    auto rowv = [&](int yy) {
        const float* p = gn + (size_t)yy * sw * 3 + c;
        if (right) return xo ? p[xi * 3] * 8.f : p[xa * 3] + p[xi * 3] * 7.f;
        return xo ? (p[xi * 3] + p[xb * 3]) * 4.f : p[xa * 3] + p[xb * 3] + 6.f * p[xi * 3];
    };
    float v;
    if (bottom) v = yo ? rowv(yi) * 8.f : rowv(ya) + rowv(yi) * 7.f;
    else v = yo ? (rowv(yi) + rowv(yb)) * 4.f : rowv(ya) + rowv(yb) + 6.f * rowv(yi);
    return v * (1.f / 64.f);
}

__global__ void k_pyrup3(const float* __restrict__ gn, int sw, int sh, float* __restrict__ out, int dw, int dh) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || y >= dh) return;
#pragma unroll
    for (int c = 0; c < 3; ++c) out[((size_t)y * dw + x) * 3 + c] = pyrup_at(gn, sw, sh, x, y, c);
}

__global__ void k_lap_level3(const float* __restrict__ g, const float* __restrict__ gn, int sw, int sh,
                             float* __restrict__ lap, int dw, int dh) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || y >= dh) return;
#pragma unroll
    for (int c = 0; c < 3; ++c) {
        size_t o = ((size_t)y * dw + x) * 3 + c;
        lap[o] = g[o] - pyrup_at(gn, sw, sh, x, y, c);
    }
}

__global__ void k_salience(const float* __restrict__ coef, int w, int h, float* __restrict__ sal) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float s = 0.f;
#pragma unroll
    for (int dy = -1; dy <= 1; ++dy) {
        int yy = refl101(y + dy, h);
#pragma unroll
        for (int dx = -1; dx <= 1; ++dx) {
            const float* p = coef + ((size_t)yy * w + refl101(x + dx, w)) * 3;
            float l = p[0] * c_luma[0] + p[1] * c_luma[1] + p[2] * c_luma[2];
            s += l * l;
        }
    }
    sal[(size_t)y * w + x] = (float)((double)s * (1.0 / 9.0));
}

template <int R>
__global__ void k_median(const float* __restrict__ in, float* __restrict__ out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const int N = (2 * R + 1) * (2 * R + 1);
    float v[N];
    int n = 0;
    for (int dy = -R; dy <= R; ++dy)
        for (int dx = -R; dx <= R; ++dx)
            v[n++] = in[(size_t)min(max(y + dy, 0), h - 1) * w + min(max(x + dx, 0), w - 1)];
    for (int a = 1; a < N; ++a) {
        float t = v[a];
        int b = a - 1;
        while (b >= 0 && v[b] > t) { v[b + 1] = v[b]; --b; }
        v[b + 1] = t;
    }
    out[(size_t)y * w + x] = v[N / 2];
}

__global__ void k_resize_linear(const float* __restrict__ src, int sw, int sh, float* __restrict__ dst, int dw, int dh,
                                double scale_x, double scale_y) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || y >= dh) return;
    float fx = (float)((x + 0.5) * scale_x - 0.5), fy = (float)((y + 0.5) * scale_y - 0.5);
    int sx = (int)floorf(fx), sy = (int)floorf(fy);
    fx -= sx; fy -= sy;
    if (sx < 0) { fx = 0.f; sx = 0; }
    if (sx >= sw - 1) { fx = 0.f; sx = sw - 1; }
    if (sy < 0) { fy = 0.f; sy = 0; }
    if (sy >= sh - 1) { fy = 0.f; sy = sh - 1; }
    int sx1 = min(sx + 1, sw - 1), sy1 = min(sy + 1, sh - 1);
    float a0 = 1.f - fx, a1 = fx, b0 = 1.f - fy, b1 = fy;
    const float* r0 = src + (size_t)sy * sw;
    const float* r1 = src + (size_t)sy1 * sw;
    float row0 = r0[sx] * a0 + r0[sx1] * a1;
    float row1 = r1[sx] * a0 + r1[sx1] * a1;
    dst[(size_t)y * dw + x] = row0 * b0 + row1 * b1;
}

__global__ void k_mul(const float* a, const float* b, float* out, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * b[i];
}
__global__ void k_add(const float* a, const float* b, float* out, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}
__global__ void k_h2f(const __half* a, float* out, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half2float(a[i]);
}
__global__ void k_f2h(const float* a, __half* out, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half_rn(a[i]);
}
__global__ void k_relu_sqrt(float* a, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = sqrtf(fmaxf(a[i], 0.f));
}
__global__ void k_clip01(float* a, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = fminf(fmaxf(a[i], 0.f), 1.f);
}

// INTER_AREA fast path (integer factors): cells summed rows outer, columns inner, then * 1/(fx fy)
template <int CH>
__global__ void k_area_fast(const float* __restrict__ src, int sw, int sh, float* __restrict__ dst, int dw, int dh, int fx, int fy) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || y >= dh) return;
    float scale = 1.f / (float)(fx * fy);
#pragma unroll
    for (int c = 0; c < CH; ++c) {
        float s = 0.f;
        for (int j = 0; j < fy; ++j) {
            const float* p = src + ((size_t)(y * fy + j) * sw + (size_t)x * fx) * CH + c;
            for (int i = 0; i < fx; ++i) s += p[i * CH];
        }
        dst[((size_t)y * dw + x) * CH + c] = s * scale;
    }
}

// INTER_AREA general path (cv::resizeArea_): per source row a weighted x-sum, rows weighted and summed
template <int CH>
__global__ void k_area_general(const float* __restrict__ src, int sw, int sh, float* __restrict__ dst, int dw, int dh,
                               const int* __restrict__ xoff, const int* __restrict__ xidx, const float* __restrict__ xw,
                               const int* __restrict__ yoff, const int* __restrict__ yidx, const float* __restrict__ yw) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || y >= dh) return;
    int x0 = xoff[x], x1 = xoff[x + 1], y0 = yoff[y], y1 = yoff[y + 1];
#pragma unroll
    for (int c = 0; c < CH; ++c) {
        float sum = 0.f;
        for (int j = y0; j < y1; ++j) {
            const float* row = src + (size_t)yidx[j] * sw * CH + c;
            float buf = 0.f;
            for (int i = x0; i < x1; ++i) buf += xw[i] * row[(size_t)xidx[i] * CH];
            sum += yw[j] * buf;
        }
        dst[((size_t)y * dw + x) * CH + c] = sum;
    }
}

// cv::computeResizeAreaTab
void area_tab(int ssize, int dsize, double scale, std::vector<int>& off, std::vector<int>& idx, std::vector<float>& wt) {
    off.assign(dsize + 1, 0);
    for (int dx = 0; dx < dsize; ++dx) {
        double fsx1 = dx * scale, fsx2 = fsx1 + scale;
        double cellWidth = std::min(scale, ssize - fsx1);
        int sx1 = (int)std::ceil(fsx1), sx2 = (int)std::floor(fsx2);
        sx2 = std::min(sx2, ssize - 1);
        sx1 = std::min(sx1, sx2);
        if (sx1 - fsx1 > 1e-3) {
            idx.push_back(sx1 - 1);
            wt.push_back((float)((sx1 - fsx1) / cellWidth));
        }
        for (int sx = sx1; sx < sx2; ++sx) {
            idx.push_back(sx);
            wt.push_back((float)(1.0 / cellWidth));
        }
        if (fsx2 - sx2 > 1e-3 && sx2 < ssize) {
            idx.push_back(sx2);
            wt.push_back((float)(std::min(std::min(fsx2 - sx2, 1.), cellWidth) / cellWidth));
        }
        off[dx + 1] = (int)idx.size();
    }
}

template <class T>
T* up(const std::vector<T>& v) {
    T* d = nullptr;
    CK(cudaMalloc(&d, v.size() * sizeof(T)));
    CK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

}  // namespace

// ----------------------------------------------------------------------------- host API

std::vector<float> gaussian_kernel(int n, double sigma) {
    std::vector<float> cf(n);
    double scale2X = -0.5 / (sigma * sigma), sum = 0;
    for (int i = 0; i < n; ++i) {
        double x = i - (n - 1) * 0.5;
        cf[i] = (float)std::exp(scale2X * x * x);
        sum += cf[i];
    }
    sum = 1. / sum;
    for (int i = 0; i < n; ++i) cf[i] = (float)(cf[i] * sum);
    return cf;
}

int blur_radius(double sigma) {
    int ks = (int)std::nearbyint(sigma * 8 + 1) | 1;
    return ks / 2;
}

void DevGaussian::build(double sigma) {
    R = blur_radius(sigma);
    auto k = gaussian_kernel(2 * R + 1, sigma);
    d_k = up(k);
}

void DevGaussian::free() {
    if (d_k) cudaFree(d_k);
    d_k = nullptr;
}

void gaussian_blur(const float* d_in, float* d_out, float* d_tmp, int w, int h, const DevGaussian& g, cudaStream_t s) {
    k_blur_row<<<grid2(w, h), BLK, 0, s>>>(d_in, d_tmp, w, h, g.d_k, g.R);
    k_blur_col<<<grid2(w, h), BLK, 0, s>>>(d_tmp, d_out, w, h, g.d_k, g.R);
}

void box_filter(const float* d_in, float* d_out, float* d_tmp, int w, int h, int r, cudaStream_t s) {
    k_box_row<<<grid2(w, h), BLK, 0, s>>>(d_in, d_tmp, w, h, r);
    k_box_col<<<grid2(w, h), BLK, 0, s>>>(d_tmp, d_out, w, h, r, 1.0 / ((2.0 * r + 1) * (2.0 * r + 1)));
}

void luma(const float* d_rgb, float* d_y, size_t n, cudaStream_t s) { k_luma<<<blocks1(n), 256, 0, s>>>(d_rgb, d_y, n); }

void laplacian3_sq(const float* d_in, float* d_out, int w, int h, cudaStream_t s) {
    k_lap3<true><<<grid2(w, h), BLK, 0, s>>>(d_in, d_out, w, h);
}
void laplacian3_abs(const float* d_in, float* d_out, int w, int h, cudaStream_t s) {
    k_lap3<false><<<grid2(w, h), BLK, 0, s>>>(d_in, d_out, w, h);
}

void pyr_down_rgb(const float* d_in, int w, int h, float* d_out, int w2, int h2, cudaStream_t s) {
    k_pyrdown3<<<grid2(w2, h2), BLK, 0, s>>>(d_in, w, h, d_out, w2, h2);
}
void pyr_up_rgb(const float* d_in, int sw, int sh, float* d_out, int dw, int dh, cudaStream_t s) {
    k_pyrup3<<<grid2(dw, dh), BLK, 0, s>>>(d_in, sw, sh, d_out, dw, dh);
}
void lap_level_rgb(const float* d_g, const float* d_gn, int sw, int sh, float* d_lap, int dw, int dh, cudaStream_t s) {
    k_lap_level3<<<grid2(dw, dh), BLK, 0, s>>>(d_g, d_gn, sw, sh, d_lap, dw, dh);
}

void salience3(const float* d_coef, int w, int h, float* d_sal, cudaStream_t s) {
    k_salience<<<grid2(w, h), BLK, 0, s>>>(d_coef, w, h, d_sal);
}
void median3(const float* d_in, float* d_out, int w, int h, cudaStream_t s) { k_median<1><<<grid2(w, h), BLK, 0, s>>>(d_in, d_out, w, h); }
void median5(const float* d_in, float* d_out, int w, int h, cudaStream_t s) { k_median<2><<<grid2(w, h), BLK, 0, s>>>(d_in, d_out, w, h); }
void resize_linear(const float* d_src, int sw, int sh, float* d_dst, int dw, int dh, cudaStream_t s) {
    k_resize_linear<<<grid2(dw, dh), BLK, 0, s>>>(d_src, sw, sh, d_dst, dw, dh, (double)sw / dw, (double)sh / dh);
}
float percentile(const float* d_in, float* d_scratch, size_t n, double p, cudaStream_t s) {
    CK(cudaMemcpyAsync(d_scratch, d_in, n * sizeof(float), cudaMemcpyDeviceToDevice, s));
    thrust::sort(thrust::cuda::par.on(s), thrust::device_ptr<float>(d_scratch), thrust::device_ptr<float>(d_scratch + n));
    double idx = p / 100.0 * (double)(n - 1);
    size_t lo = (size_t)std::floor(idx);
    size_t hi = std::min(lo + 1, n - 1);
    float v[2];
    CK(cudaMemcpyAsync(&v[0], d_scratch + lo, sizeof(float), cudaMemcpyDeviceToHost, s));
    CK(cudaMemcpyAsync(&v[1], d_scratch + hi, sizeof(float), cudaMemcpyDeviceToHost, s));
    CK(cudaStreamSynchronize(s));
    double frac = idx - lo;
    return (float)(v[0] + frac * ((double)v[1] - v[0]));
}
void mul(const float* a, const float* b, float* out, size_t n, cudaStream_t s) { k_mul<<<blocks1(n), 256, 0, s>>>(a, b, out, n); }
void add(const float* a, const float* b, float* out, size_t n, cudaStream_t s) { k_add<<<blocks1(n), 256, 0, s>>>(a, b, out, n); }
void f16_to_f32(const void* a, float* out, size_t n, cudaStream_t s) { k_h2f<<<blocks1(n), 256, 0, s>>>((const __half*)a, out, n); }
void f32_to_f16(const float* a, void* out, size_t n, cudaStream_t s) { k_f2h<<<blocks1(n), 256, 0, s>>>(a, (__half*)out, n); }
void relu_sqrt(float* a, size_t n, cudaStream_t s) { k_relu_sqrt<<<blocks1(n), 256, 0, s>>>(a, n); }
void clip01(float* a, size_t n, cudaStream_t s) { k_clip01<<<blocks1(n), 256, 0, s>>>(a, n); }

void AreaResizer::build(int sw_, int sh_, int dw_, int dh_) {
    sw = sw_; sh = sh_; dw = dw_; dh = dh_;
    double scale_x = (double)sw / dw, scale_y = (double)sh / dh;
    int ix = (int)std::nearbyint(scale_x), iy = (int)std::nearbyint(scale_y);   // saturate_cast<int> rounds
    fast_ = std::abs(scale_x - ix) < 2.2e-16 && std::abs(scale_y - iy) < 2.2e-16;
    if (fast_) { fx_ = ix; fy_ = iy; return; }
    std::vector<int> xoff, xidx, yoff, yidx;
    std::vector<float> xw, yw;
    area_tab(sw, dw, scale_x, xoff, xidx, xw);
    area_tab(sh, dh, scale_y, yoff, yidx, yw);
    d_xoff_ = up(xoff); d_xidx_ = up(xidx); d_xw_ = up(xw);
    d_yoff_ = up(yoff); d_yidx_ = up(yidx); d_yw_ = up(yw);
}

void AreaResizer::free() {
    for (auto p : {d_xoff_, d_xidx_, d_yoff_, d_yidx_}) if (p) cudaFree(p);
    for (auto p : {d_xw_, d_yw_}) if (p) cudaFree(p);
    d_xoff_ = d_xidx_ = d_yoff_ = d_yidx_ = nullptr;
    d_xw_ = d_yw_ = nullptr;
}

void AreaResizer::run(const float* d_src, float* d_dst, int ch, cudaStream_t s) const {
    if (fast_) {
        if (ch == 3) k_area_fast<3><<<grid2(dw, dh), BLK, 0, s>>>(d_src, sw, sh, d_dst, dw, dh, fx_, fy_);
        else k_area_fast<1><<<grid2(dw, dh), BLK, 0, s>>>(d_src, sw, sh, d_dst, dw, dh, fx_, fy_);
    } else {
        if (ch == 3) k_area_general<3><<<grid2(dw, dh), BLK, 0, s>>>(d_src, sw, sh, d_dst, dw, dh, d_xoff_, d_xidx_, d_xw_, d_yoff_, d_yidx_, d_yw_);
        else k_area_general<1><<<grid2(dw, dh), BLK, 0, s>>>(d_src, sw, sh, d_dst, dw, dh, d_xoff_, d_xidx_, d_xw_, d_yoff_, d_yidx_, d_yw_);
    }
}

}  // namespace stackcore
