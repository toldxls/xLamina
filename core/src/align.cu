#include "stackcore/align.h"

#include <cuComplex.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

#include "stackcore/imgops.h"

namespace stackcore {

#define CK(x)                                                                                        \
    do {                                                                                             \
        cudaError_t e_ = (x);                                                                        \
        if (e_ != cudaSuccess) throw std::runtime_error(std::string(#x) + ": " + cudaGetErrorString(e_)); \
    } while (0)
namespace {

typedef cuFloatComplex cpx;

inline unsigned blocks1(size_t n) { return (unsigned)((n + 255) / 256); }
inline dim3 grid2(int w, int h) { return dim3((w + 31) / 32, (h + 7) / 8); }
const dim3 BLK(32, 8);
const int NSUM = 36;   // 21 JtJ + 6 Jtr + rr + n + s0 s1 s00 s11 s01 + (pad)

// cv::Sobel ksize 1, scale 0.5, BORDER_REPLICATE: central differences
__global__ void k_grad(const float* __restrict__ I, float* __restrict__ gx, float* __restrict__ gy, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int xm = max(x - 1, 0), xp = min(x + 1, w - 1), ym = max(y - 1, 0), yp = min(y + 1, h - 1);
    gx[(size_t)y * w + x] = 0.5f * (I[(size_t)y * w + xp] - I[(size_t)y * w + xm]);
    gy[(size_t)y * w + x] = 0.5f * (I[(size_t)yp * w + x] - I[(size_t)ym * w + x]);
}

__device__ __forceinline__ void cubic_weights(float x, float* c) {   // OpenCV interpolateCubic, A = -0.75
    const float A = -0.75f;
    c[0] = ((A * (x + 1) - 5 * A) * (x + 1) + 8 * A) * (x + 1) - 4 * A;
    c[1] = ((A + 2) * x - (A + 3)) * x * x + 1;
    c[2] = ((A + 2) * (1 - x) - (A + 3)) * (1 - x) * (1 - x) + 1;
    c[3] = 1.f - c[0] - c[1] - c[2];
}

// cv2.warpAffine INTER_CUBIC, BORDER_REPLICATE, exact positions; single channel
__global__ void k_warp_cv(const float* __restrict__ src, int w, int h, float* __restrict__ dst,
                          double m0, double m1, double m2, double m3, double m4, double m5) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    double u = m0 * x + m1 * y + m2, v = m3 * x + m4 * y + m5;
    double fu = floor(u), fv = floor(v);
    int sx = (int)fu - 1, sy = (int)fv - 1;
    float wx[4], wy[4];
    cubic_weights((float)(u - fu), wx);
    cubic_weights((float)(v - fv), wy);
    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        int yy = min(max(sy + i, 0), h - 1);
        const float* row = src + (size_t)yy * w;
        float r = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) r += wx[j] * row[min(max(sx + j, 0), w - 1)];
        sum += wy[i] * r;
    }
    dst[(size_t)y * w + x] = sum;
}

__device__ __forceinline__ int mirror_idx(int i, int n) {   // scipy mode="mirror": d c b | a b c d | c b a
    if (n <= 1) return 0;
    int sz2 = 2 * n - 2;
    if (i < 0) i = -i;
    i %= sz2;
    return i < n ? i : sz2 - i;
}

// scipy map_coordinates order 3, prefilter=False, mode mirror, on the coefficient image
__global__ void k_warp_bspline(const float* __restrict__ coef, int w, int h, float* __restrict__ dst,
                               double m0, double m1, double m2, double m3, double m4, double m5) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    double u = m0 * x + m1 * y + m2, v = m3 * x + m4 * y + m5;
    // scipy maps an out-of-range coordinate by mirroring it first
    auto mirror_c = [](double c, int n) {
        if (n <= 1) return 0.0;
        double sz2 = 2.0 * n - 2.0;
        if (c < 0) { c = -c; }
        c = fmod(c, sz2);
        return c < n - 1 ? c : sz2 - c;
    };
    if (u < 0 || u > w - 1) u = mirror_c(u, w);
    if (v < 0 || v > h - 1) v = mirror_c(v, h);
    double fu = floor(u), fv = floor(v);
    double tu = u - fu, tv = v - fv;
    int su = (int)fu - 1, sv = (int)fv - 1;
    double wu[4], wv[4];
    auto bsp = [](double t, double* wgt) {
        double t2 = t * t, t3 = t2 * t;
        wgt[0] = (1 - 3 * t + 3 * t2 - t3) / 6.0;
        wgt[1] = (4 - 6 * t2 + 3 * t3) / 6.0;
        wgt[2] = (1 + 3 * t + 3 * t2 - 3 * t3) / 6.0;
        wgt[3] = t3 / 6.0;
    };
    bsp(tu, wu);
    bsp(tv, wv);
    double sum = 0.0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float* row = coef + (size_t)mirror_idx(sv + i, h) * w;
        double r = 0.0;
#pragma unroll
        for (int j = 0; j < 4; ++j) r += wu[j] * row[mirror_idx(su + j, w)];
        sum += wv[i] * r;
    }
    dst[(size_t)y * w + x] = (float)sum;
}

// scipy spline_filter(order 3, mode mirror) along one axis: one thread per line, the whole line in
// double like scipy's buffer (ni_splines.c: gain, mirror causal init, causal pass, mirror anticausal
// init, anticausal pass). Lines up to 4096 samples.
__global__ void k_bspline_prefilter(float* __restrict__ img, int w, int h, int along_x) {
    int line = blockIdx.x * blockDim.x + threadIdx.x;
    int n = along_x ? w : h;
    int lines = along_x ? h : w;
    if (line >= lines || n > 4096) return;
    size_t stride = along_x ? 1 : (size_t)w;
    float* src = img + (along_x ? (size_t)line * w : (size_t)line);
    double c[4096];
    const double z = -0.2679491924311227;   // sqrt(3) - 2
    const double gain = (1.0 - z) * (1.0 - 1.0 / z);   // 6
    for (int i = 0; i < n; ++i) c[i] = (double)src[(size_t)i * stride] * gain;
    // causal initial coefficient for mirror boundaries (Unser): the series is truncated at the
    // precision horizon, which also avoids z^(n-1) / z^i underflowing to 0 / 0 on long lines
    int horizon = (int)ceil(log(1e-15) / log(fabs(z)));
    double sum;
    if (horizon < n) {
        double zn = z;
        sum = c[0];
        for (int i = 1; i < horizon; ++i) { sum += zn * c[i]; zn *= z; }
    } else {
        double zn = z, iz = 1.0 / z, z2n = pow(z, (double)(n - 1));
        sum = c[0] + z2n * c[n - 1];
        z2n *= z2n * iz;
        for (int i = 1; i <= n - 2; ++i) { sum += (zn + z2n) * c[i]; zn *= z; z2n *= iz; }
        sum /= (1.0 - zn * zn);
    }
    c[0] = sum;
    for (int i = 1; i < n; ++i) c[i] += z * c[i - 1];
    c[n - 1] = z / (z * z - 1.0) * (z * c[n - 2] + c[n - 1]);
    for (int i = n - 2; i >= 0; --i) c[i] = z * (c[i + 1] - c[i]);
    for (int i = 0; i < n; ++i) src[(size_t)i * stride] = (float)c[i];
}

// per-pixel Jacobian terms and residual, reduced per block into partials[block][NSUM]
__global__ void k_gn_reduce(const float* __restrict__ I0, const float* __restrict__ gx0, const float* __restrict__ gy0,
                            const float* __restrict__ I1w, const float* __restrict__ gx1, const float* __restrict__ gy1,
                            int w, int h, double m0, double m1, double m2, double m3, double m4, double m5, float g, float beta,
                            float cx, float cy, float R, double* __restrict__ partials) {
    __shared__ double red[8][NSUM];
    size_t n = (size_t)w * h;
    double acc[NSUM];
    for (int q = 0; q < NSUM; ++q) acc[q] = 0.0;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        int x = (int)(i % w), y = (int)(i / w);
        float X = (float)(m0 * x + m1 * y + m2), Y = (float)(m3 * x + m4 * y + m5);
        if (!(X >= 3.f && X <= w - 4.f && Y >= 3.f && Y <= h - 4.f)) continue;
        float Gx = 0.5f * (g * gx1[i] + gx0[i]), Gy = 0.5f * (g * gy1[i] + gy0[i]);
        float dx = (x - cx) / R, dy = (y - cy) / R;
        float Iw = I1w[i];
        float J[6] = {Gx * dx + Gy * dy, -Gx * dy + Gy * dx, Gx, Gy, Iw, 1.0f};
        float r = g * Iw + beta - I0[i];
        int q = 0;
        for (int a = 0; a < 6; ++a)
            for (int b = a; b < 6; ++b) acc[q++] += (double)J[a] * J[b];
        for (int a = 0; a < 6; ++a) acc[q++] += (double)J[a] * r;
        acc[q++] += (double)r * r;
        acc[q++] += 1.0;
        float a0 = I0[i];
        acc[q++] += a0; acc[q++] += Iw; acc[q++] += (double)a0 * a0; acc[q++] += (double)Iw * Iw; acc[q++] += (double)a0 * Iw;
    }
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int q = 0; q < NSUM; ++q) {
        double v = acc[q];
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
        if (lane == 0) red[warp][q] = v;
    }
    __syncthreads();
    if (threadIdx.x < NSUM) {
        double v = 0.0;
        int nw = blockDim.x >> 5;
        for (int wv = 0; wv < nw; ++wv) v += red[wv][threadIdx.x];
        partials[(size_t)blockIdx.x * NSUM + threadIdx.x] = v;
    }
}

__global__ void k_final_reduce(const double* __restrict__ partials, int nblocks, double* __restrict__ out) {
    int q = threadIdx.x;
    if (q >= NSUM) return;
    double v = 0.0;
    for (int b = 0; b < nblocks; ++b) v += partials[(size_t)b * NSUM + q];   // fixed order: deterministic
    out[q] = v;
}

// Hanning window times image into a zero-padded complex array (cv::phaseCorrelate's padding)
__global__ void k_window_pad(const float* __restrict__ I, int w, int h, cpx* __restrict__ out, int pw, int ph) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= pw || y >= ph) return;
    float v = 0.f;
    if (x < w && y < h) {
        // cv::createHanningWindow: wr = 0.5 (1 - cos(2 pi i / (w - 1))), same for rows
        float wx = 0.5f * (1.f - cosf(2.f * 3.14159265358979f * x / (w - 1)));
        float wy = 0.5f * (1.f - cosf(2.f * 3.14159265358979f * y / (h - 1)));
        v = I[(size_t)y * w + x] * wx * wy;
    }
    out[(size_t)y * pw + x] = make_cuFloatComplex(v, 0.f);
}
// DFT along rows (out[y][k] = sum_x in[y][x] T[x][k]) and along columns, with a twiddle table T (n x n)
__global__ void k_dft_rows(const cpx* __restrict__ in, const cpx* __restrict__ T, cpx* __restrict__ out, int pw, int ph) {
    int k = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (k >= pw || y >= ph) return;
    cpx acc = make_cuFloatComplex(0.f, 0.f);
    const cpx* row = in + (size_t)y * pw;
    for (int x = 0; x < pw; ++x) acc = cuCaddf(acc, cuCmulf(row[x], T[(size_t)x * pw + k]));
    out[(size_t)y * pw + k] = acc;
}
__global__ void k_dft_cols(const cpx* __restrict__ in, const cpx* __restrict__ T, cpx* __restrict__ out, int pw, int ph) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, k = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= pw || k >= ph) return;
    cpx acc = make_cuFloatComplex(0.f, 0.f);
    for (int y = 0; y < ph; ++y) acc = cuCaddf(acc, cuCmulf(in[(size_t)y * pw + x], T[(size_t)y * ph + k]));
    out[(size_t)k * pw + x] = acc;
}
// cross power spectrum P = F1 conj(F2) / |F1 conj(F2)|, in place into a
__global__ void k_cross_power(cpx* a, const cpx* b, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    cpx p = cuCmulf(a[i], cuConjf(b[i]));
    float mag = sqrtf(p.x * p.x + p.y * p.y);
    if (mag > 0.f) { p.x /= mag; p.y /= mag; }
    a[i] = p;
}
__global__ void k_real_shift(const cpx* c, float* out, int pw, int ph) {   // real part, fftShift
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= pw || y >= ph) return;
    int sx = (x + pw / 2) % pw, sy = (y + ph / 2) % ph;    // out(x, y) = c(x - pw/2, y - ph/2) wrapped
    out[(size_t)y * pw + x] = c[(size_t)sy * pw + sx].x / (float)(pw * ph);
}

std::vector<cpx> twiddles(int n, bool inverse) {   // T[x][k] = exp(-+ 2 pi i x k / n)
    std::vector<cpx> T((size_t)n * n);
    for (int x = 0; x < n; ++x)
        for (int k = 0; k < n; ++k) {
            double ang = (inverse ? 2.0 : -2.0) * 3.14159265358979323846 * (double)((long long)x * k % n) / n;
            T[(size_t)x * n + k] = make_cuFloatComplex((float)std::cos(ang), (float)std::sin(ang));
        }
    return T;
}

int optimal_dft_size(int n) {   // smallest 2^a 3^b 5^c >= n (cv::getOptimalDFTSize)
    for (int m = n;; ++m) {
        int k = m;
        while (k % 2 == 0) k /= 2;
        while (k % 3 == 0) k /= 3;
        while (k % 5 == 0) k /= 5;
        if (k == 1) return m;
    }
}

// solve the 6x6 symmetric system (Gaussian elimination with partial pivoting)
bool solve6(double A[6][6], double b[6], double x[6]) {
    for (int c = 0; c < 6; ++c) {
        int p = c;
        for (int r = c + 1; r < 6; ++r) if (std::fabs(A[r][c]) > std::fabs(A[p][c])) p = r;
        if (std::fabs(A[p][c]) < 1e-300) return false;
        if (p != c) { for (int k = 0; k < 6; ++k) std::swap(A[c][k], A[p][k]); std::swap(b[c], b[p]); }
        for (int r = c + 1; r < 6; ++r) {
            double f = A[r][c] / A[c][c];
            for (int k = c; k < 6; ++k) A[r][k] -= f * A[c][k];
            b[r] -= f * b[c];
        }
    }
    for (int c = 5; c >= 0; --c) {
        double s = b[c];
        for (int k = c + 1; k < 6; ++k) s -= A[c][k] * x[k];
        x[c] = s / A[c][c];
    }
    return true;
}

bool invert6(double A[6][6], double inv[6][6]) {
    double M[6][12];
    for (int r = 0; r < 6; ++r) for (int c = 0; c < 6; ++c) { M[r][c] = A[r][c]; M[r][6 + c] = r == c ? 1.0 : 0.0; }
    for (int c = 0; c < 6; ++c) {
        int p = c;
        for (int r = c + 1; r < 6; ++r) if (std::fabs(M[r][c]) > std::fabs(M[p][c])) p = r;
        if (std::fabs(M[p][c]) < 1e-300) return false;
        if (p != c) for (int k = 0; k < 12; ++k) std::swap(M[c][k], M[p][k]);
        double d = M[c][c];
        for (int k = 0; k < 12; ++k) M[c][k] /= d;
        for (int r = 0; r < 6; ++r) if (r != c) { double f = M[r][c]; for (int k = 0; k < 12; ++k) M[r][k] -= f * M[c][k]; }
    }
    for (int r = 0; r < 6; ++r) for (int c = 0; c < 6; ++c) inv[r][c] = M[r][6 + c];
    return true;
}

// sim.to_affine for z = 1 + a + i b, t, at level size (w, h)
void to_affine(double a, double b, double tx, double ty, int w, int h, double M[6]) {
    double cx = (w - 1) / 2.0, cy = (h - 1) / 2.0, zr = 1 + a, zi = b;
    M[0] = zr; M[1] = -zi; M[2] = cx + tx - zr * cx + zi * cy;
    M[3] = zi; M[4] = zr;  M[5] = cy + ty - zi * cx - zr * cy;
}

}  // namespace

// ----------------------------------------------------------------------------- pyramid

void AlignPyramid::build(const float* d_luma, int w, int h, cudaStream_t s, int min_width) {
    Level l0;
    l0.w = w; l0.h = h;
    size_t n = (size_t)w * h;
    CK(cudaMalloc(&l0.I, n * sizeof(float)));
    CK(cudaMemcpyAsync(l0.I, d_luma, n * sizeof(float), cudaMemcpyDeviceToDevice, s));
    lv.push_back(l0);
    DevGaussian g08;
    g08.build(0.8);
    float* tmp;
    CK(cudaMalloc(&tmp, n * sizeof(float)));
    float* sm;
    CK(cudaMalloc(&sm, n * sizeof(float)));
    while (lv.back().w / 2 >= min_width) {
        Level& p = lv.back();
        Level q;
        q.w = p.w / 2; q.h = p.h / 2;
        q.fx = (double)q.w / w; q.fy = (double)q.h / h;
        gaussian_blur(p.I, sm, tmp, p.w, p.h, g08, s);
        CK(cudaMalloc(&q.I, (size_t)q.w * q.h * sizeof(float)));
        AreaResizer rz;
        rz.build(p.w, p.h, q.w, q.h);
        rz.run(sm, q.I, 1, s);
        CK(cudaStreamSynchronize(s));
        rz.free();
        lv.push_back(q);
    }
    for (auto& L : lv) {
        size_t m = (size_t)L.w * L.h;
        CK(cudaMalloc(&L.gx, m * sizeof(float)));
        CK(cudaMalloc(&L.gy, m * sizeof(float)));
        k_grad<<<grid2(L.w, L.h), BLK, 0, s>>>(L.I, L.gx, L.gy, L.w, L.h);
    }
    // level-0 B-spline coefficients (rows then columns, like spline_filter's axis loop)
    CK(cudaMalloc(&coef, n * sizeof(float)));
    CK(cudaMemcpyAsync(coef, l0.I, n * sizeof(float), cudaMemcpyDeviceToDevice, s));
    k_bspline_prefilter<<<blocks1(w), 256, 0, s>>>(coef, w, h, 0);   // scipy filters axis 0 (columns) first
    k_bspline_prefilter<<<blocks1(h), 256, 0, s>>>(coef, w, h, 1);
    CK(cudaStreamSynchronize(s));
    g08.free();
    cudaFree(tmp);
    cudaFree(sm);
}

void AlignPyramid::free() {
    for (auto& L : lv) { cudaFree(L.I); cudaFree(L.gx); cudaFree(L.gy); }
    lv.clear();
    cudaFree(coef);
    coef = nullptr;
}

void AlignWork::alloc(int w_, int h_) {
    w = w_; h = h_;
    size_t n = (size_t)w * h;
    CK(cudaMalloc(&I1w, n * sizeof(float)));
    CK(cudaMalloc(&gxw, n * sizeof(float)));
    CK(cudaMalloc(&gyw, n * sizeof(float)));
    max_blocks = 1024;
    CK(cudaMalloc(&partials, (size_t)(max_blocks + 1) * NSUM * sizeof(double)));
}

void AlignWork::free() {
    cudaFree(I1w); cudaFree(gxw); cudaFree(gyw); cudaFree(partials);
    for (void* p : {tw_fw, tw_fh, tw_iw, tw_ih, fft_a, fft_b, fft_t}) cudaFree(p);
    cudaFree(fft_c);
    tw_fw = tw_fh = tw_iw = tw_ih = fft_a = fft_b = fft_t = nullptr;
    fft_c = nullptr;
    fft_w = fft_h = 0;
}

// ----------------------------------------------------------------------------- pair

namespace {

// cv::phaseCorrelate(I0, I1, hanning): returns (dx, dy)
std::complex<double> phase_init(const AlignPyramid::Level& L0, const AlignPyramid::Level& L1, AlignWork& wk, cudaStream_t s) {
    int w = L0.w, h = L0.h;
    int pw = optimal_dft_size(w), ph = optimal_dft_size(h);
    if (wk.fft_w != pw || wk.fft_h != ph) {
        for (void* p : {wk.tw_fw, wk.tw_fh, wk.tw_iw, wk.tw_ih, wk.fft_a, wk.fft_b, wk.fft_t}) cudaFree(p);
        cudaFree(wk.fft_c);
        wk.fft_w = pw; wk.fft_h = ph;
        auto up = [](const std::vector<cpx>& v) { void* d; CK(cudaMalloc(&d, v.size() * sizeof(cpx))); CK(cudaMemcpy(d, v.data(), v.size() * sizeof(cpx), cudaMemcpyHostToDevice)); return d; };
        wk.tw_fw = up(twiddles(pw, false)); wk.tw_fh = up(twiddles(ph, false));
        wk.tw_iw = up(twiddles(pw, true));  wk.tw_ih = up(twiddles(ph, true));
        CK(cudaMalloc(&wk.fft_a, (size_t)pw * ph * sizeof(cpx)));
        CK(cudaMalloc(&wk.fft_b, (size_t)pw * ph * sizeof(cpx)));
        CK(cudaMalloc(&wk.fft_t, (size_t)pw * ph * sizeof(cpx)));
        CK(cudaMalloc(&wk.fft_c, (size_t)pw * ph * sizeof(float)));
    }
    cpx* A = (cpx*)wk.fft_a;
    cpx* B = (cpx*)wk.fft_b;
    cpx* T = (cpx*)wk.fft_t;
    auto dft2 = [&](cpx* X, bool inverse) {   // in place via T
        k_dft_rows<<<grid2(pw, ph), BLK, 0, s>>>(X, (const cpx*)(inverse ? wk.tw_iw : wk.tw_fw), T, pw, ph);
        k_dft_cols<<<grid2(pw, ph), BLK, 0, s>>>(T, (const cpx*)(inverse ? wk.tw_ih : wk.tw_fh), X, pw, ph);
    };
    k_window_pad<<<grid2(pw, ph), BLK, 0, s>>>(L0.I, w, h, A, pw, ph);
    k_window_pad<<<grid2(pw, ph), BLK, 0, s>>>(L1.I, w, h, B, pw, ph);
    dft2(A, false);
    dft2(B, false);
    k_cross_power<<<blocks1((size_t)pw * ph), 256, 0, s>>>(A, B, (size_t)pw * ph);
    dft2(A, true);
    k_real_shift<<<grid2(pw, ph), BLK, 0, s>>>(A, wk.fft_c, pw, ph);
    std::vector<float> C((size_t)pw * ph);
    CK(cudaMemcpyAsync(C.data(), wk.fft_c, C.size() * sizeof(float), cudaMemcpyDeviceToHost, s));
    CK(cudaStreamSynchronize(s));
    // peak, then the 5x5 weighted centroid around it (cv::weightedCentroid)
    size_t best = 0;
    for (size_t i = 1; i < C.size(); ++i) if (C[i] > C[best]) best = i;
    int px = (int)(best % pw), py = (int)(best / pw);
    int x0 = std::max(px - 2, 0), x1 = std::min(px + 3, pw), y0 = std::max(py - 2, 0), y1 = std::min(py + 3, ph);
    double sumx = 0, sumy = 0, sum = 0;
    for (int y = y0; y < y1; ++y)
        for (int x = x0; x < x1; ++x) {
            double v = C[(size_t)y * pw + x];
            sumx += x * v; sumy += y * v; sum += v;
        }
    double tx = sum != 0 ? sumx / sum : px, ty = sum != 0 ? sumy / sum : py;
    return {pw / 2.0 - tx, ph / 2.0 - ty};   // center - t
}

}  // namespace

PairResult align_pair(const AlignPyramid& pi, const AlignPyramid& pj, int i, int j, AlignWork& wk, const AlignParams& prm,
                      cudaStream_t s) {
    PairResult res;
    res.i = i; res.j = j;
    int nl = (int)pi.lv.size(), top = nl - 1;
    std::complex<double> t0 = phase_init(pi.lv[top], pj.lv[top], wk, s);
    double p[6] = {0.0, 0.0, t0.real(), t0.imag(), 1.0, 0.0};
    static const bool dbg = getenv("STACKER_ALIGN_DEBUG") != nullptr;
    if (dbg) printf("pair %d:%d phase init t = (%.4f, %.4f) at level %d (%dx%d)\n", i, j, t0.real(), t0.imag(), top, pi.lv[top].w, pi.lv[top].h);
    int it_total = 0;
    double sums[NSUM];
    double JtJ[6][6];
    double rr = 0, n_sel = 0, R = 1;
    int w0 = pi.lv[0].w, h0 = pi.lv[0].h;

    auto gn_level = [&](const AlignPyramid::Level& L0, const AlignPyramid::Level& L1, bool exact, int max_iter, double tol) {
        int w = L0.w, h = L0.h;
        size_t n = (size_t)w * h;
        double cx = (w - 1) / 2.0, cy = (h - 1) / 2.0;
        R = std::hypot((double)w, (double)h) / 2.0;
        int nblocks = (int)std::min<size_t>(wk.max_blocks, (n + 255) / 256);
        int it = 0;
        for (it = 1; it <= max_iter; ++it) {
            double M[6];
            to_affine(p[0], p[1], p[2], p[3], w, h, M);
            if (exact) k_warp_bspline<<<grid2(w, h), BLK, 0, s>>>(pj.coef, w, h, wk.I1w, M[0], M[1], M[2], M[3], M[4], M[5]);
            else k_warp_cv<<<grid2(w, h), BLK, 0, s>>>(L1.I, w, h, wk.I1w, M[0], M[1], M[2], M[3], M[4], M[5]);
            k_grad<<<grid2(w, h), BLK, 0, s>>>(wk.I1w, wk.gxw, wk.gyw, w, h);
            k_gn_reduce<<<nblocks, 256, 0, s>>>(L0.I, L0.gx, L0.gy, wk.I1w, wk.gxw, wk.gyw, w, h, M[0], M[1], M[2], M[3], M[4], M[5],
                                                 (float)p[4], (float)p[5], (float)cx, (float)cy, (float)R, wk.partials);
            k_final_reduce<<<1, 64, 0, s>>>(wk.partials, nblocks, wk.partials + (size_t)wk.max_blocks * NSUM);
            CK(cudaMemcpyAsync(sums, wk.partials + (size_t)wk.max_blocks * NSUM, NSUM * sizeof(double), cudaMemcpyDeviceToHost, s));
            CK(cudaStreamSynchronize(s));
            int q = 0;
            for (int a = 0; a < 6; ++a)
                for (int b = a; b < 6; ++b) { JtJ[a][b] = sums[q]; JtJ[b][a] = sums[q]; ++q; }
            double Jtr[6];
            for (int a = 0; a < 6; ++a) Jtr[a] = sums[q++];
            rr = sums[q++];
            n_sel = sums[q++];
            double A[6][6], b[6], d[6];
            std::memcpy(A, JtJ, sizeof A);
            for (int a = 0; a < 6; ++a) b[a] = Jtr[a];
            if (!solve6(A, b, d)) break;
            for (int a = 0; a < 6; ++a) d[a] = -d[a];
            p[0] += d[0] / R; p[1] += d[1] / R; p[2] += d[2]; p[3] += d[3]; p[4] += d[4]; p[5] += d[5];
            if (dbg) printf("  %s %dx%d it %d: n %.0f rr %.4g  d = %.3g %.3g %.3g %.3g %.3g %.3g  p = a %.3g b %.3g t %.4f %.4f g %.4f beta %.4f\n",
                            exact ? "exact" : "cv2  ", w, h, it, n_sel, rr, d[0], d[1], d[2], d[3], d[4], d[5], p[0], p[1], p[2], p[3], p[4], p[5]);
            if (std::max(std::max(std::fabs(d[0]), std::fabs(d[1])), std::max(std::fabs(d[2]), std::fabs(d[3]))) < tol) break;
        }
        it_total += std::min(it, max_iter);
    };

    for (int lvl = top; lvl >= 0; --lvl) {
        if (lvl != top) {
            p[2] *= pi.lv[lvl].fx / pi.lv[lvl + 1].fx;
            p[3] *= pi.lv[lvl].fy / pi.lv[lvl + 1].fy;
        }
        int mi = prm.max_iter[std::min(top - lvl, 3)];
        gn_level(pi.lv[lvl], pj.lv[lvl], false, mi, prm.tol_pyr);
        if (lvl == 0 && prm.exact_iters) gn_level(pi.lv[0], pj.lv[0], true, prm.exact_iters, prm.tol_exact);
    }
    // statistics from the last iteration's sums (they describe the state BEFORE its update, like the prototype's)
    int n = (int)n_sel;
    double s2 = rr / std::max(n - 6, 1);
    double inv[6][6];
    double A[6][6];
    std::memcpy(A, JtJ, sizeof A);
    bool okinv = invert6(A, inv);
    res.var_a = okinv ? s2 * inv[0][0] / (R * R) : INFINITY;
    res.var_b = okinv ? s2 * inv[1][1] / (R * R) : INFINITY;
    res.var_tx = okinv ? s2 * inv[2][2] : INFINITY;
    res.var_ty = okinv ? s2 * inv[3][3] : INFINITY;
    double s0 = sums[29], s1 = sums[30], s00 = sums[31], s11 = sums[32], s01 = sums[33];
    double cov01 = s01 / n - (s0 / n) * (s1 / n), v0 = s00 / n - (s0 / n) * (s0 / n), v1 = s11 / n - (s1 / n) * (s1 / n);
    res.ncc = n > 10 && v0 > 0 && v1 > 0 ? cov01 / std::sqrt(v0 * v1) : 0.0;
    res.z = {1 + p[0], p[1]};
    res.t = {p[2], p[3]};
    res.gain = p[4];
    res.n_px = n;
    res.iters = it_total;
    bool finite = std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]) && std::isfinite(p[3]);
    res.ok = finite && res.ncc > 0.3 && n > 0.3 * (double)w0 * h0;
    return res;
}

// ----------------------------------------------------------------------------- host: pairs, solve, geometry

std::vector<std::pair<int, int>> make_pairs(int n, const std::vector<int>& baselines) {
    std::vector<std::pair<int, int>> out;
    for (int d : baselines)
        for (int i = 0; i + d < n; ++i) out.push_back({i, i + d});
    std::sort(out.begin(), out.end());
    return out;
}

namespace {
using cd = std::complex<double>;

// weighted complex least squares by normal equations, column `fixed` held at 0
std::vector<cd> wlstsq(int n, const std::vector<std::vector<std::pair<int, cd>>>& rows, const std::vector<cd>& rhs,
                       const std::vector<double>& w, int fixed) {
    int m = n - 1;
    std::vector<int> col(n, -1);
    for (int k = 0, c = 0; k < n; ++k) if (k != fixed) col[k] = c++;
    std::vector<cd> N((size_t)m * m, cd(0, 0)), b(m, cd(0, 0));
    for (size_t e = 0; e < rows.size(); ++e) {
        for (auto& [ca, va] : rows[e]) {
            if (ca == fixed) continue;
            int ia = col[ca];
            b[ia] += std::conj(va) * w[e] * rhs[e];
            for (auto& [cb, vb] : rows[e]) {
                if (cb == fixed) continue;
                N[(size_t)ia * m + col[cb]] += std::conj(va) * w[e] * vb;
            }
        }
    }
    // Gaussian elimination (Hermitian positive definite in practice)
    for (int c = 0; c < m; ++c) {
        int piv = c;
        for (int r = c + 1; r < m; ++r) if (std::abs(N[(size_t)r * m + c]) > std::abs(N[(size_t)piv * m + c])) piv = r;
        if (piv != c) { for (int k = 0; k < m; ++k) std::swap(N[(size_t)c * m + k], N[(size_t)piv * m + k]); std::swap(b[c], b[piv]); }
        cd d = N[(size_t)c * m + c];
        for (int r = c + 1; r < m; ++r) {
            cd f = N[(size_t)r * m + c] / d;
            if (f == cd(0, 0)) continue;
            for (int k = c; k < m; ++k) N[(size_t)r * m + k] -= f * N[(size_t)c * m + k];
            b[r] -= f * b[c];
        }
    }
    std::vector<cd> x(m);
    for (int c = m - 1; c >= 0; --c) {
        cd sacc = b[c];
        for (int k = c + 1; k < m; ++k) sacc -= N[(size_t)c * m + k] * x[k];
        x[c] = sacc / N[(size_t)c * m + c];
    }
    std::vector<cd> full(n, cd(0, 0));
    for (int k = 0; k < n; ++k) if (k != fixed) full[k] = x[col[k]];
    return full;
}

double median(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}
}  // namespace

GlobalSolve solve_global(int n, const std::vector<PairResult>& pairs, int irls_rounds, double huber) {
    std::vector<const PairResult*> good;
    for (auto& p : pairs) if (p.ok) good.push_back(&p);
    if (good.empty()) throw std::runtime_error("no usable pairs");
    size_t m = good.size();
    std::vector<cd> lz(m), tt(m);
    std::vector<double> w_l(m), w_t(m), rob_l(m, 1.0), rob_t(m, 1.0);
    for (size_t e = 0; e < m; ++e) {
        lz[e] = std::log(good[e]->z);
        tt[e] = good[e]->t;
        w_l[e] = 1.0 / std::max(0.5 * (good[e]->var_a + good[e]->var_b), 1e-18);
        w_t[e] = 1.0 / std::max(0.5 * (good[e]->var_tx + good[e]->var_ty), 1e-12);
    }
    std::vector<std::vector<std::pair<int, cd>>> rows_l(m);
    for (size_t e = 0; e < m; ++e) rows_l[e] = {{good[e]->j, cd(1, 0)}, {good[e]->i, cd(-1, 0)}};
    std::vector<cd> L, T;
    std::vector<double> nl(m), nt(m);
    double sl = 0, st = 0;
    for (int round = 0; round < irls_rounds + 1; ++round) {
        std::vector<double> wl(m), wt(m);
        for (size_t e = 0; e < m; ++e) { wl[e] = w_l[e] * rob_l[e]; wt[e] = w_t[e] * rob_t[e]; }
        L = wlstsq(n, rows_l, lz, wl, 0);
        std::vector<cd> res_l(m), zij(m), res_t(m);
        std::vector<std::vector<std::pair<int, cd>>> rows_t(m);
        for (size_t e = 0; e < m; ++e) {
            res_l[e] = lz[e] - (L[good[e]->j] - L[good[e]->i]);
            zij[e] = std::exp(L[good[e]->j]) / std::exp(L[good[e]->i]);
            rows_t[e] = {{good[e]->j, cd(1, 0)}, {good[e]->i, -zij[e]}};
        }
        T = wlstsq(n, rows_t, tt, wt, 0);
        for (size_t e = 0; e < m; ++e) res_t[e] = tt[e] - (T[good[e]->j] - zij[e] * T[good[e]->i]);
        for (size_t e = 0; e < m; ++e) { nl[e] = std::abs(res_l[e]) * std::sqrt(w_l[e]); nt[e] = std::abs(res_t[e]) * std::sqrt(w_t[e]); }
        sl = 1.4826 * median(nl) + 1e-12;
        st = 1.4826 * median(nt) + 1e-12;
        for (size_t e = 0; e < m; ++e) {
            rob_l[e] = std::min(1.0, huber * sl / std::max(nl[e], 1e-12));
            rob_t[e] = std::min(1.0, huber * st / std::max(nt[e], 1e-12));
        }
    }
    GlobalSolve g;
    g.z.resize(n); g.t = T;
    for (int k = 0; k < n; ++k) g.z[k] = std::exp(L[k]);
    g.inlier.resize(m);
    g.outliers = 0;
    for (size_t e = 0; e < m; ++e) { g.inlier[e] = nl[e] < 5 * sl && nt[e] < 5 * st; if (!g.inlier[e]) ++g.outliers; }
    g.pairs_used = (int)m;
    g.pairs_total = (int)pairs.size();
    std::vector<double> ncc;
    for (auto p : good) ncc.push_back(p->ncc);
    g.ncc_min = *std::min_element(ncc.begin(), ncc.end());
    g.ncc_median = median(ncc);
    return g;
}

void rebase(std::vector<cd>& z, std::vector<cd>& t, int ref) {
    cd iz = 1.0 / z[ref], it = -t[ref] / z[ref];
    for (size_t k = 0; k < z.size(); ++k) {
        cd nz = z[k] * iz, nt = z[k] * it + t[k];
        z[k] = nz; t[k] = nt;
    }
}

int choose_reference(const std::vector<cd>& z0, const std::string& rule) {
    int n = (int)z0.size();
    if (rule == "first") return 0;
    if (rule == "last") return n - 1;
    if (rule == "middle") return n / 2;
    if (rule == "most-zoomed" || rule == "least-zoomed") {
        int best = 0;
        for (int k = 1; k < n; ++k) {
            bool better = rule == "most-zoomed" ? std::abs(z0[k]) > std::abs(z0[best]) : std::abs(z0[k]) < std::abs(z0[best]);
            if (better) best = k;
        }
        return best;
    }
    return ((atoi(rule.c_str()) % n) + n) % n;
}

void common_rect(const std::vector<cd>& z, const std::vector<cd>& t, int W, int H, int rect[4], double margin) {
    cd c((W - 1) / 2.0, (H - 1) / 2.0);
    cd corners[4] = {cd(0, 0) - c, cd(W - 1, 0) - c, cd(W - 1, H - 1) - c, cd(0, H - 1) - c};
    double left = -1e300, right = 1e300, top = -1e300, bottom = 1e300;
    for (size_t k = 0; k < z.size(); ++k) {
        cd iz = 1.0 / z[k], it = -t[k] / z[k];
        cd fp[4];
        for (int q = 0; q < 4; ++q) fp[q] = iz * corners[q] + it + c;
        left = std::max(left, std::max(fp[0].real(), fp[3].real()));
        right = std::min(right, std::min(fp[1].real(), fp[2].real()));
        top = std::max(top, std::max(fp[0].imag(), fp[1].imag()));
        bottom = std::min(bottom, std::min(fp[2].imag(), fp[3].imag()));
    }
    rect[0] = std::max(0, (int)std::ceil(left + margin));
    rect[1] = std::max(0, (int)std::ceil(top + margin));
    rect[2] = std::min(W - 1, (int)std::floor(right - margin));
    rect[3] = std::min(H - 1, (int)std::floor(bottom - margin));
}

}  // namespace stackcore
