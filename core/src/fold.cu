#include "stackcore/fold.h"

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <stdexcept>

namespace stackcore {

#define CK(x)                                                                                        \
    do {                                                                                             \
        cudaError_t e_ = (x);                                                                        \
        if (e_ != cudaSuccess) throw std::runtime_error(std::string(#x) + ": " + cudaGetErrorString(e_)); \
    } while (0)

std::vector<LevelShape> pyramid_shapes(int w, int h, int levels) {
    std::vector<LevelShape> s{{w, h}};
    for (int l = 1; l < levels; ++l) s.push_back({(s.back().w + 1) / 2, (s.back().h + 1) / 2});
    return s;
}

// ----------------------------------------------------------------------------- kernels

namespace {

__constant__ float c_k_small[16];   // blur sigma 0.8 (OpenCV ksize 7 -> radius 3)
__constant__ float c_k_big[64];     // blur sigma = radius (ksize = round(8 sigma + 1) | 1)
__constant__ float c_luma[3] = {0.2126f, 0.7152f, 0.0722f};

__device__ __forceinline__ int refl101(int p, int n) {   // BORDER_REFLECT_101
    if (n == 1) return 0;
    while ((unsigned)p >= (unsigned)n) p = p < 0 ? -p : 2 * n - 2 - p;
    return p;
}

__global__ void k_luma(const float* __restrict__ rgb, float* __restrict__ y, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* p = rgb + i * 3;
    y[i] = p[0] * c_luma[0] + p[1] * c_luma[1] + p[2] * c_luma[2];
}

// separable blur, kernel in constant memory (which one chosen by template), radius R at runtime
template <bool BIG>
__global__ void k_blur_row(const float* __restrict__ in, float* __restrict__ out, int w, int h, int R) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const float* k = BIG ? c_k_big : c_k_small;
    const float* row = in + (size_t)y * w;
    float s = 0.f;
    for (int i = -R; i <= R; ++i) s += k[i + R] * row[refl101(x + i, w)];   // RowFilter: left to right
    out[(size_t)y * w + x] = s;
}

template <bool BIG>
__global__ void k_blur_col(const float* __restrict__ in, float* __restrict__ out, int w, int h, int R) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const float* k = BIG ? c_k_big : c_k_small;
    float s = k[R] * in[(size_t)y * w + x];                                   // SymmColumnFilter: centre, then pairs
    for (int i = 1; i <= R; ++i)
        s += k[R + i] * (in[(size_t)refl101(y + i, h) * w + x] + in[(size_t)refl101(y - i, h) * w + x]);
    out[(size_t)y * w + x] = s;
}

// OpenCV Laplacian ksize 3: {2,0,2, 0,-8,0, 2,0,2}, then squared
__global__ void k_lap3_sq(const float* __restrict__ in, float* __restrict__ out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int xm = refl101(x - 1, w), xp = refl101(x + 1, w), ym = refl101(y - 1, h), yp = refl101(y + 1, h);
    const float* r0 = in + (size_t)ym * w;
    const float* r1 = in + (size_t)y * w;
    const float* r2 = in + (size_t)yp * w;
    float l = 2.f * r0[xm] + 2.f * r0[xp] - 8.f * r1[x] + 2.f * r2[xm] + 2.f * r2[xp];
    out[(size_t)y * w + x] = l * l;
}

// pyrDown, 3 channels, reflect-101, [1 4 6 4 1]/256
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

// lap = g - pyrUp(gn) where pyrUp has the size of g (dw = 2 sw or 2 sw - 1). cv::pyrUp
// reflects 101 at the top/left but uses replicate-style weights at the right and
// bottom borders: 7 s[n-1] + s[n-2] for the even output, 8 s[n-1] for the odd one
// (measured against cv2 5.0; FINDINGS 18).
__global__ void k_lap_level3(const float* __restrict__ g, const float* __restrict__ gn, int sw, int sh,
                             float* __restrict__ lap, int dw, int dh) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || y >= dh) return;
    int xi = x >> 1, yi = y >> 1;
    bool xo = x & 1, yo = y & 1;
    bool right = xi == sw - 1, bottom = yi == sh - 1;
    int xa = refl101(xi - 1, sw), xb = refl101(xi + 1, sw), ya = refl101(yi - 1, sh), yb = refl101(yi + 1, sh);
    xi = refl101(xi, sw);
    yi = refl101(yi, sh);
#pragma unroll
    for (int c = 0; c < 3; ++c) {
        auto rowv = [&](int yy) {
            const float* p = gn + (size_t)yy * sw * 3 + c;
            if (right) return xo ? p[xi * 3] * 8.f : p[xa * 3] + p[xi * 3] * 7.f;
            return xo ? (p[xi * 3] + p[xb * 3]) * 4.f : p[xa * 3] + p[xb * 3] + 6.f * p[xi * 3];
        };
        float v;
        if (bottom) v = yo ? rowv(yi) * 8.f : rowv(ya) + rowv(yi) * 7.f;
        else v = yo ? (rowv(yi) + rowv(yb)) * 4.f : rowv(ya) + rowv(yb) + 6.f * rowv(yi);
        size_t o = ((size_t)y * dw + x) * 3 + c;
        lap[o] = g[o] - v * (1.f / 64.f);
    }
}

// salience: 3x3 box (reflect-101, normalised like cv::boxFilter) of luma^2
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

// |coef @ LUMA| into tmp (for the noise MAD)
__global__ void k_abs_luma(const float* __restrict__ coef, float* __restrict__ out, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* p = coef + i * 3;
    out[i] = fabsf(p[0] * c_luma[0] + p[1] * c_luma[1] + p[2] * c_luma[2]);
}

// ---- state updates

__global__ void k_fold_ab(double* __restrict__ a_num, double* __restrict__ a_den, float* __restrict__ b_best,
                          int16_t* __restrict__ b_idx, float* __restrict__ b_col, double* __restrict__ b_fmsum,
                          const float* __restrict__ img, const float* __restrict__ fm, size_t n, int k, float half_power) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float f = fm[i];
    double wgt = (double)f + 1e-9;
    wgt = half_power == 2.f ? wgt * wgt : pow(wgt, (double)half_power);
    const float* p = img + i * 3;
    a_num[i * 3] += wgt * (double)p[0];
    a_num[i * 3 + 1] += wgt * (double)p[1];
    a_num[i * 3 + 2] += wgt * (double)p[2];
    a_den[i] += wgt;
    b_fmsum[i] += (double)f;
    // _take: higher value wins, ties go to the lower frame index (frames arrive in any order)
    if (f > b_best[i] || (f == b_best[i] && k < b_idx[i])) {
        b_best[i] = f;
        b_idx[i] = (int16_t)k;
        b_col[i * 3] = p[0];
        b_col[i * 3 + 1] = p[1];
        b_col[i * 3 + 2] = p[2];
    }
}

__global__ void k_fold_c(float* __restrict__ c_sal, float* __restrict__ c_coef, int16_t* __restrict__ c_idx,
                         const float* __restrict__ sal, const float* __restrict__ lap, size_t n, int k) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = sal[i];
    if (s > c_sal[i] || (s == c_sal[i] && k < c_idx[i])) {
        c_sal[i] = s;
        c_idx[i] = (int16_t)k;
        c_coef[i * 3] = lap[i * 3];
        c_coef[i * 3 + 1] = lap[i * 3 + 1];
        c_coef[i * 3 + 2] = lap[i * 3 + 2];
    }
}

__global__ void k_add_base(double* __restrict__ base, const float* __restrict__ top, size_t n3) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n3) base[i] += (double)top[i];
}

__global__ void k_fill_f32(float* p, size_t n, float v) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = v;
}

inline unsigned blocks1(size_t n) { return (unsigned)((n + 255) / 256); }
inline dim3 grid2(int w, int h) { return dim3((w + 31) / 32, (h + 7) / 8); }
const dim3 BLK(32, 8);

// OpenCV getGaussianKernel for CV_32F
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

int blur_radius(double sigma) {   // cv::GaussianBlur ksize for float images: round(sigma * 4 * 2 + 1) | 1
    int ks = (int)std::nearbyint(sigma * 8 + 1) | 1;
    return ks / 2;
}

float median_sorted(float* d, size_t n, cudaStream_t s) {
    thrust::sort(thrust::cuda::par.on(s), thrust::device_ptr<float>(d), thrust::device_ptr<float>(d + n));
    float v[2];
    CK(cudaMemcpyAsync(v, d + (n - 1) / 2, sizeof(float) * 2, cudaMemcpyDeviceToHost, s));
    CK(cudaStreamSynchronize(s));
    return n % 2 ? v[0] : (v[0] + v[1]) * 0.5f;
}

}  // namespace

// ----------------------------------------------------------------------------- allocation

void FoldState::alloc(int dev_, int w_, int h_, int levels_) {
    dev = dev_; w = w_; h = h_; levels = levels_;
    shapes = pyramid_shapes(w, h, levels);
    size_t n = (size_t)w * h;
    CK(cudaSetDevice(dev));
    CK(cudaMalloc(&a_num, n * 3 * sizeof(double)));
    CK(cudaMemset(a_num, 0, n * 3 * sizeof(double)));
    CK(cudaMalloc(&a_den, n * sizeof(double)));
    CK(cudaMemset(a_den, 0, n * sizeof(double)));
    CK(cudaMalloc(&b_best, n * sizeof(float)));
    k_fill_f32<<<blocks1(n), 256>>>(b_best, n, -1.f);
    CK(cudaMalloc(&b_idx, n * sizeof(int16_t)));
    CK(cudaMemset(b_idx, 0, n * sizeof(int16_t)));
    CK(cudaMalloc(&b_col, n * 3 * sizeof(float)));
    CK(cudaMemset(b_col, 0, n * 3 * sizeof(float)));
    CK(cudaMalloc(&b_fmsum, n * sizeof(double)));
    CK(cudaMemset(b_fmsum, 0, n * sizeof(double)));
    for (int l = 0; l < levels - 1; ++l) {
        size_t m = (size_t)shapes[l].w * shapes[l].h;
        float* s; float* c; int16_t* i;
        CK(cudaMalloc(&s, m * sizeof(float)));
        k_fill_f32<<<blocks1(m), 256>>>(s, m, -1.f);
        CK(cudaMalloc(&c, m * 3 * sizeof(float)));
        CK(cudaMemset(c, 0, m * 3 * sizeof(float)));
        CK(cudaMalloc(&i, m * sizeof(int16_t)));
        CK(cudaMemset(i, 0, m * sizeof(int16_t)));
        c_sal.push_back(s); c_coef.push_back(c); c_idx.push_back(i);
    }
    size_t t = (size_t)shapes.back().w * shapes.back().h * 3;
    CK(cudaMalloc(&c_base, t * sizeof(double)));
    CK(cudaMemset(c_base, 0, t * sizeof(double)));
    CK(cudaMalloc(&d_noise_tmp, n * sizeof(float)));
    noise_mad.assign(levels - 1, 0.f);
    CK(cudaDeviceSynchronize());
}

void FoldState::free() {
    if (dev < 0) return;
    cudaSetDevice(dev);
    cudaFree(a_num); cudaFree(a_den); cudaFree(b_best); cudaFree(b_idx); cudaFree(b_col); cudaFree(b_fmsum);
    for (auto p : c_sal) cudaFree(p);
    for (auto p : c_coef) cudaFree(p);
    for (auto p : c_idx) cudaFree(p);
    cudaFree(c_base); cudaFree(d_noise_tmp);
    c_sal.clear(); c_coef.clear(); c_idx.clear();
    dev = -1;
}

size_t FoldState::bytes() const {
    size_t n = (size_t)w * h, b = n * (24 + 8 + 4 + 2 + 12 + 8 + 4);
    for (int l = 0; l < levels - 1; ++l) b += (size_t)shapes[l].w * shapes[l].h * (4 + 12 + 2);
    return b + (size_t)shapes.back().w * shapes.back().h * 24;
}

void FoldScratch::alloc(int w_, int h_, int levels_) {
    w = w_; h = h_; levels = levels_;
    shapes = pyramid_shapes(w, h, levels);
    size_t n = (size_t)w * h;
    CK(cudaMalloc(&y, n * sizeof(float)));
    CK(cudaMalloc(&t1, n * sizeof(float)));
    CK(cudaMalloc(&t2, n * sizeof(float)));
    CK(cudaMalloc(&fm, n * sizeof(float)));
    g.assign(levels, nullptr);
    for (int l = 1; l < levels; ++l) CK(cudaMalloc(&g[l], (size_t)shapes[l].w * shapes[l].h * 3 * sizeof(float)));
    for (int l = 0; l < levels - 1; ++l) {
        size_t m = (size_t)shapes[l].w * shapes[l].h;
        float* a; float* b;
        CK(cudaMalloc(&a, m * 3 * sizeof(float)));
        CK(cudaMalloc(&b, m * sizeof(float)));
        lap.push_back(a); sal.push_back(b);
    }
}

void FoldScratch::free() {
    cudaFree(y); cudaFree(t1); cudaFree(t2); cudaFree(fm);
    for (auto p : g) if (p) cudaFree(p);
    for (auto p : lap) cudaFree(p);
    for (auto p : sal) cudaFree(p);
    g.clear(); lap.clear(); sal.clear();
}

size_t FoldScratch::bytes() const {
    size_t n = (size_t)w * h, b = n * 4 * 4;
    for (int l = 1; l < levels; ++l) b += (size_t)shapes[l].w * shapes[l].h * 12;
    for (int l = 0; l < levels - 1; ++l) b += (size_t)shapes[l].w * shapes[l].h * 16;
    return b;
}

// ----------------------------------------------------------------------------- stages

void fold_setup_device(const FoldParams& p) {
    auto ks = gaussian_kernel(2 * blur_radius(0.8) + 1, 0.8);
    auto kb = gaussian_kernel(2 * blur_radius(p.radius) + 1, p.radius);
    if (kb.size() > 64) throw std::runtime_error("fold: blur radius too large for the constant kernel");
    CK(cudaMemcpyToSymbol(c_k_small, ks.data(), ks.size() * sizeof(float)));
    CK(cudaMemcpyToSymbol(c_k_big, kb.data(), kb.size() * sizeof(float)));
}

void fold_measure(const float* d_img, const FoldParams& p, FoldScratch& sc, cudaStream_t s) {
    int w = sc.w, h = sc.h;
    size_t n = (size_t)w * h;
    int rs = blur_radius(0.8), rb = blur_radius(p.radius);
    // focus measure: luma -> blur 0.8 -> Laplacian^2 -> blur radius
    k_luma<<<blocks1(n), 256, 0, s>>>(d_img, sc.y, n);
    k_blur_row<false><<<grid2(w, h), BLK, 0, s>>>(sc.y, sc.t1, w, h, rs);
    k_blur_col<false><<<grid2(w, h), BLK, 0, s>>>(sc.t1, sc.t2, w, h, rs);
    k_lap3_sq<<<grid2(w, h), BLK, 0, s>>>(sc.t2, sc.t1, w, h);
    k_blur_row<true><<<grid2(w, h), BLK, 0, s>>>(sc.t1, sc.t2, w, h, rb);
    k_blur_col<true><<<grid2(w, h), BLK, 0, s>>>(sc.t2, sc.fm, w, h, rb);
    // Laplacian pyramid
    const float* gl = d_img;
    for (int l = 1; l < p.levels; ++l) {
        auto a = sc.shapes[l - 1], b = sc.shapes[l];
        k_pyrdown3<<<grid2(b.w, b.h), BLK, 0, s>>>(gl, a.w, a.h, sc.g[l], b.w, b.h);
        gl = sc.g[l];
    }
    gl = d_img;
    for (int l = 0; l < p.levels - 1; ++l) {
        auto a = sc.shapes[l], b = sc.shapes[l + 1];
        k_lap_level3<<<grid2(a.w, a.h), BLK, 0, s>>>(gl, sc.g[l + 1], b.w, b.h, sc.lap[l], a.w, a.h);
        k_salience<<<grid2(a.w, a.h), BLK, 0, s>>>(sc.lap[l], a.w, a.h, sc.sal[l]);
        gl = sc.g[l + 1];
    }
}

void fold_accumulate(FoldState& st, int k, const float* d_img, const FoldParams& p, FoldScratch& sc, cudaStream_t s) {
    size_t n = (size_t)st.w * st.h;
    k_fold_ab<<<blocks1(n), 256, 0, s>>>(st.a_num, st.a_den, st.b_best, st.b_idx, st.b_col, st.b_fmsum, d_img, sc.fm, n, k,
                                         p.power * 0.5f);
    for (int l = 0; l < p.levels - 1; ++l) {
        size_t m = (size_t)st.shapes[l].w * st.shapes[l].h;
        k_fold_c<<<blocks1(m), 256, 0, s>>>(st.c_sal[l], st.c_coef[l], st.c_idx[l], sc.sal[l], sc.lap[l], m, k);
    }
    size_t t = (size_t)st.shapes.back().w * st.shapes.back().h * 3;
    k_add_base<<<blocks1(t), 256, 0, s>>>(st.c_base, sc.g[p.levels - 1], t);
    st.count += 1;
    if (k < st.noise_k) {   // noise reference: lowest frame index seen (medians sync this stream)
        st.noise_k = k;
        for (int l = 0; l < p.levels - 1; ++l) {
            size_t m = (size_t)st.shapes[l].w * st.shapes[l].h;
            k_abs_luma<<<blocks1(m), 256, 0, s>>>(sc.lap[l], st.d_noise_tmp, m);
            st.noise_mad[l] = (float)(1.4826 * median_sorted(st.d_noise_tmp, m, s));
        }
        CK(cudaMemcpyAsync(st.d_noise_tmp, sc.fm, n * sizeof(float), cudaMemcpyDeviceToDevice, s));
        st.noise_fm = median_sorted(st.d_noise_tmp, n, s);
    }
}

// ----------------------------------------------------------------------------- host side

template <class T>
static std::vector<T> dl(const T* d, size_t n) {
    std::vector<T> v(n);
    CK(cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
}

HostFoldState download(const FoldState& st) {
    CK(cudaSetDevice(st.dev));
    CK(cudaDeviceSynchronize());
    HostFoldState h;
    h.w = st.w; h.h = st.h; h.levels = st.levels; h.shapes = st.shapes;
    size_t n = (size_t)st.w * st.h;
    h.a_num = dl(st.a_num, n * 3);
    h.a_den = dl(st.a_den, n);
    h.b_best = dl(st.b_best, n);
    h.b_idx = dl(st.b_idx, n);
    h.b_col = dl(st.b_col, n * 3);
    h.b_fmsum = dl(st.b_fmsum, n);
    for (int l = 0; l < st.levels - 1; ++l) {
        size_t m = (size_t)st.shapes[l].w * st.shapes[l].h;
        h.c_sal.push_back(dl(st.c_sal[l], m));
        h.c_coef.push_back(dl(st.c_coef[l], m * 3));
        h.c_idx.push_back(dl(st.c_idx[l], m));
    }
    h.c_base = dl(st.c_base, (size_t)st.shapes.back().w * st.shapes.back().h * 3);
    h.count = st.count; h.noise_k = st.noise_k; h.noise_mad = st.noise_mad; h.noise_fm = st.noise_fm;
    return h;
}

void merge_into(HostFoldState& a, const HostFoldState& b) {
    size_t n = (size_t)a.w * a.h;
    for (size_t i = 0; i < n * 3; ++i) a.a_num[i] += b.a_num[i];
    for (size_t i = 0; i < n; ++i) {
        a.a_den[i] += b.a_den[i];
        a.b_fmsum[i] += b.b_fmsum[i];
        // _take: src wins on a higher value, ties -> lower frame index
        if (b.b_best[i] > a.b_best[i] || (b.b_best[i] == a.b_best[i] && b.b_idx[i] < a.b_idx[i])) {
            a.b_best[i] = b.b_best[i];
            a.b_idx[i] = b.b_idx[i];
            for (int c = 0; c < 3; ++c) a.b_col[i * 3 + c] = b.b_col[i * 3 + c];
        }
    }
    for (int l = 0; l < a.levels - 1; ++l) {
        size_t m = a.c_sal[l].size();
        for (size_t i = 0; i < m; ++i)
            if (b.c_sal[l][i] > a.c_sal[l][i] || (b.c_sal[l][i] == a.c_sal[l][i] && b.c_idx[l][i] < a.c_idx[l][i])) {
                a.c_sal[l][i] = b.c_sal[l][i];
                a.c_idx[l][i] = b.c_idx[l][i];
                for (int c = 0; c < 3; ++c) a.c_coef[l][i * 3 + c] = b.c_coef[l][i * 3 + c];
            }
    }
    for (size_t i = 0; i < a.c_base.size(); ++i) a.c_base[i] += b.c_base[i];
    a.count += b.count;
    if (b.noise_k < a.noise_k) {
        a.noise_k = b.noise_k;
        a.noise_mad = b.noise_mad;
        a.noise_fm = b.noise_fm;
    }
}

template <class T>
static bool wr(const std::filesystem::path& p, const std::vector<T>& v) {
    FILE* f = fopen(p.string().c_str(), "wb");
    if (!f) return false;
    size_t n = fwrite(v.data(), sizeof(T), v.size(), f);
    fclose(f);
    return n == v.size();
}

bool write_state(const HostFoldState& st, const std::string& dir) {
    namespace fs = std::filesystem;
    fs::create_directories(dir);
    fs::path d(dir);
    bool ok = wr(d / "a_num.f64", st.a_num) && wr(d / "a_den.f64", st.a_den) && wr(d / "b_best.f32", st.b_best) &&
              wr(d / "b_idx.i16", st.b_idx) && wr(d / "b_col.f32", st.b_col) && wr(d / "b_fmsum.f64", st.b_fmsum) &&
              wr(d / "c_base.f64", st.c_base);
    for (int l = 0; l < st.levels - 1; ++l)
        ok = ok && wr(d / ("c_sal_" + std::to_string(l) + ".f32"), st.c_sal[l]) &&
             wr(d / ("c_coef_" + std::to_string(l) + ".f32"), st.c_coef[l]) &&
             wr(d / ("c_idx_" + std::to_string(l) + ".i16"), st.c_idx[l]);
    FILE* f = fopen((d / "manifest.txt").string().c_str(), "w");
    if (!f) return false;
    fprintf(f, "w %d\nh %d\nlevels %d\ncount %d\nnoise_k %d\nnoise_fm %.9g\nnoise_mad", st.w, st.h, st.levels, st.count,
            st.noise_k, st.noise_fm);
    for (float v : st.noise_mad) fprintf(f, " %.9g", v);
    fprintf(f, "\nshapes");
    for (auto& s : st.shapes) fprintf(f, " %dx%d", s.w, s.h);
    fprintf(f, "\n");
    fclose(f);
    return ok;
}

}  // namespace stackcore
