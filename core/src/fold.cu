#include "stackcore/fold.h"

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include <algorithm>
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

__constant__ float c_luma[3] = {0.2126f, 0.7152f, 0.0722f};

// |coef @ LUMA| into tmp (for the noise MAD)
__global__ void k_abs_luma(const float* __restrict__ coef, float* __restrict__ out, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* p = coef + i * 3;
    out[i] = fabsf(p[0] * c_luma[0] + p[1] * c_luma[1] + p[2] * c_luma[2]);
}

// ---- state updates (the _take rule: higher value wins, ties go to the lower frame index)

__global__ void k_fold_ab(double* __restrict__ a_num, double* __restrict__ a_den, float* __restrict__ b_best,
                          int16_t* __restrict__ b_idx, float* __restrict__ b_col, double* __restrict__ b_fmsum,
                          const float* __restrict__ img, const float* __restrict__ fm, size_t n, int k, float half_power) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float f = fm[i];
    const float* p = img + i * 3;
    if (a_num) {   // A's sums only when A is an output (uniform branch)
        double wgt = (double)f + 1e-9;
        wgt = half_power == 2.f ? wgt * wgt : pow(wgt, (double)half_power);
        a_num[i * 3] += wgt * (double)p[0];
        a_num[i * 3 + 1] += wgt * (double)p[1];
        a_num[i * 3 + 2] += wgt * (double)p[2];
        a_den[i] += wgt;
    }
    b_fmsum[i] += (double)f;
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

__global__ void k_preview(const double* __restrict__ num, const double* __restrict__ den, int w, int h, int down,
                          float* __restrict__ out, int pw, int ph) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= pw || y >= ph) return;
    size_t i = (size_t)min(y * down + down / 2, h - 1) * w + min(x * down + down / 2, w - 1);
    double d = den[i];
    for (int c = 0; c < 3; ++c) out[((size_t)y * pw + x) * 3 + c] = d > 0 ? (float)(num[i * 3 + c] / d) : 0.f;
}

__global__ void k_preview_img(const float* __restrict__ img, int w, int h, int x0, int y0, int down, float* __restrict__ out, int pw, int ph) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= pw || y >= ph) return;
    size_t i = (size_t)min(y0 + y * down + down / 2, h - 1) * w + min(x0 + x * down + down / 2, w - 1);
    for (int c = 0; c < 3; ++c) out[((size_t)y * pw + x) * 3 + c] = img[i * 3 + c];
}

__global__ void k_fill_f32(float* p, size_t n, float v) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = v;
}

inline unsigned blocks1(size_t n) { return (unsigned)((n + 255) / 256); }

// sum of a float image into *acc (double): per block a shared-memory tree, one atomic per block. A frame's
// total focus energy, so a frame blurred by movement during its exposure stands out against its neighbours
// (FINDINGS 49). The atomics' order varies run to run: the sum is reproducible to ~1e-15, not to the bit.
__global__ void k_sum_f32(const float* __restrict__ a, size_t n, double* __restrict__ acc) {
    __shared__ double sh[256];
    double v = 0;
    for (size_t i = blockIdx.x * 256 + threadIdx.x; i < n; i += (size_t)gridDim.x * 256) v += a[i];
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) {
        if (threadIdx.x < o) sh[threadIdx.x] += sh[threadIdx.x + o];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(acc, sh[0]);
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

void FoldState::alloc(int dev_, int w_, int h_, int levels_, bool with_a) {
    dev = dev_; w = w_; h = h_; levels = levels_;
    shapes = pyramid_shapes(w, h, levels);
    size_t n = (size_t)w * h;
    CK(cudaSetDevice(dev));
    a_num = nullptr; a_den = nullptr;
    if (with_a) {
        CK(cudaMalloc(&a_num, n * 3 * sizeof(double)));
        CK(cudaMemset(a_num, 0, n * 3 * sizeof(double)));
        CK(cudaMalloc(&a_den, n * sizeof(double)));
        CK(cudaMemset(a_den, 0, n * sizeof(double)));
    }
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
        // in the lists before they are allocated, so free() finds them when a later allocation throws
        c_sal.push_back(nullptr); c_coef.push_back(nullptr); c_idx.push_back(nullptr);
        float*& s = c_sal.back(); float*& c = c_coef.back(); int16_t*& i = c_idx.back();
        CK(cudaMalloc(&s, m * sizeof(float)));
        k_fill_f32<<<blocks1(m), 256>>>(s, m, -1.f);
        CK(cudaMalloc(&c, m * 3 * sizeof(float)));
        CK(cudaMemset(c, 0, m * 3 * sizeof(float)));
        CK(cudaMalloc(&i, m * sizeof(int16_t)));
        CK(cudaMemset(i, 0, m * sizeof(int16_t)));
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
    size_t n = (size_t)w * h, b = n * ((a_num ? 24 + 8 : 0) + 4 + 2 + 12 + 8 + 4);
    for (int l = 0; l < levels - 1; ++l) b += (size_t)shapes[l].w * shapes[l].h * (4 + 12 + 2);
    return b + (size_t)shapes.back().w * shapes.back().h * 24;
}

void FoldScratch::alloc(int w_, int h_, int levels_, float radius, float c_radius) {
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
    g_small.build(0.8);
    g_big.build(radius);
    if (c_radius > 0.f) {
        g_c.resize(levels);
        for (int l = 1; l < levels - 1; ++l) g_c[l].build(std::max(0.5, (double)c_radius / (1 << l)));
    }
}

void FoldScratch::free() {
    cudaFree(y); cudaFree(t1); cudaFree(t2); cudaFree(fm);
    for (auto p : g) if (p) cudaFree(p);
    for (auto p : lap) cudaFree(p);
    for (auto p : sal) cudaFree(p);
    g.clear(); lap.clear(); sal.clear();
    g_small.free();
    g_big.free();
    for (auto& k : g_c) k.free();
    g_c.clear();
}

size_t FoldScratch::bytes() const {
    size_t n = (size_t)w * h, b = n * 4 * 4;
    for (int l = 1; l < levels; ++l) b += (size_t)shapes[l].w * shapes[l].h * 12;
    for (int l = 0; l < levels - 1; ++l) b += (size_t)shapes[l].w * shapes[l].h * 16;
    return b;
}

// ----------------------------------------------------------------------------- stages

void fold_measure(const float* d_img, const FoldParams& p, FoldScratch& sc, cudaStream_t s) {
    int w = sc.w, h = sc.h;
    size_t n = (size_t)w * h;
    // focus measure: luma -> blur 0.8 -> Laplacian^2 -> blur radius
    luma(d_img, sc.y, n, s);
    gaussian_blur(sc.y, sc.t2, sc.t1, w, h, sc.g_small, s);
    laplacian3_sq(sc.t2, sc.t1, w, h, s);
    gaussian_blur(sc.t1, sc.fm, sc.t2, w, h, sc.g_big, s);
    // Laplacian pyramid
    const float* gl = d_img;
    for (int l = 1; l < p.levels; ++l) {
        auto a = sc.shapes[l - 1], b = sc.shapes[l];
        pyr_down_rgb(gl, a.w, a.h, sc.g[l], b.w, b.h, s);
        gl = sc.g[l];
    }
    gl = d_img;
    for (int l = 0; l < p.levels - 1; ++l) {
        auto a = sc.shapes[l], b = sc.shapes[l + 1];
        lap_level_rgb(gl, sc.g[l + 1], b.w, b.h, sc.lap[l], a.w, a.h, s);
        salience3(sc.lap[l], a.w, a.h, sc.sal[l], s);
        // aggregate the energy so the frame of a strong edge owns the band beside it (c_radius):
        // level 0 is left alone, its selection in such bands is noise that the smoothing gate handles
        if (p.c_radius > 0.f && l >= 1) gaussian_blur(sc.sal[l], sc.sal[l], sc.t1, a.w, a.h, sc.g_c[l], s);
        gl = sc.g[l + 1];
    }
}

void fold_energy_add(const float* d_fm, size_t n, double* d_acc, cudaStream_t s) {
    unsigned b = (unsigned)std::min<size_t>(blocks1(n), 4096);
    k_sum_f32<<<b, 256, 0, s>>>(d_fm, n, d_acc);
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

void fold_preview(const FoldState& st, int down, std::vector<float>& rgb, int& pw, int& ph, cudaStream_t s) {
    down = std::max(1, down);
    pw = std::max(1, st.w / down);
    ph = std::max(1, st.h / down);
    float* d;
    CK(cudaMallocAsync(&d, (size_t)pw * ph * 3 * sizeof(float), s));
    k_preview<<<dim3((pw + 31) / 32, (ph + 7) / 8), dim3(32, 8), 0, s>>>(st.a_num, st.a_den, st.w, st.h, down, d, pw, ph);
    rgb.resize((size_t)pw * ph * 3);
    CK(cudaMemcpyAsync(rgb.data(), d, rgb.size() * sizeof(float), cudaMemcpyDeviceToHost, s));
    CK(cudaFreeAsync(d, s));
    CK(cudaStreamSynchronize(s));
}

// the best focus measure found so far per pixel (grey; -1 = nothing yet -> 0), point-sampled
__global__ void k_preview_focus(const float* __restrict__ best, int w, int h, int down, float* __restrict__ out, int pw, int ph) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= pw || y >= ph) return;
    size_t i = (size_t)min(y * down + down / 2, h - 1) * w + min(x * down + down / 2, w - 1);
    float v = fmaxf(best[i], 0.f);
    for (int c = 0; c < 3; ++c) out[((size_t)y * pw + x) * 3 + c] = v;
}

void preview_size(int w, int h, int down, int& pw, int& ph) {
    down = std::max(1, down);
    pw = std::max(1, w / down);
    ph = std::max(1, h / down);
}

void fold_focus_preview_kernel(const FoldState& st, int down, float* d_out, int& pw, int& ph, cudaStream_t s) {
    down = std::max(1, down);
    preview_size(st.w, st.h, down, pw, ph);
    k_preview_focus<<<dim3((pw + 31) / 32, (ph + 7) / 8), dim3(32, 8), 0, s>>>(st.b_best, st.w, st.h, down, d_out, pw, ph);
}

void image_preview_kernel(const float* d_rgb, int w, int h, int x0, int y0, int cw, int ch, int down, float* d_out, int& pw, int& ph, cudaStream_t s) {
    down = std::max(1, down);
    preview_size(cw, ch, down, pw, ph);
    k_preview_img<<<dim3((pw + 31) / 32, (ph + 7) / 8), dim3(32, 8), 0, s>>>(d_rgb, w, h, x0, y0, down, d_out, pw, ph);
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
    if (st.a_num) { h.a_num = dl(st.a_num, n * 3); h.a_den = dl(st.a_den, n); }
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
    if (!a.a_num.empty() && !b.a_num.empty()) {
        for (size_t i = 0; i < n * 3; ++i) a.a_num[i] += b.a_num[i];
        for (size_t i = 0; i < n; ++i) a.a_den[i] += b.a_den[i];
    }
    for (size_t i = 0; i < n; ++i) {
        a.b_fmsum[i] += b.b_fmsum[i];
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

// ----------------------------------------------------------------------------- merge across devices

__global__ void k_add_f64(double* __restrict__ a, const double* __restrict__ b, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}
__global__ void k_merge_take(float* __restrict__ a_best, int16_t* __restrict__ a_idx, float* __restrict__ a_col,
                             const float* __restrict__ b_best, const int16_t* __restrict__ b_idx, const float* __restrict__ b_col,
                             size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (b_best[i] > a_best[i] || (b_best[i] == a_best[i] && b_idx[i] < a_idx[i])) {
        a_best[i] = b_best[i];
        a_idx[i] = b_idx[i];
        if (a_col) { a_col[i * 3] = b_col[i * 3]; a_col[i * 3 + 1] = b_col[i * 3 + 1]; a_col[i * 3 + 2] = b_col[i * 3 + 2]; }
    }
}
__global__ void k_base_f32(const double* __restrict__ in, float* __restrict__ out, size_t n, double den) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (float)(in[i] / den);
}

void add_f64(double* a, const double* b, size_t n, cudaStream_t s) { k_add_f64<<<blocks1(n), 256, 0, s>>>(a, b, n); }
void merge_take(float* a_best, int16_t* a_idx, float* a_col, const float* b_best, const int16_t* b_idx, const float* b_col,
                size_t n, cudaStream_t s) {
    k_merge_take<<<blocks1(n), 256, 0, s>>>(a_best, a_idx, a_col, b_best, b_idx, b_col, n);
}
void fold_base_f32(const FoldState& st, float* d_out, cudaStream_t s) {
    size_t t = (size_t)st.shapes.back().w * st.shapes.back().h * 3;
    k_base_f32<<<blocks1(t), 256, 0, s>>>(st.c_base, d_out, t, (double)std::max(st.count, 1));
}

// The two page-locked chunks are pinned once for the process and handed from one CrossCopier to the next (merges
// run one at a time): with two GPUs loaded, page-locked memory that is freed is not credited back by the driver,
// and a pair pinned for every merge counted towards its ceiling (FINDINGS 55).
static void* g_cross_pin[2] = {nullptr, nullptr};
static size_t g_cross_bytes = 0;
static bool g_cross_out = false;

CrossCopier::CrossCopier(int dst_dev, int src_dev, size_t chunk) : dst_(dst_dev), src_(src_dev), chunk_(chunk) {
    if (!g_cross_out && g_cross_pin[0] && g_cross_bytes == chunk_) {
        pin_[0] = (decltype(pin_[0]))g_cross_pin[0]; pin_[1] = (decltype(pin_[1]))g_cross_pin[1];
        g_cross_out = true;
        shared_ = true;
    } else {
        // (a second chunk that cannot be pinned gives the first back: the destructor does not run for a constructor
        // that throws)
        void* p[2] = {nullptr, nullptr};
        for (int b = 0; b < 2; ++b) {
            cudaError_t e = cudaHostAlloc(&p[b], chunk_, cudaHostAllocPortable);
            if (e != cudaSuccess) { if (p[0]) cudaFreeHost(p[0]); CK(e); }
        }
        pin_[0] = (decltype(pin_[0]))p[0]; pin_[1] = (decltype(pin_[1]))p[1];
        if (!g_cross_pin[0]) { g_cross_pin[0] = pin_[0]; g_cross_pin[1] = pin_[1]; g_cross_bytes = chunk_; g_cross_out = true; shared_ = true; }
    }
    // a stream or event that cannot be made: the shared pair is handed back (left marked as out, every later merge
    // pinned a pair of its own) and what was made here is released
    try {
        CK(cudaSetDevice(src_));
        CK(cudaStreamCreateWithFlags(&ss_, cudaStreamNonBlocking));
        CK(cudaSetDevice(dst_));
        CK(cudaStreamCreateWithFlags(&ds_, cudaStreamNonBlocking));
        for (int b = 0; b < 2; ++b) CK(cudaEventCreateWithFlags(&ev_[b], cudaEventDisableTiming));
    } catch (...) {
        for (int b = 0; b < 2; ++b) { if (ev_[b]) cudaEventDestroy(ev_[b]); if (!shared_) cudaFreeHost(pin_[b]); }
        if (shared_) g_cross_out = false;
        if (ds_) { cudaSetDevice(dst_); cudaStreamDestroy(ds_); }
        if (ss_) { cudaSetDevice(src_); cudaStreamDestroy(ss_); }
        cudaSetDevice(dst_);
        cudaGetLastError();
        throw;
    }
}
CrossCopier::~CrossCopier() {
    cudaSetDevice(dst_);
    cudaStreamSynchronize(ds_);
    for (int b = 0; b < 2; ++b) { cudaEventDestroy(ev_[b]); if (!shared_) cudaFreeHost(pin_[b]); }
    if (shared_) g_cross_out = false;
    cudaStreamDestroy(ds_);
    cudaSetDevice(src_);
    cudaStreamDestroy(ss_);
    cudaSetDevice(dst_);
}
void CrossCopier::copy(void* dst, const void* src, size_t bytes) {
    bool used[2] = {false, false};
    size_t i = 0;
    for (size_t off = 0; off < bytes; off += chunk_, ++i) {
        int b = (int)(i & 1);
        size_t len = std::min(chunk_, bytes - off);
        if (used[b]) CK(cudaEventSynchronize(ev_[b]));   // the H2D that read this buffer two chunks ago is done
        CK(cudaSetDevice(src_));
        CK(cudaMemcpyAsync(pin_[b], (const uint8_t*)src + off, len, cudaMemcpyDeviceToHost, ss_));
        CK(cudaStreamSynchronize(ss_));
        CK(cudaSetDevice(dst_));
        CK(cudaMemcpyAsync((uint8_t*)dst + off, pin_[b], len, cudaMemcpyHostToDevice, ds_));
        CK(cudaEventRecord(ev_[b], ds_));
        used[b] = true;
    }
    CK(cudaSetDevice(dst_));
    CK(cudaStreamSynchronize(ds_));
}

void merge_device(FoldState& a, const FoldState& b) {
    if (a.dev == b.dev) throw std::runtime_error("merge_device: same device");
    CK(cudaSetDevice(b.dev));
    CK(cudaDeviceSynchronize());
    CK(cudaSetDevice(a.dev));
    CK(cudaDeviceSynchronize());
    size_t n = (size_t)a.w * a.h;
    struct Tmp { uint8_t* p = nullptr; ~Tmp() { cudaFree(p); } } tbuf;
    CK(cudaMalloc(&tbuf.p, n * 3 * sizeof(double)));   // the largest array; the three B arrays together fit too (18 B/px)
    uint8_t* tmp = tbuf.p;
    CrossCopier cc(a.dev, b.dev);
    cudaStream_t s = 0;
    auto stage = [&](const void* src, size_t bytes, size_t at = 0) { cc.copy(tmp + at, src, bytes); return tmp + at; };
    if (a.a_num && b.a_num) {
        add_f64(a.a_num, (double*)stage(b.a_num, n * 3 * sizeof(double)), n * 3, s);
        CK(cudaDeviceSynchronize());
        add_f64(a.a_den, (double*)stage(b.a_den, n * sizeof(double)), n, s);
        CK(cudaDeviceSynchronize());
    }
    add_f64(a.b_fmsum, (double*)stage(b.b_fmsum, n * sizeof(double)), n, s);
    CK(cudaDeviceSynchronize());
    // the three arrays staged together: the 12-byte colour first, then the floats, then the int16s, so every
    // offset keeps its alignment whatever m is (a level with an odd pixel count would otherwise put the colour at 6m)
    auto take = [&](float* a_best, int16_t* a_idx, float* a_col, const float* bb, const int16_t* bi, const float* bc, size_t m) {
        float* sc = (float*)stage(bc, m * 3 * sizeof(float), 0);
        float* sb = (float*)stage(bb, m * sizeof(float), m * 3 * sizeof(float));
        int16_t* si = (int16_t*)stage(bi, m * sizeof(int16_t), m * 4 * sizeof(float));
        merge_take(a_best, a_idx, a_col, sb, si, sc, m, s);
        CK(cudaDeviceSynchronize());
    };
    take(a.b_best, a.b_idx, a.b_col, b.b_best, b.b_idx, b.b_col, n);
    for (int l = 0; l < a.levels - 1; ++l) {
        size_t m = (size_t)a.shapes[l].w * a.shapes[l].h;
        take(a.c_sal[l], a.c_idx[l], a.c_coef[l], b.c_sal[l], b.c_idx[l], b.c_coef[l], m);
    }
    size_t t = (size_t)a.shapes.back().w * a.shapes.back().h * 3;
    add_f64(a.c_base, (double*)stage(b.c_base, t * sizeof(double)), t, s);
    CK(cudaDeviceSynchronize());
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
