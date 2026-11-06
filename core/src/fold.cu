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

__constant__ float c_luma[3] = {0.2126f, 0.7152f, 0.0722f};

__device__ __forceinline__ int refl101(int p, int n) {
    if (n == 1) return 0;
    while ((unsigned)p >= (unsigned)n) p = p < 0 ? -p : 2 * n - 2 - p;
    return p;
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

// ---- state updates (the _take rule: higher value wins, ties go to the lower frame index)

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

void FoldScratch::alloc(int w_, int h_, int levels_, float radius) {
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
}

void FoldScratch::free() {
    cudaFree(y); cudaFree(t1); cudaFree(t2); cudaFree(fm);
    for (auto p : g) if (p) cudaFree(p);
    for (auto p : lap) cudaFree(p);
    for (auto p : sal) cudaFree(p);
    g.clear(); lap.clear(); sal.clear();
    g_small.free();
    g_big.free();
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
