#include "stackcore/compose.h"

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>

#include "stackcore/depth.h"
#include "stackcore/fold.h"
#include "stackcore/volume.h"

namespace stackcore {

#define CK(x)                                                                                        \
    do {                                                                                             \
        cudaError_t e_ = (x);                                                                        \
        if (e_ != cudaSuccess) throw std::runtime_error(std::string(#x) + ": " + cudaGetErrorString(e_)); \
    } while (0)

namespace {

__constant__ float c_luma[3] = {0.2126f, 0.7152f, 0.0722f};

inline unsigned blocks1(size_t n) { return (unsigned)((n + 255) / 256); }

template <class T>
T* dalloc(size_t n) {
    T* p;
    CK(cudaMalloc(&p, n * sizeof(T)));
    return p;
}
template <class T>
T* upload(const std::vector<T>& v) {
    T* p = dalloc<T>(v.size());
    CK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    CK(cudaStreamSynchronize(0));   // the staged DMA has landed before any stream's kernel reads it (FINDINGS 43)
    return p;
}
template <class T>
std::vector<T> dl(const T* d, size_t n) {
    std::vector<T> v(n);
    CK(cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
}

__global__ void k_fill(float* p, size_t n, float v) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = v;
}

__global__ void k_normalise(const float* fm, const float* m, float* out, size_t n, float floor2) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float mm = m[i]; out[i] = fm[i] / (mm * mm + floor2); }
}

__global__ void k_refine_acc(float* __restrict__ bw, int16_t* __restrict__ iw, const float* __restrict__ fm,
                             const float* __restrict__ D, size_t n, int k, float R) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float fw = fabsf(D[i] - (float)k) <= R ? fm[i] : -INFINITY;
    if (fw > bw[i] || (fw == bw[i] && k < iw[i])) { bw[i] = fw; iw[i] = (int16_t)k; }
}

// depth_s = where(confs > 0.5 & spill < 0.3, med3(refined), depth_s); refined = isfinite(bw) ? iw : depth_s
__global__ void k_refined(const float* bw, const int16_t* iw, const float* ds, float* out, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = isfinite(bw[i]) ? (float)iw[i] : ds[i];
}
__global__ void k_select(const float* confs, const float* spill, const float* med, float* ds, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && confs[i] > 0.5f && spill[i] < 0.3f) ds[i] = med[i];
}
__global__ void k_clipf(float* a, size_t n, float lo, float hi) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = fminf(fmaxf(a[i], lo), hi);
}

// wmap (optional): the width in frames of the smooth-depth blend, per pixel. 1 = the two neighbouring frames (the
// reference). Wider in defocused background (FINDINGS 41): there the depth ramps through the stack, and a two-frame
// blend that changes from pixel to pixel draws iso-frame rings; averaging the frames within +-w of the depth makes
// neighbouring pixels' mixtures almost the same, and the rings vanish.
__global__ void k_labels(const float* __restrict__ D, const float* __restrict__ Ds, const float* __restrict__ wmap,
                         float* __restrict__ lab, float* __restrict__ wi, size_t n, float k) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    lab[i] = fabsf(D[i] - k) < 0.5f ? 1.f : 0.f;
    float w = wmap ? fmaxf(wmap[i], 1.f) : 1.f;
    wi[i] = fminf(fmaxf(1.0f - fabsf(Ds[i] - k) / w, 0.f), 1.f);
}
// width map from the flat mask at depth resolution: 1 + flat * (wflat - 1), resized to the output grid
__global__ void k_flat_width(const float* __restrict__ f, float* __restrict__ w, size_t n, float wflat) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) w[i] = 1.f + fminf(fmaxf(f[i], 0.f), 1.f) * (wflat - 1.f);
}
__global__ void k_compose_acc(double* __restrict__ n1, double* __restrict__ d1, double* __restrict__ n2, double* __restrict__ d2,
                              const float* __restrict__ w, const float* __restrict__ wi, const float* __restrict__ img, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* p = img + i * 3;
    double a = w[i], b = wi[i];
    n1[i * 3] += a * p[0]; n1[i * 3 + 1] += a * p[1]; n1[i * 3 + 2] += a * p[2];
    d1[i] += a;
    n2[i * 3] += b * p[0]; n2[i * 3 + 1] += b * p[1]; n2[i * 3 + 2] += b * p[2];
    d2[i] += b;
}

// b = n / max(d, 1e-12) in double; y = luma of float32(b)
__global__ void k_divide(const double* n, const double* d, double* b, float* y, size_t npx) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npx) return;
    double den = fmax(d[i], 1e-12);
    double r = n[i * 3] / den, g = n[i * 3 + 1] / den, bb = n[i * 3 + 2] / den;
    b[i * 3] = r; b[i * 3 + 1] = g; b[i * 3 + 2] = bb;
    if (y) y[i] = (float)(r * (double)c_luma[0] + g * (double)c_luma[1] + bb * (double)c_luma[2]);   // (b1 @ LUMA).astype(f32)
}
// ftex = blap / (by + 0.02); c_tex = clip((ftex / floor - lo) / (hi - lo), 0, 1)  (two steps: floor needs the percentile)
__global__ void k_ftex(const float* blap, const float* by, float* ftex, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) ftex[i] = blap[i] / (by[i] + 0.02f);
}
__global__ void k_crisp(const float* ftex, const float* bconf, float* crisp, size_t n, float floor, float lo, float hi) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float c_tex = fminf(fmaxf((ftex[i] / floor - lo) / (hi - lo), 0.f), 1.f);
    float c_focus = fminf(fmaxf((bconf[i] - 0.3f) / 0.3f, 0.f), 1.f);
    crisp[i] = 1.0f - (1.0f - c_tex) * (1.0f - c_focus);
}
__global__ void k_mix(const float* cm, const double* b1, const double* b2, float* out, size_t npx) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npx) return;
    double c = cm[i];
    for (int ch = 0; ch < 3; ++ch) out[i * 3 + ch] = (float)(c * b1[i * 3 + ch] + (1.0 - c) * b2[i * 3 + ch]);
}

// ---- finalize kernels
__global__ void k_a_out(const double* num, const double* den, float* out, size_t npx) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npx) return;
    double d = fmax(den[i], 1e-30);
    for (int c = 0; c < 3; ++c) out[i * 3 + c] = (float)(num[i * 3 + c] / d);
}
__global__ void k_d2f(const double* in, float* out, size_t n, double scale) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (float)(in[i] / scale);
}
// take_b = |c_idx - dl| > agree & c_sal < dominance * sB; wb = take_b (float)
__global__ void k_take_b(const int16_t* c_idx, const float* dl, const float* c_sal, const float* sB, float* wb, size_t n,
                         float agree, float dominance) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    bool t = fabsf((float)c_idx[i] - dl[i]) > agree && c_sal[i] < dominance * sB[i];
    wb[i] = t ? 1.f : 0.f;
}
// Cx (FINDINGS 50): take_b = |c_idx - dl| > agree & fewer than min_n of the 2x2 finer pixels under this one chose a
// frame within `near` of its choice. `fine` is level l-1's chosen frame (fw x fh); the coarse grid is w x h.
__global__ void k_take_support(const int16_t* c_idx, const int16_t* fine, const float* dl, float* wb, int w, int h, int fw, int fh,
                               float agree, int near_fr, int min_n) {   // (`near` is a Windows macro)
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)w * h) return;
    int c = c_idx[i];
    if (fabsf((float)c - dl[i]) <= agree) { wb[i] = 0.f; return; }
    int x = (int)(i % w), y = (int)(i / w), n = 0;
    for (int dy = 0; dy < 2; ++dy)
        for (int dx = 0; dx < 2; ++dx) {
            int fx = min(2 * x + dx, fw - 1), fy = min(2 * y + dy, fh - 1);
            if (abs((int)fine[(size_t)fy * fw + fx] - c) <= near_fr) ++n;
        }
    wb[i] = n < min_n ? 1.f : 0.f;
}

// far = |c_idx - dl| > glint_agree
__global__ void k_far(const int16_t* c_idx, const float* dl, float* wb, size_t n, float glint) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) wb[i] = fabsf((float)c_idx[i] - dl[i]) > glint ? 1.f : 0.f;
}
// wb = max(wb, clip(1.5 - |cC @ LUMA| / (noise_k sig + 1e-12), 0, 1)); or just wn when `only`
__global__ void k_noise_w(const float* cC, float* wb, size_t n, float thr, bool only) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* p = cC + i * 3;
    float l = fabsf(p[0] * c_luma[0] + p[1] * c_luma[1] + p[2] * c_luma[2]);
    float r = l / thr;
    float wn = fminf(fmaxf(1.5f - r, 0.f), 1.f);
    wb[i] = only ? wn : fmaxf(wb[i], wn);
}
// lv = wb * cB + (1 - wb) * cC  (x3)
__global__ void k_level_mix(const float* wb, const float* cB, const float* cC, float* lv, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float w = wb[i];
    for (int c = 0; c < 3; ++c) lv[i * 3 + c] = w * cB[i * 3 + c] + (1.f - w) * cC[i * 3 + c];
}
// H = swap chroma: a = y_src @ M^T, b = c_src @ M^T, a[1:] = b[1:], out = a @ Mi^T
__constant__ float c_M[9] = {0.2126f, 0.7152f, 0.0722f, -0.1146f, -0.3854f, 0.5f, 0.5f, -0.4542f, -0.0458f};
__constant__ float c_Mi[9];
__global__ void k_swap_chroma(const float* ysrc, const float* csrc, float* out, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* a = ysrc + i * 3;
    const float* b = csrc + i * 3;
    float v[3];
    v[0] = a[0] * c_M[0] + a[1] * c_M[1] + a[2] * c_M[2];
    v[1] = b[0] * c_M[3] + b[1] * c_M[4] + b[2] * c_M[5];
    v[2] = b[0] * c_M[6] + b[1] * c_M[7] + b[2] * c_M[8];
    for (int c = 0; c < 3; ++c) out[i * 3 + c] = v[0] * c_Mi[c * 3] + v[1] * c_Mi[c * 3 + 1] + v[2] * c_Mi[c * 3 + 2];
}
__global__ void k_absdiff_luma(const float* a, const float* b, float* out, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float d0 = a[i * 3] - b[i * 3], d1 = a[i * 3 + 1] - b[i * 3 + 1], d2 = a[i * 3 + 2] - b[i * 3 + 2];
    out[i] = fabsf(d0 * c_luma[0] + d1 * c_luma[1] + d2 * c_luma[2]);
}

float median_sorted(float* d, size_t n, cudaStream_t s) {
    thrust::sort(thrust::cuda::par.on(s), thrust::device_ptr<float>(d), thrust::device_ptr<float>(d + n));
    float v[2];
    CK(cudaMemcpyAsync(v, d + (n - 1) / 2, sizeof(float) * 2, cudaMemcpyDeviceToHost, s));
    CK(cudaStreamSynchronize(s));
    return n % 2 ? v[0] : (v[0] + v[1]) * 0.5f;
}

// Laplacian pyramid of an x3 image into fresh device buffers (lap[levels-1] = top Gaussian level)
struct DevPyramid {
    std::vector<LevelShape> shapes;
    std::vector<float*> lap;
    void build(const float* d_img, int w, int h, int levels, cudaStream_t s) {
        shapes = pyramid_shapes(w, h, levels);
        std::vector<float*> g(levels, nullptr);
        for (int l = 1; l < levels; ++l) {
            g[l] = dalloc<float>((size_t)shapes[l].w * shapes[l].h * 3);
            pyr_down_rgb(l == 1 ? d_img : g[l - 1], shapes[l - 1].w, shapes[l - 1].h, g[l], shapes[l].w, shapes[l].h, s);
        }
        for (int l = 0; l < levels - 1; ++l) {
            float* lp = dalloc<float>((size_t)shapes[l].w * shapes[l].h * 3);
            lap_level_rgb(l == 0 ? d_img : g[l], g[l + 1], shapes[l + 1].w, shapes[l + 1].h, lp, shapes[l].w, shapes[l].h, s);
            lap.push_back(lp);
        }
        lap.push_back(g[levels - 1]);
        for (int l = 1; l < levels - 1; ++l) cudaFree(g[l]);
    }
    void free() { for (auto p : lap) cudaFree(p); lap.clear(); }
};

}  // namespace

// ----------------------------------------------------------------------------- refinement

// the composition so far (narrow labels: n1 / d1, black where no frame has contributed yet), point-sampled
__global__ void k_compose_preview(const double* __restrict__ num, const double* __restrict__ den, int w, int h, int down,
                                  float* __restrict__ out, int pw, int ph) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= pw || y >= ph) return;
    size_t i = (size_t)min(y * down + down / 2, h - 1) * w + min(x * down + down / 2, w - 1);
    double d = den[i];
    for (int c = 0; c < 3; ++c) out[((size_t)y * pw + x) * 3 + c] = d > 0 ? (float)(num[i * 3 + c] / d) : 0.f;
}

void compose_preview_kernel(const ComposeState& st, int down, float* d_out, int& pw, int& ph, cudaStream_t s) {
    down = std::max(1, down);
    pw = std::max(1, st.w / down);
    ph = std::max(1, st.h / down);
    k_compose_preview<<<dim3((pw + 31) / 32, (ph + 7) / 8), dim3(32, 8), 0, s>>>(st.n1, st.d1, st.w, st.h, down, d_out, pw, ph);
}

// the refined depth so far: the frame index that holds the best fine focus (grey, / (K-1)); -inf = nothing yet -> 0
__global__ void k_refine_preview(const float* __restrict__ bw, const int16_t* __restrict__ iw, int w, int h, int down, float inv,
                                 float* __restrict__ out, int pw, int ph) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= pw || y >= ph) return;
    size_t i = (size_t)min(y * down + down / 2, h - 1) * w + min(x * down + down / 2, w - 1);
    float v = bw[i] > -INFINITY ? (float)iw[i] * inv : 0.f;
    for (int c = 0; c < 3; ++c) out[((size_t)y * pw + x) * 3 + c] = v;
}

void refine_preview_kernel(const RefineState& st, int K, int down, float* d_out, int& pw, int& ph, cudaStream_t s) {
    down = std::max(1, down);
    pw = std::max(1, st.w / down);
    ph = std::max(1, st.h / down);
    k_refine_preview<<<dim3((pw + 31) / 32, (ph + 7) / 8), dim3(32, 8), 0, s>>>(st.bw, st.iw, st.w, st.h, down, 1.f / (float)std::max(K - 1, 1), d_out, pw, ph);
}

void RefineState::alloc(int dev_, int w_, int h_) {
    dev = dev_; w = w_; h = h_;
    size_t n = (size_t)w * h;
    CK(cudaSetDevice(dev));
    bw = dalloc<float>(n);
    k_fill<<<blocks1(n), 256>>>(bw, n, -INFINITY);
    iw = dalloc<int16_t>(n);
    CK(cudaMemset(iw, 0, n * sizeof(int16_t)));
    CK(cudaDeviceSynchronize());
}
void RefineState::free() { if (dev >= 0) { cudaSetDevice(dev); cudaFree(bw); cudaFree(iw); dev = -1; } }

void RefineScratch::alloc(int w_, int h_, float fm_sigma) {
    w = w_; h = h_;
    size_t n = (size_t)w * h;
    y = dalloc<float>(n); t1 = dalloc<float>(n); t2 = dalloc<float>(n); fm = dalloc<float>(n);
    g08.build(0.8); g_fm.build(fm_sigma); g_norm.build(3.0 * fm_sigma);
}
void RefineScratch::free() { cudaFree(y); cudaFree(t1); cudaFree(t2); cudaFree(fm); g08.free(); g_fm.free(); g_norm.free(); }

void refine_measure(const float* d_img, RefineScratch& sc, cudaStream_t s) {
    size_t n = (size_t)sc.w * sc.h;
    luma(d_img, sc.y, n, s);
    gaussian_blur(sc.y, sc.t2, sc.t1, sc.w, sc.h, sc.g08, s);
    laplacian3_sq(sc.t2, sc.t1, sc.w, sc.h, s);
    gaussian_blur(sc.t1, sc.fm, sc.t2, sc.w, sc.h, sc.g_fm, s);
    gaussian_blur(sc.y, sc.t1, sc.t2, sc.w, sc.h, sc.g_norm, s);
    k_normalise<<<blocks1(n), 256, 0, s>>>(sc.fm, sc.t1, sc.fm, n, 0.02f * 0.02f);
}

void refine_accumulate(RefineState& st, int k, const float* d_fm, const float* d_depth, int R, cudaStream_t s) {
    size_t n = (size_t)st.w * st.h;
    k_refine_acc<<<blocks1(n), 256, 0, s>>>(st.bw, st.iw, d_fm, d_depth, n, k, (float)R);
}

HostRefine download(const RefineState& st) {
    CK(cudaSetDevice(st.dev));
    CK(cudaDeviceSynchronize());
    HostRefine h;
    h.w = st.w; h.h = st.h;
    size_t n = (size_t)st.w * st.h;
    h.bw = dl(st.bw, n);
    h.iw = dl(st.iw, n);
    return h;
}
void merge_into(HostRefine& a, const HostRefine& b) {
    for (size_t i = 0; i < a.bw.size(); ++i)
        if (b.bw[i] > a.bw[i] || (b.bw[i] == a.bw[i] && b.iw[i] < a.iw[i])) { a.bw[i] = b.bw[i]; a.iw[i] = b.iw[i]; }
}
void merge_device(RefineState& a, const RefineState& b) {
    CK(cudaSetDevice(b.dev)); CK(cudaDeviceSynchronize());
    CK(cudaSetDevice(a.dev)); CK(cudaDeviceSynchronize());
    size_t n = (size_t)a.w * a.h;
    float* sb = dalloc<float>(n);
    int16_t* si = dalloc<int16_t>(n);
    {
        CrossCopier cc(a.dev, b.dev);
        cc.copy(sb, b.bw, n * sizeof(float));
        cc.copy(si, b.iw, n * sizeof(int16_t));
    }
    merge_take(a.bw, a.iw, nullptr, sb, si, nullptr, n, 0);
    CK(cudaDeviceSynchronize());
    cudaFree(sb); cudaFree(si);
}

void refine_depth(std::vector<float>& depth_s, const std::vector<float>& conf, const std::vector<float>& spill, int dw, int dh,
                  const RefineState& r, int ow, int oh, double scale, cudaStream_t s) {
    size_t n = (size_t)ow * oh;
    float* d_ds = upload(depth_s);
    float* d_conf = upload(conf);
    float* d_sp = upload(spill);
    const float* d_bw = r.bw;
    const int16_t* d_iw = r.iw;
    float *confs = dalloc<float>(n), *tmp = dalloc<float>(n), *spup = dalloc<float>(n), *ref = dalloc<float>(n), *med = dalloc<float>(n);
    DevGaussian g;
    g.build(std::max(1.0, 6 * scale));
    gaussian_blur(d_conf, confs, tmp, ow, oh, g, s);
    resize_linear(d_sp, dw, dh, spup, ow, oh, s);
    k_refined<<<blocks1(n), 256, 0, s>>>(d_bw, d_iw, d_ds, ref, n);
    median3(ref, med, ow, oh, s);
    k_select<<<blocks1(n), 256, 0, s>>>(confs, spup, med, d_ds, n);
    CK(cudaMemcpyAsync(depth_s.data(), d_ds, n * sizeof(float), cudaMemcpyDeviceToHost, s));
    CK(cudaStreamSynchronize(s));
    g.free();
    for (void* p : {(void*)d_ds, (void*)d_conf, (void*)d_sp, (void*)confs, (void*)tmp, (void*)spup, (void*)ref, (void*)med}) cudaFree(p);
}

std::vector<float> smooth_depth(const std::vector<float>& depth_s, const float* d_guide_full, int ow, int oh, double scale, int nfr,
                                cudaStream_t s) {
    size_t n = (size_t)ow * oh;
    float* d_ds = upload(depth_s);
    float *m1 = dalloc<float>(n), *m2 = dalloc<float>(n), *sc5 = dalloc<float>(n * 5), *out = dalloc<float>(n);
    median5(d_ds, m1, ow, oh, s);
    median5(m1, m2, ow, oh, s);
    int r = std::max(2, (int)std::nearbyint(40.0 * scale));
    guided_filter(d_guide_full, m2, out, ow, oh, r, 2e-3f, sc5, s);
    k_clipf<<<blocks1(n), 256, 0, s>>>(out, n, 0.f, (float)(nfr - 1));
    std::vector<float> h(n);
    CK(cudaMemcpyAsync(h.data(), out, n * sizeof(float), cudaMemcpyDeviceToHost, s));
    CK(cudaStreamSynchronize(s));
    for (float* p : {d_ds, m1, m2, sc5, out}) cudaFree(p);
    return h;
}

// ----------------------------------------------------------------------------- composition

void ComposeState::alloc(int dev_, int w_, int h_) {
    dev = dev_; w = w_; h = h_;
    size_t n = (size_t)w * h;
    CK(cudaSetDevice(dev));
    n1 = dalloc<double>(n * 3); d1 = dalloc<double>(n); n2 = dalloc<double>(n * 3); d2 = dalloc<double>(n);
    CK(cudaMemset(n1, 0, n * 3 * sizeof(double))); CK(cudaMemset(d1, 0, n * sizeof(double)));
    CK(cudaMemset(n2, 0, n * 3 * sizeof(double))); CK(cudaMemset(d2, 0, n * sizeof(double)));
    CK(cudaDeviceSynchronize());
}
void ComposeState::free() { if (dev >= 0) { cudaSetDevice(dev); cudaFree(n1); cudaFree(d1); cudaFree(n2); cudaFree(d2); dev = -1; } }

void ComposeScratch::alloc(int w_, int h_, float soft_sigma) {
    w = w_; h = h_;
    size_t n = (size_t)w * h;
    lab = dalloc<float>(n); wn = dalloc<float>(n); tmp = dalloc<float>(n);
    g_s1.build(soft_sigma);
}
void ComposeScratch::free() { cudaFree(lab); cudaFree(wn); cudaFree(tmp); g_s1.free(); }

std::vector<int> compose_frames(const std::vector<float>& D, const std::vector<float>& Ds, int n, float wmax) {
    std::vector<char> use(n, 0);
    const float w = std::max(1.f, wmax);
    for (size_t i = 0; i < D.size(); ++i) {
        int k = (int)std::nearbyint(D[i]);                     // |D - k| < 0.5
        if (k >= 0 && k < n && std::fabs(D[i] - k) < 0.5f) use[k] = 1;
        int lo = (int)std::floor(Ds[i] - w) + 1, hi = (int)std::ceil(Ds[i] + w) - 1;   // |Ds - k| < w
        for (int k2 = std::max(0, lo); k2 <= std::min(n - 1, hi); ++k2) if (std::fabs(Ds[i] - k2) < w) use[k2] = 1;
    }
    std::vector<int> out;
    for (int k = 0; k < n; ++k) if (use[k]) out.push_back(k);
    return out;
}

void flat_width_map(const float* d_flat, int dw, int dh, float* d_W, int ow, int oh, float wflat, cudaStream_t s) {
    resize_linear(d_flat, dw, dh, d_W, ow, oh, s);
    k_flat_width<<<blocks1((size_t)ow * oh), 256, 0, s>>>(d_W, d_W, (size_t)ow * oh, wflat);
}

void compose_frame(ComposeState& st, int k, const float* d_img, const float* d_D, const float* d_Ds, ComposeScratch& sc,
                   cudaStream_t s, const float* d_W) {
    size_t n = (size_t)st.w * st.h;
    k_labels<<<blocks1(n), 256, 0, s>>>(d_D, d_Ds, d_W, sc.lab, sc.wn, n, (float)k);
    gaussian_blur(sc.lab, sc.lab, sc.tmp, st.w, st.h, sc.g_s1, s);   // soft label (in place via tmp)
    k_compose_acc<<<blocks1(n), 256, 0, s>>>(st.n1, st.d1, st.n2, st.d2, sc.lab, sc.wn, d_img, n);
}

HostCompose download(const ComposeState& st) {
    CK(cudaSetDevice(st.dev));
    CK(cudaDeviceSynchronize());
    HostCompose h;
    h.w = st.w; h.h = st.h;
    size_t n = (size_t)st.w * st.h;
    h.n1 = dl(st.n1, n * 3); h.d1 = dl(st.d1, n); h.n2 = dl(st.n2, n * 3); h.d2 = dl(st.d2, n);
    return h;
}
void merge_into(HostCompose& a, const HostCompose& b) {
    for (size_t i = 0; i < a.n1.size(); ++i) { a.n1[i] += b.n1[i]; a.n2[i] += b.n2[i]; }
    for (size_t i = 0; i < a.d1.size(); ++i) { a.d1[i] += b.d1[i]; a.d2[i] += b.d2[i]; }
}
void merge_device(ComposeState& a, const ComposeState& b) {
    CK(cudaSetDevice(b.dev)); CK(cudaDeviceSynchronize());
    CK(cudaSetDevice(a.dev)); CK(cudaDeviceSynchronize());
    size_t n = (size_t)a.w * a.h;
    double* tmp = dalloc<double>(n * 3);
    CrossCopier cc(a.dev, b.dev);
    struct P { double* dst; const double* src; size_t cnt; } parts[4] = {{a.n1, b.n1, n * 3}, {a.d1, b.d1, n}, {a.n2, b.n2, n * 3}, {a.d2, b.d2, n}};
    for (auto& p : parts) {
        cc.copy(tmp, p.src, p.cnt * sizeof(double));
        add_f64(p.dst, tmp, p.cnt, 0);
        CK(cudaDeviceSynchronize());
    }
    cudaFree(tmp);
}

std::vector<float> compose_mix(const ComposeState& c, const std::vector<float>& conf, int ow, int oh, double scale, cudaStream_t s) {
    size_t n = (size_t)ow * oh;
    const double *n1 = c.n1, *d1 = c.d1, *n2 = c.n2, *d2 = c.d2;
    double *b1 = dalloc<double>(n * 3), *b2 = dalloc<double>(n * 3);
    float *y1 = dalloc<float>(n), *t1 = dalloc<float>(n), *t2 = dalloc<float>(n), *t3 = dalloc<float>(n), *t4 = dalloc<float>(n);
    k_divide<<<blocks1(n), 256, 0, s>>>(n1, d1, b1, y1, n);
    k_divide<<<blocks1(n), 256, 0, s>>>(n2, d2, b2, nullptr, n);
    DevGaussian g07, gfs, gc, gm;
    g07.build(0.7);
    double fs = std::max(1.5, 3.0 * scale);
    gfs.build(fs);
    gc.build(std::max(2.0, 6 * scale));
    gm.build(std::max(2.0, 8 * scale));
    gaussian_blur(y1, t2, t1, ow, oh, g07, s);
    laplacian3_abs(t2, t1, ow, oh, s);              // lapf
    gaussian_blur(t1, t3, t2, ow, oh, gfs, s);      // blur(lapf)
    gaussian_blur(y1, t1, t2, ow, oh, gfs, s);      // blur(y1)
    k_ftex<<<blocks1(n), 256, 0, s>>>(t3, t1, t4, n);   // ftex -> t4
    float floor = percentile(t4, t1, n, 20.0, s) + 1e-9f;
    float* d_conf = upload(conf);
    gaussian_blur(d_conf, t1, t2, ow, oh, gc, s);   // blur(conf) -> t1
    k_crisp<<<blocks1(n), 256, 0, s>>>(t4, t1, t2, n, floor, 1.6f, 3.0f);   // crisp -> t2
    gaussian_blur(t2, t3, t1, ow, oh, gm, s);       // cm -> t3
    float* out = dalloc<float>(n * 3);
    k_mix<<<blocks1(n), 256, 0, s>>>(t3, b1, b2, out, n);
    std::vector<float> h(n * 3);
    CK(cudaMemcpyAsync(h.data(), out, n * 3 * sizeof(float), cudaMemcpyDeviceToHost, s));
    CK(cudaStreamSynchronize(s));
    g07.free(); gfs.free(); gc.free(); gm.free();
    for (void* p : {(void*)b1, (void*)b2, (void*)y1, (void*)t1, (void*)t2, (void*)t3, (void*)t4, (void*)d_conf, (void*)out}) cudaFree(p);
    return h;
}

// ----------------------------------------------------------------------------- finalize

FinalOutputs finalize(const FoldState& st, const std::vector<float>& noise_mad, const std::vector<float>& depth_s,
                      const std::vector<float>& b_img, const FinalParams& p, cudaStream_t s) {
    FinalOutputs out;
    int w = st.w, h = st.h, levels = st.levels;
    size_t n = (size_t)w * h;
    auto want = [&](const char* name) { return p.outputs.empty() || std::find(p.outputs.begin(), p.outputs.end(), name) != p.outputs.end(); };
    auto emit = [&](const std::string& name, const float* d, size_t cnt) {
        out.names.push_back(name);
        out.images.push_back(dl(d, cnt));
    };
    // what the requested outputs need: B's pyramid for every mix with B, the depth helpers for the
    // depth-locked ones; everything is computed in the same order as before, only the unrequested
    // outputs are skipped (FINDINGS 29)
    const bool want_H = want("H"), want_CBn = want("CBn") || want_H;
    const bool need_depth = want("CB") || want_CBn || want("Cd") || want("Cx");
    const bool need_B = (p.c_smoothing > 0.f && want("C")) || want("Cn") || need_depth || want("Cg") || want("Cg2");
    // A, B0
    if (want("A")) {
        if (!st.a_num) throw std::runtime_error("A requested but the fold ran without A's sums");
        float* a = dalloc<float>(n * 3);
        k_a_out<<<blocks1(n), 256, 0, s>>>(st.a_num, st.a_den, a, n);
        CK(cudaStreamSynchronize(s));
        emit("A", a, n * 3);
        cudaFree(a);
    }
    out.names.push_back("B");
    out.images.push_back(b_img);
    if (want("B0")) emit("B0", st.b_col, n * 3);
    // lapC on the device: the state's coef levels + base / count
    std::vector<const float*> lapC;
    for (int l = 0; l < levels - 1; ++l) lapC.push_back(st.c_coef[l]);
    float* bf = dalloc<float>((size_t)st.shapes.back().w * st.shapes.back().h * 3);
    fold_base_f32(st, bf, s);
    lapC.push_back(bf);
    CK(cudaStreamSynchronize(s));
    std::vector<const int16_t*> c_idx(st.c_idx.begin(), st.c_idx.end());
    std::vector<const float*> c_sal(st.c_sal.begin(), st.c_sal.end());
    float *coll = dalloc<float>(n * 3), *tmpA = dalloc<float>(n * 3), *tmpB = dalloc<float>(n * 3);
    auto collapse_levels = [&](const std::vector<const float*>& lv) {
        collapse_pyramid(std::vector<const float*>(lv.begin(), lv.end() - 1), lv.back(), st.shapes, coll, tmpA, tmpB, s);
        CK(cudaStreamSynchronize(s));
    };
    if (p.c_smoothing <= 0.f && want("C")) {
        collapse_levels(lapC);
        emit("C", coll, n * 3);
    }
    if (!need_B && !want_H) {
        cudaFree(bf); cudaFree(coll); cudaFree(tmpA); cudaFree(tmpB);
        return out;
    }
    // lapB
    float* d_b = upload(b_img);
    DevPyramid pyrB;
    pyrB.build(d_b, w, h, levels, s);
    // per-level helpers: depth at the level size (INTER_AREA), salience of cB
    std::vector<float*> dl_lvl(levels - 1, nullptr), sB(levels - 1, nullptr);
    if (need_depth) {
        float* d_depth = upload(depth_s);
        for (int l = 0; l < levels - 1; ++l) {
            auto sh = st.shapes[l];
            size_t m = (size_t)sh.w * sh.h;
            dl_lvl[l] = dalloc<float>(m);
            AreaResizer rz;
            rz.build(w, h, sh.w, sh.h);
            rz.run(d_depth, dl_lvl[l], 1, s);
            CK(cudaStreamSynchronize(s));
            rz.free();
            sB[l] = dalloc<float>(m);
            salience3(pyrB.lap[l], sh.w, sh.h, sB[l], s);
        }
        CK(cudaStreamSynchronize(s));
        cudaFree(d_depth);
    }
    DevGaussian g1;
    g1.build(1.0);
    std::vector<float*> lv(levels), wb(levels - 1), tmpw(levels - 1);
    for (int l = 0; l < levels - 1; ++l) {
        size_t m = (size_t)st.shapes[l].w * st.shapes[l].h;
        lv[l] = dalloc<float>(m * 3);
        wb[l] = dalloc<float>(m);
        tmpw[l] = dalloc<float>(m);
    }
    lv[levels - 1] = pyrB.lap[levels - 1];   // base from B
    auto level_weights = [&](int l, bool use_depth, bool use_noise, float agree_v, float noise_k, bool use_support = false) {
        auto sh = st.shapes[l];
        size_t m = (size_t)sh.w * sh.h;
        if (use_support && l >= 1)   // Cx: the depth test with the finer level's support instead of the dominance test
            k_take_support<<<blocks1(m), 256, 0, s>>>(c_idx[l], c_idx[l - 1], dl_lvl[l], wb[l], sh.w, sh.h, st.shapes[l - 1].w, st.shapes[l - 1].h,
                                                      agree_v, p.support_near, p.support_min);
        else if (use_depth) k_take_b<<<blocks1(m), 256, 0, s>>>(c_idx[l], dl_lvl[l], c_sal[l], sB[l], wb[l], m, agree_v, p.dominance);
        else CK(cudaMemsetAsync(wb[l], 0, m * sizeof(float), s));
        // `only`: the noise rule alone sets the weight; with a depth or support rule before it, it adds to that weight
        if (use_noise) k_noise_w<<<blocks1(m), 256, 0, s>>>(lapC[l], wb[l], m, noise_k * noise_mad[l] + 1e-12f, !(use_depth || (use_support && l >= 1)));
        gaussian_blur(wb[l], wb[l], tmpw[l], sh.w, sh.h, g1, s);
        k_level_mix<<<blocks1(m), 256, 0, s>>>(wb[l], pyrB.lap[l], lapC[l], lv[l], m * 1);
    };
    std::vector<const float*> lv_c(levels);
    for (int l = 0; l < levels; ++l) lv_c[l] = lv[l];
    if (p.c_smoothing > 0.f && want("C")) {   // C with smoothing: the noise gate at the chosen strength
        for (int l = 0; l < levels - 1; ++l) level_weights(l, false, true, p.agree, p.c_smoothing);
        collapse_levels(lv_c);
        emit("C", coll, n * 3);
    }
    std::vector<float> cbn_host;
    // Cx: level 0 has no finer level, so there only the noise rule applies (use_depth false)
    struct V { const char* name; bool depth, noise, support; };
    for (V v : {V{"CB", true, false, false}, V{"Cn", false, true, false}, V{"CBn", true, true, false}, V{"Cx", false, true, true}}) {
        bool is_cbn = std::string(v.name) == "CBn";
        if (!(is_cbn ? want_CBn : want(v.name))) continue;
        for (int l = 0; l < levels - 1; ++l) level_weights(l, v.depth, v.noise, p.agree, p.noise_k, v.support);
        collapse_levels(lv_c);
        if (!is_cbn || want("CBn")) emit(v.name, coll, n * 3);
        if (is_cbn) cbn_host = want("CBn") ? out.images.back() : dl(coll, n * 3);
    }
    // Cd: depth-locked fine levels, coarse levels from B
    if (want("Cd")) {
        for (int l = 0; l < levels - 1; ++l) {
            auto sh = st.shapes[l];
            size_t m = (size_t)sh.w * sh.h;
            if (l >= p.coarse_b_level) { CK(cudaMemcpyAsync(lv[l], pyrB.lap[l], m * 3 * sizeof(float), cudaMemcpyDeviceToDevice, s)); continue; }
            k_far<<<blocks1(m), 256, 0, s>>>(c_idx[l], dl_lvl[l], wb[l], m, p.glint_agree);
            k_noise_w<<<blocks1(m), 256, 0, s>>>(lapC[l], wb[l], m, p.noise_k * noise_mad[l] + 1e-12f, false);
            gaussian_blur(wb[l], wb[l], tmpw[l], sh.w, sh.h, g1, s);
            k_level_mix<<<blocks1(m), 256, 0, s>>>(wb[l], pyrB.lap[l], lapC[l], lv[l], m);
        }
        collapse_levels(lv_c);
        emit("Cd", coll, n * 3);
    }
    // Cg, Cg2
    for (auto [L0, name] : {std::pair<int, const char*>{p.coarse_b_level, "Cg"}, std::pair<int, const char*>{p.coarse_b_level - 1, "Cg2"}}) {
        if (!want(name)) continue;
        for (int l = 0; l < levels - 1; ++l) {
            auto sh = st.shapes[l];
            size_t m = (size_t)sh.w * sh.h;
            if (l >= L0) { CK(cudaMemcpyAsync(lv[l], pyrB.lap[l], m * 3 * sizeof(float), cudaMemcpyDeviceToDevice, s)); continue; }
            k_noise_w<<<blocks1(m), 256, 0, s>>>(lapC[l], wb[l], m, p.noise_k * noise_mad[l] + 1e-12f, true);
            gaussian_blur(wb[l], wb[l], tmpw[l], sh.w, sh.h, g1, s);
            k_level_mix<<<blocks1(m), 256, 0, s>>>(wb[l], pyrB.lap[l], lapC[l], lv[l], m);
        }
        collapse_levels(lv_c);
        emit(name, coll, n * 3);
    }
    // H: luma from CBn, chroma from B
    if (want_H) {
        // Mi = inverse of M (float32), computed on the host once
        double M[9] = {0.2126, 0.7152, 0.0722, -0.1146, -0.3854, 0.5, 0.5, -0.4542, -0.0458};
        float Mf[9];
        for (int i = 0; i < 9; ++i) Mf[i] = (float)M[i];
        // inverse of the float32 matrix in double, cast to float (np.linalg.inv on float32 -> float32 result)
        double a[9];
        for (int i = 0; i < 9; ++i) a[i] = Mf[i];
        double det = a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) + a[2] * (a[3] * a[7] - a[4] * a[6]);
        double inv[9] = {(a[4] * a[8] - a[5] * a[7]) / det, (a[2] * a[7] - a[1] * a[8]) / det, (a[1] * a[5] - a[2] * a[4]) / det,
                         (a[5] * a[6] - a[3] * a[8]) / det, (a[0] * a[8] - a[2] * a[6]) / det, (a[2] * a[3] - a[0] * a[5]) / det,
                         (a[3] * a[7] - a[4] * a[6]) / det, (a[1] * a[6] - a[0] * a[7]) / det, (a[0] * a[4] - a[1] * a[3]) / det};
        float Mi[9];
        for (int i = 0; i < 9; ++i) Mi[i] = (float)inv[i];
        CK(cudaMemcpyToSymbol(c_Mi, Mi, sizeof(Mi)));
        float* d_cbn = upload(cbn_host);
        k_swap_chroma<<<blocks1(n), 256, 0, s>>>(d_cbn, d_b, coll, n);
        CK(cudaStreamSynchronize(s));
        emit("H", coll, n * 3);
        cudaFree(d_cbn);
    }
    g1.free();
    cudaFree(bf);
    for (auto q : dl_lvl) cudaFree(q);
    for (auto q : sB) cudaFree(q);
    for (int l = 0; l < levels - 1; ++l) { cudaFree(lv[l]); cudaFree(wb[l]); cudaFree(tmpw[l]); }
    pyrB.free();
    cudaFree(d_b); cudaFree(coll); cudaFree(tmpA); cudaFree(tmpB);
    return out;
}

std::vector<float> noise_from_pair(const float* d_a, const float* d_b, int w, int h, int levels, cudaStream_t s) {
    DevPyramid pa, pb;
    pa.build(d_a, w, h, levels, s);
    pb.build(d_b, w, h, levels, s);
    std::vector<float> mad(levels - 1);
    float* tmp = dalloc<float>((size_t)w * h);
    for (int l = 0; l < levels - 1; ++l) {
        size_t m = (size_t)pa.shapes[l].w * pa.shapes[l].h;
        k_absdiff_luma<<<blocks1(m), 256, 0, s>>>(pa.lap[l], pb.lap[l], tmp, m);
        mad[l] = (float)(1.4826 * median_sorted(tmp, m, s) / std::sqrt(2.0));
    }
    cudaFree(tmp);
    pa.free(); pb.free();
    return mad;
}

// Written beside the target as <path>.tmp and renamed over it when every byte is on the disk, so a write that fails
// (the disk full, the target open in another program) leaves the file that was there as it was and no half-written
// TIFF with a valid header. `why`, when given, takes the system's reason for a failure.
bool write_tiff_rgb16(const std::string& path, const std::vector<float>& rgb, int w, int h, const std::vector<uint8_t>& icc, std::string* why) {
    auto fail = [&](const std::string& what) { if (why) *why = what; return false; };
    auto sys = [](const char* doing) { return std::string(doing) + ": " + strerror(errno); };
    size_t n = (size_t)w * h * 3;
    if (rgb.size() < n) return fail("the image is incomplete");
    if (n * 2 > 0xFFFF0000u - icc.size()) return fail("the image is too large for a TIFF file (4 GB)");
    const std::string tmp = path + ".tmp";
    FILE* f = fopen(tmp.c_str(), "wb");
    if (!f) return fail(sys("cannot create the file"));
    struct Shut { FILE*& f; const std::string& tmp; bool keep = false; ~Shut() { if (f) fclose(f); if (!keep) remove(tmp.c_str()); } } shut{f, tmp};
    // classic little-endian TIFF: header, IFD, BitsPerSample / SampleFormat arrays, ICC profile, then the strip
    bool has_icc = !icc.empty();
    uint32_t ifd_off = 8, n_ent = has_icc ? 12 : 11;
    uint32_t ifd_size = 2 + n_ent * 12 + 4;
    uint32_t bps_off = ifd_off + ifd_size, sf_off = bps_off + 6, icc_off = sf_off + 6;
    uint32_t data_off = icc_off + (has_icc ? (uint32_t)icc.size() : 0);
    if (data_off & 1) ++data_off;
    uint32_t nbytes = (uint32_t)(n * 2);
    auto put16 = [&](uint16_t v) { fwrite(&v, 2, 1, f); };
    auto put32 = [&](uint32_t v) { fwrite(&v, 4, 1, f); };
    auto entry = [&](uint16_t tag, uint16_t type, uint32_t count, uint32_t value) { put16(tag); put16(type); put32(count); put32(value); };
    fwrite("II", 1, 2, f); put16(42); put32(ifd_off);
    put16((uint16_t)n_ent);
    entry(256, 4, 1, (uint32_t)w);            // ImageWidth
    entry(257, 4, 1, (uint32_t)h);            // ImageLength
    entry(258, 3, 3, bps_off);                // BitsPerSample (16,16,16)
    entry(259, 3, 1, 1);                      // Compression none
    entry(262, 3, 1, 2);                      // Photometric RGB
    entry(273, 4, 1, data_off);               // StripOffsets
    entry(277, 3, 1, 3);                      // SamplesPerPixel
    entry(278, 4, 1, (uint32_t)h);            // RowsPerStrip
    entry(279, 4, 1, nbytes);                 // StripByteCounts
    entry(284, 3, 1, 1);                      // PlanarConfig chunky
    entry(339, 3, 3, sf_off);                 // SampleFormat (1,1,1)
    if (has_icc) entry(34675, 7, (uint32_t)icc.size(), icc_off);   // ICC profile (UNDEFINED bytes)
    put32(0);                                 // next IFD
    put16(16); put16(16); put16(16);
    put16(1); put16(1); put16(1);
    if (has_icc) fwrite(icc.data(), 1, icc.size(), f);
    long pos = ftell(f);
    while ((uint32_t)pos < data_off) { fputc(0, f); ++pos; }
    // the 16-bit rows, a block at a time (the whole image as one more buffer was 190 MB per output, all outputs at once)
    const size_t block = (size_t)w * 3 * 64;
    std::vector<uint16_t> u(std::min(block, n));
    for (size_t i0 = 0; i0 < n; i0 += u.size()) {
        const size_t c = std::min(u.size(), n - i0);
        for (size_t i = 0; i < c; ++i) {
            float v = std::min(std::max(rgb[i0 + i], 0.f), 1.f);
            u[i] = (uint16_t)(v * 65535.0f + 0.5f);
        }
        if (fwrite(u.data(), 2, c, f) != c) return fail(sys("cannot write the file"));
    }
    if (ferror(f)) return fail(sys("cannot write the file"));
    FILE* g = f;
    f = nullptr;
    if (fclose(g) != 0) return fail(sys("cannot write the file"));
    // one step that replaces the target (the C rename does not on Windows, and removing the target first lost the
    // previous file as well when the rename then failed)
    std::error_code ec;
    std::filesystem::rename(std::filesystem::path(tmp), std::filesystem::path(path), ec);
    if (ec) return fail("cannot replace the file (is it open in another program?): " + ec.message());
    shut.keep = true;
    return true;
}

}  // namespace stackcore
