#include "stackcore/depth.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <chrono>
#include <cmath>
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

inline unsigned blocks1(size_t n) { return (unsigned)((n + 255) / 256); }
inline dim3 grid2(int w, int h) { return dim3((w + 31) / 32, (h + 7) / 8); }
const dim3 BLK(32, 8);

__device__ __forceinline__ int refl101(int p, int n) {
    if (n == 1) return 0;
    while ((unsigned)p >= (unsigned)n) p = p < 0 ? -p : 2 * n - 2 - p;
    return p;
}

// ---- layout

__global__ void k_planes_to_k_f32(const float* __restrict__ in, float* __restrict__ out, int n, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // pixel
    if (i >= n) return;
    for (int k = 0; k < K; ++k) out[(size_t)i * K + k] = in[(size_t)k * n + i];
}
__global__ void k_planes_to_k_f16(const __half* __restrict__ in, __half* __restrict__ out, int n, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    for (int k = 0; k < K; ++k) out[(size_t)i * K + k] = in[(size_t)k * n + i];
}

// ---- per-pixel statistics of the focus curve: vmax, median, argmax; and of Lv: min, max

__global__ void k_curve_stats(const float* __restrict__ V, const __half* __restrict__ Lv, int n, int K,
                              float* __restrict__ vmax, float* __restrict__ vmed, int* __restrict__ kmax,
                              float* __restrict__ lmin, float* __restrict__ lmax) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* v = V + (size_t)i * K;
    float buf[512];   // K <= 512
    float mx = -1e30f, lo = 1e30f, hi = -1e30f;
    int km = 0;
    for (int k = 0; k < K; ++k) {
        float x = v[k];
        buf[k] = x;
        if (x > mx) { mx = x; km = k; }   // first maximum, like np.argmax
        float l = __half2float(Lv[(size_t)i * K + k]);
        lo = fminf(lo, l);
        hi = fmaxf(hi, l);
    }
    // median by insertion sort (K ~ 160)
    for (int a = 1; a < K; ++a) {
        float x = buf[a];
        int b = a - 1;
        while (b >= 0 && buf[b] > x) { buf[b + 1] = buf[b]; --b; }
        buf[b + 1] = x;
    }
    float med = (K & 1) ? buf[K / 2] : (buf[K / 2 - 1] + buf[K / 2]) * 0.5f;
    vmax[i] = mx;
    vmed[i] = med;
    kmax[i] = km;
    lmin[i] = lo;
    lmax[i] = hi;
}

// confidence w from contrast, end-frame mask; w *= (1 - h); w = 0 on end-peaked pixels
__global__ void k_confidence(const float* __restrict__ vmax, const float* __restrict__ vmed, const int* __restrict__ kmax,
                             const float* __restrict__ h, int n, int K, float lo, float hi, int eg,
                             float* __restrict__ w, uint8_t* __restrict__ endm) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float vm = vmax[i];
    float contrast = (vm - vmed[i]) / fmaxf(vm, 1e-12f);
    float ww = fminf(fmaxf((contrast - lo) / (hi - lo), 0.f), 1.f);
    ww = ww * (1.0f - h[i]);
    int km = kmax[i];
    uint8_t e = (km < eg || km > K - 1 - eg) ? 1 : 0;
    w[i] = e ? 0.f : ww;
    endm[i] = e | (km < eg ? 2 : 0);   // bit 1: at the first frames (for the end prior's direction)
}

// cost volume, in place over V (double intermediates where numpy promoted to float64)
__global__ void k_cost(float* __restrict__ V, const __half* __restrict__ Lv, const float* __restrict__ w,
                       const uint8_t* __restrict__ endm, const float* __restrict__ vmax, const float* __restrict__ lmin,
                       const float* __restrict__ lmax, const float* __restrict__ h, int n, int K, float ep, float lam,
                       float lam_h) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float* c = V + (size_t)i * K;
    float wi = w[i], vm = fmaxf(vmax[i], 1e-12f), hi = h[i];
    bool e = endm[i] & 1, at0 = endm[i] & 2;
    float lrange = fmaxf(lmax[i] - lmin[i], 0.01f), l0 = lmin[i];
    // numpy's dtypes: (1 - w)(1 - endm) lam dark is float64 (bool array - python float -> float64);
    // lam_h h dark is float32 (python float * float32 array); C accumulates in float64, cast at the end
    double w_dark = ((double)(1.0f - wi) * (e ? 0.0 : 1.0)) * (double)lam;
    float w_halo = lam_h * hi;
    for (int k = 0; k < K; ++k) {
        float base = wi * (1.0f - c[k] / vm);                              // float32
        double acc = base;
        if (ep > 0 && e) {
            float dist = (at0 ? (float)k : (float)(K - 1 - k)) / (float)(K > 1 ? K - 1 : 1);
            acc += ((double)ep * 1.0) * (double)dist;
        }
        if (lam > 0 || lam_h > 0) {
            float dark = (__half2float(Lv[(size_t)i * K + k]) - l0) / lrange;   // float32
            if (lam > 0) acc += w_dark * (double)dark;
            if (lam_h > 0) acc += (double)(w_halo * dark);
        }
        c[k] = (float)acc;
    }
}

// ---- spill zone h (all at the depth resolution)

// erode / dilate with an explicit structuring element (offsets), outside pixels ignored (OpenCV's default border)
__global__ void k_morph(const float* __restrict__ in, float* __restrict__ out, int w, int h, const int2* __restrict__ se, int nse, bool dilate) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float v = dilate ? -3.4e38f : 3.4e38f;
    for (int i = 0; i < nse; ++i) {
        int xx = x + se[i].x, yy = y + se[i].y;
        if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
        float s = in[(size_t)yy * w + xx];
        v = dilate ? fmaxf(v, s) : fminf(v, s);
    }
    out[(size_t)y * w + x] = v;
}

__global__ void k_spill_a(const float* __restrict__ nbmax, const float* __restrict__ g, float* __restrict__ h, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = ((nbmax[i] - g[i]) / (nbmax[i] + 0.02f) - 0.3f) / 0.4f;
    h[i] = fminf(fmaxf(v, 0.f), 1.f);
}

// h *= clip((0.20 - texq) / 0.08, 0, 1) with texq = blur(lapq) / (blur(g) + 0.02)
__global__ void k_spill_b(float* __restrict__ h, const float* __restrict__ blap, const float* __restrict__ bg, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float texq = blap[i] / (bg[i] + 0.02f);
    h[i] = h[i] * fminf(fmaxf((0.20f - texq) / 0.08f, 0.f), 1.f);
}

// ---- P2 map from the guide's Sobel gradient

__global__ void k_p2(const float* __restrict__ g, float* __restrict__ P2, int w, int h, float p1, float p2, float edge) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int xm = refl101(x - 1, w), xp = refl101(x + 1, w), ym = refl101(y - 1, h), yp = refl101(y + 1, h);
    const float* r0 = g + (size_t)ym * w;
    const float* r1 = g + (size_t)y * w;
    const float* r2 = g + (size_t)yp * w;
    // Sobel 3x3: dx = [1 2 1]^T [-1 0 1], dy = [-1 0 1]^T [1 2 1]; then / 8
    float gx = ((r0[xp] - r0[xm]) + 2.f * (r1[xp] - r1[xm]) + (r2[xp] - r2[xm])) / 8.f;
    float gy = ((r2[xm] - r0[xm]) + 2.f * (r2[x] - r0[x]) + (r2[xp] - r0[xp])) / 8.f;
    float grad = sqrtf(gx * gx + gy * gy);
    float v = p2 / (1.0f + grad / edge);
    P2[(size_t)y * w + x] = fmaxf(v, 2.f * p1);
}

// ---- SGM: one block per scanline, one thread per label; shared prev row with inf guards

template <int MAXK>
__global__ void k_sgm_pass(const float* __restrict__ C, const float* __restrict__ P2, float P1, float* __restrict__ S,
                           int W, int H, int K, int dx, int dy) {
    __shared__ float prev[MAXK + 2];
    __shared__ float red[32];
    int k = threadIdx.x;
    // scanline start from the block index
    int b = blockIdx.x, x0, y0, n;
    if (dy == 0) {                         // horizontal: one per row
        y0 = b; x0 = dx > 0 ? 0 : W - 1; n = W;
    } else if (dx == 0) {                  // vertical: one per column
        x0 = b; y0 = dy > 0 ? 0 : H - 1; n = H;
    } else {                               // diagonal: starts on the entry row (b < W) or the entry column
        int ey = dy > 0 ? 0 : H - 1, ex = dx > 0 ? 0 : W - 1;
        if (b < W) { x0 = b; y0 = ey; }
        else { x0 = ex; y0 = dy > 0 ? b - W + 1 : H - 2 - (b - W); }
        int nx = dx > 0 ? W - x0 : x0 + 1, ny = dy > 0 ? H - y0 : y0 + 1;
        n = min(nx, ny);
    }
    if (k == 0) { prev[0] = INFINITY; prev[MAXK + 1] = INFINITY; }
    float pm = 0.f;
    int x = x0, y = y0;
    for (int t = 0; t < n; ++t, x += dx, y += dy) {
        size_t base = ((size_t)y * W + x) * K;
        float c = k < K ? C[base + k] : INFINITY;
        float cur;
        if (t == 0) cur = c;
        else {
            float p = prev[k + 1];
            float m = fminf(fminf(p, fminf(prev[k], prev[k + 2]) + P1), pm + P2[(size_t)y * W + x]);
            cur = (c + m) - pm;
        }
        if (k < K) S[base + k] += cur;
        __syncthreads();                      // everyone has read prev for this step
        if (k < K) prev[k + 1] = cur;
        else if (k < MAXK) prev[k + 1] = INFINITY;
        // block min of cur over k < K
        float v = k < K ? cur : INFINITY;
        for (int o = 16; o > 0; o >>= 1) v = fminf(v, __shfl_down_sync(0xffffffffu, v, o));
        if ((k & 31) == 0) red[k >> 5] = v;
        __syncthreads();
        int nw = (blockDim.x + 31) >> 5;
        v = INFINITY;
        for (int i = 0; i < nw; ++i) v = fminf(v, red[i]);
        pm = v;
        __syncthreads();                      // red / prev stable before the next step
    }
}

// ---- argmin with sub-frame parabola (sgm.argmin_subpixel)

__global__ void k_argmin(const float* __restrict__ S, int n, int K, float* __restrict__ d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* s = S + (size_t)i * K;
    int k = 0;
    float best = s[0];
    for (int j = 1; j < K; ++j) if (s[j] < best) { best = s[j]; k = j; }
    if (K < 3) { d[i] = (float)k; return; }   // no neighbours for the parabola (2-frame stacks)
    int kc = min(max(k, 1), K - 2);
    float a = s[kc - 1], b = s[kc], c = s[kc + 1];
    float den = a - 2.f * b + c;
    float off = den > 1e-9f ? 0.5f * (a - c) / den : 0.f;
    off = (k >= 1 && k <= K - 2) ? fminf(fmaxf(off, -0.5f), 0.5f) : 0.f;
    d[i] = (float)kc + off;
}

// ---- cv::resize INTER_LINEAR (float coefficients, row pass then column pass)

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

__global__ void k_rint(float* a, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = rintf(a[i]);
}

// cv::medianBlur ksize 3 on float (replicate border)
__global__ void k_median3(const float* __restrict__ in, float* __restrict__ out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float v[9];
    int n = 0;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
            v[n++] = in[(size_t)min(max(y + dy, 0), h - 1) * w + min(max(x + dx, 0), w - 1)];
    for (int a = 1; a < 9; ++a) {
        float t = v[a];
        int b = a - 1;
        while (b >= 0 && v[b] > t) { v[b + 1] = v[b]; --b; }
        v[b + 1] = t;
    }
    out[(size_t)y * w + x] = v[4];
}

__global__ void k_gf_ab(const float* __restrict__ mI, const float* __restrict__ vI, const float* __restrict__ mp,
                        const float* __restrict__ mIp, float* __restrict__ a, float* __restrict__ b, size_t n, float eps) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float cIp = mIp[i] - mI[i] * mp[i];
    float av = cIp / (vI[i] + eps);
    a[i] = av;
    b[i] = mp[i] - av * mI[i];
}
__global__ void k_gf_out(const float* __restrict__ ba, const float* __restrict__ bb, const float* __restrict__ guide,
                         float* __restrict__ out, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = ba[i] * guide[i] + bb[i];
}
__global__ void k_sub_mul(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] - b[i] * b[i];
}

// cv::getStructuringElement(MORPH_ELLIPSE, (2r+1, 2r+1)) as offsets
std::vector<int2> ellipse_se(int r) {
    std::vector<int2> se;
    int rows = 2 * r + 1, cols = 2 * r + 1, c = cols / 2;
    double inv_r2 = r ? 1. / ((double)r * r) : 0;
    for (int i = 0; i < rows; ++i) {
        int dy = i - r, j1 = 0, j2 = 0;
        if (std::abs(dy) <= r) {
            int dx = (int)std::nearbyint(c * std::sqrt((r * r - dy * dy) * inv_r2));
            j1 = std::max(c - dx, 0);
            j2 = std::min(c + dx + 1, cols);
        }
        for (int j = j1; j < j2; ++j) se.push_back({j - c, i - r});
    }
    return se;
}

float* dmalloc(size_t n) {
    float* p;
    CK(cudaMalloc(&p, n * sizeof(float)));
    return p;
}

}  // namespace

// ----------------------------------------------------------------------------- API

void planes_to_kfastest_f32(const float* d_planes, float* d_out, int dw, int dh, int K, cudaStream_t s) {
    int n = dw * dh;
    k_planes_to_k_f32<<<blocks1(n), 256, 0, s>>>(d_planes, d_out, n, K);
}
void planes_to_kfastest_f16(const void* d_planes, void* d_out, int dw, int dh, int K, cudaStream_t s) {
    int n = dw * dh;
    k_planes_to_k_f16<<<blocks1(n), 256, 0, s>>>((const __half*)d_planes, (__half*)d_out, n, K);
}

void guided_filter(const float* d_guide, const float* d_src, float* d_out, int w, int h, int r, float eps, float* sc,
                   cudaStream_t s) {
    size_t n = (size_t)w * h;
    float *mI = sc, *vI = sc + n, *mp = sc + 2 * n, *mIp = sc + 3 * n, *tmp = sc + 4 * n;
    box_filter(d_guide, mI, tmp, w, h, r, s);
    mul(d_guide, d_guide, vI, n, s);
    box_filter(vI, vI, tmp, w, h, r, s);
    k_sub_mul<<<blocks1(n), 256, 0, s>>>(vI, mI, vI, n);                // vI = box(I I) - mI^2
    box_filter(d_src, mp, tmp, w, h, r, s);
    mul(d_guide, d_src, mIp, n, s);
    box_filter(mIp, mIp, tmp, w, h, r, s);
    k_gf_ab<<<blocks1(n), 256, 0, s>>>(mI, vI, mp, mIp, mIp, mp, n, eps);   // a -> mIp, b -> mp
    box_filter(mIp, mIp, tmp, w, h, r, s);                                 // box(a)
    box_filter(mp, mp, tmp, w, h, r, s);                                   // box(b)
    k_gf_out<<<blocks1(n), 256, 0, s>>>(mIp, mp, d_guide, d_out, n);
}

DepthResult depth_from_volume(const float* d_V_in, const void* d_Lv16, const float* d_guide_full, const float* d_ga,
                              int dw, int dh, int K, int ow, int oh, const DepthParams& p, cudaStream_t s) {
    if (K > 512) throw std::runtime_error("depth: K > 512 not supported");
    DepthResult R;
    R.dw = dw; R.dh = dh; R.ow = ow; R.oh = oh; R.K = K;
    int n = dw * dh;
    size_t no = (size_t)ow * oh;
    auto t0 = std::chrono::steady_clock::now();

    // --- per-pixel maps
    float *vmax = dmalloc(n), *vmed = dmalloc(n), *lmin = dmalloc(n), *lmax = dmalloc(n), *w = dmalloc(n), *h = dmalloc(n);
    int* kmax;
    CK(cudaMalloc(&kmax, n * sizeof(int)));
    uint8_t* endm;
    CK(cudaMalloc(&endm, n));
    float* V = const_cast<float*>(d_V_in);   // the cost overwrites the volume
    k_curve_stats<<<blocks1(n), 256, 0, s>>>(V, (const __half*)d_Lv16, n, K, vmax, vmed, kmax, lmin, lmax);

    // spill zone h from the depth-res guide
    {
        float *t1 = dmalloc(n), *t2 = dmalloc(n), *t3 = dmalloc(n);
        int ro = std::max(1, (int)std::nearbyint(16.0 * p.depth_full_fraction));
        int rh = std::max(1, (int)std::nearbyint(p.halo_radius_px * p.depth_full_fraction));
        auto se_o = ellipse_se(ro), se_h = ellipse_se(rh), se_5 = ellipse_se(2);
        int2 *d_so, *d_sh, *d_s5;
        CK(cudaMalloc(&d_so, se_o.size() * sizeof(int2)));
        CK(cudaMalloc(&d_sh, se_h.size() * sizeof(int2)));
        CK(cudaMalloc(&d_s5, se_5.size() * sizeof(int2)));
        CK(cudaMemcpyAsync(d_so, se_o.data(), se_o.size() * sizeof(int2), cudaMemcpyHostToDevice, s));
        CK(cudaMemcpyAsync(d_sh, se_h.data(), se_h.size() * sizeof(int2), cudaMemcpyHostToDevice, s));
        CK(cudaMemcpyAsync(d_s5, se_5.data(), se_5.size() * sizeof(int2), cudaMemcpyHostToDevice, s));
        k_morph<<<grid2(dw, dh), BLK, 0, s>>>(d_ga, t1, dw, dh, d_so, (int)se_o.size(), false);   // erode
        k_morph<<<grid2(dw, dh), BLK, 0, s>>>(t1, t2, dw, dh, d_so, (int)se_o.size(), true);      // dilate -> opening
        k_morph<<<grid2(dw, dh), BLK, 0, s>>>(t2, t1, dw, dh, d_sh, (int)se_h.size(), true);      // nbmax
        k_spill_a<<<blocks1(n), 256, 0, s>>>(t1, d_ga, h, n);
        DevGaussian g08, g3, g1;
        g08.build(0.8); g3.build(3.0); g1.build(1.0);
        gaussian_blur(d_ga, t1, t2, dw, dh, g08, s);
        laplacian3_abs(t1, t2, dw, dh, s);
        gaussian_blur(t2, t3, t1, dw, dh, g3, s);     // blur(lapq)
        gaussian_blur(d_ga, t1, t2, dw, dh, g3, s);   // blur(g)
        k_spill_b<<<blocks1(n), 256, 0, s>>>(h, t3, t1, n);
        gaussian_blur(h, t1, t2, dw, dh, g1, s);
        k_morph<<<grid2(dw, dh), BLK, 0, s>>>(t1, h, dw, dh, d_s5, (int)se_5.size(), false);      // erode 5x5 ellipse
        CK(cudaStreamSynchronize(s));
        g08.free(); g3.free(); g1.free();
        cudaFree(t1); cudaFree(t2); cudaFree(t3); cudaFree(d_so); cudaFree(d_sh); cudaFree(d_s5);
    }
    k_confidence<<<blocks1(n), 256, 0, s>>>(vmax, vmed, kmax, h, n, K, p.c_lo, p.c_hi, p.end_guard, w, endm);
    k_cost<<<blocks1(n), 256, 0, s>>>(V, (const __half*)d_Lv16, w, endm, vmax, lmin, lmax, h, n, K, p.end_prior, p.dark_prior,
                                       p.halo_prior);
    float* P2 = dmalloc(n);
    k_p2<<<grid2(dw, dh), BLK, 0, s>>>(d_ga, P2, dw, dh, p.p1, p.p2, p.edge);
    CK(cudaStreamSynchronize(s));
    CK(cudaGetLastError());
    R.ms_cost = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    {
        std::vector<uint8_t> e(n);
        CK(cudaMemcpy(e.data(), endm, n, cudaMemcpyDeviceToHost));
        size_t c = 0;
        for (auto v : e) c += v & 1;
        R.end_fraction = (double)c / n;
    }

    // --- SGM
    auto t1 = std::chrono::steady_clock::now();
    float* S = dmalloc((size_t)n * K);
    CK(cudaMemsetAsync(S, 0, (size_t)n * K * sizeof(float), s));
    int threads = ((K + 31) / 32) * 32;
    auto pass = [&](int dx, int dy) {
        unsigned blocks = dy == 0 ? dh : (dx == 0 ? dw : dw + dh - 1);
        if (K <= 256) k_sgm_pass<256><<<blocks, threads, 0, s>>>(V, P2, p.p1, S, dw, dh, K, dx, dy);
        else k_sgm_pass<512><<<blocks, threads, 0, s>>>(V, P2, p.p1, S, dw, dh, K, dx, dy);
    };
    pass(1, 0); pass(-1, 0); pass(0, 1); pass(0, -1);   // the prototype's order: x+, x-, y+, y-
    if (p.dirs == 8) { pass(1, 1); pass(-1, 1); pass(1, -1); pass(-1, -1); }   // (dy, dx) = (1,1), (1,-1), (-1,1), (-1,-1)
    float* d = dmalloc(n);
    k_argmin<<<blocks1(n), 256, 0, s>>>(S, n, K, d);
    CK(cudaStreamSynchronize(s));
    CK(cudaGetLastError());
    R.ms_sgm = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
    cudaFree(S);

    // --- upsample, guided filter on the output grid, rint, 3x3 median; conf
    auto t2 = std::chrono::steady_clock::now();
    float *up = dmalloc(no), *gfo = dmalloc(no), *sc5 = dmalloc(no * 5);
    k_resize_linear<<<grid2(ow, oh), BLK, 0, s>>>(d, dw, dh, up, ow, oh, (double)dw / ow, (double)dh / oh);
    int r = std::max(2, (int)std::nearbyint(2.0 * ow / dw));
    guided_filter(d_guide_full, up, gfo, ow, oh, r, p.gf_eps, sc5, s);
    k_rint<<<blocks1(no), 256, 0, s>>>(gfo, no);
    k_median3<<<grid2(ow, oh), BLK, 0, s>>>(gfo, up, ow, oh);
    R.depth_s.resize(no);
    CK(cudaMemcpyAsync(R.depth_s.data(), up, no * sizeof(float), cudaMemcpyDeviceToHost, s));
    k_resize_linear<<<grid2(ow, oh), BLK, 0, s>>>(w, dw, dh, gfo, ow, oh, (double)dw / ow, (double)dh / oh);
    R.conf.resize(no);
    CK(cudaMemcpyAsync(R.conf.data(), gfo, no * sizeof(float), cudaMemcpyDeviceToHost, s));
    R.spill.resize(n);
    CK(cudaMemcpyAsync(R.spill.data(), h, n * sizeof(float), cudaMemcpyDeviceToHost, s));
    R.depth_d.resize(n);
    CK(cudaMemcpyAsync(R.depth_d.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost, s));
    CK(cudaStreamSynchronize(s));
    CK(cudaGetLastError());
    R.ms_post = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t2).count();
    for (float* q : {vmax, vmed, lmin, lmax, w, h, P2, d, up, gfo, sc5}) cudaFree(q);
    cudaFree(kmax);
    cudaFree(endm);
    return R;
}

}  // namespace stackcore
