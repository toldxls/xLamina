#include "stackcore/depth.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "stackcore/imgops.h"
#include "stackcore/pipeline.h"   // logf

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
__global__ void k_planes_to_k_f32_chunk(const float* __restrict__ in, float* __restrict__ out, int n, int K, int k0, int kc) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    for (int j = 0; j < kc; ++j) out[(size_t)i * K + k0 + j] = in[(size_t)j * n + i];
}
__global__ void k_planes_to_k_f16_chunk(const __half* __restrict__ in, __half* __restrict__ out, int n, int K, int k0, int kc) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    for (int j = 0; j < kc; ++j) out[(size_t)i * K + k0 + j] = in[(size_t)j * n + i];
}

// ---- per-pixel statistics of the focus curve: vmax, median, argmax; and of Lv: min, max

__global__ void k_curve_stats(const float* __restrict__ V, const __half* __restrict__ Lv, int n, int K,
                              float* __restrict__ vmax, float* __restrict__ vmed, int* __restrict__ kmax,
                              float* __restrict__ lmin, float* __restrict__ lmax, float* __restrict__ trend) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* v = V + (size_t)i * K;
    float buf[512];   // K <= 512
    float mx = -1e30f, lo = 1e30f, hi = -1e30f;
    // the luma of the first and last thirds: an out-of-range pixel goes to the darker end (what the dark prior
    // chose for the whole background on the 1/4 grid; the focus curve's own slope pointed the other way there)
    float l_first = 0.f, l_last = 0.f;
    int km = 0;
    const int third = max(1, K / 3);
    for (int k = 0; k < K; ++k) {
        float x = v[k];
        buf[k] = x;
        if (x > mx) { mx = x; km = k; }   // first maximum, like np.argmax
        float l = __half2float(Lv[(size_t)i * K + k]);
        if (k < third) l_first += l;
        if (k >= K - third) l_last += l;
        lo = fminf(lo, l);
        hi = fmaxf(hi, l);
    }
    if (trend) trend[i] = l_last - l_first;   // > 0: the first frames are darker -> beyond the first end
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
                             const float* __restrict__ h, const float* __restrict__ trend, int n, int K, float lo, float hi, int eg,
                             float* __restrict__ w, uint8_t* __restrict__ endm, float* __restrict__ wc, float* __restrict__ cont) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float vm = vmax[i];
    float contrast = (vm - vmed[i]) / fmaxf(vm, 1e-12f);
    if (cont) cont[i] = contrast;   // the raw focus-curve contrast (flat_keep: weak features keep their own depth)
    float ww = fminf(fmaxf((contrast - lo) / (hi - lo), 0.f), 1.f);
    ww = ww * (1.0f - h[i]);
    int km = kmax[i];
    uint8_t e = (km < eg || km > K - 1 - eg) ? 1 : 0;
    w[i] = e ? 0.f : ww;
    if (wc) wc[i] = ww;   // the curve's contrast whatever its peak's position: sharp detail, end frames included
    // bit 1: the pixel lies beyond the first frames (the end prior's direction): the peak's side in 0.5.x, the
    // curve's trend when given (robust to a noisy peak)
    bool at0 = trend ? (trend[i] > 0.f) : (km < eg);
    endm[i] = e | (at0 ? 2 : 0);
}

// the end-frame vote: +1 for an out-of-range pixel peaking at the first frames, -1 at the last, 0 otherwise
__global__ void k_end_dir(const uint8_t* __restrict__ endm, float* __restrict__ dir, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint8_t e = endm[i];
    dir[i] = (e & 1) ? ((e & 2) ? 1.f : -1.f) : 0.f;
}
// ... and the smoothed vote decides the direction for every out-of-range pixel
__global__ void k_end_vote(uint8_t* __restrict__ endm, const float* __restrict__ vote, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint8_t e = endm[i];
    if (e & 1) endm[i] = 1 | (vote[i] > 0.f ? 2 : 0);
}

// cost volume, in place over V (double intermediates where numpy promoted to float64)
__global__ void k_cost(float* __restrict__ V, const __half* __restrict__ Lv, const float* __restrict__ w,
                       const uint8_t* __restrict__ endm, const float* __restrict__ vmax, const float* __restrict__ lmin,
                       const float* __restrict__ lmax, const float* __restrict__ h, const float* __restrict__ wnb,
                       const float* __restrict__ flat, int n, int K, float ep, float lam, float lam_h, float lam_f) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float* c = V + (size_t)i * K;
    float wi = w[i], vm = fmaxf(vmax[i], 1e-12f), hi = h[i];
    bool e = endm[i] & 1, at0 = endm[i] & 2;
    float lrange = fmaxf(lmax[i] - lmin[i], 0.01f), l0 = lmin[i];
    // a flat pixel (defocused background, no texture within reach) has no focus peak to follow; what it wants is
    // the frame where it shows the LEAST structure (the rings of a nearby object's defocus are structure), so its
    // data term is lam_f x the normalised focus energy, and the end and dark priors step back by the same share
    float fl = (flat && lam_f > 0) ? flat[i] : 0.f;
    // numpy's dtypes: (1 - w)(1 - endm) lam dark is float64 (bool array - python float -> float64);
    // lam_h h dark is float32 (python float * float32 array); C accumulates in float64, cast at the end
    double w_dark = ((double)(1.0f - wi) * (e ? 0.0 : 1.0)) * (double)lam;
    if (wnb) w_dark *= (double)(1.0f - wnb[i]);   // no dark prior beside sharp content (wnb: the neighbourhood's confidence)
    w_dark *= (double)(1.0f - fl);
    float w_halo = lam_h * hi;
    for (int k = 0; k < K; ++k) {
        float base = wi * (1.0f - c[k] / vm);                              // float32
        double acc = base;
        if (fl > 0) acc = (1.0f - fl) * base + fl * lam_f * (c[k] / vm);
        if (ep > 0 && e) {
            float dist = (at0 ? (float)k : (float)(K - 1 - k)) / (float)(K > 1 ? K - 1 : 1);
            acc += ((double)ep * 1.0) * (double)dist * (double)(1.0f - fl);
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
    // guards below label 0 and above label K-1. prev[K + 1] must be set here: when K is a multiple of 32 the block
    // has exactly K threads, so no thread k >= K exists to write it in the loop (stale shared memory otherwise)
    if (k == 0) { prev[0] = INFINITY; prev[K + 1] = INFINITY; prev[MAXK + 1] = INFINITY; }
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
    if (k == 0 || k == K - 1) { d[i] = (float)k; return; }   // an end frame has one neighbour: no parabola, and the
    float a = s[k - 1], b = s[k], c = s[k + 1];              // frame itself (0.6.9 and the prototype returned 1 / K-2)
    float den = a - 2.f * b + c;
    float off = den > 1e-9f ? 0.5f * (a - c) / den : 0.f;
    d[i] = (float)k + fminf(fmaxf(off, -0.5f), 0.5f);
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
void planes_to_kfastest_f32_chunk(const float* d_planes, float* d_out, int dw, int dh, int K, int k0, int kc, cudaStream_t s) {
    int n = dw * dh;
    k_planes_to_k_f32_chunk<<<blocks1(n), 256, 0, s>>>(d_planes, d_out, n, K, k0, kc);
}
void planes_to_kfastest_f16_chunk(const void* d_planes, void* d_out, int dw, int dh, int K, int k0, int kc, cudaStream_t s) {
    int n = dw * dh;
    k_planes_to_k_f16_chunk<<<blocks1(n), 256, 0, s>>>((const __half*)d_planes, (__half*)d_out, n, K, k0, kc);
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

// ---- flat regions (FINDINGS 34): within reach of textured pixels the depth is theirs (a band, as a sharp object's
// own frame shows the least of its defocus beside it); beyond the band it is the quieter end of the stack (the one
// where the coarse structure of the defocused background is weakest, measured on the luma volume's end frames)

__global__ void k_plane_mean(const __half* __restrict__ Lv, int n, int K, int k0, int k1, float* __restrict__ out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float a = 0.f;
    for (int k = k0; k < k1; ++k) a += __half2float(Lv[(size_t)i * K + k]);
    out[i] = a / (float)(k1 - k0);
}
__global__ void k_abs_diff(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = fabsf(a[i] - b[i]);
}
__global__ void k_textured(const float* __restrict__ flat, const float* __restrict__ d, float* __restrict__ wgt, float* __restrict__ num, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float w = 1.f - flat[i];
    wgt[i] = w;
    num[i] = d[i] * w;
}
__global__ void k_down2(const float* __restrict__ in, int w, int h, float* __restrict__ out, int w2, int h2) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w2 || y >= h2) return;
    int x0 = 2 * x, y0 = 2 * y, x1 = min(x0 + 1, w - 1), y1 = min(y0 + 1, h - 1);
    out[y * w2 + x] = 0.25f * (in[y0 * w + x0] + in[y0 * w + x1] + in[y1 * w + x0] + in[y1 * w + x1]);
}
// pull-push: where the level has data (den -> 1) its own mean, elsewhere the coarser level's fill
__global__ void k_push(const float* __restrict__ num, const float* __restrict__ den, const float* __restrict__ up, float* __restrict__ fill, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = num[i] / fmaxf(den[i], 1e-9f);
    if (!up) { fill[i] = v; return; }
    float a = fminf(den[i], 1.f);
    fill[i] = a * v + (1.f - a) * up[i];
}
// flat_keep: a pixel with focus evidence of its own keeps that share of its SGM depth. The override ov (1 beyond
// the reach of confident detail) is scaled by 1 - keep * min(contrast / c_lo, 1): a weak feature at half the
// confidence threshold keeps half its own depth, one at the threshold all of it, featureless background none
__global__ void k_keep(float* __restrict__ ov, const float* __restrict__ cont, int n, float c_lo, float keep) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float kw = fminf(fmaxf(cont[i] / fmaxf(c_lo, 1e-6f), 0.f), 1.f);
    ov[i] *= 1.f - keep * kw;
}
__global__ void k_flat_target(const float* __restrict__ fill, const float* __restrict__ band_g, const float* __restrict__ e0,
                              const float* __restrict__ e1, float* __restrict__ out, int n, float kfar, float gain) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float b = fminf(band_g[i] * gain, 1.f);
    float far = e0[i] < e1[i] ? 0.f : kfar;
    out[i] = b * fill[i] + (1.f - b) * far;
}

// coarse structure of one luma plane: |G(3 px) - G(12 px)| (sensor px), averaged over the flat blur width
static void coarse_energy(const float* plane, float* out, float* t1, float* t2, int dw, int dh, float frac, float big_px, cudaStream_t s) {
    DevGaussian ga, gb, gc;
    ga.build(std::max(0.5, 3.0 * (double)frac));
    gb.build(std::max(1.0, 12.0 * (double)frac));
    gc.build(std::max(1.0, (double)big_px * (double)frac));
    gaussian_blur(plane, t1, out, dw, dh, ga, s);
    gaussian_blur(plane, t2, out, dw, dh, gb, s);
    k_abs_diff<<<blocks1((size_t)dw * dh), 256, 0, s>>>(t1, t2, out, dw * dh);
    gaussian_blur(out, t1, t2, dw, dh, gc, s);
    CK(cudaMemcpyAsync(out, t1, (size_t)dw * dh * sizeof(float), cudaMemcpyDeviceToDevice, s));
    CK(cudaStreamSynchronize(s));
    ga.free(); gb.free(); gc.free();
}

// the weighted pixels' depth spread into the rest (pull-push over an area pyramid); fill is dw*dh
static void pull_push_fill(const float* d, const float* wgt, float* fill, int dw, int dh, cudaStream_t s) {
    std::vector<float*> num, den;
    std::vector<int> ws, hs;
    int w = dw, h = dh;
    float *n0 = dmalloc((size_t)w * h), *d0 = dmalloc((size_t)w * h);
    mul(d, wgt, n0, (size_t)w * h, s);
    CK(cudaMemcpyAsync(d0, wgt, (size_t)w * h * sizeof(float), cudaMemcpyDeviceToDevice, s));
    num.push_back(n0); den.push_back(d0); ws.push_back(w); hs.push_back(h);
    while (std::min(w, h) > 8) {
        int w2 = (w + 1) / 2, h2 = (h + 1) / 2;
        float *nn = dmalloc((size_t)w2 * h2), *dd = dmalloc((size_t)w2 * h2);
        k_down2<<<grid2(w2, h2), BLK, 0, s>>>(num.back(), w, h, nn, w2, h2);
        k_down2<<<grid2(w2, h2), BLK, 0, s>>>(den.back(), w, h, dd, w2, h2);
        num.push_back(nn); den.push_back(dd); ws.push_back(w2); hs.push_back(h2);
        w = w2; h = h2;
    }
    int L = (int)num.size();
    float* prev = dmalloc((size_t)ws[L - 1] * hs[L - 1]);
    k_push<<<blocks1((size_t)ws[L - 1] * hs[L - 1]), 256, 0, s>>>(num[L - 1], den[L - 1], nullptr, prev, ws[L - 1] * hs[L - 1]);
    for (int l = L - 2; l >= 0; --l) {
        size_t nl = (size_t)ws[l] * hs[l];
        float* up = dmalloc(nl);
        resize_linear(prev, ws[l + 1], hs[l + 1], up, ws[l], hs[l], s);
        float* cur = (l == 0) ? fill : dmalloc(nl);
        k_push<<<blocks1(nl), 256, 0, s>>>(num[l], den[l], up, cur, (int)nl);
        CK(cudaStreamSynchronize(s));
        cudaFree(up); cudaFree(prev);
        prev = cur;
    }
    CK(cudaStreamSynchronize(s));
    for (float* q : num) cudaFree(q);
    for (float* q : den) cudaFree(q);
}

DepthResult depth_from_volume(const float* d_V_in, void* d_Lv16, const float* d_guide_full, const float* d_ga,
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
    // the end direction stays the peak's side: both the focus curve's slope and the luma trend pointed to the far end
    // in the defocused background where the 1/4 map and the reference chose the near end (FINDINGS 34)
    float* trend = nullptr;
    k_curve_stats<<<blocks1(n), 256, 0, s>>>(V, (const __half*)d_Lv16, n, K, vmax, vmed, kmax, lmin, lmax, trend);

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
    const bool flat_fill = p.flat && p.flat_blur_px > 0;
    float* wc = flat_fill ? dmalloc(n) : nullptr;   // sharp detail, for the flat regions' band (after the SGM)
    float* cont = (flat_fill && p.flat_keep > 0.f) ? dmalloc(n) : nullptr;   // the raw contrast, for flat_keep
    k_confidence<<<blocks1(n), 256, 0, s>>>(vmax, vmed, kmax, h, trend, n, K, p.c_lo, p.c_hi, p.end_guard, w, endm, wc, cont);
    if (trend) { CK(cudaStreamSynchronize(s)); cudaFree(trend); }
    // out-of-range pixels follow their neighbourhood's vote on which end they are beyond
    if (p.end_vote_px > 0 && p.end_prior > 0) {
        float *dir = dmalloc(n), *t1 = dmalloc(n), *t2 = dmalloc(n);
        k_end_dir<<<blocks1(n), 256, 0, s>>>(endm, dir, n);
        DevGaussian gv;
        gv.build(std::max(1.0, (double)(p.end_vote_px * p.depth_full_fraction)));
        gaussian_blur(dir, t1, t2, dw, dh, gv, s);
        k_end_vote<<<blocks1(n), 256, 0, s>>>(endm, t1, n);
        CK(cudaStreamSynchronize(s));
        gv.free();
        cudaFree(dir); cudaFree(t1); cudaFree(t2);
    }
    // the neighbourhood's confidence: the dark prior is gated off within dark_reach_px of sharp content
    float* wnb = nullptr;
    if (p.dark_prior > 0 && p.dark_reach_px > 0) {
        int rr = std::max(1, (int)std::nearbyint(p.dark_reach_px * p.depth_full_fraction));
        auto se = ellipse_se(rr);
        int2* d_se;
        CK(cudaMalloc(&d_se, se.size() * sizeof(int2)));
        CK(cudaMemcpyAsync(d_se, se.data(), se.size() * sizeof(int2), cudaMemcpyHostToDevice, s));
        wnb = dmalloc(n);
        k_morph<<<grid2(dw, dh), BLK, 0, s>>>(w, wnb, dw, dh, d_se, (int)se.size(), true);   // dilate
        CK(cudaStreamSynchronize(s));
        cudaFree(d_se);
    }
    k_cost<<<blocks1(n), 256, 0, s>>>(V, (const __half*)d_Lv16, w, endm, vmax, lmin, lmax, h, wnb, p.flat, n, K, p.end_prior,
                                       p.dark_prior, p.halo_prior, p.flat_min);
    if (wnb) { CK(cudaStreamSynchronize(s)); cudaFree(wnb); }
    float* P2 = dmalloc(n);
    k_p2<<<grid2(dw, dh), BLK, 0, s>>>(d_ga, P2, dw, dh, p.p1, p.p2, p.edge);
    CK(cudaStreamSynchronize(s));
    CK(cudaGetLastError());
    // the coarse structure of the background at the two ends of the stack (for the flat regions, after the SGM)
    float *e0 = nullptr, *e1 = nullptr;
    if (flat_fill) {
        float *pl = dmalloc(n), *t1 = dmalloc(n), *t2 = dmalloc(n);
        int ke = std::min(8, std::max(1, K / 4));
        e0 = dmalloc(n); e1 = dmalloc(n);
        k_plane_mean<<<blocks1(n), 256, 0, s>>>((const __half*)d_Lv16, n, K, 0, ke, pl);
        coarse_energy(pl, e0, t1, t2, dw, dh, p.depth_full_fraction, p.flat_blur_px, s);
        k_plane_mean<<<blocks1(n), 256, 0, s>>>((const __half*)d_Lv16, n, K, K - ke, K, pl);
        coarse_energy(pl, e1, t1, t2, dw, dh, p.depth_full_fraction, p.flat_blur_px, s);
        cudaFree(pl); cudaFree(t1); cudaFree(t2);
    }
    cudaFree(d_Lv16);   // consumed: the luma volume's memory goes to the SGM
    R.ms_cost = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    {
        std::vector<uint8_t> e(n);
        CK(cudaMemcpy(e.data(), endm, n, cudaMemcpyDeviceToHost));
        size_t c = 0;
        for (auto v : e) c += v & 1;
        R.end_fraction = (double)c / n;
    }

    // --- SGM, in horizontal bands when the path-sum volume does not fit beside the cost volume (at 1/2
    // resolution 163 frames are 4.9 GB each; FINDINGS 31). The bands overlap by `ov` rows: a path's cost is
    // capped by P2 at every step, so what it carries from rows farther away than that is nil. One band is
    // the whole image, which is the code as it always was.
    auto t1 = std::chrono::steady_clock::now();
    const int ov = 64;
    size_t S_bytes = (size_t)n * K * sizeof(float);
    {   // the stream-ordered allocator keeps pass 1's freed slices in its pool: hand them back before measuring
        int dev = 0;
        cudaMemPool_t pool;
        if (cudaGetDevice(&dev) == cudaSuccess && cudaDeviceGetDefaultMemPool(&pool, dev) == cudaSuccess) cudaMemPoolTrimTo(pool, 0);
    }
    size_t fr = 0, tot = 0;
    CK(cudaMemGetInfo(&fr, &tot));
    // the band layout decides the result (a path's memory is cut at the band's edge), so it must not depend on
    // what else happens to be on the card: the path-sum volume gets a fixed 1.5 GB unless even that is short
    size_t budget = std::min<size_t>((size_t)1536 << 20, fr > (768u << 20) ? fr - (768u << 20) : 0);
    if (budget < S_bytes && fr - (768u << 20) < ((size_t)1536 << 20))
        logf("depth: only %.2f GB free for the SGM volume; the band layout, and so the result, depends on it this time\n", fr / 1e9);
    int bands = 1;
    if (S_bytes > budget) {
        size_t row_bytes = (size_t)dw * K * sizeof(float);
        int rows_fit = (int)(budget / row_bytes);
        if (rows_fit <= 2 * ov + 16) {
            char msg[200];
            snprintf(msg, sizeof msg, "depth: not enough GPU memory for the SGM volume (%.2f GB free, %d rows of %d fit)", fr / 1e9, rows_fit, dh);
            throw std::runtime_error(msg);
        }
        bands = (int)((dh + (rows_fit - 2 * ov) - 1) / (rows_fit - 2 * ov));
    }
    const int band_rows = (dh + bands - 1) / bands;
    const int s_rows = std::min(dh, band_rows + 2 * ov);
    float* S = dmalloc((size_t)s_rows * dw * K);
    float* d = dmalloc(n);
    int threads = ((K + 31) / 32) * 32;
    for (int b = 0; b < bands; ++b) {
        const int y0 = b * band_rows, y1 = std::min(dh, y0 + band_rows);
        const int ya = std::max(0, y0 - ov), yb = std::min(dh, y1 + ov), H = yb - ya;
        const float* Vb = V + (size_t)ya * dw * K;
        const float* P2b = P2 + (size_t)ya * dw;
        CK(cudaMemsetAsync(S, 0, (size_t)H * dw * K * sizeof(float), s));
        auto pass = [&](int dx, int dy) {
            unsigned blocks = dy == 0 ? H : (dx == 0 ? dw : dw + H - 1);
            if (K <= 256) k_sgm_pass<256><<<blocks, threads, 0, s>>>(Vb, P2b, p.p1, S, dw, H, K, dx, dy);
            else k_sgm_pass<512><<<blocks, threads, 0, s>>>(Vb, P2b, p.p1, S, dw, H, K, dx, dy);
        };
        pass(1, 0); pass(-1, 0); pass(0, 1); pass(0, -1);   // the prototype's order: x+, x-, y+, y-
        if (p.dirs == 8) { pass(1, 1); pass(-1, 1); pass(1, -1); pass(-1, -1); }   // (dy, dx) = (1,1), (1,-1), (-1,1), (-1,-1)
        k_argmin<<<blocks1((y1 - y0) * dw), 256, 0, s>>>(S + (size_t)(y0 - ya) * dw * K, (y1 - y0) * dw, K, d + (size_t)y0 * dw);
    }
    CK(cudaStreamSynchronize(s));
    CK(cudaGetLastError());
    R.ms_sgm = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
    R.sgm_bands = bands;
    cudaFree(S);

    // --- pixels without sharp detail within reach: the band beside sharp detail takes its depth (spread from the
    // confident pixels), beyond the band the quieter end of the stack; soft transitions. Sharp detail keeps the SGM.
    if (flat_fill) {
        float *fill = dmalloc(n), *band = dmalloc(n), *f1 = dmalloc(n), *f2 = dmalloc(n), *ov = dmalloc(n), *wd = dmalloc(n);
        pull_push_fill(d, wc, fill, dw, dh, s);
        // sharp detail dilated by the reach: the override is 1 - that; the band is its Gaussian spread
        {
            float reach = p.dark_reach_px > 0 ? p.dark_reach_px : 24.f;
            int rr = std::max(1, (int)std::nearbyint(reach * p.depth_full_fraction));
            auto se = ellipse_se(rr);
            int2* d_se;
            CK(cudaMalloc(&d_se, se.size() * sizeof(int2)));
            CK(cudaMemcpyAsync(d_se, se.data(), se.size() * sizeof(int2), cudaMemcpyHostToDevice, s));
            k_morph<<<grid2(dw, dh), BLK, 0, s>>>(wc, wd, dw, dh, d_se, (int)se.size(), true);   // dilate
            k_textured<<<blocks1(n), 256, 0, s>>>(wd, d, ov, f1, n);                                // ov = 1 - dilated
            if (cont) k_keep<<<blocks1(n), 256, 0, s>>>(ov, cont, n, p.c_lo, p.flat_keep);          // weak features keep theirs
            CK(cudaStreamSynchronize(s));
            cudaFree(d_se);
        }
        // the band's width: flat_band_px when given, else the blur width (0.6.4: one number for both)
        const float bandw = p.flat_band_px > 0.f ? p.flat_band_px : p.flat_blur_px;
        DevGaussian gb, gt;
        gb.build(std::max(1.0, 1.5 * bandw * p.depth_full_fraction));            // the band's reach
        gaussian_blur(wd, band, f2, dw, dh, gb, s);
        k_flat_target<<<blocks1(n), 256, 0, s>>>(fill, band, e0, e1, f1, n, (float)(K - 1), 4.f);
        gt.build(std::max(1.0, 0.25 * bandw * p.depth_full_fraction));          // soft band-to-far transition
        gaussian_blur(f1, f2, fill, dw, dh, gt, s);
        lerp_mask(d, f2, ov, d, n, s);
        CK(cudaStreamSynchronize(s));
        gb.free(); gt.free();
        for (float* q : {fill, band, f1, f2, ov, wd, e0, e1, wc}) cudaFree(q);
        if (cont) cudaFree(cont);
    }

    // --- upsample, guided filter on the output grid, rint, 3x3 median; conf
    auto t2 = std::chrono::steady_clock::now();
    float *up = dmalloc(no), *gfo = dmalloc(no), *sc5 = dmalloc(no * 5);
    k_resize_linear<<<grid2(ow, oh), BLK, 0, s>>>(d, dw, dh, up, ow, oh, (double)dw / ow, (double)dh / oh);
    // the guided filter's radius on the output grid: 8 px for a full-size output whatever the volume's grid (it used
    // to be 2 volume pixels, which halved on the 1/2-resolution grid and left the finer map less smoothed)
    int r = std::max(2, (int)std::nearbyint(2.0 * ow / dw * (p.depth_full_fraction / 0.25)));
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
