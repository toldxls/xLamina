#include "stackcore/volume.h"

#include <stdexcept>
#include <string>

namespace stackcore {

#define CK(x)                                                                                        \
    do {                                                                                             \
        cudaError_t e_ = (x);                                                                        \
        if (e_ != cudaSuccess) throw std::runtime_error(std::string(#x) + ": " + cudaGetErrorString(e_)); \
    } while (0)

namespace {

inline unsigned blocks1(size_t n) { return (unsigned)((n + 255) / 256); }

// fm / (m^2 + floor^2)
__global__ void k_normalise(const float* __restrict__ fm, const float* __restrict__ m, float* __restrict__ out, size_t n, float floor2) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float mm = m[i];
        out[i] = fm[i] / (mm * mm + floor2);
    }
}

// guided filter coefficients: a = (mIp - mI mp) / (vI + eps), b = mp - a mI
__global__ void k_gf_ab(const float* __restrict__ mI, const float* __restrict__ vI, const float* __restrict__ mp,
                        const float* __restrict__ mIp, float* __restrict__ a, float* __restrict__ b, size_t n, float eps) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float cIp = mIp[i] - mI[i] * mp[i];
    float av = cIp / (vI[i] + eps);
    a[i] = av;
    b[i] = mp[i] - av * mI[i];
}

// out = sqrt(max(box_a * guide + box_b, 0))
__global__ void k_gf_out(const float* __restrict__ ba, const float* __restrict__ bb, const float* __restrict__ guide,
                         float* __restrict__ out, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = sqrtf(fmaxf(ba[i] * guide[i] + bb[i], 0.f));
}

// vI = box(I^2) - mI^2
__global__ void k_var(const float* __restrict__ mII, const float* __restrict__ mI, float* __restrict__ vI, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) vI[i] = mII[i] - mI[i] * mI[i];
}

}  // namespace

void VolumeScratch::alloc(int ow_, int oh_, const VolumeParams& p) {
    ow = ow_; oh = oh_; dw = p.dw; dh = p.dh;
    size_t n = (size_t)dw * dh;
    CK(cudaMalloc(&img_d, n * 3 * sizeof(float)));
    CK(cudaMalloc(&y, n * sizeof(float)));
    CK(cudaMalloc(&t1, n * sizeof(float)));
    CK(cudaMalloc(&t2, n * sizeof(float)));
    CK(cudaMalloc(&t3, n * sizeof(float)));
    g08.build(0.8);
    g_fm.build(p.fm_radius);
    g_norm.build(3.0 * p.fm_radius);
    g_lum.build(2.0 * p.fm_radius);
    rz.build(ow, oh, dw, dh);
}

void VolumeScratch::free() {
    cudaFree(img_d); cudaFree(y); cudaFree(t1); cudaFree(t2); cudaFree(t3);
    g08.free(); g_fm.free(); g_norm.free(); g_lum.free();
    rz.free();
}

void volume_slice(const float* d_img_out, const VolumeParams& p, VolumeScratch& sc, void* d_fm16, void* d_lum16, cudaStream_t s) {
    size_t n = (size_t)sc.dw * sc.dh;
    sc.rz.run(d_img_out, sc.img_d, 3, s);                       // cv2.resize INTER_AREA
    luma(sc.img_d, sc.y, n, s);
    gaussian_blur(sc.y, sc.t2, sc.t1, sc.dw, sc.dh, sc.g08, s);   // focus_measure_rgb(normalized=True)
    laplacian3_sq(sc.t2, sc.t1, sc.dw, sc.dh, s);
    gaussian_blur(sc.t1, sc.t3, sc.t2, sc.dw, sc.dh, sc.g_fm, s);
    gaussian_blur(sc.y, sc.t1, sc.t2, sc.dw, sc.dh, sc.g_norm, s);
    k_normalise<<<blocks1(n), 256, 0, s>>>(sc.t3, sc.t1, sc.t2, n, 0.02f * 0.02f);
    f32_to_f16(sc.t2, d_fm16, n, s);
    gaussian_blur(sc.y, sc.t1, sc.t2, sc.dw, sc.dh, sc.g_lum, s);   // lum = blur(luma, 2 fm_radius)
    f32_to_f16(sc.t1, d_lum16, n, s);
}

void collapse_pyramid(const std::vector<const float*>& d_coef, const float* d_base_f, const std::vector<LevelShape>& shapes,
                      float* d_out, float* d_tmp, float* d_up, cudaStream_t s) {
    int levels = (int)shapes.size();
    // cur alternates between d_out and d_tmp; pyrUp always goes through d_up; the top level is the base itself
    const float* cur = d_base_f;
    float* bufs[2] = {d_out, d_tmp};
    int which = 0;
    for (int l = levels - 2; l >= 0; --l) {
        float* dst = bufs[which];
        pyr_up_rgb(cur, shapes[l + 1].w, shapes[l + 1].h, d_up, shapes[l].w, shapes[l].h, s);
        size_t n3 = (size_t)shapes[l].w * shapes[l].h * 3;
        add(d_up, d_coef[l], dst, n3, s);
        cur = dst;
        which ^= 1;
    }
    if (cur != d_out) CK(cudaMemcpyAsync(d_out, cur, (size_t)shapes[0].w * shapes[0].h * 3 * sizeof(float), cudaMemcpyDeviceToDevice, s));
}

void GuideStats::build(const float* d_guide, int w_, int h_, int r_, cudaStream_t s) {
    w = w_; h = h_; r = r_;
    size_t n = (size_t)w * h;
    for (float** p : {&mI, &vI, &t1, &t2, &t3, &t4, &t5}) CK(cudaMalloc(p, n * sizeof(float)));
    box_filter(d_guide, mI, t1, w, h, r, s);
    mul(d_guide, d_guide, t2, n, s);
    box_filter(t2, t3, t1, w, h, r, s);
    k_var<<<blocks1(n), 256, 0, s>>>(t3, mI, vI, n);
}

void GuideStats::free() {
    for (float* p : {mI, vI, t1, t2, t3, t4, t5}) cudaFree(p);
    mI = vI = t1 = t2 = t3 = t4 = t5 = nullptr;
}

void aggregate_slice(const float* d_guide, const GuideStats& gs, const void* d_fm16, float* d_V, float eps, cudaStream_t s) {
    size_t n = (size_t)gs.w * gs.h;
    float *src = gs.t1, *mp = gs.t2, *mIp = gs.t3, *a = gs.t4, *b = gs.t5;
    f16_to_f32(d_fm16, src, n, s);
    box_filter(src, mp, d_V, gs.w, gs.h, gs.r, s);          // mp = box(p)       (d_V as scratch)
    mul(d_guide, src, mIp, n, s);
    box_filter(mIp, mIp, d_V, gs.w, gs.h, gs.r, s);         // mIp = box(I p)    (in place via the row pass into d_V)
    k_gf_ab<<<blocks1(n), 256, 0, s>>>(gs.mI, gs.vI, mp, mIp, a, b, n, eps);
    box_filter(a, src, d_V, gs.w, gs.h, gs.r, s);           // box(a) -> src
    box_filter(b, mp, d_V, gs.w, gs.h, gs.r, s);            // box(b) -> mp
    k_gf_out<<<blocks1(n), 256, 0, s>>>(src, mp, d_guide, d_V, n);
}

}  // namespace stackcore
