// After the depth map: the refinement pass (fusion._w_refine + the assembly
// in run()), the smoothed depth, the dual composition pass (_w_b2dual) and
// its mix, and finalize (A, B0, C and the pyramid variants).
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

#include "stackcore/fold.h"
#include "stackcore/imgops.h"

namespace stackcore {

// ---- refinement pass: fine-scale focus argmax within +-R frames of the SGM depth

struct RefineState {           // one per GPU, output resolution
    int dev = -1, w = 0, h = 0;
    float* bw = nullptr;       // best fine fm within range, init -inf
    int16_t* iw = nullptr;     // its frame
    void alloc(int dev, int w, int h);
    void free();
};

struct RefineScratch {         // per slot
    int w = 0, h = 0;
    float *y = nullptr, *t1 = nullptr, *t2 = nullptr, *fm = nullptr;
    DevGaussian g08, g_fm, g_norm;
    void alloc(int w, int h, float fm_sigma);
    void free();
};

// fm = focus_measure_rgb(img, refine_fm_r, normalized=True)
void refine_measure(const float* d_img, RefineScratch& sc, cudaStream_t s);
// the restricted argmax: fw = |D - k| <= R ? fm : -inf; _take rule
void refine_accumulate(RefineState& st, int k, const float* d_fm, const float* d_depth, int R, cudaStream_t s);

// the refined depth so far at 1/down (frame index / (K-1), grey x3), for a live view
void refine_preview(const RefineState& st, int K, int down, std::vector<float>& rgb, int& pw, int& ph, cudaStream_t s);

struct HostRefine {
    int w = 0, h = 0;
    std::vector<float> bw;
    std::vector<int16_t> iw;
};
HostRefine download(const RefineState& st);
void merge_into(HostRefine& a, const HostRefine& b);
void merge_device(RefineState& a, const RefineState& b);   // a = merge(a, b) on a's device, same rule

// depth_s <- where(blur(conf) > 0.5 & spill_up < 0.3, median3(where(isfinite(bw), iw, depth_s)), depth_s)
// (all on the current device; host vectors in and out)
void refine_depth(std::vector<float>& depth_s, const std::vector<float>& conf, const std::vector<float>& spill, int dw, int dh,
                  const RefineState& r, int ow, int oh, double scale, cudaStream_t s);   // r on the current device

// depth_smooth = clip(guided_filter(guide, median5(median5(depth_s)), max(2, round(40 scale)), 2e-3), 0, n-1)
std::vector<float> smooth_depth(const std::vector<float>& depth_s, const float* d_guide_full, int ow, int oh, double scale, int n,
                                cudaStream_t s);

// ---- composition pass

struct ComposeState {          // one per GPU
    int dev = -1, w = 0, h = 0;
    double *n1 = nullptr, *d1 = nullptr, *n2 = nullptr, *d2 = nullptr;   // narrow (x3, x1), smooth (x3, x1)
    void alloc(int dev, int w, int h);
    void free();
};

struct ComposeScratch {        // per slot
    int w = 0, h = 0;
    float *lab = nullptr, *wn = nullptr, *tmp = nullptr;
    DevGaussian g_s1;
    void alloc(int w, int h, float soft_sigma);
    void free();
};

// frames that contribute: any |D - k| < 0.5 or any |Ds - k| < 1
std::vector<int> compose_frames(const std::vector<float>& depth_s, const std::vector<float>& depth_smooth, int n);
// lab = |D - k| < 0.5 blurred (soft labels); wi = clip(1 - |Ds - k|, 0, 1); accumulate both compositions
void compose_frame(ComposeState& st, int k, const float* d_img, const float* d_D, const float* d_Ds, ComposeScratch& sc,
                   cudaStream_t s);

// the composition so far at 1/down (narrow labels), for a live view
void compose_preview(const ComposeState& st, int down, std::vector<float>& rgb, int& pw, int& ph, cudaStream_t s);

struct HostCompose {
    int w = 0, h = 0;
    std::vector<double> n1, d1, n2, d2;
};
HostCompose download(const ComposeState& st);
void merge_into(HostCompose& a, const HostCompose& b);
void merge_device(ComposeState& a, const ComposeState& b);   // a += b on a's device

// b1 = n1 / d1, b2 = n2 / d2, mixed by fine detail and focus confidence -> B (float RGB, host)
std::vector<float> compose_mix(const ComposeState& c, const std::vector<float>& conf, int ow, int oh, double scale, cudaStream_t s);   // c on the current device

// ---- finalize

struct FinalParams {
    float agree = 3.f, dominance = 4.f, noise_k = 3.f, glint_agree = 6.f;
    int coarse_b_level = 3;
    // C's smoothing: pyramid detail below c_smoothing x the noise sigma of its level is taken
    // from B instead of the argmax frame (which in flat areas picks the noisiest frame).
    // 0 = the raw selection; 3 = the same gate as Cn.
    float c_smoothing = 0.f;
    std::vector<std::string> outputs;   // which outputs to compute (plus what they depend on); empty = all
};

// Outputs by name (A, B, B0, C, CB, Cn, CBn, Cd, Cg, Cg2, H), float RGB on the host.
struct FinalOutputs {
    std::vector<std::string> names;
    std::vector<std::vector<float>> images;
};
FinalOutputs finalize(const FoldState& st, const std::vector<float>& noise_mad, const std::vector<float>& depth_s,
                      const std::vector<float>& b_img, const FinalParams& p, cudaStream_t s);   // st on the current device

// noise per level from two adjacent frames' Laplacian pyramids: 1.4826 median(|(pa - pb) @ LUMA|) / sqrt 2
std::vector<float> noise_from_pair(const float* d_img_a, const float* d_img_b, int w, int h, int levels, cudaStream_t s);

// 16-bit RGB TIFF (uncompressed, one strip), values clipped to [0, 1] like frames.write_rgb16;
// `icc` (may be empty) is embedded as the ICC profile so the colours keep their meaning
bool write_tiff_rgb16(const std::string& path, const std::vector<float>& rgb, int w, int h,
                      const std::vector<uint8_t>& icc = {});

}  // namespace stackcore
