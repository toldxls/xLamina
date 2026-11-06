// Pass 1 of the fusion: every frame (already warped onto the output grid)
// folds into a partial State that merges with any other partial in any
// order (proto/stackproto/fusion.py: new_state / fold / merge). One State per
// GPU; the host merges them.
//
//   A   weighted average:  a_num += (fm + 1e-9)^(power/2) * img,  a_den += w   (double)
//   B0  per-pixel argmax of the focus measure: b_best / b_idx / b_col, b_fmsum (double)
//   C   Laplacian pyramid: per level keep the coefficient with the highest 3x3
//       luma energy, remember its frame; base level summed (double)
//   noise reference: the lowest frame index seen, its per-level MAD and median fm
//
// Ties break toward the lower frame index, so the merged result does not
// depend on how frames were split (test_fusion). Image-processing details
// follow OpenCV's float paths (reflect-101 borders, getGaussianKernel, the
// 3x3 Laplacian {2,0,2,0,-8,0,2,0,2}, pyrDown/pyrUp 5-tap, 3x3 box).
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

#include "stackcore/imgops.h"

namespace stackcore {

struct FoldParams {
    int levels = 0;        // pyramid levels, fusion.n_levels(w, h)
    float radius = 3.0f;   // focus-measure blur sigma in OUTPUT px (radius_px * scale, >= 0.8)
    float power = 4.0f;    // A weight exponent
};

inline int n_levels(int w, int h, int top = 32) {
    int m = w < h ? w : h;
    int l = 0;
    while ((m >> (l + 1)) >= top) ++l;   // floor(log2(m / top)) + 1, like the prototype
    l += 1;
    return l < 2 ? 2 : l;
}

struct LevelShape { int w, h; };
std::vector<LevelShape> pyramid_shapes(int w, int h, int levels);

// Device-resident partial result for one GPU.
struct FoldState {
    int dev = -1;
    int w = 0, h = 0, levels = 0;
    std::vector<LevelShape> shapes;
    double* a_num = nullptr;       // h*w*3
    double* a_den = nullptr;       // h*w
    float* b_best = nullptr;       // h*w, init -1
    int16_t* b_idx = nullptr;      // h*w
    float* b_col = nullptr;        // h*w*3
    double* b_fmsum = nullptr;     // h*w
    std::vector<float*> c_sal;     // levels-1 arrays, init -1
    std::vector<float*> c_coef;    // levels-1 arrays, x3
    std::vector<int16_t*> c_idx;   // levels-1 arrays
    double* c_base = nullptr;      // top level, x3
    int count = 0;
    int noise_k = 1 << 30;
    std::vector<float> noise_mad;  // levels-1
    float noise_fm = 0.f;
    float* d_noise_tmp = nullptr;  // scratch for the medians (h*w)

    void alloc(int dev, int w, int h, int levels);   // zero / -1 initialised
    void free();
    size_t bytes() const;
};

// Per-slot temporaries for one frame's measures (sized for w x h), plus the
// two blur kernels the focus measure needs (built on the current device).
struct FoldScratch {
    int w = 0, h = 0, levels = 0;
    std::vector<LevelShape> shapes;
    float *y = nullptr, *t1 = nullptr, *t2 = nullptr, *fm = nullptr;
    std::vector<float*> g;     // Gaussian levels 1..levels-1 (level 0 is the frame)
    std::vector<float*> lap;   // levels-1 Laplacian levels, x3 (top level = g[levels-1])
    std::vector<float*> sal;   // levels-1 salience maps
    DevGaussian g_small, g_big;
    void alloc(int w, int h, int levels, float radius);
    void free();
    size_t bytes() const;
};

// Stage 1 (per-slot stream): focus measure + Laplacian pyramid + salience of
// d_img (h*w*3 float RGB on the output grid) into `sc`.
void fold_measure(const float* d_img, const FoldParams& p, FoldScratch& sc, cudaStream_t s);

// Stage 2 (the device's single fold stream): fold frame k's measures into st.
void fold_accumulate(FoldState& st, int k, const float* d_img, const FoldParams& p, FoldScratch& sc, cudaStream_t s);

// Host copy of a state, for merging and for writing out.
struct HostFoldState {
    int w = 0, h = 0, levels = 0;
    std::vector<LevelShape> shapes;
    std::vector<double> a_num, a_den, b_fmsum, c_base;
    std::vector<float> b_best, b_col;
    std::vector<int16_t> b_idx;
    std::vector<std::vector<float>> c_sal, c_coef;
    std::vector<std::vector<int16_t>> c_idx;
    int count = 0;
    int noise_k = 1 << 30;
    std::vector<float> noise_mad;
    float noise_fm = 0.f;
};

HostFoldState download(const FoldState& st);
void merge_into(HostFoldState& a, const HostFoldState& b);   // a = merge(a, b)
// Writes every array as a raw file (<dir>/<name>.<f32|f64|i16>) plus manifest.txt.
bool write_state(const HostFoldState& st, const std::string& dir);

}  // namespace stackcore
