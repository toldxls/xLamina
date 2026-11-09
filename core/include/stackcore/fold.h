// Pass 1 of the fusion: every frame (already warped onto the output grid)
// folds into a partial State that merges with any other partial in any
// order (proto/stackproto/fusion.py: new_state / fold / merge). One State per
// GPU; the host merges them.
//
//   A   weighted average:  a_num += (fm + 1e-9)^(power/2) * img,  a_den += w   (double)
//   B0  per-pixel argmax of the focus measure: b_best / b_idx / b_col, b_fmsum (double)
//   C   Laplacian pyramid: per level keep the coefficient with the highest 3x3
//       luma energy, remember its frame; base level summed (double). With
//       c_radius > 0 the energy of levels >= 1 is first aggregated spatially
//       (Gaussian, sigma c_radius / 2^level), so a strong edge's frame wins the
//       band beside it instead of the defocus rims of that edge from frames
//       far away (FINDINGS 25).
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
    float c_radius = 0.f;  // C's energy aggregation sigma in OUTPUT px at level 0 (halved per level); 0 = off
    bool with_a = true;    // accumulate A's weighted sums (off when A is not an output: 64 B/px of traffic per frame)
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
    double* a_num = nullptr;       // h*w*3 (null when allocated without A)
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

    void alloc(int dev, int w, int h, int levels, bool with_a = true);   // zero / -1 initialised
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
    std::vector<DevGaussian> g_c;   // per level, C's energy aggregation (empty when c_radius == 0)
    void alloc(int w, int h, int levels, float radius, float c_radius = 0.f);
    void free();
    size_t bytes() const;
};

// Stage 1 (per-slot stream): focus measure + Laplacian pyramid + salience of
// d_img (h*w*3 float RGB on the output grid) into `sc`.
void fold_measure(const float* d_img, const FoldParams& p, FoldScratch& sc, cudaStream_t s);

// Stage 2 (the device's single fold stream): fold frame k's measures into st.
void fold_accumulate(FoldState& st, int k, const float* d_img, const FoldParams& p, FoldScratch& sc, cudaStream_t s);
// *d_acc += sum of the n floats (the frame's focus measure, sc.fm, after fold_measure): the per-frame focus energy
// behind the soft-frame flag (pipeline.h frame_sharpness). d_acc is a device double, zeroed by the caller.
void fold_energy_add(const float* d_fm, size_t n, double* d_acc, cudaStream_t s);

// Live preview of the running A (a_num / a_den) at 1/down of the grid, float RGB on the host (syncs the stream).
void fold_preview(const FoldState& st, int down, std::vector<float>& rgb, int& pw, int& ph, cudaStream_t s);
// The best focus measure found so far per pixel (b_best, grey x3), same layout: the detail map building.
// Preview kernels: the small image (pw x ph x 3 floats) into d_out on stream s, nothing else. The copy to
// the host is the caller's (pipeline.cpp AsyncPreview: its own stream, so a slow D2H under a saturated
// link never holds the slot or fold stream; FINDINGS 30).
void preview_size(int w, int h, int down, int& pw, int& ph);
void fold_focus_preview_kernel(const FoldState& st, int down, float* d_out, int& pw, int& ph, cudaStream_t s);
// A float RGB device image at 1/down (point samples), to the host (syncs the stream).
// every down-th pixel of the crop (x0, y0, cw, ch) of the w x h image -> pw x ph float RGB
void image_preview_kernel(const float* d_rgb, int w, int h, int x0, int y0, int cw, int ch, int down, float* d_out, int& pw, int& ph, cudaStream_t s);

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

// Device-to-device copies between cards through pinned host memory (no P2P between GeForce
// cards), chunked and double-buffered: the D2H of one chunk overlaps the H2D of the previous.
class CrossCopier {
public:
    CrossCopier(int dst_dev, int src_dev, size_t chunk = (size_t)64 << 20);
    ~CrossCopier();
    void copy(void* dst, const void* src, size_t bytes);   // returns when dst holds the data
private:
    int dst_, src_;
    size_t chunk_;
    uint8_t* pin_[2] = {nullptr, nullptr};
    cudaStream_t ss_ = nullptr, ds_ = nullptr;
    cudaEvent_t ev_[2] = {nullptr, nullptr};
};

// a = merge(a, b) on a's device, b staged across: the same arithmetic as merge_into(HostFoldState),
// so the merged state is bit-identical whichever card holds it (FINDINGS 29).
void merge_device(FoldState& a, const FoldState& b);
// the argmax merge rule on device arrays (greater, or equal with the lower index); col may be null
void merge_take(float* a_best, int16_t* a_idx, float* a_col, const float* b_best, const int16_t* b_idx, const float* b_col,
                size_t n, cudaStream_t s);
void add_f64(double* a, const double* b, size_t n, cudaStream_t s);   // a += b
// out = (float)(c_base / max(count, 1)): the collapsed pyramid's base, as the host merge computed it
void fold_base_f32(const FoldState& st, float* d_out, cudaStream_t s);
// Writes every array as a raw file (<dir>/<name>.<f32|f64|i16>) plus manifest.txt.
bool write_state(const HostFoldState& st, const std::string& dir);

}  // namespace stackcore
