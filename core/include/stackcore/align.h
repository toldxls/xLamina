// Alignment: pairwise similarity by Gauss-Newton on luma pyramids
// (proto/stackproto/align.py) and the exact global solve (sim.py / align.py).
//
// Pair (i, j): find P with g I_j(P(p)) + beta ~= I_i(p), P(p) = z p + t in
// centred coordinates. Coarse: phase correlation at the top level. Fine:
// Gauss-Newton on (a, b, tx, ty, g, beta), z = 1 + a + i b, ESM mean
// gradient, coarse to fine with OpenCV-style bicubic warps, then 4 iterations
// at level 0 with exact cubic B-spline interpolation (software, FP32
// weights: texture units' 1/256 px weights biased scale, FINDINGS 1).
// Reductions are deterministic (fixed-order partials), so a pair gives the
// same answer on any GPU.
#pragma once

#include <cuda_runtime.h>

#include <complex>
#include <cstdint>
#include <vector>

namespace stackcore {

struct PairResult {
    int i = 0, j = 0;
    std::complex<double> z{1, 0}, t{0, 0};   // frame-i coords -> frame-j coords, work px
    double var_a = 0, var_b = 0, var_tx = 0, var_ty = 0;
    double gain = 1, ncc = 0;
    int n_px = 0, iters = 0;
    bool ok = false;
};

// One frame's luma pyramid on the device (level 0 = the align-scale luma).
struct AlignPyramid {
    struct Level {
        int w = 0, h = 0;
        float *I = nullptr, *gx = nullptr, *gy = nullptr;
        double fx = 1, fy = 1;       // level size / level-0 size
    };
    std::vector<Level> lv;
    float* coef = nullptr;           // level-0 cubic B-spline coefficients (scipy spline_filter, mirror)
    void build(const float* d_luma, int w, int h, cudaStream_t s, int min_width = 200);
    void free();
};

// Working buffers for pairs of a given level-0 size (one per worker thread).
struct AlignWork {
    int w = 0, h = 0;
    float *I1w = nullptr, *gxw = nullptr, *gyw = nullptr;
    double* partials = nullptr;      // block partial sums
    int max_blocks = 0;
    // phase correlation at the top level: padded size, twiddle tables (forward/inverse, rows/cols), buffers
    int fft_w = 0, fft_h = 0;
    void *tw_fw = nullptr, *tw_fh = nullptr, *tw_iw = nullptr, *tw_ih = nullptr;
    void *fft_a = nullptr, *fft_b = nullptr, *fft_t = nullptr;
    float* fft_c = nullptr;
    float2* rowoff = nullptr;        // band_shifts: per-row extra offsets of the warp (h entries, allocated on first use)
    int rowoff_h = 0;
    void alloc(int w, int h);
    void free();
};

struct AlignParams {
    int max_iter[4] = {40, 25, 15, 12};   // per level from the top
    int exact_iters = 4;
    double tol_pyr = 1e-3, tol_exact = 1e-5;
};

// Align pair (i, j) on the current device.
PairResult align_pair(const AlignPyramid& pi, const AlignPyramid& pj, int i, int j, AlignWork& wk, const AlignParams& p,
                      cudaStream_t s);

// ---- rolling shutter (FINDINGS 51)

// One frame's level-0 data without the pyramid (luma copy and gradients), for band_shifts.
void level0_build(AlignPyramid::Level& L, const float* d_luma, int w, int h, cudaStream_t s);
void level0_free(AlignPyramid::Level& L);

struct BandShift {
    std::vector<double> dx, dy;   // per band: the extra shift (work px) frame j's content needs in that band of frame i's rows
    std::vector<double> ncc;      // per band: the correlation after the solve
    std::vector<int> n;           // per band: pixels used
};
// Frame j warped onto frame i by the pair similarity (z, t: frame-i coords -> frame-j coords, centred work px, the
// PairResult convention), then per horizontal band of i's rows a Gauss-Newton solve of translation + gain + bias
// with the similarity held. A shift that differs from band to band is a row-dependent displacement the similarity
// cannot model: the rolling shutter reading the sensor while the stage still vibrated.
BandShift band_shifts(const AlignPyramid::Level& Li, const AlignPyramid::Level& Lj, std::complex<double> z, std::complex<double> t,
                      int nbands, AlignWork& wk, cudaStream_t s, int iters = 8);
// Per-frame offsets d_k from pair measurements m_e = d_j - d_i (one band, dx + i dy), weighted least squares with
// Huber reweighting, the mean over the frames held at 0 (the common deformation is not observable and harmless);
// a frame without a usable pair (w = 0) gets 0. Throws nothing.
std::vector<std::complex<double>> chain_offsets(int n, const std::vector<std::pair<int, int>>& pairs,
                                                const std::vector<std::complex<double>>& m, const std::vector<double>& w);

// ---- host side: the global solve and the geometry

std::vector<std::pair<int, int>> make_pairs(int n, const std::vector<int>& baselines);

struct GlobalSolve {
    std::vector<std::complex<double>> z, t;   // per frame, relative to frame 0 (work px)
    std::vector<char> inlier;                 // per good pair
    int pairs_used = 0, pairs_total = 0, outliers = 0;
    double ncc_min = 0, ncc_median = 0;
};
GlobalSolve solve_global(int n, const std::vector<PairResult>& pairs, int irls_rounds = 3, double huber = 3.0);

// T'_k = T_k o T_ref^-1
void rebase(std::vector<std::complex<double>>& z, std::vector<std::complex<double>>& t, int ref);
int choose_reference(const std::vector<std::complex<double>>& z0, const std::string& rule);
// largest integer rect inside every frame's footprint (geometry.common_rect), inclusive
void common_rect(const std::vector<std::complex<double>>& z, const std::vector<std::complex<double>>& t, int W, int H,
                 int rect[4], double margin = 2.0);

}  // namespace stackcore
