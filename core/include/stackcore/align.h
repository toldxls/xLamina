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
