// Similarity warp of a frame onto the output grid: the GPU version of
// proto/stackproto/fusion.load_aligned (conventions in proto/stackproto/sim.py).
//
//   frame (float RGB) -> [INTER_AREA x0.5 if the grid's scale is 0.5] -> bicubic
//   warp sampling the frame at T_k(p_ref), BORDER_REFLECT -> output grid
//
// The default samples at exact positions (double) with OpenCV's cubic kernel
// (a = -0.75) and matches cv2 5.0's warpAffine to ~1e-6 rms (FINDINGS 17).
// `quantize` reproduces OpenCV 4's fixed-point warpAffine (positions on a
// 1/32 px grid, its tables and summation order), the behaviour CLAUDE.md
// warns about; OpenCV 5 no longer does this for float images, so on this PC
// the quantized mode is the one that differs from the prototype.
#pragma once

#include <cuda_runtime.h>

namespace stackcore {

struct Similarity {
    double zr = 1, zi = 0;   // z = s * exp(i theta)
    double tx = 0, ty = 0;   // t, in full-resolution centred pixels
};

struct Grid {
    int W = 0, H = 0;            // full-resolution frame size
    int x0 = 0, y0 = 0, x1 = 0, y1 = 0;   // output rectangle, inclusive, reference full-res px
    double scale = 1.0;          // working scale: 1.0 or 0.5 (INTER_AREA 2x2 box)
    // the resampling filter of the warp (Interp): bicubic a = -0.75 is the reference (the prototype's cv2
    // warpAffine); Lanczos keeps more fine detail at a cost in time (FINDINGS 39)
    int interp = 1;
    int work_w() const;          // frame size at the working scale (round(W * scale))
    int work_h() const;
    int out_w() const;           // output size (round((x1 - x0 + 1) * scale))
    int out_h() const;
};

// Per-row corrections of a frame, applied inside the warp at the frame's own (working-scale) rows (FINDINGS 51):
// a row gain that flattens the LED flicker's brightness banding, and a row offset (work px) that undoes the
// rolling shutter's row-dependent displacement, both measured by the alignment. Null = none; the quantized
// (OpenCV 4 replica) bicubic ignores the offset.
struct RowCorr {
    const float* gain = nullptr;   // work_h floats, 1 = no change
    const float2* off = nullptr;   // work_h offsets added to the sample position (x, y)
};

enum Interp : int { kBilinear = 0, kBicubic = 1, kLanczos3 = 2, kLanczos8 = 3 };
const char* interp_name(int interp);      // "bilinear", "bicubic", "lanczos3", "lanczos8"
int interp_from_name(const char* name);   // -1 when unknown

// The 2x3 matrix (row-major, double) mapping output pixels to working-scale
// frame pixels, exactly as load_aligned builds it for cv2.warpAffine.
void affine_for(const Similarity& T, const Grid& g, double M[6]);

// INTER_AREA downsample by exactly 2 (w, h even), float RGB interleaved.
void box_down2_rgb(const float* d_src, int w, int h, float* d_dst, cudaStream_t s);

// Bicubic (a = -0.75) warp with BORDER_REFLECT. src: sw x sh float RGB; dst: ow x oh.
void warp_bicubic_rgb(const float* d_src, int sw, int sh, const double M[6], float* d_dst, int ow, int oh,
                      bool quantize, cudaStream_t s, RowCorr rc = {});

// Any Interp: bilinear, the bicubic above (quantize applies to it alone), Lanczos 3 (6x6 taps) or Lanczos 8
// (16x16 taps), all with BORDER_REFLECT and exact double positions.
void warp_rgb(const float* d_src, int sw, int sh, const double M[6], float* d_dst, int ow, int oh, int interp,
              bool quantize, cudaStream_t s, RowCorr rc = {});

// Convenience: full-res float RGB frame -> output grid with the grid's Interp, using a scratch buffer
// of work_w * work_h * 3 floats when scale == 0.5 (unused otherwise). rc: the frame's row corrections at the
// working scale (see RowCorr).
void load_aligned(const float* d_frame, const Grid& g, const Similarity& T, float* d_scratch, float* d_out,
                  bool quantize, cudaStream_t s, RowCorr rc = {});

}  // namespace stackcore
