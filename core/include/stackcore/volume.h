// The depth-map input (fusion._w_fmvol and the first part of sgm_from_volume):
//
//   per frame, at the fixed depth resolution (1/4 of the sensor whatever the
//   output scale): INTER_AREA resize of the warped frame, contrast-normalised
//   focus measure  lap^2 blur / (mean^2 + 0.02^2)  and blurred luma, both
//   stored as float16 (the prototype's cache format);
//   then every focus slice aggregated with a guided filter (guide = the luma
//   of the collapsed C image), clamped at 0 and square-rooted.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

#include "stackcore/fold.h"
#include "stackcore/imgops.h"

namespace stackcore {

struct VolumeParams {
    int dw = 0, dh = 0;        // depth resolution: round(out * depth_full_fraction / scale), >= 16
    float fm_radius = 1.5f;    // max(0.8, depth_fm_px * depth_full_fraction)
    int agg_r = 8;             // round(vol_agg_px * depth_full_fraction); 0 = no aggregation
    float agg_eps = 1e-4f;
    // the focus measure at two scales: the grid's own, and half of it (each px the mean of 2 x 2), the larger of the
    // two taken. Detail of one scale beside strong detail is missed at the other scale, and its place goes to the
    // strong detail's frame: fine needles behind a group of grains at 1/4 of the sensor, broad striations at 1/2.
    bool two_scale = false;
};

// Per-slot temporaries for one frame's slice (built on the current device).
struct VolumeScratch {
    int ow = 0, oh = 0, dw = 0, dh = 0;
    float *img_d = nullptr, *y = nullptr, *t1 = nullptr, *t2 = nullptr, *t3 = nullptr;
    DevGaussian g08, g_fm, g_norm, g_lum;
    AreaResizer rz;
    // the half grid of the two-scale measure (dw2 = 0: one scale)
    int dw2 = 0, dh2 = 0;
    float *img2 = nullptr, *y2 = nullptr, *u1 = nullptr, *u2 = nullptr, *u3 = nullptr;
    DevGaussian g_fm2, g_norm2;
    AreaResizer rz2;
    void alloc(int ow, int oh, const VolumeParams& p);
    void free();
};

// fm (normalised focus) and lum of the warped output-grid frame, as float16 planes of dw*dh.
void volume_slice(const float* d_img_out, const VolumeParams& p, VolumeScratch& sc, void* d_fm16, void* d_lum16, cudaStream_t s);

// The noise floor of the depth volume's focus measure (FINDINGS 38): the same measure taken of the difference of
// two adjacent warped frames (pure noise, / sqrt 2), normalised by the first frame's luma and box-aggregated over
// agg_r like a slice, square-rooted; returns its median over the depth grid. The median of a pixel's focus curve
// in defocused or flat areas equals this floor, so the confidence ramp is in effect an SNR test against it.
float depth_noise_floor(const float* d_a_out, const float* d_b_out, const VolumeParams& p, VolumeScratch& sc, cudaStream_t s);

// collapse(c_coef + [c_base / count]) on the device: coef levels (levels-1, x3 float), base x3 float
// already divided; d_tmp and d_up are scratch buffers of the level-0 size (x3). Result in d_out.
void collapse_pyramid(const std::vector<const float*>& d_coef, const float* d_base_f, const std::vector<LevelShape>& shapes,
                      float* d_out, float* d_tmp, float* d_up, cudaStream_t s);

// Guide statistics shared by every slice (box means of the guide and its variance).
struct GuideStats {
    int w = 0, h = 0, r = 0;
    float *mI = nullptr, *vI = nullptr;   // mean, variance (box r)
    float *t1 = nullptr, *t2 = nullptr, *t3 = nullptr, *t4 = nullptr, *t5 = nullptr;   // slice scratch
    void build(const float* d_guide, int w, int h, int r, cudaStream_t s);
    void free();
};

// One slice: V = sqrt(max(guided_filter(guide, fm, r, eps), 0)); fm16 -> float plane d_V.
void aggregate_slice(const float* d_guide, const GuideStats& gs, const void* d_fm16, float* d_V, float eps, cudaStream_t s);

}  // namespace stackcore
