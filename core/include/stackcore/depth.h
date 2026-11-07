// Regularised depth from the aggregated focus volume: fusion.sgm_from_volume
// from the sqrt onward, and sgm.py.
//
//   per pixel: contrast of the focus curve -> confidence w (ramp c_lo..c_hi);
//   curves peaking at an end frame get no confidence and a pull to that end;
//   spill zones beside extended bright objects (opening, dilation, texture
//   veto) discount w and prefer darker frames; darkness prior for weak,
//   featureless pixels; cost C = w (1 - V / Vmax) + priors;
//   SGM over 8 directions with P2 reduced across guide edges; sub-frame
//   argmin; bilinear upsample; guided filter on the output grid; rint;
//   3x3 median -> depth_s. conf = w upsampled.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace stackcore {

struct DepthParams {
    float p1 = 0.03f, p2 = 0.4f, edge = 0.02f;
    int dirs = 8;
    float c_lo = 0.45f, c_hi = 0.65f;
    int end_guard = 3;
    float end_prior = 1.0f;
    float dark_prior = 1.0f;
    float halo_prior = 2.0f;
    float halo_radius_px = 80.f;
    float depth_full_fraction = 0.25f;
    float gf_eps = 1e-4f;
};

struct DepthResult {
    int dw = 0, dh = 0, ow = 0, oh = 0, K = 0;
    int sgm_bands = 1;            // > 1 when the path-sum volume was processed in overlapping bands (memory)
    std::vector<float> depth_s;   // oh*ow, integer-valued frame index
    std::vector<float> conf;      // oh*ow
    std::vector<float> spill;     // dh*dw, the h map (kept for the refinement pass)
    std::vector<float> depth_d;   // dh*dw, the sub-frame argmin before upsampling (diagnostic)
    double end_fraction = 0;      // share of pixels peaking at an end frame
    double ms_cost = 0, ms_sgm = 0, ms_post = 0;
};

// d_V: dh*dw*K float32, k fastest (post-aggregation, post-sqrt). d_Lv16: same layout, float16.
// d_guide_full: oh*ow float (luma of collapsed C, clipped); d_ga: dh*dw (the same at depth res).
// Runs on the current device and stream; needs ~2 x the volume in free memory.
DepthResult depth_from_volume(const float* d_V, const void* d_Lv16, const float* d_guide_full, const float* d_ga,
                              int dw, int dh, int K, int ow, int oh, const DepthParams& p, cudaStream_t s);

// plane-major (K, dh, dw) -> k-fastest (dh, dw, K), float32 and float16 variants; the _chunk forms take
// planes k0 .. k0 + kc - 1 (a staging buffer of kc planes) and write them into their k slots of the full volume
void planes_to_kfastest_f32_chunk(const float* d_planes, float* d_out, int dw, int dh, int K, int k0, int kc, cudaStream_t s);
void planes_to_kfastest_f16_chunk(const void* d_planes, void* d_out, int dw, int dh, int K, int k0, int kc, cudaStream_t s);
void planes_to_kfastest_f32(const float* d_planes, float* d_out, int dw, int dh, int K, cudaStream_t s);
void planes_to_kfastest_f16(const void* d_planes, void* d_out, int dw, int dh, int K, cudaStream_t s);

// generic guided filter (He et al., grey guide, box r, eps); 5 scratch planes of w*h
void guided_filter(const float* d_guide, const float* d_src, float* d_out, int w, int h, int r, float eps, float* scratch5,
                   cudaStream_t s);

}  // namespace stackcore
