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

#include <cstdint>
#include <functional>

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace stackcore {

struct DepthParams {
    float p1 = 0.03f, p2 = 0.4f, edge = 0.02f;
    int dirs = 8;
    float c_lo = 0.45f, c_hi = 0.65f;
    // the sub-frame argmin at an end frame: true = that frame (0.6.10, FINDINGS 37a); false = its neighbour (1 or
    // K-2), the prototype's clamp, for the 0.5.x reference B
    bool end_frames = true;
    int end_guard = 3;
    float end_prior = 1.0f;
    float dark_prior = 1.0f;
    // the dark prior is switched off within this many sensor px of a confident (sharp) pixel: a featureless pixel
    // there is a gap between needles or the inside of a rim, not background, and must follow its neighbours
    // (FINDINGS 34). 0 = the prior everywhere the pixel itself is featureless (the 0.5.x behaviour).
    float dark_reach_px = 24.f;
    // out-of-range pixels (focus curve peaking at an end) take the end their neighbourhood votes for, the vote
    // smoothed over this many sensor px: on a fine grid the per-pixel choice flips from patch to patch and the
    // patch borders show as wisps in defocused background (FINDINGS 34). 0 = each pixel its own end (0.5.x).
    float end_vote_px = 48.f;
    float halo_prior = 2.0f;
    float halo_radius_px = 80.f;
    float depth_full_fraction = 0.25f;
    float gf_eps = 1e-4f;
    // the flat-region mask (dh*dw on the device, 1 = no texture within reach; Pass1Options::flat_blur_px): where it
    // is set, the SGM depth is blurred over half flat_blur_px before upsampling, so defocused background changes
    // frame gradually (as a blend of frames) instead of at a hard border between two patches (FINDINGS 34)
    const float* flat = nullptr;
    float flat_blur_px = 0.f;
    // in flat regions the data term prefers the frame with the least focus energy (defocused background shows
    // the fewest ghosts of nearby objects there), weighted by this; the end and dark priors step back by the
    // mask's share. 0 = flat pixels keep the 0.6.3 cost (no data term, priors decide). Experimental: the fine
    // focus measure is noise in defocused background, so this term settled on intermediate frames (FINDINGS 34)
    float flat_min = 0.f;
    // the width of the band beside sharp detail that takes its frame (and of the band-to-far transition), in
    // sensor px; 0 = flat_blur_px, the 0.6.4 behaviour where one number set both the background's smoothness and
    // how far a sharp object reaches (FINDINGS 36)
    float flat_band_px = 0.f;
    // weak features keep this share x min(contrast / c_lo, 1) of their own SGM depth where the flat fill would
    // otherwise replace it: features with focus evidence of their own, far from confident detail, survive a wide
    // background blur. 0 = off (the fill replaces everything beyond the reach of confident detail)
    float flat_keep = 0.f;
    // what the background beyond the band takes (FINDINGS 40): 0 = the quiet end of the stack (0.6.4, the fewest
    // ghosts, a flat wax); 1 = the loud end, sliding between the ends where the end frames' coarse energies are
    // within flat_end_soft (relative) of each other; 2 = the frame where each pixel's focus measure (the aggregated
    // slice, pooled over 24 px) peaks over the whole stack, energy-weighted and smoothed over 32 px (the blob-wise
    // field Helicon's background follows, FINDINGS 37c); 3 = that field snapped to the ends over a transition of
    // 2 x flat_end_soft (fraction of the stack).
    // 2 is the default since 0.6.16 (FINDINGS 40); 0 = --flat-end quiet, the 0.6.4 .. 0.6.15 depth-stage background
    // (the composition's flat blend, 0.6.18, is separate: --flat-blend 0 for the 0.6.15 B)
    int flat_end = 2;
    float flat_end_soft = 0.3f;
    // progress for a front end (name, fraction of the depth stage), called after a sync at each step: the focus curves,
    // the cost volume, each SGM band, the upsampling. Four seconds of silence at full resolution read as a hang
    std::function<void(const char*, double)> progress;
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

// d_V: dh*dw*K float32, k fastest (post-aggregation, post-sqrt), on the device; overwritten by the cost.
// h_lum: the luma volume, float16 bits, plane-major (K planes of dh*dw) on the HOST (pinned for speed): it is
// streamed through a 256 MB staging buffer, twice, and never sits on the device (FINDINGS 46: 3 GB less GPU
// memory at 1/2 resolution and 200 frames; the 201-frame stack paged on a 12 GB card before).
// d_guide_full: oh*ow float (luma of collapsed C, clipped); d_ga: dh*dw (the same at depth res).
// Runs on the current device and stream; needs the volume plus the SGM's path sums (1.5 GB, or bands) free.
DepthResult depth_from_volume(const float* d_V, const uint16_t* h_lum, const float* d_guide_full, const float* d_ga,
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
