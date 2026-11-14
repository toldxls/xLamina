// The dust map: a frame of an evenly lit white background taken through the same optics, which shows the shadows of
// the dust on the sensor and in the optical path. Dust sits at the same pixels of every frame, so once the frames are
// aligned (each scaled and shifted a little differently) one speck becomes a trail in the stack. With a dust map the
// specks are taken out of every frame as it is read, before anything else sees it:
//
//   - where the map is darker than its own smooth background (a dust shadow), the frame's pixels are multiplied by
//     background / map: the shadow is a share of the light, and that share comes back;
//   - where the shadow is too deep for that (under 40 % of the light left: the gain would be mostly noise), the pixel
//     is filled from the nearest clean pixels around the speck.
//
// Everything outside the detected specks is left exactly as it is (a gain of 1): the map is not divided into the
// frame as a whole, which would also press the map's own noise, its 8-bit steps and its vignetting into every frame.
// The correction is therefore sparse: a list of pixels with their gains, and a list of pixels to fill.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace stackcore {

struct DustFix {
    int w = 0, h = 0;                     // the map's size: it applies to frames of this size only
    std::vector<uint32_t> idx;            // pixels (y * w + x) with a gain
    std::vector<float> gain;              // ... and the gain, > 1
    std::vector<uint32_t> heal;           // pixels filled from their surroundings
    std::vector<uint32_t> heal_src;       // 8 source pixels per healed pixel (the nearest clean pixel in 8 directions)
    std::vector<float> heal_w;            // ... and their weights (1 / distance, summing to 1)
    // the mask itself, whatever the correction made of it: every pixel in or beside a dust shadow, the 8 clean pixels
    // its surroundings are read from (with weights summing to 1), and the share of light the map says it keeps
    std::vector<uint32_t> px, px_src;
    std::vector<float> px_w, map_ratio;
    bool calibrated = false;              // the gains come from the stack's own frames (calibrate_dust), not from the map
    float map_deepest = 1.f;              // the darkest shadow in the map (deepest is the one in effect)
    // what was found, for the front end to say
    int spots = 0;                        // connected dust shadows
    float deepest = 1.f;                  // the darkest shadow as a share of the background (0.6 = 40 % of the light lost)
    float residual = 0.f;                 // the largest deviation left in the map itself after its own correction
    float vignette = 0.f;                 // the background's fall-off, corner to brightest (not corrected; for the record)
    std::string name;                     // the file's name
    bool empty() const { return idx.empty() && heal.empty(); }
};

// Reads the map (any image Windows or the TIFF reader decodes) and finds the dust. False with `err` set when the
// file cannot be read or is not an even bright field (too dark, or more of it "dust" than a dust map has).
bool build_dust_fix(const std::filesystem::path& map, DustFix& out, std::string* err);

// the correction from the share of light each mask pixel keeps (fix.px's order)
void set_dust_correction(DustFix& fix, const std::vector<float>& kept);
// The same mask with its depths measured on the stack's frames: a white frame says where the dust is, but not how
// deep its shadows are in a photograph (a camera's tone curve flattens towards white, which makes every shadow on
// a white frame look shallower than it is at mid-tones; measured, 31 % lost on the map was 55 % lost in the frames).
// ratios: frames x fix.px.size() values from DevDustProbe, frame after frame.
std::shared_ptr<const DustFix> calibrate_dust(const DustFix& fix, const std::vector<float>& ratios, int frames);

// measures, for one frame on its stream, each mask pixel's luma over that of its surroundings (n floats to d_out)
struct DevDustProbe {
    uint32_t *px = nullptr, *src = nullptr;
    float* wt = nullptr;
    int n = 0;
    void upload(const DustFix* fix, int w, int h);
    void free();
    void measure(const float* d_rgb, float* d_out, cudaStream_t s) const;
};

// the correction on one GPU: uploaded once per pass, applied to each frame (float RGB, interleaved) on its stream
struct DevDust {
    uint32_t *idx = nullptr, *heal = nullptr, *src = nullptr;
    float *gain = nullptr, *hw = nullptr;
    int n = 0, nh = 0;
    void upload(const DustFix* fix, int w, int h);   // nothing when fix is null, empty, or of another size
    void free();
    void apply(float* d_rgb, cudaStream_t s) const;
};

}  // namespace stackcore
