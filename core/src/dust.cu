#include "stackcore/dust.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>

#include "stackcore/tiff.h"

namespace stackcore {

#define CK(x)                                                                                        \
    do {                                                                                             \
        cudaError_t e_ = (x);                                                                        \
        if (e_ != cudaSuccess) throw std::runtime_error(std::string(#x) + ": " + cudaGetErrorString(e_)); \
    } while (0)

// ----------------------------------------------------------------------------- finding the dust (host)

namespace {
// three box blurs in a row (close to a Gaussian), with weights: pixels of weight 0 take no part, so the background
// under a speck is what surrounds the speck
void smooth_weighted(std::vector<float>& v, std::vector<float>& wt, int w, int h, int r) {
    std::vector<float> tv(v.size()), tw(v.size());
    for (size_t i = 0; i < v.size(); ++i) v[i] *= wt[i];
    for (int pass = 0; pass < 3; ++pass) {
        for (int y = 0; y < h; ++y) {   // rows
            double sv = 0, sw = 0;
            const float *pv = &v[(size_t)y * w], *pw = &wt[(size_t)y * w];
            for (int x = -r; x <= r; ++x) { int c = std::min(std::max(x, 0), w - 1); sv += pv[c]; sw += pw[c]; }
            for (int x = 0; x < w; ++x) {
                tv[(size_t)y * w + x] = (float)sv; tw[(size_t)y * w + x] = (float)sw;
                int a = std::max(x - r, 0), b = std::min(x + r + 1, w - 1);
                sv += pv[b] - pv[a]; sw += pw[b] - pw[a];
            }
        }
        for (int x = 0; x < w; ++x) {   // columns
            double sv = 0, sw = 0;
            for (int y = -r; y <= r; ++y) { int c = std::min(std::max(y, 0), h - 1); sv += tv[(size_t)c * w + x]; sw += tw[(size_t)c * w + x]; }
            for (int y = 0; y < h; ++y) {
                v[(size_t)y * w + x] = (float)sv; wt[(size_t)y * w + x] = (float)sw;
                int a = std::max(y - r, 0), b = std::min(y + r + 1, h - 1);
                sv += tv[(size_t)b * w + x] - tv[(size_t)a * w + x]; sw += tw[(size_t)b * w + x] - tw[(size_t)a * w + x];
            }
        }
    }
    for (size_t i = 0; i < v.size(); ++i) { v[i] = wt[i] > 1e-6f ? v[i] / wt[i] : 0.f; }
}
}  // namespace

bool build_dust_fix(const std::filesystem::path& map, DustFix& out, std::string* err) {
    auto fail = [&](const std::string& why) { if (err) *err = why; return false; };
    out = DustFix{};
    std::vector<uint8_t> rgb;
    int w = 0, h = 0;
    std::string why;
    if (!decode_rgb8(map, rgb, w, h, &why)) return fail("the dust map could not be read: " + why);
    if (w < 64 || h < 64) return fail("the dust map is too small");
    const size_t n = (size_t)w * h;
    std::vector<float> lum(n);
    double mean = 0;
    for (size_t i = 0; i < n; ++i) {
        lum[i] = (0.2126f * rgb[i * 3] + 0.7152f * rgb[i * 3 + 1] + 0.0722f * rgb[i * 3 + 2]) / 255.f;
        mean += lum[i];
    }
    rgb.clear(); rgb.shrink_to_fit();
    mean /= (double)n;
    if (mean < 0.25) return fail("this image is too dark for a dust map (a dust map is a photograph of an evenly lit white background)");

    // the smooth background at 1/8: twice, the second time without the pixels that came out darker than it
    const int ds = 8, sw = (w + ds - 1) / ds, sh = (h + ds - 1) / ds, r = std::max(4, std::min(sw, sh) / 40);
    std::vector<float> small((size_t)sw * sh);
    for (int y = 0; y < sh; ++y)
        for (int x = 0; x < sw; ++x) {
            double s = 0; int c = 0;
            for (int yy = y * ds; yy < std::min(h, (y + 1) * ds); ++yy)
                for (int xx = x * ds; xx < std::min(w, (x + 1) * ds); ++xx) { s += lum[(size_t)yy * w + xx]; ++c; }
            small[(size_t)y * sw + x] = (float)(s / std::max(c, 1));
        }
    std::vector<float> bg = small, wt(small.size(), 1.f);
    smooth_weighted(bg, wt, sw, sh, r);
    {
        std::vector<float> bg2 = small;
        for (size_t i = 0; i < small.size(); ++i) wt[i] = small[i] < 0.985f * bg[i] ? 0.f : 1.f;
        smooth_weighted(bg2, wt, sw, sh, r);
        for (size_t i = 0; i < bg.size(); ++i) if (bg2[i] > 0) bg[i] = bg2[i];
    }
    {
        float lo = 1e9f, hi = 0;
        for (float v : bg) { lo = std::min(lo, v); hi = std::max(hi, v); }
        out.vignette = hi > 0 ? 1.f - lo / hi : 0.f;
        if (out.vignette > 0.6f) return fail("this image is not evenly lit enough for a dust map (its background varies by more than half)");
    }
    auto bg_at = [&](int x, int y) {   // bilinear, the small grid's pixel centres at (i + 0.5) * ds
        float fx = std::min(std::max((x + 0.5f) / ds - 0.5f, 0.f), (float)(sw - 1)), fy = std::min(std::max((y + 0.5f) / ds - 0.5f, 0.f), (float)(sh - 1));
        int x0 = (int)fx, y0 = (int)fy, x1 = std::min(x0 + 1, sw - 1), y1 = std::min(y0 + 1, sh - 1);
        float ax = fx - x0, ay = fy - y0;
        return (bg[(size_t)y0 * sw + x0] * (1 - ax) + bg[(size_t)y0 * sw + x1] * ax) * (1 - ay) + (bg[(size_t)y1 * sw + x0] * (1 - ax) + bg[(size_t)y1 * sw + x1] * ax) * ay;
    };
    // the share of the light each pixel kept: the map, lightly smoothed against its own noise (3 x 3), over the background
    std::vector<float> ratio(n);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            float s = 0;
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) s += lum[(size_t)std::min(std::max(y + dy, 0), h - 1) * w + std::min(std::max(x + dx, 0), w - 1)];
            ratio[(size_t)y * w + x] = (s / 9.f) / std::max(bg_at(x, y), 1e-3f);
        }
    lum.clear(); lum.shrink_to_fit();

    // dust: 3 % darker than the background or more. Each such pixel takes its surroundings with it (the shadow's soft
    // edge), out to 9 px; inside that, a pixel is corrected when it lost more than half a percent.
    const float seed = 0.97f;
    const int grow = 9;   // the shadow's soft edge, and the few pixels a speck sits away from where the map shows it
    std::vector<uint8_t> mask(n, 0);   // 1 = dust seed, 2 = its surroundings
    size_t seeds = 0;
    for (size_t i = 0; i < n; ++i) if (ratio[i] < seed) { mask[i] = 1; ++seeds; }
    if (seeds > n / 50) return fail("this image does not look like an even white background: " + std::to_string(100 * seeds / n) + " % of it is darker than its surroundings");
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            if (mask[(size_t)y * w + x] != 1) continue;
            for (int dy = -grow; dy <= grow; ++dy)
                for (int dx = -grow; dx <= grow; ++dx) {
                    int yy = y + dy, xx = x + dx;
                    if (yy < 0 || xx < 0 || yy >= h || xx >= w || dx * dx + dy * dy > grow * grow) continue;
                    uint8_t& m = mask[(size_t)yy * w + xx];
                    if (!m) m = 2;
                }
        }
    // the spots, counted as connected groups of the grown mask
    {
        std::vector<uint8_t> seen(n, 0);
        std::vector<uint32_t> stack;
        for (size_t i = 0; i < n; ++i) {
            if (!mask[i] || seen[i]) continue;
            ++out.spots;
            stack.push_back((uint32_t)i);
            seen[i] = 1;
            while (!stack.empty()) {
                uint32_t p = stack.back(); stack.pop_back();
                int px = (int)(p % w), py = (int)(p / w);
                const int nx[4] = {px - 1, px + 1, px, px}, ny[4] = {py, py, py - 1, py + 1};
                for (int k = 0; k < 4; ++k) {
                    if (nx[k] < 0 || ny[k] < 0 || nx[k] >= w || ny[k] >= h) continue;
                    size_t q = (size_t)ny[k] * w + nx[k];
                    if (mask[q] && !seen[q]) { seen[q] = 1; stack.push_back((uint32_t)q); }
                }
            }
        }
    }
    // every pixel of the mask, with the clean pixels its surroundings are read from: the nearest one outside the mask in
    // each of 8 directions, weighted by 1 / distance. That estimate of "the frame here without the dust" is what a
    // frame's own shadow depth is measured against, and what a pixel too dark to scale back is filled with.
    static const int dirs[8][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {-1, -1}, {1, -1}, {-1, 1}};
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const size_t i = (size_t)y * w + x;
            if (!mask[i]) continue;
            uint32_t src[8]; float wgt[8]; float sum = 0;
            for (int k = 0; k < 8; ++k) {
                int xx = x, yy = y, d = 0;
                bool ok = false;
                while (true) {
                    xx += dirs[k][0]; yy += dirs[k][1]; ++d;
                    if (xx < 0 || yy < 0 || xx >= w || yy >= h || d > 400) break;
                    if (!mask[(size_t)yy * w + xx]) { ok = true; break; }
                }
                src[k] = ok ? (uint32_t)((size_t)yy * w + xx) : (uint32_t)i;
                wgt[k] = ok ? 1.f / (float)d : 0.f;
                sum += wgt[k];
            }
            if (sum <= 0) continue;   // (no clean pixel within 400 px in any direction: left as it is)
            out.px.push_back((uint32_t)i);
            out.map_ratio.push_back(ratio[i]);
            for (int k = 0; k < 8; ++k) { out.px_src.push_back(src[k]); out.px_w.push_back(wgt[k] / sum); }
        }
    out.w = w; out.h = h;
    out.name = map.filename().string();
    set_dust_correction(out, out.map_ratio);   // the map's own depths: a first guess, replaced by the frames' (calibrate_dust)
    out.map_deepest = out.deepest;
    return true;
}

// The correction from the share of light each mask pixel keeps: a gain where the light can be scaled back, a fill
// from the surroundings where too little is left, nothing where the pixel lost under half a percent.
void set_dust_correction(DustFix& fix, const std::vector<float>& kept) {
    const float edge = 0.995f, heal_under = 0.40f;
    fix.idx.clear(); fix.gain.clear(); fix.heal.clear(); fix.heal_src.clear(); fix.heal_w.clear();
    fix.deepest = 1.f;
    for (size_t i = 0; i < fix.px.size(); ++i) {
        const float a = kept[i];
        if (!(a == a)) continue;
        fix.deepest = std::min(fix.deepest, a);
        if (a < heal_under) {
            fix.heal.push_back(fix.px[i]);
            for (int k = 0; k < 8; ++k) { fix.heal_src.push_back(fix.px_src[i * 8 + k]); fix.heal_w.push_back(fix.px_w[i * 8 + k]); }
        } else if (a < edge) {
            fix.idx.push_back(fix.px[i]);
            fix.gain.push_back(1.f / a);
        }
    }
}

// The depths measured on the stack's own frames. `ratios` holds, frame after frame, each mask pixel's luma over the
// luma of its surroundings (DevDustProbe). In most frames a given place is out of focus and smooth, so the ratio is
// the dust's shadow and nothing else; where a frame is sharp there it is the subject's texture as well. The median
// over the frames keeps the first and is not moved by the second. Under 8 frames a median says too little: every
// pixel of the mask is filled from its surroundings instead.
std::shared_ptr<const DustFix> calibrate_dust(const DustFix& fix, const std::vector<float>& ratios, int frames) {
    auto out = std::make_shared<DustFix>(fix);
    const size_t n = fix.px.size();
    if (n == 0 || frames < 1 || ratios.size() < n * (size_t)frames) return out;
    std::vector<float> kept(n), col(frames);
    for (size_t i = 0; i < n; ++i) {
        if (frames < 8) { kept[i] = 0.f; continue; }
        for (int k = 0; k < frames; ++k) col[k] = ratios[(size_t)k * n + i];
        std::nth_element(col.begin(), col.begin() + frames / 2, col.end());
        kept[i] = std::min(col[frames / 2], 1.f);   // dust takes light away: a pixel that reads brighter than its surroundings is left alone
    }
    set_dust_correction(*out, kept);
    out->calibrated = true;
    return out;
}
// ----------------------------------------------------------------------------- applying it (device)

__global__ void k_dust_gain(float* __restrict__ rgb, const uint32_t* __restrict__ idx, const float* __restrict__ gain, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const size_t p = (size_t)idx[i] * 3;
    const float g = gain[i];
    rgb[p] *= g; rgb[p + 1] *= g; rgb[p + 2] *= g;
}

// the sources are clean pixels (outside every speck), which neither kernel writes: no order between the two matters
__global__ void k_dust_heal(float* __restrict__ rgb, const uint32_t* __restrict__ heal, const uint32_t* __restrict__ src, const float* __restrict__ wt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float r = 0, g = 0, b = 0;
    for (int k = 0; k < 8; ++k) {
        const size_t q = (size_t)src[i * 8 + k] * 3;
        const float w = wt[i * 8 + k];
        r += w * rgb[q]; g += w * rgb[q + 1]; b += w * rgb[q + 2];
    }
    const size_t p = (size_t)heal[i] * 3;
    rgb[p] = r; rgb[p + 1] = g; rgb[p + 2] = b;
}

void DevDust::upload(const DustFix* fix, int w, int h) {
    free();
    if (!fix || fix->empty() || fix->w != w || fix->h != h) return;
    auto up = [](const void* host, size_t bytes) -> void* {
        if (!bytes) return nullptr;
        void* d = nullptr;
        CK(cudaMalloc(&d, bytes));
        CK(cudaMemcpy(d, host, bytes, cudaMemcpyHostToDevice));
        return d;
    };
    n = (int)fix->idx.size();
    nh = (int)fix->heal.size();
    idx = (uint32_t*)up(fix->idx.data(), fix->idx.size() * sizeof(uint32_t));
    gain = (float*)up(fix->gain.data(), fix->gain.size() * sizeof(float));
    heal = (uint32_t*)up(fix->heal.data(), fix->heal.size() * sizeof(uint32_t));
    src = (uint32_t*)up(fix->heal_src.data(), fix->heal_src.size() * sizeof(uint32_t));
    hw = (float*)up(fix->heal_w.data(), fix->heal_w.size() * sizeof(float));
}

void DevDust::free() {
    cudaFree(idx); cudaFree(gain); cudaFree(heal); cudaFree(src); cudaFree(hw);
    idx = heal = src = nullptr;
    gain = hw = nullptr;
    n = nh = 0;
}

void DevDust::apply(float* d_rgb, cudaStream_t s) const {
    if (n > 0) k_dust_gain<<<(n + 255) / 256, 256, 0, s>>>(d_rgb, idx, gain, n);
    if (nh > 0) k_dust_heal<<<(nh + 255) / 256, 256, 0, s>>>(d_rgb, heal, src, hw, nh);
}

// each mask pixel's luma over the luma of its surroundings (the 8 clean pixels around its speck), for calibrate_dust
__global__ void k_dust_probe(const float* __restrict__ rgb, const uint32_t* __restrict__ px, const uint32_t* __restrict__ src, const float* __restrict__ wt,
                             float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float bg = 0;
    for (int k = 0; k < 8; ++k) {
        const size_t q = (size_t)src[i * 8 + k] * 3;
        bg += wt[i * 8 + k] * (0.2126f * rgb[q] + 0.7152f * rgb[q + 1] + 0.0722f * rgb[q + 2]);
    }
    const size_t p = (size_t)px[i] * 3;
    const float v = 0.2126f * rgb[p] + 0.7152f * rgb[p + 1] + 0.0722f * rgb[p + 2];
    out[i] = bg > 0.02f ? v / bg : 1.f;   // (nearly black surroundings say nothing about a shadow)
}

void DevDustProbe::upload(const DustFix* fix, int w, int h) {
    free();
    if (!fix || fix->px.empty() || fix->w != w || fix->h != h) return;
    n = (int)fix->px.size();
    CK(cudaMalloc(&px, n * sizeof(uint32_t)));
    CK(cudaMalloc(&src, (size_t)n * 8 * sizeof(uint32_t)));
    CK(cudaMalloc(&wt, (size_t)n * 8 * sizeof(float)));
    CK(cudaMemcpy(px, fix->px.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(src, fix->px_src.data(), (size_t)n * 8 * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(wt, fix->px_w.data(), (size_t)n * 8 * sizeof(float), cudaMemcpyHostToDevice));
}

void DevDustProbe::free() {
    cudaFree(px); cudaFree(src); cudaFree(wt);
    px = src = nullptr;
    wt = nullptr;
    n = 0;
}

void DevDustProbe::measure(const float* d_rgb, float* d_out, cudaStream_t s) const {
    if (n > 0) k_dust_probe<<<(n + 255) / 256, 256, 0, s>>>(d_rgb, px, src, wt, d_out, n);
}

}  // namespace stackcore