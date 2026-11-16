// The pipeline as a library: alignment, pass 1 (+ depth volume), the fusion
// and the folder-level job. The CLI and the GUI both call this. Progress and
// preview callbacks let a GUI show the stack building (DESIGN.md: progressive
// preview); a cancel flag stops a job between frames.
#pragma once

#include <memory>
#include <atomic>
#include <complex>
#include <cstring>
#include <filesystem>
#include <functional>
#include <new>
#include <string>
#include <thread>
#include <vector>

#include "stackcore/dust.h"
#include "stackcore/align.h"
#include "stackcore/compose.h"
#include "stackcore/fold.h"
#include "stackcore/streamer.h"
#include "stackcore/volume.h"
#include "stackcore/warp.h"

namespace stackcore {

// Log line sink (default: stdout). Installed once per process.
using LogFn = std::function<void(const std::string&)>;
// Why this PC cannot render, in words for the user, or an empty string when it can: no NVIDIA GPU or driver, or
// a GPU (of `devs`, or of all when that is empty) older than the architecture the kernels are compiled for.
// With launch, a small kernel is also run on each of the GPUs: a driver too old for the code this build carries for a
// card answers every other call and fails only there ("the provided PTX was compiled with an unsupported toolchain").
// That starts the cards' contexts (a few tenths of a second each), so a front end asks for it when a render begins.
std::string gpu_problem(const std::vector<int>& devs = {}, bool launch = false);
void set_log(LogFn fn);
void logf(const char* fmt, ...);

// Why a job failed, for a front end to put into words (the return codes stay as they are: 0 done, 1 failed,
// 2 a bad option, 3 cancelled, 4 the GPU memory ran out, 6 the outputs were computed but could not be written).
// The engine notes the first cause of a job; a front end calls take_fail() before a job (to clear) and after it.
enum class Fail {
    None,
    HostMemory,      // the PC's own memory (RAM) ran out
    GpuFault,        // a CUDA call failed for another reason than memory: the driver was reset or stopped answering
    FileRead,        // a frame could not be read; detail names it
    FileWrite,       // the output folder or a file in it could not be written; detail names it and says why
    Alignment,       // a frame has no usable pair; detail names it
    TooManyFrames,   // more frames than the depth stage takes
    Other,
};
struct FailInfo { Fail kind = Fail::None; std::string detail; };
void note_fail(Fail kind, const std::string& detail);
FailInfo take_fail();
// the most frames a stack may have (the depth stage's kernels hold one value per frame in fixed storage)
constexpr int kMaxFrames = 512;

// progress(stage, fraction 0..1); preview(name, rgb float interleaved, w, h);
// frame(k, rgb, w, h): the frame just processed (warped onto the output grid, small)
using ProgressFn = std::function<void(const std::string& stage, double fraction)>;
// rgb points at w x h x 3 floats that are valid only during the call: copy what is needed. (The hooks
// run on the GPU worker threads; a fresh 10 MB vector per call cost ~3 ms of page faults each, FINDINGS 30.)
using PreviewFn = std::function<void(const std::string& name, const float* rgb, int w, int h)>;
// The window of the frame a view shows (0.6.33, FINDINGS 47): a crop x0, y0, w, h in the frame's pixels, shown as
// pw x ph pixels (each the mean of the frame's pixels it covers), or at 1/down when pw is 0; fw, fh = the frame's
// size. A frame view carries the window it was taken with, so the front end can place it. w = 0 asks for the whole
// frame at frame_down.
struct ViewWindow { int x0 = 0, y0 = 0, w = 0, h = 0, down = 1, fw = 0, fh = 0, pw = 0, ph = 0; };
// A frame view: w x h pixels of 8 bits a channel, 0xffRRGGBB, in the frames' own colour space; valid only during the
// call. (Float RGB before: three times the bytes off the card and through the hook, and a conversion on the host.)
using FrameViewFn = std::function<void(int k, const uint32_t* px, int w, int h, const ViewWindow& win)>;

struct Hooks {
    ProgressFn progress;
    PreviewFn preview;
    FrameViewFn frame;
    // when set, each frame view renders what the screen shows: the GUI's visible window at the screen's resolution,
    // about 2 Mpx whatever the zoom, instead of the whole frame at frame_down (a Full frame is 31 Mpx, 388 MB)
    std::function<ViewWindow(int fullW, int fullH)> view_window;
    size_t view_max_px = 4000000;   // a frame view's buffer; a window asking for more is sampled coarser
    std::atomic<bool>* cancel = nullptr;
    int preview_down = 8;        // preview images are 1/preview_down of the output grid
    int preview_every = 8;       // frames between pass-1 previews
    int frame_down = 16;         // per-frame views are 1/frame_down of the output grid
    // preview_min_ms: at most one building preview per device within this many ms (every copy off a GPU costs its
    // uploads link time, FINDINGS 30). frame_min_ms: the frame views come no closer together than this on average,
    // by showing every n-th frame of the stack, n from the pass's frame rate (even steps through the stack; a view
    // per so many ms took frames at uneven steps); 0 = a view of every frame
    int frame_min_ms = 60, preview_min_ms = 150;
    bool cancelled() const { return cancel && cancel->load(); }
};

// ---- pass 1: fold + depth volume over a frame list

struct Pass1Options {
    std::filesystem::path list;                   // frame list file (path zr zi tx ty per line), or...
    std::vector<std::filesystem::path> paths;     // ...the frames and transforms in memory
    std::vector<Similarity> T;
    Grid g;
    double radius_px = 6.0, power = 4.0;
    double c_radius_px = 0.0;                     // C's energy aggregation sigma in sensor px (FoldParams::c_radius); 0 = raw selection
    double b_radius_px = 0.0;                     // B: isotropic Gaussian over each aggregated focus slice, sigma in sensor px; 0 = off
    std::vector<int> frames_sel;
    StreamerOptions opt;
    bool volume = false;
    bool with_a = true;                           // accumulate A (off when A is not an output)
    int levels = 0;                               // pyramid levels; 0 = from the output size (a strip takes the whole frame's)
    // a single-GPU pass whose volume slices cannot all fit returns 4 before it allocates its slots, instead of
    // running out of memory part-way (run_stack sets it when it may answer with strips)
    bool stop_if_short = false;
    bool host_state = false;                      // also download the merged state into Pass1::merged (the fold command)
    // the depth volume's resolution as a fraction of the sensor. 1/4 again since 1.2.0 (1/2 from 0.6.1, for the
    // doubled plate edges of 1/4, FINDINGS 32). With one scale, at 1/2 broad weak detail beside strong detail
    // (striations beside grains) loses its depth to the strong detail, and at 1/4 fine detail does (needles behind a
    // group of grains): the blurred band of a halo, either way. 1/2 with the measure at two scales
    // (depth_two_scale) keeps both in the depth map, but its B was judged the poorer one by eye on two stacks:
    // off, and 1/4
    // vol_agg_px: the edge-aware aggregation of the volume, sensor px. 0 (off) since 1.2.0: at 32 a crystal's depth
    // reached 20 to 60 px further into the background than the crystal does (measured against another stacker's
    // depth map on a group of grains), and that band of background, taken from the crystal's frame, is the halo
    double depth_fraction = 0.25, depth_fm_px = 6.0, vol_agg_px = 0.0;
    bool depth_two_scale = false;                 // VolumeParams::two_scale
    // defocused background (FINDINGS 34): in flat regions of the guide (no texture within flat_reach_px) every
    // focus slice is box-blurred over flat_blur_px, and in the depth stage pixels without sharp detail within reach
    // take the nearest sharp detail's frame in a band about flat_blur_px wide, the quieter end of the stack beyond
    // it (DepthParams::flat, flat_blur_px). 0 = off (the priors decide). 128 from 0.6.4, 256 from 0.6.22; off since
    // 1.2.0: the band took real detail with it (crystals in haze, a needle group in the background, checked against
    // the frames), and what it was for, the seams between patches of background, B's wide fade now deals with
    // (StackOptions::b_steps)
    double flat_blur_px = 0.0, flat_reach_px = 24.0;
    double flat_min = 0.0;                        // DepthParams::flat_min (least-structure data term in flat regions); 0 = off
    // the flat mask's two texture thresholds, given instead of ranked over this rectangle (flat_hi > flat_lo >= 0): a
    // strip takes the whole frame's, or its mask would follow its own content
    float flat_lo = -1.f, flat_hi = -1.f;
    // per-row corrections (FINDINGS 51), K x H at full resolution, applied inside every warp of the frame (RowCorr);
    // empty = none. row_gain flattens the frame's brightness banding; rs_dx / rs_dy are the frame's rolling-shutter
    // displacement per sensor row (full-res px), added to the sample position
    std::vector<float> row_gain, rs_dx, rs_dy;
    // the dust map's correction (dust.h), applied to every frame as it is read; null = none
    std::shared_ptr<const DustFix> dust;
    Hooks hooks;
};

// Page-locked host array: GPU copies to and from it run asynchronously at the link rate (a pageable
// destination is staged by the driver and is effectively synchronous).
template <class T>
class PinnedVec {
public:
    PinnedVec() = default;
    ~PinnedVec() { clear(); }
    PinnedVec(const PinnedVec&) = delete;
    PinnedVec& operator=(const PinnedVec&) = delete;
    PinnedVec(PinnedVec&& o) noexcept : p_(o.p_), n_(o.n_), pinned_(o.pinned_) { o.p_ = nullptr; o.n_ = 0; }
    PinnedVec& operator=(PinnedVec&& o) noexcept {
        if (this != &o) { clear(); p_ = o.p_; n_ = o.n_; pinned_ = o.pinned_; o.p_ = nullptr; o.n_ = 0; }
        return *this;
    }
    // NOT zeroed (a memset of 1.8 GB costs 0.15 s); the caller fills every element. pinned = false takes ordinary
    // memory instead: the copies are then staged by the driver (slower), and nothing is added to the page-locked
    // total, which has a ceiling of its own when two GPUs are in use (FINDINGS 54)
    void resize(size_t n, bool pinned = true) {
        clear();
        if (!n) return;
        if (pinned) {
            if (cudaHostAlloc((void**)&p_, n * sizeof(T), cudaHostAllocPortable) != cudaSuccess) throw std::bad_alloc();
        } else {
            p_ = (T*)std::malloc(n * sizeof(T));
            if (!p_) throw std::bad_alloc();
        }
        pinned_ = pinned;
        n_ = n;
    }
    void clear() {
        if (p_) { if (pinned_) cudaFreeHost(p_); else std::free(p_); }
        p_ = nullptr;
        n_ = 0;
    }
    // unpinning 1.8 GB takes ~0.2 s: hand the buffer to a thread that frees it off the critical path
    void release_async() {
        if (!p_) return;
        T* p = p_;
        const bool pinned = pinned_;
        p_ = nullptr;
        n_ = 0;
        std::thread([p, pinned] { if (pinned) cudaFreeHost(p); else std::free(p); }).detach();
    }
    bool pinned() const { return pinned_; }
    T* data() { return p_; }
    const T* data() const { return p_; }
    size_t size() const { return n_; }
    bool empty() const { return n_ == 0; }

private:
    T* p_ = nullptr;
    size_t n_ = 0;
    bool pinned_ = true;
};

struct Pass1 {
    Grid g;
    FoldParams fp;
    VolumeParams vp;
    std::vector<std::filesystem::path> paths;
    std::vector<Similarity> T;
    std::vector<int> devs;
    FoldState dst;                 // the merged state, resident on devs[0]; the caller frees it
    HostFoldState merged;          // only when host_state
    double merge_seconds = 0, setup_seconds = 0;
    std::vector<float> guide, ga;
    std::vector<float> flat;               // the flat-region mask at depth resolution (when flat_blur_px > 0)
    float flat_lo = -1.f, flat_hi = -1.f;  // the mask's two texture thresholds as used (ranked over the rectangle, or given)
    // each depth pixel's own focus peak over the frames, from the aggregated slices pooled over 24 px BEFORE the flat
    // regions' wide blur (when flat_blur_px > 0): the largest value, its frame, and the sum over the frames
    std::vector<float> peak_best, peak_arg, peak_sum;
    PinnedVec<uint16_t> fm_all, lum_all;   // fm_all only when host_state (the fold command writes it; the fusion never reads it)
    PinnedVec<float> V_all;
    int slices = 0;
    StreamerStats st;
    double vol_seconds = 0;
    // per frame index (the list's order): the mean of the focus measure over the output grid, summed on the GPU while
    // the frame folds (fold_energy_add); 0 for a frame that was not run. The soft-frame flag reads it (frame_sharpness).
    std::vector<double> frame_energy;
};

bool read_list(const std::filesystem::path& list, std::vector<std::filesystem::path>& paths, std::vector<Similarity>& T);
int run_pass1(const Pass1Options& po, Pass1& P);

// The soft-frame rule (FINDINGS 49). A frame blurred by movement during its exposure loses fine detail everywhere,
// while the focus energy of its neighbours in the stack is nearly the same as its own would have been (the focus
// profile across frames is smooth), so each run frame's energy is set against the median of the five run frames
// centred on it. `ratio` 1 = as expected; a frame is `soft` when its ratio is under `cutoff` (0.90 by default: a
// tenth under its neighbours). A spread-based rule (median minus robust sigmas of the log ratios) was tried first
// and is degenerate here: the focus profile is locally monotone, so most frames ARE the median of their five and
// have the ratio 1 exactly, which makes the robust sigma 0. The wobble flags the frames whose position jumped; this
// flags the frames that are blurred, which FINDINGS 44/45 found to be different frames.
struct FrameSharpness {
    std::vector<double> energy;   // per frame index: Pass1::frame_energy as given (0 = not run)
    std::vector<double> ratio;    // per frame index: energy / median of the five run frames centred on it; 0 = not run
    std::vector<char> soft;       // per frame index: 1 = flagged soft
    double threshold = 0;         // the ratio below which a frame is soft (0 when fewer than 4 frames were run)
    double peak = 0;              // the largest energy (the sharpest frame's)
    int n_soft = 0;
};
FrameSharpness frame_sharpness(const std::vector<double>& energy, double cutoff = 0.9);

// ---- the fusion (pass 1 + depth + refine + compose + finalize)

struct StackOptions {
    Pass1Options po;
    std::string out;                   // output folder; empty = write nothing
    // refine_r: how many frames either way of the coarse depth the fine focus may look. -1 = from the stack: 22 % of
    // its frames, at least 12 and at most 60. A wide range lets the background beside a crystal reach its own frame
    // where the coarse depth gave it to the crystal, but at 60 on 276 frames it drew arcs around needle tips on a
    // dark background and lost small grains, and at 120 on 261 it looked through translucent plates to the matrix
    // behind: 12 stays the default.
    int streams = 2, refine_r = 12;
    // the fine focus's window widens as the stack gets shorter than 192 frames (refine_fm_px x 192 / frames, at most
    // twice): see fuse_from_pass1
    bool refine_scale = true;
    // the fine focus's rim check (RefineState): a pair of matching peaks either side of the smooth depth is the rim of
    // a speck out of focus, and the pixel keeps the centre; false = the strongest frame in range, whatever it is.
    // Both checks run on stacks under 192 frames only (large steps between frames): see fuse_from_pass1
    bool refine_twin = true;
    // ... and its island check (refine_depth): a move of more than a quarter of the range away from the smooth depth
    // stands only where it joins a neighbouring surface; needs the rim check's planes
    bool refine_island = true;
    double refine_fm_px = 9.0, soft_px = 3.0;
    // B's composition (compose.h compose_frame_steps), since 1.2.0: every pixel from the frame its fine depth names,
    // with a fade between neighbouring pixels' frames of b_fade (sigma, sensor px; 2 left the edge of a halo as a line) where there is detail and b_wide
    // where there is none. false = the composition up to 1.1.1: soft labels (soft_px) mixed by confidence with a
    // second image whose depth slides smoothly through the stack; that one blurred weak detail beside strong detail
    // and drew a band there.
    bool b_steps = true;
    double b_fade = 5.0, b_wide = 24.0;
    double c_smoothing = 0.0;          // C's smoothing (FinalParams::c_smoothing); 0 = the raw selection
    double soft_cutoff = 0.9;          // the soft-frame flag's cutoff (frame_sharpness): a frame under this share of its neighbours' focus energy
    // FINDINGS 51: apply the alignment's per-frame row corrections in every warp (run_job copies them into po):
    // the brightness-banding gain and the rolling-shutter row offsets. The warp correction is on by default (it removes
    // most doubled edges); the banding correction is off until judged
    bool fix_banding = false, fix_rs = true;
    // B's depth priors (DepthParams). On the 1/2 grid the dark prior put dark far frames into the gaps of needle
    // fans and inside bright rims (FINDINGS 33), so since 0.6.3 it is gated off within dark_reach_px of sharp
    // content and kept for defocused background (FINDINGS 34). The 0.5.x reference B is --b-reference 1.
    double halo_prior = 2.0, dark_prior = 1.0;
    double dark_reach_px = 24.0;   // DepthParams::dark_reach_px; 0 = the 0.5.x behaviour
    double end_vote_px = 48.0;     // DepthParams::end_vote_px; 0 = the 0.5.x behaviour
    // SGM's smoothness penalties and the focus-curve confidence ramp (DepthParams); 0 = the DepthParams default.
    // sgm_scale: multiply P1 and P2 by (depth_fraction / 0.25), so the regularisation per micron does not halve
    // when the volume's grid is twice as fine (FINDINGS 33)
    double sgm_p1 = 0, sgm_p2 = 0, conf_lo = 0, conf_hi = 0;
    bool sgm_scale = true;
    // noise-adaptive confidence (FINDINGS 38): the ramp is in effect an SNR test against the focus measure's noise
    // floor, calibrated on denoised frames whose floor (the median over the depth grid) was conf_noise_ref. On a
    // noisier stack with a floor N the ramp is scaled by (conf_noise_ref / N)^conf_noise_gamma (never above 1, at
    // least 0.5, and exactly 1 within 5 % of the reference), so raw frames keep their weak detail instead of
    // handing it to the priors and the fill. conf_noise_gamma 0 = off (the ramp as given, whatever the noise)
    double conf_noise_ref = 0.028, conf_noise_gamma = 1.0;
    // the SGM argmin returns an end frame as itself (0.6.10, FINDINGS 37a); false = the end frame's neighbour, 1 or
    // K-2, as 0.5.x and the prototype (--b-reference sets it)
    bool sgm_end_frames = true;
    // flat regions, depth stage (FINDINGS 36): the band's width decoupled from the blur width (0 = the blur width,
    // as 0.6.4), and the share of its own SGM depth a weak feature keeps, by its focus-curve contrast (0 = none)
    double flat_band_px = 0, flat_keep = 0;
    // what the background beyond the band takes (DepthParams::flat_end, FINDINGS 40): 2 = the smoothed frame where
    // each pixel's focus measure peaks (default since 0.6.16), 0 = the quiet end (the 0.6.4 .. 0.6.15 depth stage; the
    // 0.6.15 B also needs --flat-blend 0), 1 = the loud end, 3 = the field snapped to the ends; flat_end_soft = half
    // the transition of 1 and 3 (the frame slides over -soft .. +soft)
    int flat_end = 2;
    double flat_end_soft = 0.3;
    // the composition's smooth-depth blend widened to this many frames in flat regions (FINDINGS 41): the background's
    // depth ramps through the stack there and a two-frame blend draws iso-frame rings; 0 = off (two frames everywhere,
    // the 0.6.17 composition). 16 from 0.6.18 (8 already removes the rings, 16 and 32 look alike);
    // 32 since 0.6.22, with flat_blur_px 256 (FINDINGS 44)
    double flat_blend = 32;
    // surroundings nearest to focus at an end of the stack take that end frame (DepthParams::claim): a flat pixel
    // whose own focus peak lies within a sixteenth of the stack of an end and stands end_claim_lo .. end_claim_hi
    // times above its mean over the stack (0 .. 1). A featureless background has no such peak (a ratio near 1) and
    // is left as it was. false (the default: not judged yet) = the background as before (the smoothed field and the
    // band's fill everywhere)
    bool end_claim = false;
    double end_claim_lo = 2.0, end_claim_hi = 4.0;
    // strips, for stacks whose depth volume does not fit the GPUs: the output rectangle is cut along its longer side
    // into overlapping strips, each rendered on its own at the full depth resolution (one pass 1 + fusion per strip,
    // with the whole frame's noise estimate and pyramid depth), and the outputs are feathered together across the
    // middle third of each overlap. 0 = only when the memory runs short (before the fall back to the 1/4 volume),
    // 1 = never, N >= 2 = always N strips. strip_overlap_px: the overlap of neighbouring strips in sensor px
    int strips = 0;
    double strip_overlap_px = 512;
    std::vector<std::string> outputs;  // which images to write / keep; empty = all
    bool cache_frames = true;          // keep the frames pinned in host memory between the passes (--cache 0: read every pass from disk)
    bool write_outputs = true;         // false: compute everything (and fill `result`) but write no TIFFs (the GUI's benchmark)
    bool diagnostics = true;           // the raw maps beside the TIFFs (depth.f32, depth_smooth.f32, conf.f32, flat.f32); the GUI turns them off
    Hooks hooks;
    // results, filled when `result` is set
    struct Result {
        int w = 0, h = 0;
        std::vector<std::string> names;
        std::vector<std::vector<float>> images;   // float RGB, output grid, only the requested outputs
        std::vector<float> depth, conf;
        std::vector<float> noise_mad;
        // what else a render measures over its whole rectangle, and a region of it rendered again takes as given
        // (run_region): the focus measure's noise floor behind the confidence ramp, the flat mask's two texture
        // thresholds (-1 = none: flat regions off, or not known) and the pyramid depth
        float depth_floor = 0.f, flat_lo = -1.f, flat_hi = -1.f;
        int levels = 0;
        std::vector<double> frame_energy;         // per frame (stack order): pass 1's focus energy (Pass1::frame_energy)
        std::vector<uint8_t> icc;                 // the frames' ICC profile, embedded in the outputs
        double seconds = 0;
        // where the time went (s): pass 1 (stream + merge + volume), depth, refinement, composition, finalize, writing
        double pass1_seconds = 0, depth_seconds = 0, refine_seconds = 0, compose_seconds = 0, finalize_seconds = 0, write_seconds = 0;
        double frames_per_second = 0;             // pass 1's streaming rate
    };
    Result* result = nullptr;
    std::string note;                  // free text recorded in stack.txt as "options ...": the sweep's variant options
};
bool parse_stack_option(const std::string& a, const std::string& v, StackOptions& so);
int run_stack(StackOptions& so);

// ---- the sweep: one alignment, one pass 1 + volume per group of pass-1 settings, then the depth map, refinement,
// composition and B for every variant (about 6 s each against 12 s for a whole `stack`). Variants are the base
// StackOptions with their options applied through parse_stack_option; each writes to out_dir/<name>/ and the
// sweep writes out_dir/sweep.json (name, dir, options, pass-1 group, rc, seconds) after every variant.
struct SweepVariant {
    std::string name;
    std::vector<std::pair<std::string, std::string>> options;   // ("--sgm-p1", "0.06"), ...
};
int run_sweep(const StackOptions& base, const std::vector<SweepVariant>& variants, const std::string& out_dir);

// ---- regions of a finished render, rendered again with other settings (the two-up viewer's Refine). A tile is a
// rectangle of the render's own; it is rendered as a strip is: with the whole frame's noise, flat thresholds and
// pyramid depth (`whole`: the render's Result, of which only those values are read), so that with unchanged settings
// it gives the render's own pixels away from its edges. One pass 1 + volume per tile and group of pass-1 settings,
// then the tile's output per variant; nothing is written. `base` is the render's options as it ran them (the frames,
// their transforms and row corrections, the dust map, the rectangle in po.g).
struct RegionTile {
    int x0 = 0, y0 = 0, x1 = 0, y1 = 0;   // sensor px, inclusive; run_region replaces them with the rectangle it rendered
    std::string output = "B";             // the output rendered (any of the render's but A, whose sums a tile does not keep)
    std::vector<SweepVariant> variants;   // a variant without options = the render's own settings
    struct Out {
        std::string name;
        int rc = 0;                       // 0, or why this variant has no image (the codes of run_stack)
        std::vector<float> b;             // `output`, float RGB, w x h
        std::vector<float> depth;         // the frame per pixel, w x h
    };
    int ox = 0, oy = 0, w = 0, h = 0;     // where the rendered rectangle lies in the render's output grid, and its size
    std::vector<Out> out;                 // one per variant, in their order
};
// The rectangle a tile is rendered as: the given one grown by margin_px (sensor px) each way, its start moved down to
// the grid of the render's coarsest pyramid level, its sides multiples of 4, all within the render's rectangle `g`
// (the rule the strips follow, FINDINGS 53/55). False when nothing of it lies inside.
bool region_rect(const Grid& g, int levels, double margin_px, int& x0, int& y0, int& x1, int& y1);
int run_region(const StackOptions& base, const StackOptions::Result& whole, std::vector<RegionTile>& tiles, double margin_px = 512);

// One frame of a finished render as the render saw it: read, its dust taken out, and warped onto the render's output
// grid with its transform and row corrections (`base`: the render's options, as for run_region; k: the frame's place
// in base.po.paths). rgb: float RGB, w x h = the outputs' size. For painting a single frame into an output (the
// two-up viewer's Retouch): a frame as it is on disk is another size and lies a little beside the output.
// Returns 0, or a code of run_stack's (1 the file, 4 the card's memory).
int warp_frame(const StackOptions& base, int k, std::vector<float>& rgb, int& w, int& h);

// How sharp every frame of a finished render is at one place: energy[k] = the mean squared second difference of
// frame k's green samples over a window of `radius` px around output pixel (x, y), taken at the frame's own place for
// it (its transform; the row corrections, a few px, are left out). Read from the frames resident in the render's
// cache, so it takes milliseconds; false when the render has no cache or a frame is not resident in it. For finding
// the frame to retouch from: the depth map says which frame the render chose, this says which one is sharp.
bool focus_at(const StackOptions& base, int x, int y, int radius, std::vector<double>& energy);

// ---- alignment

struct AlignOptions {
    std::filesystem::path dir;
    std::vector<std::filesystem::path> paths;   // the frames; empty = every TIFF in dir starting with prefix
    std::string out, prefix, ref_rule = "middle", pairs_sel;
    double align_scale = 0.25;
    std::vector<int> baselines = {1, 2, 4, 8}, devs;
    int readers = 16;
    int pair_threads = 3;             // pairs in flight per GPU (each on its own stream); the per-pair arithmetic is unchanged
    int max_frames = 0;               // > 0: more frames than this end the job at once (run_job sets kMaxFrames)
    bool lock_scale = false;
    double um_per_px = 0;             // 0 = unknown; negative = -FOV in mm, resolved once the width is known
    FrameCache* cache = nullptr;      // resident frames (pass 0 fills it); null = stream from disk
    std::shared_ptr<const DustFix> dust;   // the dust map's correction: the alignment reads the frames without their dust too
    // rolling shutter (FINDINGS 51): the row bands per frame whose shift is solved against the neighbours (0 = off),
    // and the row-frequency range (cycles per frame height) taken as LED flicker in the brightness profiles
    int rs_bands = 8;
    int band_lo = 3, band_hi = 12;
    Hooks hooks;
};
bool parse_align_option(const std::string& a, const std::string& v, AlignOptions& ao);

struct AlignOut {
    std::vector<std::filesystem::path> paths;
    std::vector<Similarity> T;
    int W = 0, H = 0, ref = 0, rect[4] = {0, 0, 0, 0};
    GlobalSolve G;
    double smin = 1, smax = 1, seconds = 0;
    std::vector<PairResult> pairs;    // only when pairs_sel was given
    // rolling shutter (FINDINGS 51). Warping: rs_dx/rs_dy, N x rs_bands, each frame's extra displacement (full-res px) at
    // the centre of each row band, the stack's mean removed; rs_px per frame = the larger of the two peak-to-peak
    // spreads over the bands (0 = flat). Banding: the LED flicker in each frame's row-mean luma, as a share of the
    // stack's median row profile, in the band_lo..band_hi cycles-per-frame range: band_pct = its peak-to-peak in
    // percent; row_gain, N x H, the gain that flattens it (1 = none). band_peak_cycles / band_peak_power: the
    // strongest row frequency across the stack (power of the mean spectrum). Empty when not measured.
    int rs_bands = 0;
    std::vector<float> rs_dx, rs_dy, rs_px;
    std::vector<float> band_pct, row_gain;
    int band_peak_cycles = 0;
    std::shared_ptr<const DustFix> dust;   // the dust map with its depths measured on the frames (null without a dust map)
    double band_peak_power = 0, band_median_pct = 0, rs_median_px = 0;
};
int run_align(AlignOptions& ao, AlignOut& A);
void write_report(const AlignOptions& ao, const AlignOut& A, const std::filesystem::path& od);
std::string report_json(const AlignOptions& ao, const AlignOut& A);

// ---- a job: folder in, report + outputs out

struct JobOptions {
    AlignOptions align;
    StackOptions stack;
    std::string out;                  // empty = <dir>_stacker
    std::shared_ptr<const DustFix> dust;   // the dust map's correction for the whole job (a map of another size than the frames is left out, and said)
    FrameCache* cache = nullptr;      // a caller-owned frame cache that outlives the job (the GUI keeps the stack resident
                                      // across runs); null = one for this job when stack.cache_frames is set
};
int run_job(JobOptions& jo, AlignOut& A, Hooks hooks);
// the GUI's sweep: align the frames as run_job does (report.json + list.txt into out_dir), then run_sweep over them
int run_sweep_job(JobOptions& jo, const std::vector<SweepVariant>& variants, const std::string& out_dir, AlignOut& A, Hooks hooks);

}  // namespace stackcore
