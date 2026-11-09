// The pipeline as a library: alignment, pass 1 (+ depth volume), the fusion
// and the folder-level job. The CLI and the GUI both call this. Progress and
// preview callbacks let a GUI show the stack building (DESIGN.md: progressive
// preview); a cancel flag stops a job between frames.
#pragma once

#include <atomic>
#include <complex>
#include <cstring>
#include <filesystem>
#include <functional>
#include <new>
#include <string>
#include <thread>
#include <vector>

#include "stackcore/align.h"
#include "stackcore/compose.h"
#include "stackcore/fold.h"
#include "stackcore/streamer.h"
#include "stackcore/volume.h"
#include "stackcore/warp.h"

namespace stackcore {

// Log line sink (default: stdout). Installed once per process.
using LogFn = std::function<void(const std::string&)>;
void set_log(LogFn fn);
void logf(const char* fmt, ...);

// progress(stage, fraction 0..1); preview(name, rgb float interleaved, w, h);
// frame(k, rgb, w, h): the frame just processed (warped onto the output grid, small)
using ProgressFn = std::function<void(const std::string& stage, double fraction)>;
// rgb points at w x h x 3 floats that are valid only during the call: copy what is needed. (The hooks
// run on the GPU worker threads; a fresh 10 MB vector per call cost ~3 ms of page faults each, FINDINGS 30.)
using PreviewFn = std::function<void(const std::string& name, const float* rgb, int w, int h)>;
// The window of the frame a view shows (0.6.33, FINDINGS 47): a crop x0, y0, w, h in the frame's pixels, sampled every
// `down`-th pixel; fw, fh = the frame's size. A frame view carries the window it was taken with, so the front end can
// place it. w = 0 asks for the whole frame at frame_down.
struct ViewWindow { int x0 = 0, y0 = 0, w = 0, h = 0, down = 1, fw = 0, fh = 0; };
using FrameViewFn = std::function<void(int k, const float* rgb, int w, int h, const ViewWindow& win)>;

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
    // at most one frame view per slot and one building preview per device within this many ms: every copy
    // off a GPU costs its uploads ~3 ms of link time whatever its size (FINDINGS 30)
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
    bool host_state = false;                      // also download the merged state into Pass1::merged (the fold command)
    // the depth volume's resolution as a fraction of the sensor: 1/2 since 0.6.1 (FINDINGS 32: the doubled plate
    // edges of 1/4 are gone at 1:1); --depth-fraction 0.25 gives the 0.5.x reference B
    double depth_fraction = 0.5, depth_fm_px = 6.0, vol_agg_px = 32.0;
    // defocused background (FINDINGS 34): in flat regions of the guide (no texture within flat_reach_px) every
    // focus slice is box-blurred over flat_blur_px, and in the depth stage pixels without sharp detail within reach
    // take the nearest sharp detail's frame in a band about flat_blur_px wide, the quieter end of the stack beyond
    // it (DepthParams::flat, flat_blur_px). 0 = off (the 0.5.x behaviour: the priors decide). 128 from 0.6.4; 256 since
    // 0.6.22 (FINDINGS 44: a fifth less ring energy in the background, detail untouched; Travis preferred it at 1:1)
    double flat_blur_px = 256.0, flat_reach_px = 24.0;
    double flat_min = 0.0;                        // DepthParams::flat_min (least-structure data term in flat regions); 0 = off
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
    PinnedVec(PinnedVec&& o) noexcept : p_(o.p_), n_(o.n_) { o.p_ = nullptr; o.n_ = 0; }
    PinnedVec& operator=(PinnedVec&& o) noexcept {
        if (this != &o) { clear(); p_ = o.p_; n_ = o.n_; o.p_ = nullptr; o.n_ = 0; }
        return *this;
    }
    void resize(size_t n) {   // NOT zeroed (a memset of 1.8 GB costs 0.15 s); the caller fills every element
        clear();
        if (!n) return;
        if (cudaHostAlloc((void**)&p_, n * sizeof(T), cudaHostAllocPortable) != cudaSuccess) throw std::bad_alloc();
        n_ = n;
    }
    void clear() { if (p_) cudaFreeHost(p_); p_ = nullptr; n_ = 0; }
    // unpinning 1.8 GB takes ~0.2 s: hand the buffer to a thread that frees it off the critical path
    void release_async() {
        if (!p_) return;
        T* p = p_;
        p_ = nullptr;
        n_ = 0;
        std::thread([p] { cudaFreeHost(p); }).detach();
    }
    T* data() { return p_; }
    const T* data() const { return p_; }
    size_t size() const { return n_; }
    bool empty() const { return n_ == 0; }

private:
    T* p_ = nullptr;
    size_t n_ = 0;
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
    PinnedVec<uint16_t> fm_all, lum_all;   // fm_all only when host_state (the fold command writes it; the fusion never reads it)
    PinnedVec<float> V_all;
    int slices = 0;
    StreamerStats st;
    double vol_seconds = 0;
};

bool read_list(const std::filesystem::path& list, std::vector<std::filesystem::path>& paths, std::vector<Similarity>& T);
int run_pass1(const Pass1Options& po, Pass1& P);

// ---- the fusion (pass 1 + depth + refine + compose + finalize)

struct StackOptions {
    Pass1Options po;
    std::string out;                   // output folder; empty = write nothing
    int streams = 2, refine_r = 12;
    double refine_fm_px = 9.0, soft_px = 3.0;
    double c_smoothing = 0.0;          // C's smoothing (FinalParams::c_smoothing); 0 = the raw selection
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
    // the 0.6.17 composition). 16 from 0.6.18 (8 already removes the rings, 16 and 32 look alike and nearer Helicon);
    // 32 since 0.6.22, with flat_blur_px 256 (FINDINGS 44)
    double flat_blend = 32;
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

// ---- alignment

struct AlignOptions {
    std::filesystem::path dir;
    std::vector<std::filesystem::path> paths;   // the frames; empty = every TIFF in dir starting with prefix
    std::string out, prefix, ref_rule = "middle", pairs_sel;
    double align_scale = 0.25;
    std::vector<int> baselines = {1, 2, 4, 8}, devs;
    int readers = 16;
    int pair_threads = 3;             // pairs in flight per GPU (each on its own stream); the per-pair arithmetic is unchanged
    bool lock_scale = false;
    double um_per_px = 0;             // 0 = unknown; negative = -FOV in mm, resolved once the width is known
    FrameCache* cache = nullptr;      // resident frames (pass 0 fills it); null = stream from disk
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
};
int run_align(AlignOptions& ao, AlignOut& A);
void write_report(const AlignOptions& ao, const AlignOut& A, const std::filesystem::path& od);
std::string report_json(const AlignOptions& ao, const AlignOut& A);

// ---- a job: folder in, report + outputs out

struct JobOptions {
    AlignOptions align;
    StackOptions stack;
    std::string out;                  // empty = <dir>_stacker
    FrameCache* cache = nullptr;      // a caller-owned frame cache that outlives the job (the GUI keeps the stack resident
                                      // across runs); null = one for this job when stack.cache_frames is set
};
int run_job(JobOptions& jo, AlignOut& A, Hooks hooks);
// the GUI's sweep: align the frames as run_job does (report.json + list.txt into out_dir), then run_sweep over them
int run_sweep_job(JobOptions& jo, const std::vector<SweepVariant>& variants, const std::string& out_dir, AlignOut& A, Hooks hooks);

}  // namespace stackcore
