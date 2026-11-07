// The pipeline as a library: alignment, pass 1 (+ depth volume), the fusion
// and the folder-level job. The CLI and the GUI both call this. Progress and
// preview callbacks let a GUI show the stack building (DESIGN.md: progressive
// preview); a cancel flag stops a job between frames.
#pragma once

#include <atomic>
#include <complex>
#include <filesystem>
#include <functional>
#include <string>
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
using PreviewFn = std::function<void(const std::string& name, const std::vector<float>& rgb, int w, int h)>;
using FrameViewFn = std::function<void(int k, const std::vector<float>& rgb, int w, int h)>;

struct Hooks {
    ProgressFn progress;
    PreviewFn preview;
    FrameViewFn frame;
    std::atomic<bool>* cancel = nullptr;
    int preview_down = 8;        // preview images are 1/preview_down of the output grid
    int preview_every = 8;       // frames between pass-1 previews
    int frame_down = 16;         // per-frame views are 1/frame_down of the output grid
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
    double depth_fraction = 0.25, depth_fm_px = 6.0, vol_agg_px = 32.0;
    Hooks hooks;
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
    std::vector<uint16_t> fm_all, lum_all;
    std::vector<float> V_all;
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
    double halo_prior = 2.0, dark_prior = 1.0;   // B's depth priors (DepthParams); the validated defaults
    std::vector<std::string> outputs;  // which images to write / keep; empty = all
    bool cache_frames = true;          // keep the frames pinned in host memory between the passes (--cache 0: read every pass from disk)
    bool write_outputs = true;         // false: compute everything (and fill `result`) but write no TIFFs (the GUI's benchmark)
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
};
bool parse_stack_option(const std::string& a, const std::string& v, StackOptions& so);
int run_stack(StackOptions& so);

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

}  // namespace stackcore
