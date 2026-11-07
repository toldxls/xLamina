// The pipeline as a library (see pipeline.h). Grown out of stackcli's commands.
#include "stackcore/pipeline.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>

#include "stackcore/convert.h"
#include "stackcore/depth.h"
#include "stackcore/tiff.h"

namespace stackcore {

namespace fs = std::filesystem;

#define CK(x)                                                                                   \
    do {                                                                                        \
        cudaError_t e_ = (x);                                                                   \
        if (e_ != cudaSuccess) {                                                                \
            logf("CUDA error: %s (%s:%d)", cudaGetErrorString(e_), __FILE__, __LINE__);         \
            return 1;                                                                           \
        }                                                                                       \
    } while (0)
#define CK_THROW(x)                                                                             \
    do {                                                                                        \
        cudaError_t e_ = (x);                                                                   \
        if (e_ != cudaSuccess) throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(e_)); \
    } while (0)

static LogFn g_log;
void set_log(LogFn fn) { g_log = std::move(fn); }
void logf(const char* fmt, ...) {
    char buf[4096];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    std::string s(buf);
    while (!s.empty() && s.back() == '\n') s.pop_back();
    if (g_log) g_log(s);
    else { fputs(s.c_str(), stdout); fputc('\n', stdout); fflush(stdout); }
}

static std::vector<int> parse_list(const std::string& s) {
    std::vector<int> v;
    size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find(',', i);
        if (j == std::string::npos) j = s.size();
        if (j > i) v.push_back(atoi(s.substr(i, j - i).c_str()));
        i = j + 1;
    }
    return v;
}

// where a pass's frames came from, for its log line
static std::string source_note(const StreamerStats& st) {
    if (st.from_cache == 0 && st.to_cache == 0) return " read";
    if (st.from_cache == st.frames) return " from memory";
    if (st.to_cache == st.frames) return " read, now resident";
    return " read, " + std::to_string(st.from_cache) + " from memory";
}

// read FILE into a device float RGB frame; returns 0 and fills info/d_rgb on success
static int load_frame_to_device(const fs::path& in, TiffInfo& info, float*& d_rgb, FrameCache* cache = nullptr) {
    info = tiff_probe(in);
    if (!info.ok()) { logf("%s: %s", in.string().c_str(), info.error.c_str()); return 1; }
    size_t cap = tiff_buffer_capacity(info.pixel_bytes());
    uint8_t* h = nullptr;
    const uint8_t* src = nullptr;
    size_t bytes = 0;
    if (const FrameCache::Entry* e = cache ? cache->find(in) : nullptr) { src = e->buf + e->offset; bytes = e->bytes; }
    else {
        CK(cudaHostAlloc((void**)&h, cap, cudaHostAllocPortable));
        TiffRead rd = tiff_read_pixels(in, info, h, cap);
        if (!rd.ok()) { logf("read failed: %s", rd.error.c_str()); return 1; }
        src = h + rd.data_offset_in_buf;
        bytes = rd.bytes;
    }
    uint8_t* d_src;
    size_t npx = (size_t)info.width * info.height;
    CK(cudaMalloc(&d_src, bytes));
    CK(cudaMalloc(&d_rgb, npx * 3 * sizeof(float)));
    CK(cudaMemcpy(d_src, src, bytes, cudaMemcpyHostToDevice));
    convert_to_rgb_f32(d_src, info, d_rgb, nullptr, 0);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    CK(cudaFree(d_src));
    if (h) CK(cudaFreeHost(h));
    return 0;
}
// ----------------------------------------------------------------------------- fold (pass 1)

struct FoldSlot {
    float *d_rgb = nullptr, *d_scratch = nullptr, *d_warp = nullptr;
    FoldScratch sc;
    VolumeScratch vsc;
    cudaEvent_t ready = nullptr, done = nullptr;
    bool used = false;
};

struct FoldDev {
    int dev = -1;
    cudaStream_t fold_s = nullptr;
    FoldState st;
    std::vector<FoldSlot> slots;
    std::vector<void*> fm16, lum16;   // per frame index: this GPU's volume slices (null = not here)
};

template <class T>
static bool write_raw(const fs::path& p, const T* data, size_t n) {
    FILE* f = fopen(p.string().c_str(), "wb");
    if (!f) return false;
    size_t m = fwrite(data, sizeof(T), n, f);
    fclose(f);
    return m == n;
}

bool read_list(const fs::path& list, std::vector<fs::path>& paths, std::vector<Similarity>& T) {
    // path zr zi tx ty per line (paths may contain spaces: the last four fields are numbers)
    FILE* f = fopen(list.string().c_str(), "r");
    if (!f) { logf("cannot read %s\n", list.string().c_str()); return false; }
    char line[4096];
    while (fgets(line, sizeof line, f)) {
        std::string s(line);
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) s.pop_back();
        if (s.empty() || s[0] == '#') continue;
        double v[4];
        size_t end = s.size();
        for (int j = 3; j >= 0; --j) {
            size_t sp = s.rfind(' ', end - 1);
            if (sp == std::string::npos) { logf("bad list line: %s\n", s.c_str()); fclose(f); return false; }
            v[j] = atof(s.substr(sp + 1, end - sp - 1).c_str());
            end = sp;
        }
        paths.push_back(s.substr(0, end));
        T.push_back({v[0], v[1], v[2], v[3]});
    }
    fclose(f);
    return !paths.empty();
}


int run_pass1(const Pass1Options& po, Pass1& P) {
    Grid g = po.g;
    StreamerOptions opt = po.opt;
    const bool volume = po.volume;
    const Hooks& hooks = po.hooks;
    std::atomic<int> frames_done{0};
    if (g.scale != 1.0 && g.scale != 0.5) { logf("--scale must be 1 or 0.5\n"); return 2; }
    std::vector<fs::path>& paths = P.paths;
    std::vector<Similarity>& T = P.T;
    if (!po.paths.empty()) { paths = po.paths; T = po.T; }
    else if (!read_list(po.list, paths, T)) return 1;
    const std::vector<int>& frames_sel = po.frames_sel;
    {
    }
    // frame k keeps its index from the list even when a subset is run (the index is what the state records)
    std::vector<int> ks;
    if (frames_sel.empty()) for (int k = 0; k < (int)paths.size(); ++k) ks.push_back(k);
    else ks = frames_sel;
    std::vector<fs::path> run_paths;
    for (int k : ks) run_paths.push_back(paths[k]);

    TiffInfo t0 = tiff_probe(paths[ks[0]]);
    if (!t0.ok()) { logf("%s: %s\n", paths[ks[0]].string().c_str(), t0.error.c_str()); return 1; }
    g.W = (int)t0.width;
    g.H = (int)t0.height;
    int ow = g.out_w(), oh = g.out_h();
    FoldParams fp;
    fp.levels = n_levels(ow, oh);
    fp.radius = (float)std::max(0.8, po.radius_px * g.scale);
    fp.power = (float)po.power;
    fp.c_radius = po.c_radius_px > 0 ? (float)(po.c_radius_px * g.scale) : 0.f;
    fp.with_a = po.with_a;
    VolumeParams vp;
    {
        double ds = po.depth_fraction / g.scale;   // depth res = fixed fraction of the sensor
        vp.dw = std::max(16, (int)std::nearbyint(ow * ds));
        vp.dh = std::max(16, (int)std::nearbyint(oh * ds));
        vp.fm_radius = (float)std::max(0.8, po.depth_fm_px * po.depth_fraction);
        vp.agg_r = (int)std::nearbyint(po.vol_agg_px * po.depth_fraction);
        if (po.vol_agg_px * po.depth_fraction < 1.0) vp.agg_r = 0;
    }
    const int K = (int)paths.size();
    P.g = g; P.fp = fp; P.vp = vp;

    int ndev = 0;
    CK(cudaGetDeviceCount(&ndev));
    std::vector<int> devs = opt.devices;
    if (devs.empty()) for (int i = 0; i < ndev; ++i) devs.push_back(i);
    P.devs = devs;
    std::vector<FoldDev> ctx(ndev);
    size_t npx_full = (size_t)g.W * g.H, npx_out = (size_t)ow * oh;
    opt.streams_by_device.clear();
    auto t_setup = std::chrono::steady_clock::now();
    for (int d : devs) {
        auto& D = ctx[d];
        D.dev = d;
        CK(cudaSetDevice(d));
        CK(cudaStreamCreateWithFlags(&D.fold_s, cudaStreamNonBlocking));
        D.st.alloc(d, ow, oh, fp.levels, fp.with_a);
        D.fm16.assign(K, nullptr);
        D.lum16.assign(K, nullptr);
        // slots per GPU from the memory left after the state: each needs the frame buffers,
        // the measure scratch and the streamer's raw upload buffer, plus a working margin
        FoldScratch probe;
        probe.w = ow; probe.h = oh; probe.levels = fp.levels; probe.shapes = pyramid_shapes(ow, oh, fp.levels);
        size_t slot_bytes = npx_full * 12 + (g.scale != 1.0 ? (size_t)g.work_w() * g.work_h() * 12 : 0) + npx_out * 12 + probe.bytes();
        size_t per_slot = slot_bytes + t0.pixel_bytes() + 2 * kReadAlign;
        size_t fr, tot;
        CK(cudaMemGetInfo(&fr, &tot));
        // the depth-volume slices stay on the GPU until the aggregation: budget for every frame landing here
        size_t margin = (512u << 20) + (volume ? (size_t)K * vp.dw * vp.dh * 4 : 0);
        int fit = fr > margin ? (int)((fr - margin) / per_slot) : 0;
        int S = std::max(1, std::min(opt.streams_per_gpu, fit));
        if (fit < 1) logf("GPU %d: not enough memory for the state plus one slot (%.2f GB free)\n", d, fr / 1e9);
        opt.streams_by_device.push_back(S);
        D.slots.resize(S);
        for (auto& S : D.slots) {
            CK(cudaMalloc(&S.d_rgb, npx_full * 3 * sizeof(float)));
            if (g.scale != 1.0) CK(cudaMalloc(&S.d_scratch, (size_t)g.work_w() * g.work_h() * 3 * sizeof(float)));
            CK(cudaMalloc(&S.d_warp, npx_out * 3 * sizeof(float)));
            S.sc.alloc(ow, oh, fp.levels, fp.radius, fp.c_radius);
            if (volume) S.vsc.alloc(ow, oh, vp);
            CK(cudaEventCreateWithFlags(&S.ready, cudaEventDisableTiming));
            CK(cudaEventCreateWithFlags(&S.done, cudaEventDisableTiming));
        }
        CK(cudaMemGetInfo(&fr, &tot));
        logf("GPU %d: state %.2f GB + %d slot(s) x %.2f GB (%d requested, %d fit); %.2f of %.2f GB free after allocation\n", d,
               D.st.bytes() / 1e9, S, per_slot / 1e9, opt.streams_per_gpu, fit, fr / 1e9, tot / 1e9);
    }

    auto on_frame = [&](const DeviceFrame& f) {
        auto& D = ctx[f.dev];
        auto& S = D.slots[f.slot];
        if (S.used) cudaStreamWaitEvent(f.stream, S.done, 0);   // the fold stream must be done with this slot
        convert_to_rgb_f32(f.d_pixels(), *f.info, S.d_rgb, nullptr, f.stream);
        int k = ks[f.k];
        load_aligned(S.d_rgb, g, T[k], S.d_scratch, S.d_warp, false, f.stream);
        fold_measure(S.d_warp, fp, S.sc, f.stream);
        if (volume) {   // this frame's depth-resolution slices stay on this GPU until the aggregation
            size_t nb = (size_t)vp.dw * vp.dh * 2;
            CK_THROW(cudaMallocAsync(&D.fm16[k], nb, f.stream));   // a failure here must stop the pass, not fault a kernel
            CK_THROW(cudaMallocAsync(&D.lum16[k], nb, f.stream));
            volume_slice(S.d_warp, vp, S.vsc, D.fm16[k], D.lum16[k], f.stream);
        }
        cudaEventRecord(S.ready, f.stream);
        cudaStreamWaitEvent(D.fold_s, S.ready, 0);
        fold_accumulate(D.st, k, S.d_warp, fp, S.sc, D.fold_s);
        cudaEventRecord(S.done, D.fold_s);
        S.used = true;
        int done = ++frames_done;
        if (hooks.progress) hooks.progress("pass 1: fold", (double)done / (double)run_paths.size());
        if (hooks.frame) {   // the frame just aligned, small, for the GUI's frame view
            std::vector<float> img;
            int pw, ph;
            image_preview(S.d_warp, ow, oh, hooks.frame_down, img, pw, ph, f.stream);
            hooks.frame(k, img, pw, ph);
        }
        // live preview: this GPU's best focus measure so far at a fraction of the output grid
        if (hooks.preview && f.dev == devs[0] && (done % hooks.preview_every == 0)) {
            std::vector<float> img;
            int pw, ph;
            fold_focus_preview(D.st, hooks.preview_down, img, pw, ph, D.fold_s);
            hooks.preview("focus (building)", img, pw, ph);   // the detail map so far
        }
    };

    logf("fold: %zu frames, grid %dx%d -> %dx%d at scale %g, %d levels, blur sigma %.2f px, %zu GPU(s)\n",
           run_paths.size(), g.W, g.H, ow, oh, g.scale, fp.levels, fp.radius, devs.size());
    P.setup_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_setup).count();
    opt.cancel = hooks.cancel;
    StreamerStats st = stream_frames(run_paths, opt, on_frame);
    for (auto& e : st.errors) logf("  error: %s\n", e.c_str());
    for (int d : devs) {
        CK(cudaSetDevice(d));
        CK(cudaStreamSynchronize(ctx[d].fold_s));
        CK(cudaGetLastError());
    }
    logf("%d frames in %.3f s: %.1f frames/s, %.2f GB/s%s\n", st.frames, st.seconds, st.frames / st.seconds,
           st.bytes / st.seconds / 1e9, source_note(st).c_str());
    for (size_t i = 0; i < devs.size(); ++i) printf("  GPU %d: %d frames\n", devs[i], st.frames_per_device[i]);

    // merge on the first device, in device order (any order gives the same B / C; A to double rounding):
    // the other cards' states cross through pinned memory, and the merged state stays on the GPU for
    // the guide and the finalize (FINDINGS 29); the host copy is only made for the fold command
    {
        auto t_m = std::chrono::steady_clock::now();
        int prim = devs[0];
        for (size_t i = 1; i < devs.size(); ++i) merge_device(ctx[prim].st, ctx[devs[i]].st);
        P.dst = ctx[prim].st;
        ctx[prim].st = FoldState{};   // moved out: P.dst outlives this pass
        if (po.host_state) P.merged = download(P.dst);
        P.merge_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_m).count();
    }
    P.st = st;

    if (volume) {
        auto t0 = std::chrono::steady_clock::now();
        // guide: luma of the collapsed merged C, clipped to [0, 1], on the first device; then at the depth res
        int d0 = devs[0];
        CK(cudaSetDevice(d0));
        cudaStream_t s0 = ctx[d0].fold_s;
        std::vector<const float*> d_coef(P.dst.c_coef.begin(), P.dst.c_coef.end());   // the merged state's levels, in place
        float *d_base, *d_coll, *d_t1, *d_t2, *d_guide, *d_ga;
        CK(cudaMalloc(&d_base, (size_t)P.dst.shapes.back().w * P.dst.shapes.back().h * 3 * sizeof(float)));
        fold_base_f32(P.dst, d_base, s0);
        CK(cudaMalloc(&d_coll, npx_out * 3 * sizeof(float)));
        CK(cudaMalloc(&d_t1, npx_out * 3 * sizeof(float)));
        CK(cudaMalloc(&d_t2, npx_out * 3 * sizeof(float)));
        CK(cudaMalloc(&d_guide, npx_out * sizeof(float)));
        size_t nd = (size_t)vp.dw * vp.dh;
        CK(cudaMalloc(&d_ga, nd * sizeof(float)));
        collapse_pyramid(d_coef, d_base, P.dst.shapes, d_coll, d_t1, d_t2, s0);
        luma(d_coll, d_guide, npx_out, s0);
        clip01(d_guide, npx_out, s0);
        AreaResizer rz;
        rz.build(ow, oh, vp.dw, vp.dh);
        rz.run(d_guide, d_ga, 1, s0);
        std::vector<float> guide_h(npx_out), ga_h(nd);
        CK(cudaMemcpyAsync(guide_h.data(), d_guide, npx_out * sizeof(float), cudaMemcpyDeviceToHost, s0));
        CK(cudaMemcpyAsync(ga_h.data(), d_ga, nd * sizeof(float), cudaMemcpyDeviceToHost, s0));
        CK(cudaStreamSynchronize(s0));
        rz.free();
        cudaFree(d_base); cudaFree(d_coll); cudaFree(d_t1); cudaFree(d_t2); cudaFree(d_guide); cudaFree(d_ga);

        // every GPU aggregates its own slices against the shared guide; the host collects plane-major volumes
        std::vector<uint16_t>& fm_all = P.fm_all;
        std::vector<uint16_t>& lum_all = P.lum_all;
        std::vector<float>& V_all = P.V_all;
        fm_all.assign(nd * K, 0); lum_all.assign(nd * K, 0); V_all.assign(nd * K, 0.f);
        int done = 0;
        for (int d : devs) {
            auto& D = ctx[d];
            CK(cudaSetDevice(d));
            cudaStream_t s = D.fold_s;
            float *d_g, *d_V;
            CK(cudaMalloc(&d_g, nd * sizeof(float)));
            CK(cudaMemcpyAsync(d_g, ga_h.data(), nd * sizeof(float), cudaMemcpyHostToDevice, s));
            CK(cudaMalloc(&d_V, nd * sizeof(float)));
            GuideStats gs;
            if (vp.agg_r > 0) gs.build(d_g, vp.dw, vp.dh, vp.agg_r, s);
            // B radius: an isotropic blur of each aggregated slice, so the frame of a strong edge also
            // claims the band beside it (the edge-aware aggregation stops at the edge, which is exactly
            // where the defocus rims of far frames win otherwise; FINDINGS 25)
            DevGaussian gb;
            float* d_T = nullptr;
            bool b_blur = po.b_radius_px > 0;
            if (b_blur) { gb.build(std::max(0.5, po.b_radius_px * po.depth_fraction)); CK(cudaMalloc(&d_T, nd * sizeof(float))); }
            for (int k = 0; k < K; ++k) {
                if (!D.fm16[k]) continue;
                if (vp.agg_r > 0) aggregate_slice(d_g, gs, D.fm16[k], d_V, vp.agg_eps, s);
                else { f16_to_f32(D.fm16[k], d_V, nd, s); relu_sqrt(d_V, nd, s); }
                if (b_blur) gaussian_blur(d_V, d_V, d_T, vp.dw, vp.dh, gb, s);
                CK(cudaMemcpyAsync(V_all.data() + nd * k, d_V, nd * sizeof(float), cudaMemcpyDeviceToHost, s));
                CK(cudaMemcpyAsync(fm_all.data() + nd * k, D.fm16[k], nd * 2, cudaMemcpyDeviceToHost, s));
                CK(cudaMemcpyAsync(lum_all.data() + nd * k, D.lum16[k], nd * 2, cudaMemcpyDeviceToHost, s));
                CK(cudaStreamSynchronize(s));
                cudaFreeAsync(D.fm16[k], s);
                cudaFreeAsync(D.lum16[k], s);
                D.fm16[k] = D.lum16[k] = nullptr;
                ++done;
            }
            CK(cudaStreamSynchronize(s));
            if (vp.agg_r > 0) gs.free();
            if (b_blur) { gb.free(); cudaFree(d_T); }
            cudaFree(d_g);
            cudaFree(d_V);
        }
        P.vol_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        P.guide = std::move(guide_h);
        P.ga = std::move(ga_h);
        P.slices = done;
    }

    for (int d : devs) {
        auto& D = ctx[d];
        CK(cudaSetDevice(d));
        for (auto& S : D.slots) {
            cudaFree(S.d_rgb); cudaFree(S.d_scratch); cudaFree(S.d_warp); S.sc.free();
            if (volume) S.vsc.free();
            cudaEventDestroy(S.ready); cudaEventDestroy(S.done);
        }
        D.st.free();
        cudaStreamDestroy(D.fold_s);
    }
    return st.errors.empty() ? 0 : 1;
}

// ----------------------------------------------------------------------------- stack: the whole pipeline

// A streaming pass after pass 1: per slot convert + warp, then a per-slot stage on the slot stream and a
// per-device accumulate stage on the device's fold stream (the same event hand-over as pass 1).
struct PassSlot {
    float *d_rgb = nullptr, *d_scratch = nullptr, *d_warp = nullptr;
    cudaEvent_t ready = nullptr, done = nullptr;
    bool used = false;
    int index = 0;   // slot index within its device
};
struct PassDev {
    int dev = -1;
    cudaStream_t fold_s = nullptr;
    std::vector<PassSlot> slots;
};

template <class SlotStage, class AccStage>
static StreamerStats run_pass(const std::vector<fs::path>& run_paths, const std::vector<int>& ks, const std::vector<Similarity>& T,
                              const Grid& g, const std::vector<int>& devs, int streams, int readers, FrameCache* cache,
                              std::vector<PassDev>& ctx, SlotStage slot_stage, AccStage acc_stage, const char* what, const Hooks& hooks) {
    int ndev = 0;
    CK_THROW(cudaGetDeviceCount(&ndev));
    ctx.assign(ndev, PassDev{});
    size_t npx_full = (size_t)g.W * g.H, npx_out = (size_t)g.out_w() * g.out_h();
    StreamerOptions opt;
    opt.devices = devs;
    opt.streams_per_gpu = streams;
    opt.readers = readers;
    opt.cancel = hooks.cancel;
    opt.cache = cache;
    std::atomic<int> done{0};
    for (int d : devs) {
        auto& D = ctx[d];
        D.dev = d;
        CK_THROW(cudaSetDevice(d));
        CK_THROW(cudaStreamCreateWithFlags(&D.fold_s, cudaStreamNonBlocking));
        D.slots.resize(streams);
        int idx = 0;
        for (auto& S : D.slots) {
            S.index = idx++;
            CK_THROW(cudaMalloc(&S.d_rgb, npx_full * 3 * sizeof(float)));
            if (g.scale != 1.0) CK_THROW(cudaMalloc(&S.d_scratch, (size_t)g.work_w() * g.work_h() * 3 * sizeof(float)));
            CK_THROW(cudaMalloc(&S.d_warp, npx_out * 3 * sizeof(float)));
            CK_THROW(cudaEventCreateWithFlags(&S.ready, cudaEventDisableTiming));
            CK_THROW(cudaEventCreateWithFlags(&S.done, cudaEventDisableTiming));
        }
    }
    auto on_frame = [&](const DeviceFrame& f) {
        auto& D = ctx[f.dev];
        auto& S = D.slots[f.slot];
        if (S.used) cudaStreamWaitEvent(f.stream, S.done, 0);
        convert_to_rgb_f32(f.d_pixels(), *f.info, S.d_rgb, nullptr, f.stream);
        int k = ks[f.k];
        load_aligned(S.d_rgb, g, T[k], S.d_scratch, S.d_warp, false, f.stream);
        slot_stage(f.dev, S, k, f.stream);
        cudaEventRecord(S.ready, f.stream);
        cudaStreamWaitEvent(D.fold_s, S.ready, 0);
        acc_stage(f.dev, S, k, D.fold_s);
        cudaEventRecord(S.done, D.fold_s);
        S.used = true;
        if (hooks.progress) hooks.progress(what, (double)(++done) / (double)run_paths.size());
        if (hooks.frame) {
            std::vector<float> img;
            int pw, ph;
            image_preview(S.d_warp, g.out_w(), g.out_h(), hooks.frame_down, img, pw, ph, f.stream);
            hooks.frame(k, img, pw, ph);
        }
    };
    StreamerStats st = stream_frames(run_paths, opt, on_frame);
    for (auto& e : st.errors) logf("  error: %s\n", e.c_str());
    for (int d : devs) {
        CK_THROW(cudaSetDevice(d));
        CK_THROW(cudaStreamSynchronize(ctx[d].fold_s));
        CK_THROW(cudaGetLastError());
    }
    std::string split;
    for (size_t i = 0; i < devs.size(); ++i) split += (i ? ", GPU " : "; GPU ") + std::to_string(devs[i]) + " " + std::to_string(st.frames_per_device[i]);
    logf("  %s: %d frames in %.2f s (%.1f frames/s%s)%s", what, st.frames, st.seconds, st.frames / st.seconds, source_note(st).c_str(),
         split.c_str());
    for (int d : devs) {
        auto& D = ctx[d];
        CK_THROW(cudaSetDevice(d));
        for (auto& S : D.slots) {
            cudaFree(S.d_rgb); cudaFree(S.d_scratch); cudaFree(S.d_warp);
            cudaEventDestroy(S.ready); cudaEventDestroy(S.done);
        }
        cudaStreamDestroy(D.fold_s);
    }
    return st;
}

// parses the options shared by `stack` and `run`; returns false on an unknown option
bool parse_stack_option(const std::string& a, const std::string& v, StackOptions& so) {
    Grid& g = so.po.g;
    if (a == "--scale") g.scale = atof(v.c_str());
    else if (a == "--gpus") so.po.opt.devices = parse_list(v);
    else if (a == "--streams") { so.streams = atoi(v.c_str()); so.po.opt.streams_per_gpu = so.streams; }
    else if (a == "--readers") so.po.opt.readers = atoi(v.c_str());
    else if (a == "--cache") so.cache_frames = atoi(v.c_str()) != 0;
    else if (a == "--refine-r") so.refine_r = atoi(v.c_str());
    else if (a == "--refine-fm-px") so.refine_fm_px = atof(v.c_str());
    else if (a == "--soft-px") so.soft_px = atof(v.c_str());
    else if (a == "--smoothing") so.c_smoothing = atof(v.c_str());
    else if (a == "--c-radius") so.po.c_radius_px = atof(v.c_str());
    else if (a == "--b-radius") so.po.b_radius_px = atof(v.c_str());
    else if (a == "--halo-prior") so.halo_prior = atof(v.c_str());
    else if (a == "--dark-prior") so.dark_prior = atof(v.c_str());
    else if (a == "--outputs") {
        so.outputs.clear();
        size_t i = 0;
        while (i < v.size()) { size_t j = v.find(',', i); if (j == std::string::npos) j = v.size(); if (j > i) so.outputs.push_back(v.substr(i, j - i)); i = j + 1; }
    } else return false;
    return true;
}


// a preview of a float RGB image at 1/down (block means), for the hooks
static void downsample_preview(const std::vector<float>& img, int w, int h, int down, std::vector<float>& out, int& pw, int& ph) {
    pw = std::max(1, w / down); ph = std::max(1, h / down);
    out.assign((size_t)pw * ph * 3, 0.f);
    for (int y = 0; y < ph; ++y)
        for (int x = 0; x < pw; ++x) {
            double acc[3] = {0, 0, 0};
            int n = 0;
            for (int yy = y * down; yy < std::min(h, (y + 1) * down); ++yy)
                for (int xx = x * down; xx < std::min(w, (x + 1) * down); ++xx, ++n)
                    for (int c = 0; c < 3; ++c) acc[c] += img[((size_t)yy * w + xx) * 3 + c];
            for (int c = 0; c < 3; ++c) out[((size_t)y * pw + x) * 3 + c] = n ? (float)(acc[c] / n) : 0.f;
        }
}

int run_stack(StackOptions& so) {
    Pass1Options& po = so.po;
    po.volume = true;
    po.hooks = so.hooks;
    const Hooks& hooks = so.hooks;
    const std::string& out = so.out;
    const int refine_r = so.refine_r, streams = so.streams;
    const double refine_fm_px = so.refine_fm_px, soft_px = so.soft_px;
    auto t_all = std::chrono::steady_clock::now();
    auto since = [](std::chrono::steady_clock::time_point t) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count(); };
    auto stage = [&](const char* name, double f) { if (hooks.progress) hooks.progress(name, f); };

    // the frames stay pinned in host memory from the first pass that reads them (FINDINGS 26); a job
    // without a caller-owned cache gets one of its own, released with the options' pointer at the end
    std::unique_ptr<FrameCache> own_cache;
    struct Unlink { FrameCache*& p; bool mine; ~Unlink() { if (mine) p = nullptr; } } unlink{po.opt.cache, false};
    if (!so.cache_frames) po.opt.cache = nullptr;
    else if (!po.opt.cache) { own_cache = std::make_unique<FrameCache>(); po.opt.cache = own_cache.get(); unlink.mine = true; }
    if (po.opt.cache) {
        std::vector<fs::path> paths = po.paths;
        std::vector<Similarity> T;
        if (paths.empty() && !po.list.empty()) read_list(po.list, paths, T);
        po.opt.cache->prepare(paths);
        if (own_cache) logf("frames resident in host memory: %d of %zu, budget %.1f GB\n", po.opt.cache->count(), paths.size(), po.opt.cache->budget() / 1e9);
    }

    // ---- pass 1 + volume
    auto wants = [&](const char* name) { return so.outputs.empty() || std::find(so.outputs.begin(), so.outputs.end(), name) != so.outputs.end(); };
    po.with_a = wants("A");
    Pass1 P;
    struct FreeState { FoldState& s; ~FreeState() { s.free(); } } free_state{P.dst};   // the merged state lives on devs[0] until the end
    if (int r = run_pass1(po, P)) return r;
    if (hooks.cancelled()) return 3;
    const int K = (int)P.paths.size();
    const int ow = P.g.out_w(), oh = P.g.out_h();
    const size_t no = (size_t)ow * oh, nd = (size_t)P.vp.dw * P.vp.dh;
    const int prim = P.devs[0];
    logf("pass 1: %.2f s setup, %.2f s stream + %.2f s merge + %.2f s volume\n", P.setup_seconds, P.st.seconds, P.merge_seconds, P.vol_seconds);
    if (so.result) {
        so.result->pass1_seconds = since(t_all);
        so.result->frames_per_second = P.st.seconds > 0 ? P.st.frames / P.st.seconds : 0;
    }

    // ---- depth (SGM) on the primary GPU
    stage("depth map", 0.0);
    auto t1 = std::chrono::steady_clock::now();
    CK(cudaSetDevice(prim));
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    DepthResult R;
    float* d_guide = nullptr;
    {
        float *d_planes, *d_V, *d_ga;
        void *d_L16p, *d_L16;
        CK(cudaMalloc(&d_planes, nd * K * sizeof(float)));
        CK(cudaMalloc(&d_V, nd * K * sizeof(float)));
        CK(cudaMalloc(&d_L16p, nd * K * 2));
        CK(cudaMalloc(&d_L16, nd * K * 2));
        CK(cudaMalloc(&d_guide, no * sizeof(float)));
        CK(cudaMalloc(&d_ga, nd * sizeof(float)));
        CK(cudaMemcpy(d_planes, P.V_all.data(), nd * K * sizeof(float), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_L16p, P.lum_all.data(), nd * K * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_guide, P.guide.data(), no * sizeof(float), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_ga, P.ga.data(), nd * sizeof(float), cudaMemcpyHostToDevice));
        planes_to_kfastest_f32(d_planes, d_V, P.vp.dw, P.vp.dh, K, s);
        planes_to_kfastest_f16(d_L16p, d_L16, P.vp.dw, P.vp.dh, K, s);
        CK(cudaStreamSynchronize(s));
        cudaFree(d_planes); cudaFree(d_L16p);
        std::vector<float>().swap(P.V_all);
        std::vector<uint16_t>().swap(P.fm_all);
        std::vector<uint16_t>().swap(P.lum_all);
        DepthParams dp;
        dp.halo_prior = (float)so.halo_prior;
        dp.dark_prior = (float)so.dark_prior;
        R = depth_from_volume(d_V, d_L16, d_guide, d_ga, P.vp.dw, P.vp.dh, K, ow, oh, dp, s);
        cudaFree(d_V); cudaFree(d_L16); cudaFree(d_ga);
    }
    logf("depth: %.2f s (cost %.0f ms, SGM %.0f ms, post %.0f ms); end-peaked %.1f%%\n", since(t1), R.ms_cost, R.ms_sgm, R.ms_post,
           100 * R.end_fraction);
    if (so.result) so.result->depth_seconds = since(t1);
    auto preview_depth = [&](const char* name) {
        if (!hooks.preview) return;
        std::vector<float> rgb(no * 3);
        for (size_t i = 0; i < no; ++i) { float v = R.depth_s[i] / (float)std::max(K - 1, 1); rgb[i * 3] = rgb[i * 3 + 1] = rgb[i * 3 + 2] = v; }
        std::vector<float> small; int pw, ph;
        downsample_preview(rgb, ow, oh, hooks.preview_down, small, pw, ph);
        hooks.preview(name, small, pw, ph);
    };
    preview_depth("depth map");
    if (hooks.cancelled()) { cudaFree(d_guide); cudaStreamDestroy(s); return 3; }

    // ---- pass 2: refinement
    auto t2 = std::chrono::steady_clock::now();
    std::vector<int> all_k(K);
    for (int k = 0; k < K; ++k) all_k[k] = k;
    float refine_sigma = (float)std::max(0.6, refine_fm_px * P.g.scale);
    {
        int ndev = 0;
        CK(cudaGetDeviceCount(&ndev));
        std::vector<RefineState> rst(ndev);
        std::vector<std::vector<RefineScratch>> rsc(ndev);
        std::vector<float*> d_depth(ndev, nullptr);
        for (int d : P.devs) {
            CK(cudaSetDevice(d));
            rst[d].alloc(d, ow, oh);
            rsc[d].resize(streams);
            for (auto& sc : rsc[d]) sc.alloc(ow, oh, refine_sigma);
            CK(cudaMalloc(&d_depth[d], no * sizeof(float)));
            CK(cudaMemcpy(d_depth[d], R.depth_s.data(), no * sizeof(float), cudaMemcpyHostToDevice));
        }
        // last frame first: the live view runs back through the stack (the argmax is order-free)
        std::vector<int> back_k(all_k.rbegin(), all_k.rend());
        std::vector<fs::path> back_paths(P.paths.rbegin(), P.paths.rend());
        std::vector<PassDev> ctx;
        std::atomic<int> refined{0};
        run_pass(back_paths, back_k, P.T, P.g, P.devs, streams, po.opt.readers, po.opt.cache, ctx,
                 [&](int dev, PassSlot& S, int, cudaStream_t ss) { refine_measure(S.d_warp, rsc[dev][S.index], ss); },
                 [&](int dev, PassSlot& S, int k, cudaStream_t fs) {
                     refine_accumulate(rst[dev], k, rsc[dev][S.index].fm, d_depth[dev], refine_r, fs);
                     if (hooks.preview && dev == P.devs[0] && (++refined % std::max(1, hooks.preview_every) == 0)) {
                         std::vector<float> img;
                         int pw, ph;
                         refine_preview(rst[dev], K, hooks.preview_down, img, pw, ph, fs);
                         hooks.preview("depth (building)", img, pw, ph);
                     }
                 },
                 "pass 2: fine focus", hooks);
        for (size_t i = 1; i < P.devs.size(); ++i) merge_device(rst[prim], rst[P.devs[i]]);
        for (int d : P.devs) {
            CK(cudaSetDevice(d));
            for (auto& sc : rsc[d]) sc.free();
            cudaFree(d_depth[d]);
        }
        CK(cudaSetDevice(prim));
        refine_depth(R.depth_s, R.conf, R.spill, P.vp.dw, P.vp.dh, rst[prim], ow, oh, P.g.scale, s);
        for (int d : P.devs) rst[d].free();   // leaves the last device selected
        CK(cudaSetDevice(prim));
    }
    std::vector<float> Ds = smooth_depth(R.depth_s, d_guide, ow, oh, P.g.scale, K, s);
    logf("refine: %.2f s\n", since(t2));
    if (so.result) so.result->refine_seconds = since(t2);
    preview_depth("depth map (refined)");
    if (hooks.cancelled()) { cudaFree(d_guide); cudaStreamDestroy(s); return 3; }

    // ---- pass 3: composition over the frames that carry a label
    auto t3 = std::chrono::steady_clock::now();
    std::vector<float> b_img;
    {
        std::vector<int> ks = compose_frames(R.depth_s, Ds, K);
        std::vector<fs::path> sub;
        for (int k : ks) sub.push_back(P.paths[k]);
        int ndev = 0;
        CK(cudaGetDeviceCount(&ndev));
        std::vector<ComposeState> cst(ndev);
        std::vector<std::vector<ComposeScratch>> csc(ndev);
        std::vector<float*> d_D(ndev, nullptr), d_Ds(ndev, nullptr);
        float s1 = (float)std::max(0.8, soft_px * P.g.scale);
        for (int d : P.devs) {
            CK(cudaSetDevice(d));
            cst[d].alloc(d, ow, oh);
            csc[d].resize(streams);
            for (auto& sc : csc[d]) sc.alloc(ow, oh, s1);
            CK(cudaMalloc(&d_D[d], no * sizeof(float)));
            CK(cudaMalloc(&d_Ds[d], no * sizeof(float)));
            CK(cudaMemcpy(d_D[d], R.depth_s.data(), no * sizeof(float), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_Ds[d], Ds.data(), no * sizeof(float), cudaMemcpyHostToDevice));
        }
        std::vector<PassDev> ctx;
        std::atomic<int> composed{0};
        run_pass(sub, ks, P.T, P.g, P.devs, streams, po.opt.readers, po.opt.cache, ctx,
                 [&](int, PassSlot&, int, cudaStream_t) {},
                 [&](int dev, PassSlot& S, int k, cudaStream_t fs) {
                     compose_frame(cst[dev], k, S.d_warp, d_D[dev], d_Ds[dev], csc[dev][S.index], fs);
                     // live view: the first GPU's composition so far, every preview_every frames
                     if (hooks.preview && dev == P.devs[0] && (++composed % std::max(1, hooks.preview_every) == 0)) {
                         std::vector<float> img;
                         int pw, ph;
                         compose_preview(cst[dev], hooks.preview_down, img, pw, ph, fs);
                         hooks.preview("B (building)", img, pw, ph);
                     }
                 },
                 "pass 3: composition", hooks);
        for (size_t i = 1; i < P.devs.size(); ++i) merge_device(cst[prim], cst[P.devs[i]]);
        for (int d : P.devs) {
            CK(cudaSetDevice(d));
            for (auto& sc : csc[d]) sc.free();
            cudaFree(d_D[d]); cudaFree(d_Ds[d]);
        }
        CK(cudaSetDevice(prim));
        b_img = compose_mix(cst[prim], R.conf, ow, oh, P.g.scale, s);
        for (int d : P.devs) cst[d].free();   // leaves the last device selected
        CK(cudaSetDevice(prim));
        logf("compose: %zu of %d frames carry a label; %.2f s\n", ks.size(), K, since(t3));
    }
    if (so.result) so.result->compose_seconds = since(t3);
    if (hooks.preview) { std::vector<float> small; int pw, ph; downsample_preview(b_img, ow, oh, hooks.preview_down, small, pw, ph); hooks.preview("B", small, pw, ph); }
    if (hooks.cancelled()) { cudaFree(d_guide); cudaStreamDestroy(s); return 3; }
    stage("finalize", 0.0);

    // ---- noise from two adjacent middle frames, then finalize
    auto t4 = std::chrono::steady_clock::now();
    std::vector<float> noise_mad;
    {
        int mid = std::max(0, K / 2 - 1);
        float* d_pair[2];
        for (int j = 0; j < 2; ++j) {
            TiffInfo info;
            float* d_rgb = nullptr;
            if (int r = load_frame_to_device(P.paths[mid + j], info, d_rgb, po.opt.cache)) return r;
            float* d_scr = nullptr;
            if (P.g.scale != 1.0) CK(cudaMalloc(&d_scr, (size_t)P.g.work_w() * P.g.work_h() * 3 * sizeof(float)));
            CK(cudaMalloc(&d_pair[j], no * 3 * sizeof(float)));
            load_aligned(d_rgb, P.g, P.T[mid + j], d_scr, d_pair[j], false, s);
            CK(cudaStreamSynchronize(s));
            cudaFree(d_rgb); cudaFree(d_scr);
        }
        noise_mad = noise_from_pair(d_pair[0], d_pair[1], ow, oh, P.fp.levels, s);
        cudaFree(d_pair[0]); cudaFree(d_pair[1]);
    }
    FinalParams fpar;
    fpar.c_smoothing = (float)so.c_smoothing;
    fpar.outputs = so.outputs;
    CK(cudaSetDevice(prim));
    FinalOutputs F = finalize(P.dst, noise_mad, R.depth_s, b_img, fpar, s);
    {
        std::string sig;
        char b[32];
        for (float v : noise_mad) { snprintf(b, sizeof b, " %.2f", v * 1e4); sig += b; }
        logf("finalize: %.2f s; noise sigma per level (x1e-4):%s", since(t4), sig.c_str());
    }
    if (so.result) so.result->finalize_seconds = since(t4);
    auto wanted = [&](const std::string& n) { return so.outputs.empty() || std::find(so.outputs.begin(), so.outputs.end(), n) != so.outputs.end(); };
    if (hooks.preview)
        for (size_t i = 0; i < F.names.size(); ++i)
            if (wanted(F.names[i])) { std::vector<float> small; int pw, ph; downsample_preview(F.images[i], ow, oh, hooks.preview_down, small, pw, ph); hooks.preview(F.names[i], small, pw, ph); }
    std::vector<uint8_t> icc = tiff_probe(P.paths[0]).icc;   // the frames' colour profile goes into every output
    if (so.result) {
        auto& Rr = *so.result;
        Rr.w = ow; Rr.h = oh;
        Rr.names.clear(); Rr.images.clear();
        for (size_t i = 0; i < F.names.size(); ++i)
            if (wanted(F.names[i])) { Rr.names.push_back(F.names[i]); Rr.images.push_back(std::move(F.images[i])); }
        Rr.depth = R.depth_s; Rr.conf = R.conf; Rr.noise_mad = noise_mad; Rr.icc = icc;
    }

    // ---- write
    auto t5 = std::chrono::steady_clock::now();
    size_t written = 0;
    if (!out.empty()) {
    fs::create_directories(out);
    fs::path od(out);
    for (size_t i = 0; i < F.names.size(); ++i) {
        if (!wanted(F.names[i])) continue;
        const std::vector<float>* img = &F.images[i];
        if (so.result) {   // the image moved into the result
            size_t j = std::find(so.result->names.begin(), so.result->names.end(), F.names[i]) - so.result->names.begin();
            img = &so.result->images[j];
        }
        if (!write_tiff_rgb16((od / (F.names[i] + ".tif")).string(), *img, ow, oh, icc)) { logf("cannot write %s\n", F.names[i].c_str()); return 1; }
        ++written;
    }
    write_raw(od / "depth.f32", R.depth_s.data(), R.depth_s.size());
    write_raw(od / "depth_smooth.f32", Ds.data(), Ds.size());
    write_raw(od / "conf.f32", R.conf.data(), R.conf.size());
    FILE* mf = fopen((od / "stack.txt").string().c_str(), "w");
    if (mf) {
        fprintf(mf, "w %d\nh %d\nframes %d\nlevels %d\nscale %g\nrect %d %d %d %d\nnoise_mad", ow, oh, K, P.fp.levels, P.g.scale, P.g.x0, P.g.y0, P.g.x1, P.g.y1);
        for (float v : noise_mad) fprintf(mf, " %.9g", v);
        fprintf(mf, "\noutputs");
        for (auto& nme : F.names) fprintf(mf, " %s", nme.c_str());
        fprintf(mf, "\n");
        fclose(mf);
    }
    logf("wrote %zu outputs (%dx%d) in %.2f s -> %s", written, ow, oh, since(t5), out.c_str());
    if (so.result) so.result->write_seconds = since(t5);
    }
    cudaFree(d_guide);
    cudaStreamDestroy(s);
    if (so.result) so.result->seconds = since(t_all);
    logf("fusion total %.2f s", since(t_all));
    stage("done", 1.0);
    return 0;
}

// ----------------------------------------------------------------------------- align

bool parse_align_option(const std::string& a, const std::string& v, AlignOptions& ao) {
    if (a == "--prefix") ao.prefix = v;
    else if (a == "--align-scale") ao.align_scale = atof(v.c_str());
    else if (a == "--baselines") ao.baselines = parse_list(v);
    else if (a == "--ref") ao.ref_rule = v;
    else if (a == "--gpus") ao.devs = parse_list(v);
    else if (a == "--pairs") ao.pairs_sel = v;
    else if (a == "--readers") ao.readers = atoi(v.c_str());
    else if (a == "--pair-threads") ao.pair_threads = atoi(v.c_str());
    else if (a == "--lock-scale") ao.lock_scale = v != "0" && v != "false";
    else if (a == "--um-per-px") ao.um_per_px = atof(v.c_str());
    else if (a == "--fov-mm") ao.um_per_px = -atof(v.c_str());   // resolved once the width is known
    else return false;
    return true;
}


int run_align(AlignOptions& ao, AlignOut& A) {
    const fs::path& dir = ao.dir;
    const std::string &out = ao.out, &prefix = ao.prefix, &ref_rule = ao.ref_rule, &pairs_sel = ao.pairs_sel;
    const double align_scale = ao.align_scale;
    const std::vector<int>& baselines = ao.baselines;
    std::vector<int> devs = ao.devs;
    const int readers = ao.readers;
    const Hooks& hooks = ao.hooks;
    std::vector<fs::path>& paths = A.paths;
    paths.clear();
    if (!ao.paths.empty()) {
        paths = ao.paths;
    } else {
        for (auto& e : fs::directory_iterator(dir)) {
            auto name = e.path().filename().string();
            auto ext = e.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (e.is_regular_file() && (ext == ".tif" || ext == ".tiff") && name.rfind(prefix, 0) == 0) paths.push_back(e.path());
        }
        std::sort(paths.begin(), paths.end());
    }
    // frames are the TIFFs of the most common size; anything else in the folder (a previous stack's
    // outputs, say) is left out and reported
    {
        std::vector<TiffInfo> infos;
        std::map<std::pair<unsigned, unsigned>, int> sizes;
        for (auto& p : paths) {
            infos.push_back(tiff_probe(p));
            if (infos.back().ok()) ++sizes[{infos.back().width, infos.back().height}];
        }
        std::pair<unsigned, unsigned> best{0, 0};
        int bestn = 0;
        for (auto& [sz, n] : sizes) if (n > bestn) { best = sz; bestn = n; }
        std::vector<fs::path> keep;
        for (size_t i = 0; i < paths.size(); ++i) {
            if (infos[i].ok() && infos[i].width == best.first && infos[i].height == best.second) keep.push_back(paths[i]);
            else logf("  skipping %s (%s)", paths[i].filename().string().c_str(),
                      infos[i].ok() ? (std::to_string(infos[i].width) + " x " + std::to_string(infos[i].height) + ", not the stack's size").c_str()
                                    : infos[i].error.c_str());
        }
        paths.swap(keep);
    }
    const int N = (int)paths.size();
    if (N < 2) { logf("need >= 2 frames, found %d\n", N); return 1; }
    int ndev = 0;
    CK(cudaGetDeviceCount(&ndev));
    if (devs.empty()) for (int i = 0; i < ndev; ++i) devs.push_back(i);
    TiffInfo t0 = tiff_probe(paths[0]);
    if (!t0.ok()) { logf("%s: %s\n", paths[0].string().c_str(), t0.error.c_str()); return 1; }
    const int W = (int)t0.width, H = (int)t0.height;
    const int w = std::max(16, (int)std::nearbyint(W * align_scale)), h = std::max(16, (int)std::nearbyint(H * align_scale));
    const double fx = (double)w / W, fy = (double)h / H;
    auto t_all = std::chrono::steady_clock::now();

    // ---- pass 0: luma at the align scale, for every frame, to the host
    std::vector<float> luma_stack((size_t)N * w * h);
    {
        struct Slot { float *d_rgb = nullptr, *d_y = nullptr, *d_small = nullptr; AreaResizer rz; };
        std::vector<std::vector<Slot>> slots(ndev);
        StreamerOptions opt;
        opt.devices = devs;
        opt.streams_per_gpu = 3;
        opt.readers = readers;
        opt.cancel = hooks.cancel;
        opt.cache = ao.cache;
        if (ao.cache) {
            ao.cache->prepare(paths);
            logf("frames resident in host memory: %d of %d, budget %.1f GB\n", ao.cache->count(), N, ao.cache->budget() / 1e9);
        }
        std::atomic<int> done{0};
        for (int d : devs) {
            CK(cudaSetDevice(d));
            slots[d].resize(opt.streams_per_gpu);
            for (auto& S : slots[d]) {
                CK(cudaMalloc(&S.d_rgb, (size_t)W * H * 3 * sizeof(float)));
                CK(cudaMalloc(&S.d_y, (size_t)W * H * sizeof(float)));
                CK(cudaMalloc(&S.d_small, (size_t)w * h * sizeof(float)));
                S.rz.build(W, H, w, h);
            }
        }
        auto on_frame = [&](const DeviceFrame& f) {
            auto& S = slots[f.dev][f.slot];
            convert_to_rgb_f32(f.d_pixels(), *f.info, S.d_rgb, nullptr, f.stream);
            luma(S.d_rgb, S.d_y, (size_t)W * H, f.stream);
            S.rz.run(S.d_y, S.d_small, 1, f.stream);
            cudaMemcpyAsync(luma_stack.data() + (size_t)f.k * w * h, S.d_small, (size_t)w * h * sizeof(float), cudaMemcpyDeviceToHost, f.stream);
            if (hooks.progress) hooks.progress("reading frames", (double)(++done) / N);
            if (hooks.frame) {   // the frame as read (unaligned), small
                std::vector<float> img;
                int pw, ph;
                image_preview(S.d_rgb, W, H, hooks.frame_down, img, pw, ph, f.stream);
                hooks.frame(f.k, img, pw, ph);
            }
        };
        StreamerStats st = stream_frames(paths, opt, on_frame);
        for (auto& e : st.errors) logf("  error: %s\n", e.c_str());
        for (int d : devs) {
            CK(cudaSetDevice(d));
            CK(cudaDeviceSynchronize());
            for (auto& S : slots[d]) { cudaFree(S.d_rgb); cudaFree(S.d_y); cudaFree(S.d_small); S.rz.free(); }
        }
        logf("align: %d frames %dx%d -> luma %dx%d in %.2f s%s\n", N, W, H, w, h, st.seconds, source_note(st).c_str());
        if (getenv("STACKER_MEM_DEBUG"))
            for (int d : devs) { size_t fr, tot; cudaSetDevice(d); cudaMemGetInfo(&fr, &tot); logf("  [mem] GPU %d after pass 0: %.2f GB free", d, fr / 1e9); }
        if (hooks.cancelled()) return 3;
    }

    // ---- pairs, dynamically shared by the GPUs and by several threads per GPU (a pair's Gauss-Newton
    // loop syncs the host every iteration, so one pair at a time leaves the card mostly idle; FINDINGS 27).
    // Each GPU keeps one pyramid cache for its threads: LRU past `cap`, an entry in use is never evicted.
    std::vector<std::pair<int, int>> pairs;
    if (pairs_sel.empty()) pairs = make_pairs(N, baselines);
    else {
        size_t i = 0;
        while (i < pairs_sel.size()) {
            size_t j = pairs_sel.find(',', i);
            if (j == std::string::npos) j = pairs_sel.size();
            std::string tok = pairs_sel.substr(i, j - i);
            size_t c = tok.find(':');
            if (c != std::string::npos) pairs.push_back({atoi(tok.substr(0, c).c_str()), atoi(tok.substr(c + 1).c_str())});
            i = j + 1;
        }
    }
    std::vector<PairResult> results(pairs.size());
    std::atomic<int> next{0}, pairs_done{0};
    auto t_pairs = std::chrono::steady_clock::now();
    const int T = std::max(1, ao.pair_threads);
    struct PyrCache {
        struct Entry { AlignPyramid P; int users = 0; bool ready = false; };
        std::unordered_map<int, Entry> map;   // node-based: references stay valid while entries are added
        std::list<int> lru;
        std::mutex m;
        std::condition_variable cv;
        int cap = 24, builds = 0;
        bool failed = false;   // a thread died: waiters on an unfinished build give up
    };
    std::vector<PyrCache> caches(ndev);
    for (auto& C : caches) C.cap = 16 + 8 * T;   // each thread's pairs touch ~9 frames (baselines up to 8)
    std::vector<std::thread> workers;
    std::vector<std::string> worker_err(ndev);
    std::vector<std::atomic<int>> per_dev(ndev);
    for (auto& c : per_dev) c = 0;
    for (int d : devs)
        for (int t = 0; t < T; ++t)
            workers.emplace_back([&, d, t] {
                PyrCache& C = caches[d];
                try {
                    CK_THROW(cudaSetDevice(d));
                    cudaStream_t s;
                    CK_THROW(cudaStreamCreate(&s));
                    AlignWork wk;
                    wk.alloc(w, h);
                    float* d_luma;
                    CK_THROW(cudaMalloc(&d_luma, (size_t)w * h * sizeof(float)));
                    // a pyramid, built on this thread's stream if nobody has it; waits if another thread is building it
                    auto get = [&](int k) -> AlignPyramid& {
                        std::unique_lock<std::mutex> lk(C.m);
                        auto it = C.map.find(k);
                        if (it != C.map.end()) {
                            PyrCache::Entry& E = it->second;
                            ++E.users;
                            C.lru.remove(k); C.lru.push_back(k);
                            C.cv.wait(lk, [&] { return E.ready || C.failed; });
                            if (!E.ready) throw std::runtime_error("pyramid build failed on another thread");
                            return E.P;
                        }
                        while ((int)C.map.size() >= C.cap) {   // evict the least recently used entry nobody holds
                            auto victim = C.lru.end();
                            for (auto v = C.lru.begin(); v != C.lru.end(); ++v) {
                                PyrCache::Entry& V = C.map[*v];
                                if (V.users == 0 && V.ready) { victim = v; break; }
                            }
                            if (victim == C.lru.end()) break;
                            C.map[*victim].P.free();
                            C.map.erase(*victim);
                            C.lru.erase(victim);
                        }
                        PyrCache::Entry& E = C.map[k];
                        E.users = 1;
                        C.lru.push_back(k);
                        lk.unlock();
                        CK_THROW(cudaMemcpyAsync(d_luma, luma_stack.data() + (size_t)k * w * h, (size_t)w * h * sizeof(float), cudaMemcpyHostToDevice, s));
                        E.P.build(d_luma, w, h, s);   // synchronises its stream before returning
                        lk.lock();
                        E.ready = true;
                        ++C.builds;
                        C.cv.notify_all();
                        return E.P;
                    };
                    auto release = [&](int k) {
                        std::lock_guard<std::mutex> lk(C.m);
                        --C.map[k].users;
                    };
                    AlignParams ap;
                    for (;;) {
                        int e = next++;
                        if (e >= (int)pairs.size() || hooks.cancelled()) break;
                        auto [i, j] = pairs[e];
                        AlignPyramid& Pi = get(i);
                        AlignPyramid& Pj = get(j);
                        results[e] = align_pair(Pi, Pj, i, j, wk, ap, s);
                        release(i);
                        release(j);
                        if (hooks.progress) hooks.progress("aligning pairs", (double)(++pairs_done) / (double)pairs.size());
                        ++per_dev[d];
                    }
                    wk.free();
                    cudaFree(d_luma);
                    cudaStreamDestroy(s);
                } catch (const std::exception& ex) {
                    std::lock_guard<std::mutex> lk(C.m);
                    worker_err[d] = ex.what();
                    C.failed = true;
                    C.cv.notify_all();
                }
            });
    for (auto& t : workers) t.join();
    std::vector<int> builds(ndev, 0);
    for (int d : devs) {
        cudaSetDevice(d);
        for (auto& kv : caches[d].map) kv.second.P.free();
        caches[d].map.clear();
        builds[d] = caches[d].builds;
    }
    for (int d : devs) if (!worker_err[d].empty()) { logf("GPU %d: %s\n", d, worker_err[d].c_str()); return 1; }
    if (getenv("STACKER_MEM_DEBUG"))
        for (int d : devs) { size_t fr, tot; cudaSetDevice(d); cudaMemGetInfo(&fr, &tot); logf("  [mem] GPU %d after pairs: %.2f GB free, %d pyramid builds", d, fr / 1e9, builds[d]); }
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_pairs).count();
    {
        std::string split;
        for (int d : devs) split += (d == devs[0] ? "; GPU " : ", GPU ") + std::to_string(d) + " " + std::to_string(per_dev[d].load());
        logf("  %zu pairs in %.2f s (%.1f ms/pair, %d in flight per GPU)%s", pairs.size(), secs, 1000 * secs / pairs.size(), T, split.c_str());
    }
    if (hooks.cancelled()) return 3;
    if (!pairs_sel.empty()) {   // per-pair dump for validation
        fs::create_directories(out);
        fs::path od(out);
        A.pairs = results;
        FILE* f = fopen((od / "pairs.txt").string().c_str(), "w");
        for (auto& r : results) {
            logf("  pair %d:%d  z %.9f%+.9fi  t %.5f %.5f  gain %.5f  ncc %.5f  n %d  iters %d  var %.3e %.3e %.3e %.3e  %s\n", r.i, r.j,
                   r.z.real(), r.z.imag(), r.t.real(), r.t.imag(), r.gain, r.ncc, r.n_px, r.iters, r.var_a, r.var_b, r.var_tx, r.var_ty,
                   r.ok ? "ok" : "BAD");
            if (f) fprintf(f, "%d %d %.17g %.17g %.17g %.17g %.17g %.17g %d %d %.17g %.17g %.17g %.17g %d\n", r.i, r.j, r.z.real(), r.z.imag(),
                           r.t.real(), r.t.imag(), r.gain, r.ncc, r.n_px, r.iters, r.var_a, r.var_b, r.var_tx, r.var_ty, r.ok ? 1 : 0);
        }
        if (f) fclose(f);
        return 0;
    }

    // ---- global solve, reference, geometry
    A.G = solve_global(N, results);
    std::vector<std::complex<double>> z0 = A.G.z, tf(N);
    for (int k = 0; k < N; ++k) tf[k] = {A.G.t[k].real() / fx, A.G.t[k].imag() / fy};   // work px -> full-res px
    A.ref = choose_reference(z0, ref_rule);
    std::vector<std::complex<double>> za = z0;
    if (ao.lock_scale) for (auto& z : za) z /= std::abs(z);   // the geometry uses the APPLIED transforms
    std::vector<std::complex<double>> zr = za, tr = tf;
    rebase(zr, tr, A.ref);
    common_rect(zr, tr, W, H, A.rect);
    A.W = W; A.H = H;
    A.smin = 1e300; A.smax = -1e300;
    for (auto& z : zr) { A.smin = std::min(A.smin, std::abs(z)); A.smax = std::max(A.smax, std::abs(z)); }
    A.T.resize(N);
    for (int k = 0; k < N; ++k) A.T[k] = {zr[k].real(), zr[k].imag(), tr[k].real(), tr[k].imag()};
    if (ao.um_per_px < 0) ao.um_per_px = -ao.um_per_px * 1000.0 / W;   // from --fov-mm
    A.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_all).count();
    logf("  solve: %d/%d pairs usable, %d outliers, ncc min %.4f median %.4f; ref #%d; native crop %dx%d (L%d R%d T%d B%d); scale spread %.3f%%%s; %.2f s\n",
           A.G.pairs_used, A.G.pairs_total, A.G.outliers, A.G.ncc_min, A.G.ncc_median, A.ref, A.rect[2] - A.rect[0] + 1, A.rect[3] - A.rect[1] + 1,
           A.rect[0], W - 1 - A.rect[2], A.rect[1], H - 1 - A.rect[3], 100 * (A.smax / A.smin - 1), ao.lock_scale ? " (scale locked)" : "", A.seconds);
    return 0;
}

std::string report_json(const AlignOptions& ao, const AlignOut& A) {
    const int N = (int)A.paths.size(), W = A.W, H = A.H;
    const int* rect = A.rect;
    std::string s;
    char b[1024];
    auto add = [&](const char* fmt, auto... args) { snprintf(b, sizeof b, fmt, args...); s += b; };
    add("{\n \"stack\": \"%s\",\n \"alignment\": {\"align_scale\": %g, \"baselines\": [", ao.dir.filename().string().c_str(), ao.align_scale);
    for (size_t i = 0; i < ao.baselines.size(); ++i) add("%s%d", i ? ", " : "", ao.baselines[i]);
    add("], \"scale_locked\": %s, \"pairs_used\": %d, \"pairs_total\": %d, \"outliers\": %d, \"ncc_min\": %.17g, \"ncc_median\": %.17g},\n",
        ao.lock_scale ? "true" : "false", A.G.pairs_used, A.G.pairs_total, A.G.outliers, A.G.ncc_min, A.G.ncc_median);
    int cw = rect[2] - rect[0] + 1, ch = rect[3] - rect[1] + 1;
    add(" \"geometry\": {\n  \"input\": {\"width\": %d, \"height\": %d, \"frames\": %d, \"um_per_px_calibrated\": ", W, H, N);
    if (ao.um_per_px > 0) add("%.17g},\n", ao.um_per_px); else add("%s", "null},\n");
    add("  \"reference\": {\"index\": %d, \"rule\": \"%s\"},\n  \"selected_policy\": \"native-crop\",\n  \"policies\": {\"native-crop\": {\"width\": %d, \"height\": %d, "
        "\"crop_px\": {\"left\": %d, \"right\": %d, \"top\": %d, \"bottom\": %d}, \"pixels_kept_pct\": %.17g",
        A.ref, ao.ref_rule.c_str(), cw, ch, rect[0], W - 1 - rect[2], rect[1], H - 1 - rect[3], 100.0 * cw * ch / ((double)W * H));
    if (ao.um_per_px > 0)
        add(", \"um_per_px\": %.17g, \"um_per_px_range_over_depth\": [%.17g, %.17g], \"range_pct\": [%.17g, %.17g], \"fov_mm\": [%.17g, %.17g]",
            ao.um_per_px, ao.um_per_px * A.smin, ao.um_per_px * A.smax, 100 * (A.smin - 1), 100 * (A.smax - 1), cw * ao.um_per_px / 1000.0,
            ch * ao.um_per_px / 1000.0);
    add("}},\n  \"scale_spread\": {\"s_min\": %.17g, \"s_max\": %.17g, \"total_pct\": %.17g, \"corner_shift_px\": %.17g}\n },\n \"per_frame\": [\n",
        A.smin, A.smax, 100 * (A.smax / A.smin - 1), (A.smax / A.smin - 1) * std::hypot((double)W, (double)H) / 2.0);
    for (int k = 0; k < N; ++k) {
        std::complex<double> z(A.T[k].zr, A.T[k].zi);
        add("  {\"i\": %d, \"file\": \"%s\", \"s\": %.17g, \"theta_deg\": %.17g, \"tx\": %.17g, \"ty\": %.17g}%s\n", k,
            A.paths[k].filename().string().c_str(), std::abs(z), std::arg(z) * 180.0 / 3.14159265358979323846, A.T[k].tx, A.T[k].ty,
            k + 1 < N ? "," : "");
    }
    s += " ],\n \"transform_convention\": \"reference-grid centred px -> frame centred px: p_k = s e^{i theta} p + t\"\n}\n";
    return s;
}

void write_report(const AlignOptions& ao, const AlignOut& A, const fs::path& od) {
    const int N = (int)A.paths.size();
    const int* rect = A.rect;
    fs::create_directories(od);
    FILE* f = fopen((od / "report.json").string().c_str(), "w");
    if (!f) { logf("cannot write %s\n", (od / "report.json").string().c_str()); return; }
    std::string js = report_json(ao, A);
    fwrite(js.data(), 1, js.size(), f);
    fclose(f);
    FILE* lf = fopen((od / "list.txt").string().c_str(), "w");
    if (lf) {
        for (int k = 0; k < N; ++k) fprintf(lf, "%s %.17g %.17g %.17g %.17g\n", A.paths[k].string().c_str(), A.T[k].zr, A.T[k].zi, A.T[k].tx, A.T[k].ty);
        fclose(lf);
    }
    logf("  report.json + list.txt -> %s (rect %d,%d,%d,%d)\n", od.string().c_str(), rect[0], rect[1], rect[2], rect[3]);
}

int run_job(JobOptions& jo, AlignOut& A, Hooks hooks) {
    AlignOptions& ao = jo.align;
    StackOptions& so = jo.stack;
    std::string out = jo.out;
    if (out.empty()) out = (ao.dir.parent_path() / (ao.dir.filename().string() + "_stacker")).string();
    if (so.po.opt.devices.empty()) so.po.opt.devices = ao.devs; else if (ao.devs.empty()) ao.devs = so.po.opt.devices;
    so.po.opt.readers = ao.readers;
    ao.out = out;
    ao.pairs_sel.clear();
    ao.hooks = hooks;
    so.hooks = hooks;
    std::unique_ptr<FrameCache> own_cache;
    FrameCache* cache = so.cache_frames ? jo.cache : nullptr;
    if (so.cache_frames && !cache) { own_cache = std::make_unique<FrameCache>(); cache = own_cache.get(); }
    ao.cache = cache;
    so.po.opt.cache = cache;
    struct Unlink { FrameCache*& a; FrameCache*& b; bool mine; ~Unlink() { if (mine) a = b = nullptr; } } unlink{ao.cache, so.po.opt.cache, own_cache != nullptr};
    auto t_all = std::chrono::steady_clock::now();
    if (int r = run_align(ao, A)) return r;
    write_report(ao, A, out);
    so.po.paths = A.paths;
    so.po.T = A.T;
    so.po.g.x0 = A.rect[0]; so.po.g.y0 = A.rect[1]; so.po.g.x1 = A.rect[2]; so.po.g.y1 = A.rect[3];
    so.out = so.write_outputs ? out : std::string();   // the benchmark computes everything but writes nothing
    if (int r = run_stack(so)) return r;
    logf("run: %zu frames aligned and stacked in %.2f s -> %s", A.paths.size(),
         std::chrono::duration<double>(std::chrono::steady_clock::now() - t_all).count(), out.c_str());
    return 0;
}

}  // namespace stackcore

