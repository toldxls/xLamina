// The pipeline as a library (see pipeline.h). Grown out of stackcli's commands.
#include "stackcore/pipeline.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
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
#include <cwctype>

#include "stackcore/convert.h"
#include "stackcore/depth.h"
#include "stackcore/imgops.h"
#include "stackcore/tiff.h"

namespace stackcore {

namespace fs = std::filesystem;

// a failed CUDA call ends the function: 4 when it was out of memory (run_stack answers that with a render in strips or
// at 1/4 depth resolution), 1 otherwise (noted as a GPU fault: a reset driver fails every later call of the process)
#define CK(x)                                                                                   \
    do {                                                                                        \
        cudaError_t e_ = (x);                                                                   \
        if (e_ != cudaSuccess) {                                                                \
            logf("CUDA error: %s (%s:%d)", cudaGetErrorString(e_), "pipeline.cpp", __LINE__);         \
            if (e_ == cudaErrorMemoryAllocation) { cudaGetLastError(); return 4; }              \
            note_fail(Fail::GpuFault, cudaGetErrorString(e_));                                  \
            return 1;                                                                           \
        }                                                                                       \
    } while (0)
#define CK_THROW(x)                                                                             \
    do {                                                                                        \
        cudaError_t e_ = (x);                                                                   \
        if (e_ != cudaSuccess) {                                                                \
            if (e_ != cudaErrorMemoryAllocation) note_fail(Fail::GpuFault, cudaGetErrorString(e_)); \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(e_) + " (pipeline.cpp:" + std::to_string(__LINE__) + ")"); \
        }                                                                                       \
    } while (0)
// an allocation in the fusion: out of memory returns 4, which run_stack answers with a render at 1/4 depth resolution
#define CK_MEM(x)                                                                               \
    do {                                                                                        \
        cudaError_t e_ = (x);                                                                   \
        if (e_ == cudaErrorMemoryAllocation) {                                                  \
            cudaGetLastError();                                                                 \
            logf("GPU memory ran out (%s:%d)", "pipeline.cpp", __LINE__);                             \
            return 4;                                                                           \
        }                                                                                       \
        if (e_ != cudaSuccess) {                                                                \
            logf("CUDA error: %s (%s:%d)", cudaGetErrorString(e_), "pipeline.cpp", __LINE__);         \
            note_fail(Fail::GpuFault, cudaGetErrorString(e_));                                  \
            return 1;                                                                           \
        }                                                                                       \
    } while (0)

// the first cause of a job's failure (pipeline.h); a memory shortfall of the PC is kept over a later consequence of it
static std::mutex g_fail_m;
static FailInfo g_fail;
void note_fail(Fail kind, const std::string& detail) {
    std::lock_guard<std::mutex> lk(g_fail_m);
    if (g_fail.kind == Fail::None) g_fail = {kind, detail};
}
FailInfo take_fail() {
    std::lock_guard<std::mutex> lk(g_fail_m);
    FailInfo f = g_fail;
    g_fail = {};
    return f;
}

// What a pass's stream errors mean for the job: 0 none, 4 a GPU ran out of memory (run_stack renders again in strips
// or at 1/4), 1 otherwise, with the cause noted: a frame that could not be read ("IMG_1.tif: short read") or a GPU
// worker that died ("device 0: ..."). A pass with an error has not seen every frame and its result is not used.
static int stream_fail(const StreamerStats& st) {
    if (st.errors.empty()) return 0;
    for (auto& e : st.errors) if (strstr(e.c_str(), "out of memory")) return 4;
    const std::string& e = st.errors.front();
    const bool gpu = e.rfind("device ", 0) == 0 || e.find("cuda") != std::string::npos || e.find("CUDA") != std::string::npos;
    note_fail(gpu ? Fail::GpuFault : Fail::FileRead, e);
    return 1;
}

// The output folder, made if it is not there. False, with the reason logged and noted, when it cannot be made.
static bool make_out_dir(const fs::path& od) {
    std::error_code ec;
    fs::create_directories(od, ec);
    if (!ec && fs::is_directory(od, ec)) return true;
    const std::string why = od.string() + ": " + (ec ? ec.message() : std::string("it is not a folder"));
    logf("cannot create the output folder %s", why.c_str());
    note_fail(Fail::FileWrite, why);
    return false;
}

// The outputs as 16-bit TIFFs in `od`, one thread each (the conversion and the write of each: 0.68 s for three in a
// row). Returns how many were written; one that was not is logged and noted with its path and the system's reason.
// Nothing a writer does can end the process: a throw in a thread without a catch did.
static size_t write_outputs(const fs::path& od, const std::vector<std::string>& names, const std::vector<const std::vector<float>*>& imgs,
                            int ow, int oh, const std::vector<uint8_t>& icc) {
    std::atomic<size_t> written{0};
    std::vector<std::thread> writers;
    for (size_t j = 0; j < names.size(); ++j)
        writers.emplace_back([&, j] {
            const std::string path = (od / (names[j] + ".tif")).string();
            std::string why;
            bool ok = false;
            try { ok = write_tiff_rgb16(path, *imgs[j], ow, oh, icc, &why); }
            catch (const std::bad_alloc&) { why = "out of memory"; }
            catch (const std::exception& e) { why = e.what(); }
            if (ok) { ++written; return; }
            logf("cannot write %s: %s", path.c_str(), why.c_str());
            note_fail(Fail::FileWrite, path + ": " + why);
        });
    for (auto& t : writers) t.join();
    return written.load();
}

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

std::string gpu_problem(const std::vector<int>& devs, bool launch) {
#ifndef XLAMINA_MIN_CUDA_ARCH
#define XLAMINA_MIN_CUDA_ARCH 75
#endif
    const int need = XLAMINA_MIN_CUDA_ARCH;   // 75 = compute capability 7.5
    const char* cards = need <= 75 ? "GeForce GTX 16 / RTX 20 series or newer" : need <= 86 ? "GeForce RTX 30 series or newer"
                      : need <= 89 ? "GeForce RTX 40 series or newer" : "GeForce RTX 50 series or newer";
    int n = 0;
    cudaError_t e = cudaGetDeviceCount(&n);
    if (e != cudaSuccess || n == 0) {
        cudaGetLastError();   // cleared: the next CUDA call's check must not report this one again
        std::string why = e == cudaErrorInsufficientDriver ? "The NVIDIA driver on this PC is too old; install the current one."
                                                           : "No usable NVIDIA GPU was found on this PC.";
        return std::string("xLamina needs an NVIDIA GPU (") + cards + ") with a current NVIDIA driver. " + why;
    }
    std::vector<int> ds = devs;
    if (ds.empty()) for (int i = 0; i < n; ++i) ds.push_back(i);
    for (int d : ds) {
        if (d < 0 || d >= n) return "GPU " + std::to_string(d) + " does not exist on this PC (it has " + std::to_string(n) + "): choose the GPUs again (Preferences > Performance, or --gpus).";
        cudaDeviceProp p{};
        if (cudaGetDeviceProperties(&p, d) != cudaSuccess) { cudaGetLastError(); continue; }
        if (p.major * 10 + p.minor < need) {
            std::string s = std::string("xLamina needs an NVIDIA GPU (") + cards + "). GPU " + std::to_string(d) + " (" + p.name + ") is too old for it.";
            if (n > 1) s += " Untick it in Preferences > Performance to render on the other GPU.";
            return s;
        }
        if (!launch) continue;
        // one small kernel: the first thing a driver too old for this build's code for the card refuses
        float *d_rgb = nullptr, *d_y = nullptr;
        cudaError_t e2 = cudaSetDevice(d);
        if (e2 == cudaSuccess) e2 = cudaMalloc(&d_rgb, 3 * 256 * sizeof(float));
        if (e2 == cudaSuccess) e2 = cudaMalloc(&d_y, 256 * sizeof(float));
        if (e2 == cudaSuccess) e2 = cudaMemset(d_rgb, 0, 3 * 256 * sizeof(float));
        if (e2 == cudaSuccess) {
            try { luma(d_rgb, d_y, 256, 0); } catch (const std::exception&) {}
            e2 = cudaDeviceSynchronize();
            if (e2 == cudaSuccess) e2 = cudaGetLastError();
        }
        cudaFree(d_rgb); cudaFree(d_y);
        cudaGetLastError();
        if (e2 != cudaSuccess && e2 != cudaErrorMemoryAllocation) {
            const std::string es = cudaGetErrorString(e2);
            logf("GPU %d (%s): the test kernel failed: %s", d, p.name, es.c_str());
            if (es.find("PTX") != std::string::npos || es.find("toolchain") != std::string::npos || es.find("JIT") != std::string::npos || es.find("river") != std::string::npos)
                return std::string("The NVIDIA driver on this PC is too old for xLamina on the ") + p.name + ". Install the current driver (nvidia.com/drivers, or the NVIDIA app) and start xLamina again.";
            return std::string("The graphics card ") + p.name + " does not run xLamina's code (" + es + "). Install the current NVIDIA driver and start xLamina again; if that does not help, send a report (Help > Report a problem).";
        }
    }
    return {};
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
static int load_frame_to_device(const fs::path& in, TiffInfo& info, float*& d_rgb, FrameCache* cache = nullptr, const DustFix* dust = nullptr) {
    info = tiff_probe(in);
    if (!info.ok()) { logf("%s: %s", in.string().c_str(), info.error.c_str()); return 1; }
    size_t cap = tiff_buffer_capacity(info.pixel_bytes());
    uint8_t* h = nullptr;
    const uint8_t* src = nullptr;
    size_t bytes = 0;
    if (const FrameCache::Entry* e = cache ? cache->find(in) : nullptr) { src = e->buf + e->offset; bytes = e->bytes; }
    else {
        // ordinary memory: a frame pinned and unpinned for every call counted towards the driver's ceiling for
        // page-locked memory with two GPUs (FINDINGS 55), and one frame's upload does not need it
        h = (uint8_t*)_aligned_malloc(cap, 4096);
        if (!h) { logf("out of host memory for a frame (%zu bytes)", cap); return 1; }
        TiffRead rd = tiff_read_pixels(in, info, h, cap);
        if (!rd.ok()) { logf("read failed: %s", rd.error.c_str()); _aligned_free(h); return 1; }
        src = h + rd.data_offset_in_buf;
        bytes = rd.bytes;
    }
    uint8_t* d_src = nullptr;
    d_rgb = nullptr;
    size_t npx = (size_t)info.width * info.height;
    // whatever fails, the upload buffer and the pinned copy go, and a failed load leaves no float image behind
    struct Tidy { uint8_t*& d_src; uint8_t*& h; float*& d_rgb; bool ok = false;
                  ~Tidy() { cudaFree(d_src); if (h) _aligned_free(h); if (!ok) { cudaFree(d_rgb); d_rgb = nullptr; } } } tidy{d_src, h, d_rgb};
    size_t fr = 0, tot = 0;
    CK(cudaMemGetInfo(&fr, &tot));
    if (fr < bytes + npx * 3 * sizeof(float) + ((size_t)256 << 20)) {   // never past the card's free memory (FINDINGS 55)
        logf("GPU memory ran out (a %u x %u frame needs %.2f GB, %.2f GB free)", info.width, info.height, (bytes + npx * 12) / 1e9, fr / 1e9);
        return 4;
    }
    CK(cudaMalloc(&d_src, bytes));
    CK(cudaMalloc(&d_rgb, npx * 3 * sizeof(float)));
    CK(cudaMemcpy(d_src, src, bytes, cudaMemcpyHostToDevice));
    convert_to_rgb_f32(d_src, info, d_rgb, nullptr, 0);
    DevDust dd;   // the dust map's correction, as in the passes
    struct DustTidy { DevDust& d; ~DustTidy() { d.free(); } } dust_tidy{dd};
    dd.upload(dust, (int)info.width, (int)info.height);
    dd.apply(d_rgb, 0);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    tidy.ok = true;
    return 0;
}
// ----------------------------------------------------------------------------- fold (pass 1)

// A live-view image copied off the GPU without holding up the work. The preview kernel runs on the
// preview's own stream once the source stream has produced the image (an event), the 10 MB copy to
// page-locked memory follows it there, and only the kernel is waited for (guard) before the source
// buffer is rewritten; the copy itself, slow while the uploads saturate the PCIe link, is nobody's
// business. The image goes to the hook when its owner (a slot, or the device for the "building" maps)
// next comes round and the copy has landed, or at the end of the pass; a frame whose predecessor is
// still in flight gets no view. The previews used to synchronise the slot stream inside the streamer's
// callback and copied on it: at the GUI's Medium live view the fusion took 13 s instead of 8 (FINDINGS 30).
struct AsyncPreview {
    float* d = nullptr;        // device: the small image
    float* h = nullptr;        // page-locked host copy
    cudaStream_t ps = nullptr;
    cudaEvent_t ready = nullptr, kdone = nullptr, ev = nullptr;   // source ready; kernel done; copy done
    bool pending = false, guarded = false;
    std::chrono::steady_clock::time_point last{};   // when the last preview was enqueued
    // Every preview copy, of any size, costs the uploads about 3 ms of link time (measured: Low and Medium
    // live views cost the same), so views are rate-limited per owner; the eye needs no more than this anyway.
    // Large views (Full: 365 MB each) are limited further, to about 1 GB/s per owner, or the GUI thread drowns.
    bool due(int min_ms) const {
        int ms = std::max(min_ms, (int)(bytes_ / 1000000));
        return std::chrono::steady_clock::now() - last >= std::chrono::milliseconds(ms);
    }
    size_t bytes_ = 0;
    int k = -1, pw = 0, ph = 0;
    const char* name = nullptr;   // a "building" preview's name; null = a frame view (hooks.frame)
    ViewWindow win;               // a frame view's window

    size_t capacity_px() const { return bytes_ / (3 * sizeof(float)); }
    // a frame view's buffer: the whole frame at frame_down, or view_max_px when the GUI asks for windows
    static size_t view_bytes(const Hooks& hooks, int w, int hgt) {
        return hooks.view_window ? hooks.view_max_px * 3 * sizeof(float) : bytes(w, hgt, hooks.frame_down);
    }
    void alloc_view(const Hooks& hooks, int w, int hgt) { alloc_bytes(view_bytes(hooks, w, hgt)); }
    void alloc(int w, int hgt, int down) {
        int pw_, ph_;
        preview_size(w, hgt, down, pw_, ph_);
        alloc_bytes((size_t)pw_ * ph_ * 3 * sizeof(float));
    }
    void alloc_bytes(size_t bytes) {
        bytes_ = bytes;
        // (not mapped host memory: a kernel writing the preview over the link made the fusion 4x slower)
        CK_THROW(cudaMalloc(&d, bytes));
        CK_THROW(cudaHostAlloc((void**)&h, bytes, cudaHostAllocPortable));
        CK_THROW(cudaStreamCreateWithFlags(&ps, cudaStreamNonBlocking));
        CK_THROW(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
        CK_THROW(cudaEventCreateWithFlags(&kdone, cudaEventDisableTiming));
        CK_THROW(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
    }
    static size_t bytes(int w, int hgt, int down) {
        int pw_, ph_;
        preview_size(w, hgt, down, pw_, ph_);
        return (size_t)pw_ * ph_ * 3 * sizeof(float);
    }
    void free() {
        if (ps) cudaStreamSynchronize(ps);
        if (d) cudaFree(d);
        if (h) cudaFreeHost(h);
        if (ps) cudaStreamDestroy(ps);
        for (cudaEvent_t e : {ready, kdone, ev}) if (e) cudaEventDestroy(e);
        d = h = nullptr;
        ps = nullptr;
        ready = kdone = ev = nullptr;
        pending = guarded = false;
    }
    // the source stream `src` has the image ready at this point; `launch(d_out, stream, pw, ph)` runs the kernel
    template <class Launch>
    void enqueue(cudaStream_t src, int k_, const char* name_, Launch launch, const ViewWindow& win_ = ViewWindow{}) {
        cudaEventRecord(ready, src);
        cudaStreamWaitEvent(ps, ready, 0);
        launch(d, ps, pw, ph);
        cudaEventRecord(kdone, ps);
        cudaMemcpyAsync(h, d, (size_t)pw * ph * 3 * sizeof(float), cudaMemcpyDeviceToHost, ps);
        cudaEventRecord(ev, ps);
        last = std::chrono::steady_clock::now();
        pending = guarded = true;
        k = k_; name = name_; win = win_;
    }
    // before `src` overwrites what the kernel reads: wait for the kernel (not the copy)
    void guard(cudaStream_t src) {
        if (guarded) cudaStreamWaitEvent(src, kdone, 0);
        guarded = false;
    }
    // hands the pending image to the hook, waiting for its copy if it is still in flight
    void flush(const Hooks& hooks) {
        if (!pending) return;
        cudaEventSynchronize(ev);
        deliver(hooks);
    }
    // the non-blocking form for the streaming callbacks (which must never wait on the GPU): delivers the
    // pending image if its copy has landed and says whether the buffer is free for the next one; a frame
    // whose predecessor is still in flight simply gets no live view
    bool tryFlush(const Hooks& hooks) {
        if (!pending) return true;
        if (cudaEventQuery(ev) != cudaSuccess) return false;
        deliver(hooks);
        return true;
    }

private:
    void deliver(const Hooks& hooks) {   // straight from the pinned buffer; the hook copies
        if (name) { if (hooks.preview) hooks.preview(name, h, pw, ph); }
        else if (hooks.frame) hooks.frame(k, h, pw, ph, win);
        pending = false;
    }
};

// Frame views (the live view's sweep, FINDINGS 47). One view per frame_min_ms across every slot and card, not per
// slot: the player shows 30 a second, and every copy off a GPU costs the uploads link time (FINDINGS 30), so views
// nobody will see are not taken. The window is the GUI's, or the whole frame at frame_down.
static std::atomic<long long> g_last_view_ms{0};
static bool view_due(const Hooks& hooks, AsyncPreview& fp) {
    if (!hooks.frame || !fp.tryFlush(hooks) || !fp.due(hooks.frame_min_ms)) return false;
    long long now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    long long last = g_last_view_ms.load();
    if (now - last < hooks.frame_min_ms) return false;
    g_last_view_ms.store(now);   // a race between two slots costs one extra view, nothing else
    return true;
}
static void enqueue_view(const Hooks& hooks, AsyncPreview& fp, cudaStream_t src, int k, const float* d_img, int W, int H) {
    ViewWindow vw = hooks.view_window ? hooks.view_window(W, H) : ViewWindow{};
    if (vw.w <= 0 || vw.h <= 0) vw = ViewWindow{0, 0, W, H, std::max(1, hooks.frame_down)};
    vw.x0 = std::clamp(vw.x0, 0, W - 1);
    vw.y0 = std::clamp(vw.y0, 0, H - 1);
    vw.w = std::clamp(vw.w, 1, W - vw.x0);
    vw.h = std::clamp(vw.h, 1, H - vw.y0);
    vw.down = std::max(1, vw.down);
    vw.fw = W; vw.fh = H;
    while ((size_t)std::max(1, vw.w / vw.down) * std::max(1, vw.h / vw.down) > fp.capacity_px()) ++vw.down;   // the buffer's capacity
    fp.enqueue(src, k, nullptr, [&](float* d, cudaStream_t ps, int& pw, int& ph) {
        image_preview_kernel(d_img, W, H, vw.x0, vw.y0, vw.w, vw.h, vw.down, d, pw, ph, ps);
    }, vw);
}

// The per-row corrections of Pass1Options on one device at the working scale (FINDINGS 51): K x work_h gains and
// offsets, one RowCorr per frame for load_aligned. At scale 0.5 two sensor rows make one working row (their gains
// averaged, their offsets averaged and halved). Nothing is allocated when the options carry no tables.
struct DevRowCorr {
    float* gain = nullptr;
    float2* off = nullptr;
    int hw = 0, K = 0;
    void upload(const Pass1Options& po, const Grid& g) {
        free();
        K = (int)po.T.size();
        hw = g.work_h();
        const int H = g.H;
        const bool have_g = K > 0 && (int)po.row_gain.size() == K * H;
        const bool have_o = K > 0 && (int)po.rs_dx.size() == K * H && (int)po.rs_dy.size() == K * H;
        if (!have_g && !have_o) return;
        const double step = (double)H / hw;   // sensor rows per working row: 1 or 2
        auto rows_of = [&](int r, int& y0, int& y1) { y0 = std::min(H - 1, (int)(r * step)); y1 = std::max(y0, std::min(H - 1, (int)((r + 1) * step) - 1)); };
        if (have_g) {
            std::vector<float> t((size_t)K * hw);
            for (int k = 0; k < K; ++k)
                for (int r = 0; r < hw; ++r) {
                    int y0, y1; rows_of(r, y0, y1);
                    double s = 0; for (int y = y0; y <= y1; ++y) s += po.row_gain[(size_t)k * H + y];
                    t[(size_t)k * hw + r] = (float)(s / (y1 - y0 + 1));
                }
            CK_THROW(cudaMalloc(&gain, t.size() * sizeof(float)));
            CK_THROW(cudaMemcpy(gain, t.data(), t.size() * sizeof(float), cudaMemcpyHostToDevice));
        }
        if (have_o) {
            const float sc = (float)(hw / (double)H);
            std::vector<float2> t((size_t)K * hw);
            for (int k = 0; k < K; ++k)
                for (int r = 0; r < hw; ++r) {
                    int y0, y1; rows_of(r, y0, y1);
                    double sx = 0, sy = 0;
                    for (int y = y0; y <= y1; ++y) { sx += po.rs_dx[(size_t)k * H + y]; sy += po.rs_dy[(size_t)k * H + y]; }
                    t[(size_t)k * hw + r] = make_float2((float)(sx / (y1 - y0 + 1)) * sc, (float)(sy / (y1 - y0 + 1)) * sc);
                }
            CK_THROW(cudaMalloc(&off, t.size() * sizeof(float2)));
            CK_THROW(cudaMemcpy(off, t.data(), t.size() * sizeof(float2), cudaMemcpyHostToDevice));
        }
    }
    RowCorr at(int k) const {
        RowCorr rc;
        if (gain && k >= 0 && k < K) rc.gain = gain + (size_t)k * hw;
        if (off && k >= 0 && k < K) rc.off = off + (size_t)k * hw;
        return rc;
    }
    void free() {
        cudaFree(gain); cudaFree(off);
        gain = nullptr; off = nullptr;
    }
};

struct FoldSlot {
    float *d_rgb = nullptr, *d_scratch = nullptr, *d_warp = nullptr;
    FoldScratch sc;
    VolumeScratch vsc;
    cudaEvent_t ready = nullptr, done = nullptr;
    bool used = false;
    AsyncPreview fp;   // the frame view
};

struct FoldDev {
    int dev = -1;
    cudaStream_t fold_s = nullptr;
    FoldState st;
    std::vector<FoldSlot> slots;
    std::vector<void*> fm16, lum16;   // per frame index: this GPU's volume slices (null = not here)
    double* d_energy = nullptr;       // per frame index: the sum of the frame's focus measure (fold_energy_add), this GPU's frames
    DevDust dust;      // the dust map's correction, when the options carry one (dust.h)
    DevRowCorr rc;                    // the per-row corrections of every frame (FINDINGS 51), when the options carry them
    AsyncPreview bp;                  // the detail map being built (first device only)
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

FrameSharpness frame_sharpness(const std::vector<double>& energy, double cutoff) {
    FrameSharpness S;
    const int K = (int)energy.size();
    S.energy = energy;
    S.ratio.assign(K, 0.0);
    S.soft.assign(K, 0);
    std::vector<int> run;   // the frames that were folded, in list order
    for (int k = 0; k < K; ++k) if (energy[k] > 0) { run.push_back(k); S.peak = std::max(S.peak, energy[k]); }
    const int R = (int)run.size();
    if (R < 4) return S;
    auto median = [](std::vector<double> v) { std::sort(v.begin(), v.end()); size_t n = v.size(); return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]); };
    for (int i = 0; i < R; ++i) {
        // the frame against the median of the five run frames centred on it (itself included): a frame far above
        // its neighbours (a jump in content while the stage still moved, a different exposure) leaves the median
        // alone instead of making its neighbours look soft, and two soft frames in a row are both still under it
        std::vector<double> nb;
        for (int j = i - 2; j <= i + 2; ++j) if (j >= 0 && j < R) nb.push_back(energy[run[j]]);
        double ref = median(nb);
        S.ratio[run[i]] = ref > 0 ? energy[run[i]] / ref : 0;
    }
    S.threshold = std::clamp(cutoff, 0.01, 0.999);
    for (int k : run) if (S.ratio[k] > 0 && S.ratio[k] < S.threshold) { S.soft[k] = 1; ++S.n_soft; }
    return S;
}


// the depth volume's grid and radii for an output grid
static VolumeParams volume_params(const Pass1Options& po, const Grid& g) {
    VolumeParams vp;
    double ds = po.depth_fraction / g.scale;   // depth res = fixed fraction of the sensor
    vp.dw = std::max(16, (int)std::nearbyint(g.out_w() * ds));
    vp.dh = std::max(16, (int)std::nearbyint(g.out_h() * ds));
    vp.fm_radius = (float)std::max(0.8, po.depth_fm_px * po.depth_fraction);
    vp.agg_r = (int)std::nearbyint(po.vol_agg_px * po.depth_fraction);
    if (po.vol_agg_px * po.depth_fraction < 1.0) vp.agg_r = 0;
    return vp;
}

// the share of the output rectangle the first GPU had room for in the last pass 1 (1 = all of it): run_stack sizes
// the strips from it
static std::atomic<double> g_slices_fit{1.0};

// FoldState::bytes() for a state not yet allocated
static size_t fold_state_bytes(int w, int h, int levels, bool with_a) {
    const std::vector<LevelShape> shapes = pyramid_shapes(w, h, levels);
    size_t b = (size_t)w * h * ((with_a ? 24 + 8 : 0) + 4 + 2 + 12 + 8 + 4);
    for (int l = 0; l < levels - 1; ++l) b += (size_t)shapes[l].w * shapes[l].h * (4 + 12 + 2);
    return b + (size_t)shapes.back().w * shapes.back().h * 24;
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
    // the list and the selection are checked before anything is indexed or handed to a kernel: a list can be
    // written by hand, and a transform that is not a number or far out of range has no meaning as an alignment
    if (ks.empty()) { logf("no frames to stack\n"); return 1; }
    if (T.size() != paths.size()) { logf("the frame list has %zu paths and %zu transforms\n", paths.size(), T.size()); return 1; }
    for (int k : ks)
        if (k < 0 || k >= (int)paths.size()) { logf("frame %d is not in the list (it has %zu frames)\n", k, paths.size()); return 1; }
    for (size_t k = 0; k < T.size(); ++k) {
        const double s = std::hypot(T[k].zr, T[k].zi);
        if (!std::isfinite(s) || !std::isfinite(T[k].tx) || !std::isfinite(T[k].ty) || s < 0.01 || s > 100 || std::abs(T[k].tx) > 1e6 || std::abs(T[k].ty) > 1e6) {
            logf("frame %zu (%s): its transform is not a usable alignment (scale %g, shift %g, %g)\n", k, paths[k].filename().string().c_str(), s, T[k].tx, T[k].ty);
            return 1;
        }
    }
    std::vector<fs::path> run_paths;
    for (int k : ks) run_paths.push_back(paths[k]);

    TiffInfo t0 = tiff_probe(paths[ks[0]]);
    if (!t0.ok()) { logf("%s: %s\n", paths[ks[0]].string().c_str(), t0.error.c_str()); return 1; }
    g.W = (int)t0.width;
    g.H = (int)t0.height;
    if (g.W < 16 || g.H < 16) { logf("the frames are %d x %d px: at least 16 x 16 is needed\n", g.W, g.H); return 1; }
    // the output rectangle lies inside the frame and is at least 16 px each way (the kernels index it unchecked)
    if (g.x0 < 0 || g.y0 < 0 || g.x1 >= g.W || g.y1 >= g.H || g.x1 - g.x0 + 1 < 16 || g.y1 - g.y0 + 1 < 16) {
        logf("the output rectangle %d,%d,%d,%d does not fit the %d x %d frame (or is under 16 px)\n", g.x0, g.y0, g.x1, g.y1, g.W, g.H);
        return 1;
    }
    int ow = g.out_w(), oh = g.out_h();
    FoldParams fp;
    fp.levels = n_levels(ow, oh);
    if (po.levels > 0) {   // a strip of a larger render: the whole frame's depth, as far as the strip's size carries it
        int most = 2;
        while ((std::min(ow, oh) >> most) >= 4) ++most;
        fp.levels = std::clamp(po.levels, 2, most);
    }
    fp.radius = (float)std::max(0.8, po.radius_px * g.scale);
    fp.power = (float)po.power;
    fp.c_radius = po.c_radius_px > 0 ? (float)(po.c_radius_px * g.scale) : 0.f;
    fp.with_a = po.with_a;
    VolumeParams vp = volume_params(po, g);
    const int K = (int)paths.size();
    P.g = g; P.fp = fp; P.vp = vp;

    if (std::string gp = gpu_problem(opt.devices); !gp.empty()) { logf("%s", gp.c_str()); return 1; }
    int ndev = 0;
    CK(cudaGetDeviceCount(&ndev));
    std::vector<int> devs = opt.devices;
    if (devs.empty()) for (int i = 0; i < ndev; ++i) devs.push_back(i);
    P.devs = devs;
    std::vector<FoldDev> ctx(ndev);
    // However this pass ends (finished, a CUDA error, an exception out of a kernel wrapper, a cancel), the cards get
    // back everything it allocated: the render that follows a pass that ran out of memory (in strips, or at 1/4)
    // needs that memory, and found the cards still full before
    struct Release {
        std::vector<FoldDev>& ctx; std::vector<int>& devs; const bool volume;
        ~Release() {
            for (auto& D : ctx) {
                if (D.dev < 0) continue;
                cudaSetDevice(D.dev);
                // a driver that has run out of page-locked memory answers every call with "out of memory" and never
                // finishes what is queued: nothing can be freed then, and a synchronize would not return (FINDINGS 54)
                size_t fr_ = 0, tot_ = 0;
                if (cudaMemGetInfo(&fr_, &tot_) != cudaSuccess) { D.dev = -1; continue; }
                if (D.fold_s) cudaStreamSynchronize(D.fold_s);
                for (auto& S : D.slots) {
                    cudaFree(S.d_rgb); cudaFree(S.d_scratch); cudaFree(S.d_warp); S.sc.free();
                    if (volume) S.vsc.free();
                    if (S.ready) cudaEventDestroy(S.ready);
                    if (S.done) cudaEventDestroy(S.done);
                    S.fp.free();
                }
                D.slots.clear();
                for (auto& p : D.fm16) if (p) { cudaFree(p); p = nullptr; }
                for (auto& p : D.lum16) if (p) { cudaFree(p); p = nullptr; }
                D.bp.free();
                D.st.free();
                D.rc.free();
                D.dust.free();
                cudaFree(D.d_energy);
                D.d_energy = nullptr;
                if (D.fold_s) cudaStreamDestroy(D.fold_s);
                D.fold_s = nullptr;
                D.dev = -1;
            }
            cudaGetLastError();
            if (!devs.empty()) cudaSetDevice(devs[0]);
        }
    } release{ctx, devs, volume};
    size_t npx_full = (size_t)g.W * g.H, npx_out = (size_t)ow * oh;
    opt.streams_by_device.clear();
    auto t_setup = std::chrono::steady_clock::now();
    std::vector<int> dropped;   // cards that cannot hold the volume slices beside their state (the 3070 at 1/2 resolution)
    std::vector<int> caps;      // per kept card: the most frames it may take (0 = any number)
    long long room_total = 0;   // the frames whose volume slices the kept cards hold between them
    for (int d : devs) {
        auto& D = ctx[d];
        D.dev = d;
        CK(cudaSetDevice(d));
        // slots per GPU from the memory left after the state: each needs the frame buffers,
        // the measure scratch and the streamer's raw upload buffer, plus a working margin
        FoldScratch probe;
        probe.w = ow; probe.h = oh; probe.levels = fp.levels; probe.shapes = pyramid_shapes(ow, oh, fp.levels);
        // what a slot holds whatever the output rectangle (the frame as read and as float RGB), and what follows it
        const size_t slot_fixed = npx_full * 12 + (g.scale != 1.0 ? (size_t)g.work_w() * g.work_h() * 12 : 0) + t0.pixel_bytes() + 2 * kReadAlign;
        size_t slot_bytes = npx_full * 12 + (g.scale != 1.0 ? (size_t)g.work_w() * g.work_h() * 12 : 0) + npx_out * 12 + probe.bytes();
        if (hooks.frame) slot_bytes += AsyncPreview::view_bytes(hooks, ow, oh);
        size_t per_slot = slot_bytes + t0.pixel_bytes() + 2 * kReadAlign;
        const size_t slice = volume ? (size_t)vp.dw * vp.dh * 4 : 0;
        size_t fr, tot;
        CK(cudaMemGetInfo(&fr, &tot));
        // Nothing is allocated that the card does not have: past its free memory the Windows driver hands out
        // system memory instead of failing, and with tens of GB of frames page-locked that ended with every CUDA
        // call refused for the rest of the process (FINDINGS 54, 55). The state and one slot are the least a card
        // needs; the share of the rectangle that would fit (the state, the slot's output-sized part and the slices
        // all scale with its area) tells run_stack how many strips to cut
        const size_t state_bytes = fold_state_bytes(ow, oh, fp.levels, fp.with_a);
        {
            const double room = (double)fr - (double)(512u << 20) - (double)slot_fixed;
            const double area = (double)state_bytes + (double)(per_slot - slot_fixed) + (double)slice * K;
            const bool sole = d == devs[0] || devs.size() - dropped.size() == 1;
            if (sole && area > 0) g_slices_fit = std::clamp(room / area, 0.02, 1.0);
            if ((double)fr < (double)state_bytes + (double)per_slot + (double)(512u << 20)) {
                if (sole) {
                    logf("GPU %d: %.2f GB free, short of the %.2f GB for the %d x %d fold state and one frame slot\n", d, fr / 1e9,
                         (state_bytes + per_slot) / 1e9, ow, oh);
                    return 4;
                }
                logf("GPU %d: left out of this job, not enough memory for the fold state and one frame slot (%.2f GB free)\n", d, fr / 1e9);
                D.dev = -1;
                dropped.push_back(d);
                continue;
            }
        }
        CK(cudaStreamCreateWithFlags(&D.fold_s, cudaStreamNonBlocking));
        D.st.alloc(d, ow, oh, fp.levels, fp.with_a);
        D.rc.upload(po, g);
        D.dust.upload(po.dust.get(), g.W, g.H);
        D.fm16.assign(K, nullptr);
        D.lum16.assign(K, nullptr);
        CK(cudaMalloc(&D.d_energy, (size_t)K * sizeof(double)));
        CK(cudaMemset(D.d_energy, 0, (size_t)K * sizeof(double)));
        CK(cudaMemGetInfo(&fr, &tot));
        // the depth-volume slices stay on the GPU until the aggregation (4 bytes per depth pixel per frame). A
        // card keeps as many slots as leave room for every frame's slice. Failing that, the first card runs on
        // one slot regardless (the depth stage needs it), and another card takes a capped share of the frames
        // (what its slice room allows beside one slot) or, if that is under a quarter of the stack, sits the
        // job out rather than run out of memory mid-pass (the 3070 at 1/2 resolution; FINDINGS 32)
        const size_t avail = fr > (512u << 20) ? fr - (512u << 20) : 0;
        int S = 0, cap = 0, holds = K;   // holds: the frames whose slices this card has room for
        for (int s = opt.streams_per_gpu; s >= 1 && S == 0; --s)
            if (avail >= (size_t)s * per_slot + slice * K) S = s;
        if (S == 0) {
            size_t room = avail > per_slot ? avail - per_slot : 0;
            int frames_fit = slice ? (int)std::min<size_t>((size_t)K, room / slice) : K;
            holds = frames_fit;
            if (d == devs[0] || devs.size() - dropped.size() == 1) {
                S = 1;
                if (frames_fit < K) {
                    logf("GPU %d: short of memory for the volume slices (%.2f GB free): %d of %d frames fit\n", d, fr / 1e9, frames_fit, K);
                    // One GPU holds every slice, so this pass would run out of memory part-way: stop before the slots
                    // are allocated and the frames read, and let the caller render in strips. With a second GPU the
                    // slices are shared and the pass fits, but the depth stage then puts the whole volume on this
                    // card: with 92 % of the frames fitting here (241 of 261, 1.0 GB free before the SGM) eight
                    // renders of eight went through; with 86 % (225 of 261, the same stack with output A, 0.5 GB
                    // free) renders ended with the driver refusing every call (FINDINGS 54). Under nine tenths the
                    // render goes to strips as well
                    // (that rule was on from 9b5c54b and is off again: with the GUI's own buffers on the card the
                    // everyday 261-frame render with seven outputs fell under nine tenths and took twice as long in
                    // strips, where it had always rendered whole)
                    if (po.stop_if_short && devs.size() == 1) return 4;
                }
            } else if (frames_fit >= K / 4) {
                S = 1;
                cap = frames_fit;
                logf("GPU %d: takes at most %d of %d frames (%.2f GB of room for the volume slices beside one slot)\n", d, cap, K, room / 1e9);
            } else {
                logf("GPU %d: left out of this job, not enough memory for the volume slices plus one slot (%.2f GB free)\n", d, fr / 1e9);
                D.st.free();
                D.rc.free();
                D.dust.free();
                cudaFree(D.d_energy);
                D.d_energy = nullptr;
                cudaStreamDestroy(D.fold_s);
                D.fold_s = nullptr;
                dropped.push_back(d);
                continue;
            }
        }
        const int fit = S;
        room_total += holds;
        opt.streams_by_device.push_back(S);
        caps.push_back(cap);
        D.slots.resize(S);
        for (auto& S : D.slots) {
            CK(cudaMalloc(&S.d_rgb, npx_full * 3 * sizeof(float)));
            if (g.scale != 1.0) CK(cudaMalloc(&S.d_scratch, (size_t)g.work_w() * g.work_h() * 3 * sizeof(float)));
            CK(cudaMalloc(&S.d_warp, npx_out * 3 * sizeof(float)));
            S.sc.alloc(ow, oh, fp.levels, fp.radius, fp.c_radius);
            if (volume) S.vsc.alloc(ow, oh, vp);
            CK(cudaEventCreateWithFlags(&S.ready, cudaEventDisableTiming));
            CK(cudaEventCreateWithFlags(&S.done, cudaEventDisableTiming));
            if (hooks.frame) S.fp.alloc_view(hooks, ow, oh);
        }
        if (hooks.preview && d == devs[0]) D.bp.alloc(ow, oh, hooks.preview_down);
        CK(cudaMemGetInfo(&fr, &tot));
        logf("GPU %d: state %.2f GB + %d slot(s) x %.2f GB (%d requested, %d fit); %.2f of %.2f GB free after allocation\n", d,
               D.st.bytes() / 1e9, S, per_slot / 1e9, opt.streams_per_gpu, fit, fr / 1e9, tot / 1e9);
    }
    if (!dropped.empty()) {   // the rest of the job (merge, depth, passes 2 and 3) runs on the cards that stayed
        std::vector<int> keep;
        for (int d : devs) if (std::find(dropped.begin(), dropped.end(), d) == dropped.end()) keep.push_back(d);
        devs = keep;
        P.devs = devs;
    }
    opt.devices = devs;   // streams_by_device and the caps are parallel to this list
    opt.max_frames_by_device = caps;
    if (volume && po.stop_if_short && room_total < K) {
        // the cards together cannot hold every frame's slice: the pass would run out of memory part-way, so it stops
        // here, before a frame is read, and the caller renders in strips
        logf("the GPUs hold the volume slices of %lld of %d frames between them: not in one piece\n", room_total, K);
        g_slices_fit = room_total / (double)K;
        return 4;
    }

    auto on_frame = [&](const DeviceFrame& f) {
        auto& D = ctx[f.dev];
        auto& S = D.slots[f.slot];
        const bool view = view_due(hooks, S.fp);   // the previous view has landed and it is time
        if (S.used) cudaStreamWaitEvent(f.stream, S.done, 0);   // the fold stream must be done with this slot
        S.fp.guard(f.stream);                                    // and the previous view's kernel with d_warp
        convert_to_rgb_f32(f.d_pixels(), *f.info, S.d_rgb, nullptr, f.stream);
        D.dust.apply(S.d_rgb, f.stream);   // the frame without its dust, before anything else sees it
        int k = ks[f.k];
        load_aligned(S.d_rgb, g, T[k], S.d_scratch, S.d_warp, false, f.stream, D.rc.at(k));
        fold_measure(S.d_warp, fp, S.sc, f.stream);
        fold_energy_add(S.sc.fm, npx_out, D.d_energy + k, f.stream);   // the frame's focus energy (soft-frame flag, FINDINGS 49)
        if (volume) {   // this frame's depth-resolution slices stay on this GPU until the aggregation
            size_t nb = (size_t)vp.dw * vp.dh * 2;
            // a failure here must stop the pass, not fault a kernel; the streamer reports it and run_pass1 turns
            // "out of memory" into the 1/4-resolution retry
            CK_THROW(cudaMallocAsync(&D.fm16[k], nb, f.stream));
            if (cudaError_t e = cudaMallocAsync(&D.lum16[k], nb, f.stream); e != cudaSuccess) {
                cudaFreeAsync(D.fm16[k], f.stream);
                D.fm16[k] = nullptr;
                throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(e));
            }
            volume_slice(S.d_warp, vp, S.vsc, D.fm16[k], D.lum16[k], f.stream);
        }
        if (view) enqueue_view(hooks, S.fp, f.stream, k, S.d_warp, ow, oh);   // the frame just aligned, for the live view (delivered at the slot's next turn)
        cudaEventRecord(S.ready, f.stream);
        cudaStreamWaitEvent(D.fold_s, S.ready, 0);
        D.bp.guard(D.fold_s);   // the previous detail-map preview must have read the state before it changes
        fold_accumulate(D.st, k, S.d_warp, fp, S.sc, D.fold_s);
        cudaEventRecord(S.done, D.fold_s);
        S.used = true;
        int done = ++frames_done;
        if (hooks.progress) hooks.progress("pass 1: fold", (double)done / (double)run_paths.size());
        // live preview: this GPU's best focus measure so far at a fraction of the output grid
        if (hooks.preview && f.dev == devs[0] && (done % hooks.preview_every == 0) && D.bp.tryFlush(hooks) && D.bp.due(hooks.preview_min_ms))
            D.bp.enqueue(D.fold_s, -1, "focus (building)", [&](float* d, cudaStream_t ps, int& pw, int& ph) { fold_focus_preview_kernel(D.st, hooks.preview_down, d, pw, ph, ps); });
    };

    logf("fold: %zu frames, grid %dx%d -> %dx%d at scale %g, %d levels, blur sigma %.2f px, %zu GPU(s)\n",
           run_paths.size(), g.W, g.H, ow, oh, g.scale, fp.levels, fp.radius, devs.size());
    P.setup_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_setup).count();
    opt.cancel = hooks.cancel;
    StreamerStats st = stream_frames(run_paths, opt, on_frame);
    for (auto& e : st.errors) logf("  error: %s\n", e.c_str());
    // a card ran out of memory part-way: the pass is not used (the caller renders again in strips or at 1/4), and
    // the streams are left as they are, since after the driver's page-locked ceiling they never finish
    for (auto& e : st.errors) if (strstr(e.c_str(), "out of memory")) return 4;
    for (int d : devs) {
        CK(cudaSetDevice(d));
        CK(cudaStreamSynchronize(ctx[d].fold_s));
        CK(cudaGetLastError());
        for (auto& S : ctx[d].slots) S.fp.flush(hooks);   // the last frame of every slot
        ctx[d].bp.flush(hooks);
    }
    logf("%d frames in %.3f s: %.1f frames/s, %.2f GB/s%s\n", st.frames, st.seconds, st.frames / st.seconds,
           st.bytes / st.seconds / 1e9, source_note(st).c_str());
    for (size_t i = 0; i < devs.size(); ++i) printf("  GPU %d: %d frames\n", devs[i], st.frames_per_device[i]);
    // every frame folded on exactly one card: the per-frame focus energies add up across the cards (0 elsewhere)
    P.frame_energy.assign(K, 0.0);
    {
        std::vector<double> e(K);
        for (int d : devs) {
            CK(cudaSetDevice(d));
            CK(cudaMemcpy(e.data(), ctx[d].d_energy, (size_t)K * sizeof(double), cudaMemcpyDeviceToHost));
            for (int k = 0; k < K; ++k) P.frame_energy[k] += e[k] / (double)npx_out;
            cudaFree(ctx[d].d_energy);
            ctx[d].d_energy = nullptr;
        }
    }
    // the slots (frame, warp and measure buffers, 2.2 GB each) are not needed past the streaming: free them now so
    // the volume aggregation and the depth stage have the memory (a 1/2-resolution volume's slices are 4.9 GB)
    for (int d : devs) {
        auto& D = ctx[d];
        CK(cudaSetDevice(d));
        for (auto& S : D.slots) {
            cudaFree(S.d_rgb); cudaFree(S.d_scratch); cudaFree(S.d_warp); S.sc.free();
            if (volume) S.vsc.free();
            cudaEventDestroy(S.ready); cudaEventDestroy(S.done);
            S.fp.free();
        }
        D.slots.clear();
    }

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
        // in page-locked memory, so the copies stream behind the kernels instead of staging one slice at a
        // time (0.59 -> ~0.3 s). The raw measure is only kept for the fold command's validation files.
        PinnedVec<uint16_t>& fm_all = P.fm_all;
        PinnedVec<uint16_t>& lum_all = P.lum_all;
        PinnedVec<float>& V_all = P.V_all;
        const bool keep_fm = po.host_state;
        // ordinary memory, not page-locked: with the frames resident (50 GB for 261 16-bit frames) another 11 GB of
        // pinned volume took the two-GPU job past what the driver can map for the second card, after which every
        // CUDA call in the process fails; the copies are no slower for it (FINDINGS 54)
        if (keep_fm) fm_all.resize(nd * K, false);
        lum_all.resize(nd * K, false);
        V_all.resize(nd * K, false);
        std::vector<char> filled(K, 0);   // a frame no GPU saw (probe failed) leaves its plane to be zeroed below
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
            // flat regions (no texture in the guide within flat_reach_px): every slice box-blurred over flat_blur_px
            // there, so defocused background gets one consistent focus curve instead of a noisy one per pixel
            float *d_flat = nullptr, *d_Vb = nullptr, *d_T2 = nullptr;
            DevGaussian gr1, gr2;   // the flat mask's two kernels (freed after the stage's sync)
            int rbox = 0;
            if (po.flat_blur_px > 0) {
                CK(cudaMalloc(&d_flat, nd * sizeof(float)));
                CK(cudaMalloc(&d_Vb, nd * sizeof(float)));
                CK(cudaMalloc(&d_T2, nd * sizeof(float)));
                // texture of the guide at the scale of the reach: the guide blurred first (so its noise does not
                // count as texture), |Laplacian|, spread over the reach; ranked over the image, the flattest fifth
                // is fully flat and anything above the midpoint to the 90th percentile is textured
                // two kernels, freed after the stage's sync: freeing (and rebuilding at the same address) right after
                // the launch let the first blur read the second kernel's taps, which made the flat mask, and so the
                // depth map, differ run to run with two streams and the frame cache (FINDINGS 38)
                // STACKER_FLAT_DEBUG=DIR: the chain's stages on the first GPU, snapshotted on the stream (device copies,
                // no extra host syncs that would change the timing) and written to DIR as .f32 after the stage
                const char* dbg = getenv("STACKER_FLAT_DEBUG");
                const char* dbg_names[5] = {"ga", "blur1", "lap", "blur2", "flat"};
                float* dbg_buf = nullptr;
                int dbg_n = 0;
                if (dbg && d == devs[0]) CK(cudaMalloc(&dbg_buf, 5 * nd * sizeof(float)));
                auto snap = [&](const float* d_src) {
                    if (dbg_buf) CK(cudaMemcpyAsync(dbg_buf + (size_t)dbg_n++ * nd, d_src, nd * sizeof(float), cudaMemcpyDeviceToDevice, s));
                };
                snap(d_g);
                gr1.build(std::max(1.0, po.flat_reach_px * po.depth_fraction / 4));
                gaussian_blur(d_g, d_flat, d_T2, vp.dw, vp.dh, gr1, s);
                snap(d_flat);
                laplacian3_abs(d_flat, d_Vb, vp.dw, vp.dh, s);
                snap(d_Vb);
                gr2.build(std::max(1.0, po.flat_reach_px * po.depth_fraction));
                gaussian_blur(d_Vb, d_flat, d_T2, vp.dw, vp.dh, gr2, s);
                snap(d_flat);
                float lo = percentile(d_flat, d_T2, nd, 20.0, s), hi = percentile(d_flat, d_T2, nd, 90.0, s);
                if (po.flat_lo >= 0 && po.flat_hi > po.flat_lo) { lo = po.flat_lo; hi = po.flat_hi; }   // the whole frame's, for a strip
                flatness(d_flat, lo, std::max(0.5f * (hi - lo), 1e-6f), d_flat, nd, s);
                snap(d_flat);
                if (dbg_buf) {
                    CK(cudaStreamSynchronize(s));
                    std::vector<float> h(nd);
                    for (int i = 0; i < dbg_n; ++i) {
                        CK(cudaMemcpy(h.data(), dbg_buf + (size_t)i * nd, nd * sizeof(float), cudaMemcpyDeviceToHost));
                        write_raw(fs::path(dbg) / (std::string(dbg_names[i]) + ".f32"), h.data(), nd);
                    }
                    cudaFree(dbg_buf);
                    logf("flat debug: lo %.9g hi %.9g R %d / %d, %d stages -> %s\n", lo, hi, gr1.R, gr2.R, dbg_n, dbg);
                }
                rbox = std::max(1, (int)std::nearbyint(po.flat_blur_px * po.depth_fraction));
                if (d == devs[0]) {   // for inspection: the fusion writes it as flat.f32 when it writes the depth
                    P.flat.resize(nd);
                    CK(cudaMemcpyAsync(P.flat.data(), d_flat, nd * sizeof(float), cudaMemcpyDeviceToHost, s));
                    CK(cudaStreamSynchronize(s));
                }
            }
            for (int k = 0; k < K; ++k) {
                if (!D.fm16[k]) continue;
                if (vp.agg_r > 0) aggregate_slice(d_g, gs, D.fm16[k], d_V, vp.agg_eps, s);
                else { f16_to_f32(D.fm16[k], d_V, nd, s); relu_sqrt(d_V, nd, s); }
                if (b_blur) gaussian_blur(d_V, d_V, d_T, vp.dw, vp.dh, gb, s);
                if (d_flat) {
                    box_filter(d_V, d_Vb, d_T2, vp.dw, vp.dh, rbox, s);
                    lerp_mask(d_V, d_Vb, d_flat, d_V, nd, s);
                }
                // stream order: the copies read d_V and the slices before the next slice's kernels and the frees
                CK(cudaMemcpyAsync(V_all.data() + nd * k, d_V, nd * sizeof(float), cudaMemcpyDeviceToHost, s));
                if (keep_fm) CK(cudaMemcpyAsync(fm_all.data() + nd * k, D.fm16[k], nd * 2, cudaMemcpyDeviceToHost, s));
                CK(cudaMemcpyAsync(lum_all.data() + nd * k, D.lum16[k], nd * 2, cudaMemcpyDeviceToHost, s));
                cudaFreeAsync(D.fm16[k], s);
                cudaFreeAsync(D.lum16[k], s);
                D.fm16[k] = D.lum16[k] = nullptr;
                filled[k] = 1;
                ++done;
            }
            CK(cudaStreamSynchronize(s));
            if (vp.agg_r > 0) gs.free();
            if (b_blur) { gb.free(); cudaFree(d_T); }
            if (d_flat) { cudaFree(d_flat); cudaFree(d_Vb); cudaFree(d_T2); gr1.free(); gr2.free(); }
            cudaFree(d_g);
            cudaFree(d_V);
        }
        for (int k = 0; k < K; ++k)
            if (!filled[k]) {
                memset(V_all.data() + nd * k, 0, nd * sizeof(float));
                memset(lum_all.data() + nd * k, 0, nd * 2);
                if (keep_fm) memset(fm_all.data() + nd * k, 0, nd * 2);
            }
        P.vol_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        P.guide = std::move(guide_h);
        P.ga = std::move(ga_h);
        P.slices = done;
    }

    if (st.errors.empty()) return 0;   // (`release` frees the cards' buffers on every way out)
    // a GPU that ran out of memory for the depth slices (the most likely place at 1/2 resolution): tell run_stack
    // to render again at 1/4 instead of failing the job
    return stream_fail(st);   // 4 for a card out of memory, 1 with the cause noted
}

// The flat mask's two texture thresholds from a merged fold state on the current device: the guide and the chain of
// run_pass1's volume stage (guide blurred, |Laplacian|, spread over the reach, the 20th and 90th percentiles), without
// the volume. The strips take them from a fold of the whole frame.
static int flat_limits(const FoldState& st, const Pass1Options& po, const VolumeParams& vp, int ow, int oh, cudaStream_t s, float& lo, float& hi) {
    const size_t npx_out = (size_t)ow * oh, nd = (size_t)vp.dw * vp.dh;
    struct DevFree { std::vector<void*> p; ~DevFree() { for (void* q : p) cudaFree(q); } } dfree;
    auto dev = [&](size_t bytes, float*& d) -> cudaError_t {
        size_t fr = 0, tot = 0;   // not past the card's free memory (FINDINGS 55)
        if (cudaMemGetInfo(&fr, &tot) == cudaSuccess && fr < bytes + ((size_t)256 << 20)) return cudaErrorMemoryAllocation;
        cudaError_t e = cudaMalloc(&d, bytes);
        if (e == cudaSuccess) dfree.p.push_back(d);
        return e;
    };
    float *d_base, *d_coll, *d_t1, *d_t2, *d_guide, *d_ga, *d_flat, *d_lap, *d_tmp;
    CK_MEM(dev((size_t)st.shapes.back().w * st.shapes.back().h * 3 * sizeof(float), d_base));
    CK_MEM(dev(npx_out * 3 * sizeof(float), d_coll));
    CK_MEM(dev(npx_out * 3 * sizeof(float), d_t1));
    CK_MEM(dev(npx_out * 3 * sizeof(float), d_t2));
    CK_MEM(dev(npx_out * sizeof(float), d_guide));
    CK_MEM(dev(nd * sizeof(float), d_ga));
    CK_MEM(dev(nd * sizeof(float), d_flat));
    CK_MEM(dev(nd * sizeof(float), d_lap));
    CK_MEM(dev(nd * sizeof(float), d_tmp));
    std::vector<const float*> d_coef(st.c_coef.begin(), st.c_coef.end());
    fold_base_f32(st, d_base, s);
    collapse_pyramid(d_coef, d_base, st.shapes, d_coll, d_t1, d_t2, s);
    luma(d_coll, d_guide, npx_out, s);
    clip01(d_guide, npx_out, s);
    AreaResizer rz;
    rz.build(ow, oh, vp.dw, vp.dh);
    rz.run(d_guide, d_ga, 1, s);
    DevGaussian gr1, gr2;   // both alive until the stream has run (FINDINGS 38)
    gr1.build(std::max(1.0, po.flat_reach_px * po.depth_fraction / 4));
    gr2.build(std::max(1.0, po.flat_reach_px * po.depth_fraction));
    gaussian_blur(d_ga, d_flat, d_tmp, vp.dw, vp.dh, gr1, s);
    laplacian3_abs(d_flat, d_lap, vp.dw, vp.dh, s);
    gaussian_blur(d_lap, d_flat, d_tmp, vp.dw, vp.dh, gr2, s);
    lo = percentile(d_flat, d_tmp, nd, 20.0, s);
    hi = percentile(d_flat, d_tmp, nd, 90.0, s);
    cudaError_t e = cudaStreamSynchronize(s);
    rz.free();
    gr1.free();
    gr2.free();
    CK(e);
    return 0;
}

// ----------------------------------------------------------------------------- stack: the whole pipeline

// A streaming pass after pass 1: per slot convert + warp, then a per-slot stage on the slot stream and a
// per-device accumulate stage on the device's fold stream (the same event hand-over as pass 1).
struct PassSlot {
    float *d_rgb = nullptr, *d_scratch = nullptr, *d_warp = nullptr;
    cudaEvent_t ready = nullptr, done = nullptr;
    bool used = false;
    int index = 0;   // slot index within its device
    AsyncPreview fp;   // the frame view
};
struct PassDev {
    int dev = -1;
    cudaStream_t fold_s = nullptr;
    std::vector<PassSlot> slots;
    DevDust dust;      // the dust map's correction, when the options carry one (dust.h)
    DevRowCorr rc;     // the per-row corrections of every frame (FINDINGS 51), when the options carry them
    AsyncPreview bp;   // the map being built (first device only); the accumulate stage enqueues into it
};

template <class SlotStage, class AccStage>
static StreamerStats run_pass(const std::vector<fs::path>& run_paths, const std::vector<int>& ks, const std::vector<Similarity>& T,
                              const Grid& g, const Pass1Options& po, const std::vector<int>& devs, int streams, int readers, FrameCache* cache,
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
    // Whatever ends the pass, a throw from the set-up included, the cards' buffers go: a failed attempt that kept
    // them left the render in strips or at 1/4 that follows it with gigabytes less to work in. On the way out of a
    // failure the handles that were never created are "freed" too, so the error state is cleared afterwards.
    struct Release {
        std::vector<PassDev>& ctx; const std::vector<int>& devs; bool ok = false;
        ~Release() {
            for (int d : devs) {
                if (d < 0 || d >= (int)ctx.size()) continue;
                auto& D = ctx[d];
                if (cudaSetDevice(d) != cudaSuccess) continue;
                for (auto& S : D.slots) {
                    cudaFree(S.d_rgb); cudaFree(S.d_scratch); cudaFree(S.d_warp);
                    if (S.ready) cudaEventDestroy(S.ready);
                    if (S.done) cudaEventDestroy(S.done);
                    S.fp.free();
                }
                D.slots.clear();
                D.bp.free();
                D.rc.free();
                D.dust.free();
                if (D.fold_s) cudaStreamDestroy(D.fold_s);
                D.fold_s = nullptr;
            }
            if (!ok) cudaGetLastError();
        }
    } release{ctx, devs};
    // Slots by the memory each card has free now, as pass 1 counts them (a frame as read, the warp's scratch and
    // the warped frame per slot): the fixed count ran a small card past its memory, where Windows pages GPU memory
    // to the host instead of refusing. A card without room for one slot ends the attempt as out of memory.
    const size_t slot_bytes = npx_full * 12 + (g.scale != 1.0 ? (size_t)g.work_w() * g.work_h() * 12 : 0) + npx_out * 12 +
                              (hooks.frame ? npx_out * 12 : 0);
    opt.streams_by_device.assign(devs.size(), streams);
    for (size_t i = 0; i < devs.size(); ++i) {
        size_t fr = 0, tot = 0;
        CK_THROW(cudaSetDevice(devs[i]));
        CK_THROW(cudaMemGetInfo(&fr, &tot));
        const size_t keep = (size_t)384 << 20;   // the kernels' own scratch and the previews
        const int fit = fr > keep ? (int)std::min<size_t>((size_t)streams, (fr - keep) / slot_bytes) : 0;
        if (fit < 1) {
            logf("GPU %d memory ran out (%s needs %.2f GB for one frame slot, %.2f GB free)", devs[i], what, slot_bytes / 1e9, fr / 1e9);
            throw std::runtime_error("GPU memory ran out for a frame slot");
        }
        if (fit < streams) logf("  %s: GPU %d takes %d stream%s (%.2f GB free)", what, devs[i], fit, fit == 1 ? "" : "s", fr / 1e9);
        opt.streams_by_device[i] = fit;
    }
    for (size_t di = 0; di < devs.size(); ++di) {
        const int d = devs[di];
        auto& D = ctx[d];
        D.dev = d;
        CK_THROW(cudaSetDevice(d));
        CK_THROW(cudaStreamCreateWithFlags(&D.fold_s, cudaStreamNonBlocking));
        D.rc.upload(po, g);
        D.dust.upload(po.dust.get(), g.W, g.H);
        D.slots.resize(opt.streams_by_device[di]);
        int idx = 0;
        for (auto& S : D.slots) {
            S.index = idx++;
            CK_THROW(cudaMalloc(&S.d_rgb, npx_full * 3 * sizeof(float)));
            if (g.scale != 1.0) CK_THROW(cudaMalloc(&S.d_scratch, (size_t)g.work_w() * g.work_h() * 3 * sizeof(float)));
            CK_THROW(cudaMalloc(&S.d_warp, npx_out * 3 * sizeof(float)));
            CK_THROW(cudaEventCreateWithFlags(&S.ready, cudaEventDisableTiming));
            CK_THROW(cudaEventCreateWithFlags(&S.done, cudaEventDisableTiming));
            if (hooks.frame) S.fp.alloc_view(hooks, g.out_w(), g.out_h());
        }
        if (hooks.preview && d == devs[0]) D.bp.alloc(g.out_w(), g.out_h(), hooks.preview_down);
    }
    auto on_frame = [&](const DeviceFrame& f) {
        auto& D = ctx[f.dev];
        auto& S = D.slots[f.slot];
        const bool view = view_due(hooks, S.fp);
        if (S.used) cudaStreamWaitEvent(f.stream, S.done, 0);
        S.fp.guard(f.stream);
        convert_to_rgb_f32(f.d_pixels(), *f.info, S.d_rgb, nullptr, f.stream);
        D.dust.apply(S.d_rgb, f.stream);   // the frame without its dust, before anything else sees it
        int k = ks[f.k];
        load_aligned(S.d_rgb, g, T[k], S.d_scratch, S.d_warp, false, f.stream, D.rc.at(k));
        slot_stage(f.dev, S, k, f.stream);
        if (view) enqueue_view(hooks, S.fp, f.stream, k, S.d_warp, g.out_w(), g.out_h());
        cudaEventRecord(S.ready, f.stream);
        cudaStreamWaitEvent(D.fold_s, S.ready, 0);
        D.bp.guard(D.fold_s);   // the previous building preview must have read the state before it changes
        acc_stage(f.dev, S, k, D.fold_s);
        cudaEventRecord(S.done, D.fold_s);
        S.used = true;
        if (hooks.progress) hooks.progress(what, (double)(++done) / (double)run_paths.size());
    };
    StreamerStats st = stream_frames(run_paths, opt, on_frame);
    for (auto& e : st.errors) logf("  error: %s\n", e.c_str());
    for (int d : devs) {
        CK_THROW(cudaSetDevice(d));
        CK_THROW(cudaStreamSynchronize(ctx[d].fold_s));
        CK_THROW(cudaGetLastError());
        for (auto& S : ctx[d].slots) S.fp.flush(hooks);
        ctx[d].bp.flush(hooks);
    }
    std::string split;
    for (size_t i = 0; i < devs.size(); ++i) split += (i ? ", GPU " : "; GPU ") + std::to_string(devs[i]) + " " + std::to_string(st.frames_per_device[i]);
    logf("  %s: %d frames in %.2f s (%.1f frames/s%s)%s", what, st.frames, st.seconds, st.frames / st.seconds, source_note(st).c_str(),
         split.c_str());
    release.ok = true;   // (the buffers are freed by `release`, as on every other way out)
    return st;
}

// Upper limits on the size parameters, far above any useful value (the GUI's own ranges end well below them):
// a radius of 10^9 px typed on the command line ran a kernel for minutes instead of failing, and 0 streams
// indexed an empty slot list. A value that is not a number takes the lower end.
static void limit_stack_options(StackOptions& so) {
    auto lim = [](const char* name, double& v, double hi) {
        if (v != v) { logf("  %s is not a number: 0 is used", name); v = 0; }
        if (v > hi) { logf("  %s is limited to %g", name, hi); v = hi; }
    };
    Pass1Options& po = so.po;
    lim("--c-radius", po.c_radius_px, 1024);
    lim("--b-radius", po.b_radius_px, 1024);
    lim("--radius-px", po.radius_px, 128);
    lim("--power", po.power, 64);
    lim("--depth-fm-px", po.depth_fm_px, 128);
    lim("--vol-agg-px", po.vol_agg_px, 1024);
    lim("--flat-blur", po.flat_blur_px, 2048);
    lim("--flat-reach", po.flat_reach_px, 1024);
    lim("--refine-fm-px", so.refine_fm_px, 128);
    lim("--soft-px", so.soft_px, 64);
    lim("--dark-reach", so.dark_reach_px, 1024);
    lim("--end-vote", so.end_vote_px, 1024);
    lim("--strip-overlap", so.strip_overlap_px, 4096);
    if (so.strip_overlap_px < 64) so.strip_overlap_px = 64;
    so.strips = std::clamp(so.strips, 0, 16);
    so.refine_r = std::clamp(so.refine_r, 0, 1000);
    so.streams = std::clamp(so.streams, 1, 16);
    po.opt.streams_per_gpu = std::clamp(po.opt.streams_per_gpu, 1, 16);
    po.opt.readers = std::clamp(po.opt.readers, 1, 256);
}

static bool parse_stack_option_raw(const std::string& a, const std::string& v, StackOptions& so);

// parses the options shared by `stack` and `run`; returns false on an unknown option
bool parse_stack_option(const std::string& a, const std::string& v, StackOptions& so) {
    if (!parse_stack_option_raw(a, v, so)) return false;
    limit_stack_options(so);
    return true;
}

static bool parse_stack_option_raw(const std::string& a, const std::string& v, StackOptions& so) {
    Grid& g = so.po.g;
    if (a == "--scale") g.scale = atof(v.c_str());
    else if (a == "--interp") {   // the warp's resampling filter (FINDINGS 39)
        int i = interp_from_name(v.c_str());
        if (i < 0) { logf("--interp: bilinear, bicubic, lanczos3 or lanczos8, not '%s'\n", v.c_str()); return false; }
        g.interp = i;
    }
    else if (a == "--gpus") so.po.opt.devices = parse_list(v);
    else if (a == "--streams") { so.streams = atoi(v.c_str()); so.po.opt.streams_per_gpu = so.streams; }
    else if (a == "--readers") so.po.opt.readers = atoi(v.c_str());
    else if (a == "--cache") so.cache_frames = atoi(v.c_str()) != 0;
    else if (a == "--fix-banding") so.fix_banding = atoi(v.c_str()) != 0;
    else if (a == "--fix-rs") so.fix_rs = atoi(v.c_str()) != 0;
    else if (a == "--refine-r") so.refine_r = atoi(v.c_str());
    else if (a == "--refine-fm-px") so.refine_fm_px = atof(v.c_str());
    else if (a == "--soft-px") so.soft_px = atof(v.c_str());
    else if (a == "--smoothing") so.c_smoothing = atof(v.c_str());
    else if (a == "--soft-cutoff") so.soft_cutoff = atof(v.c_str());
    else if (a == "--c-radius") so.po.c_radius_px = atof(v.c_str());
    else if (a == "--b-radius") so.po.b_radius_px = atof(v.c_str());
    else if (a == "--halo-prior") so.halo_prior = atof(v.c_str());
    else if (a == "--dark-prior") so.dark_prior = atof(v.c_str());
    else if (a == "--dark-reach") so.dark_reach_px = atof(v.c_str());
    else if (a == "--end-vote") so.end_vote_px = atof(v.c_str());
    else if (a == "--flat-blur") so.po.flat_blur_px = atof(v.c_str());
    else if (a == "--flat-reach") so.po.flat_reach_px = atof(v.c_str());
    else if (a == "--flat-min") so.po.flat_min = atof(v.c_str());
    else if (a == "--flat-band") so.flat_band_px = atof(v.c_str());
    else if (a == "--flat-keep") so.flat_keep = atof(v.c_str());
    else if (a == "--flat-end") {   // quiet | loud | argmax | snap (FINDINGS 40)
        if (v == "quiet" || v == "0") so.flat_end = 0;
        else if (v == "loud" || v == "1") so.flat_end = 1;
        else if (v == "argmax" || v == "2") so.flat_end = 2;
        else if (v == "snap" || v == "3") so.flat_end = 3;
        else { logf("--flat-end: quiet, loud, argmax or snap, not '%s'\n", v.c_str()); return false; }
    }
    else if (a == "--flat-end-soft") so.flat_end_soft = atof(v.c_str());
    else if (a == "--flat-blend") so.flat_blend = atof(v.c_str());   // frames (FINDINGS 41)
    else if (a == "--strips") so.strips = atoi(v.c_str());
    else if (a == "--strip-overlap") so.strip_overlap_px = atof(v.c_str());
    else if (a == "--b-reference") {   // the 0.5.x B, bit-identical: the 1/4 volume with the priors and no 0.6.x additions
        so.po.depth_fraction = 0.25; so.dark_prior = 1.0; so.dark_reach_px = 0; so.end_vote_px = 0; so.po.flat_blur_px = 0; so.sgm_scale = false;
        so.conf_noise_gamma = 0; so.sgm_end_frames = false;   // 0.6.11's noise-adaptive ramp and 0.6.10's end-frame argmin off too
    }
    else if (a == "--sgm-p1") so.sgm_p1 = atof(v.c_str());
    else if (a == "--sgm-p2") so.sgm_p2 = atof(v.c_str());
    else if (a == "--sgm-scale") so.sgm_scale = atoi(v.c_str()) != 0;
    else if (a == "--conf-lo") so.conf_lo = atof(v.c_str());
    else if (a == "--conf-hi") so.conf_hi = atof(v.c_str());
    else if (a == "--conf-noise-ref") so.conf_noise_ref = atof(v.c_str());
    else if (a == "--conf-noise-gamma") so.conf_noise_gamma = atof(v.c_str());
    // the depth volume's parameters (the GUI's Advanced panel; the `fold` command has the same names)
    else if (a == "--radius-px") so.po.radius_px = atof(v.c_str());
    else if (a == "--power") so.po.power = atof(v.c_str());
    else if (a == "--depth-fraction") so.po.depth_fraction = atof(v.c_str());
    else if (a == "--depth-fm-px") so.po.depth_fm_px = atof(v.c_str());
    else if (a == "--vol-agg-px") so.po.vol_agg_px = atof(v.c_str());
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

struct NoiseEstimate;
static int run_stack_once(StackOptions& so, NoiseEstimate* noise_cache);
static int run_stack_strips(StackOptions& so, int n);
// the frames' noise, measured once per stack (and once per sweep): the finalize's pyramid MADs and the depth
// volume's focus-measure floor (FINDINGS 38)
struct NoiseEstimate {
    std::vector<float> mad;
    float depth_floor = 0.f;
};
static int fuse_from_pass1(StackOptions& so, Pass1& P, bool release_volume, std::chrono::steady_clock::time_point t_all,
                           NoiseEstimate* noise_cache);
static int measure_noise(const StackOptions& so, const Pass1& P, cudaStream_t s, NoiseEstimate& ne);

// A job's frame cache made ready before the GPUs are loaded: stale frames out, the budget set and, for a stack
// that is large for two GPUs, the buffers its frames will be read into pinned now. With two cards loaded the driver
// has a ceiling of some 73 GiB on all the host memory the process has ever page-locked, freed or not; past it every
// CUDA call fails for the rest of the process. What is pinned before the cards are loaded does not count (FINDINGS
// 55). A stack under 48 GiB pins as it reads, as it always did (pinning 50 GB up front costs 10 s that the reading
// otherwise hides); the 261 16-bit frames of 6960 x 4640 are 47 GiB.
static void ready_cache(FrameCache* cache, const std::vector<fs::path>& paths, const std::vector<int>& devices) {
    if (!cache) return;
    cache->prepare(paths);
    if (paths.empty()) return;
    int n = (int)devices.size();
    if (n == 0 && cudaGetDeviceCount(&n) != cudaSuccess) { cudaGetLastError(); return; }
    if (n < 2) return;
    TiffInfo t0 = tiff_probe(paths[0]);
    if (!t0.ok()) return;
    const size_t cap = tiff_buffer_capacity(t0.pixel_bytes());
    if ((double)cap * (double)paths.size() > 48.0 * 1073741824.0) {
        auto t0p = std::chrono::steady_clock::now();
        const size_t before = cache->spare_bytes();
        cache->prepin(paths, cap);
        const size_t got = cache->spare_bytes() - std::min(before, cache->spare_bytes());
        if (got) logf("frames: %.1f GB of buffers page-locked before the GPUs are loaded, %.1f s\n", got / 1e9,
                      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0p).count());
    }
}

// Returns the GPUs' pooled memory (cudaMallocAsync keeps freed blocks for the process): another xLamina or
// stackcli on the same cards would otherwise find them short between renders.
static void trim_pools(const std::vector<int>& devs) {   // the job's cards (empty = all): a card left out of the job is not touched
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) return;
    for (int d = 0; d < n; ++d) {
        if (!devs.empty() && std::find(devs.begin(), devs.end(), d) == devs.end()) continue;
        cudaMemPool_t pool;
        if (cudaDeviceGetDefaultMemPool(&pool, d) == cudaSuccess) cudaMemPoolTrimTo(pool, 0);
    }
    cudaGetLastError();
}

// The fusion, with a fallback: when the GPU memory runs short for the depth volume (the 1/2-resolution default
// needs ~5 GB beside the state; another process may hold part of the card), the render is repeated at 1/4
// instead of failing (FINDINGS 32).
// one attempt at the fusion; running out of GPU memory comes back as 4, whichever way it surfaced
static int try_stack(StackOptions& so, int strips) {
    try {
        return strips >= 2 ? run_stack_strips(so, strips) : run_stack_once(so, nullptr);
    } catch (const std::bad_alloc&) {
        // the PC's own memory: the depth volume on the host is the large piece (frames x depth pixels x 6 bytes),
        // and strips or the 1/4 volume make it smaller, so the same fallbacks are tried as for the cards' memory
        logf("the PC's memory (RAM) ran out during the fusion");
        note_fail(Fail::HostMemory, "during the fusion");
        return 4;
    } catch (const std::exception& e) {
        if (!strstr(e.what(), "memory")) throw;
        logf("%s", e.what());
        return 4;
    }
}

// False when the first GPU cannot give 1 MB after a failed attempt has freed everything: the state the driver is left
// in once page-locked host memory ran out (FINDINGS 54). No further attempt can work in this process then.
static bool cuda_usable(const StackOptions& so) {
    const auto& dv = so.po.opt.devices;
    void* p = nullptr;
    size_t fr = 0, tot = 0;
    cudaError_t e = cudaSetDevice(dv.empty() ? 0 : dv[0]);
    if (e == cudaSuccess) e = cudaMemGetInfo(&fr, &tot);
    // a card that answers but is full (another program holds it) is not this state: the next attempt may still fit
    if (e == cudaSuccess && fr >= ((size_t)2 << 20)) e = cudaMalloc(&p, (size_t)1 << 20);
    if (p) cudaFree(p);
    cudaGetLastError();
    return e == cudaSuccess;
}

int run_stack(StackOptions& so) {
    limit_stack_options(so);   // options set directly (the GUI, a job built in code) pass the same limits as parsed ones
    int rc;
    auto dead = [&]() {   // after an attempt that ran out of memory: is there anything left to try with?
        trim_pools(so.po.opt.devices);
        if (cuda_usable(so)) return false;
        logf("the GPU driver gives no memory back to this process any more (out of page-locked memory): close xLamina, open it again and render once more");
        return true;
    };
    if (so.strips >= 2) {
        rc = try_stack(so, so.strips);
        if (rc == 5) return 2;   // the strip count asked for does not fit the rectangle: an option error, said in the log
    } else {
        g_slices_fit = 1.0;
        rc = try_stack(so, 1);
        if (rc == 4 && dead()) return 1;
        if (rc == 4 && so.strips == 0 && !so.hooks.cancelled()) {
            // strips before a coarser volume: the first count from the share of the volume that fitted, then one more
            // strip per attempt while the memory still runs short
            const Grid& g = so.po.g;
            const double L = std::max(g.x1 - g.x0, g.y1 - g.y0) + 1.0, ov = so.strip_overlap_px;
            const double room = 0.9 * g_slices_fit.load() * L;   // the longest strip that should fit
            int n = room > ov + 1 ? (int)std::ceil((L - ov) / (room - ov)) : 2;
            for (n = std::clamp(n, 2, 16); n <= 16 && rc == 4 && !so.hooks.cancelled(); ++n) {
                logf("GPU memory ran short for the whole frame: rendering it in %d strips", n);
                trim_pools(so.po.opt.devices);
                rc = try_stack(so, n);
                if (rc == 4 && dead()) return 1;
            }
            if (rc == 5) rc = 4;   // no strip count fits the rectangle: on to the coarser volume
        }
    }
    if (rc == 4 && dead()) return 1;
    if (rc == 4 && so.po.depth_fraction > 0.25) {
        logf("GPU memory ran short at depth resolution %g: rendering again at 1/4", so.po.depth_fraction);
        trim_pools(so.po.opt.devices);
        so.po.depth_fraction = 0.25;
        rc = try_stack(so, so.strips >= 2 ? so.strips : 1);
    }
    trim_pools(so.po.opt.devices);
    return rc;
}

// The fusion in strips (StackOptions::strips): the output rectangle cut along its longer side into n overlapping
// strips, each a whole run_stack_once of its own rectangle, then feathered together and written as one render.
// What depends on the whole frame is shared: the noise estimate (the confidence ramp and C's gate follow it, and
// a strip's own estimate differs by several percent from its neighbour's) and the pyramid depth.
static int run_stack_strips(StackOptions& so, int n) {
    auto t_all = std::chrono::steady_clock::now();
    auto since = [](std::chrono::steady_clock::time_point t) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count(); };
    const Hooks hooks = so.hooks;
    Pass1Options& po = so.po;
    // The rectangle's sides become multiples of 4 sensor px (up to 3 px off the right and the bottom). The depth
    // grid is 1/2 or 1/4 of the output grid, rounded: for an odd side the ratio is not whole (6585 px to 3292 is
    // 2.0003), the depth grid slides against the output grid across the frame, and a strip, cut at even px, has
    // another grid than the whole frame. With odd sides the confidence was the same as the unstripped render's in
    // 45 % of the pixels, with even sides in 96 % (FINDINGS 55).
    {
        const int cut_x = (po.g.x1 - po.g.x0 + 1) % 4, cut_y = (po.g.y1 - po.g.y0 + 1) % 4;
        if (cut_x || cut_y) {
            po.g.x1 -= cut_x;
            po.g.y1 -= cut_y;
            logf("strips: the rectangle is trimmed by %d x %d px to sides that are multiples of 4", cut_x, cut_y);
        }
    }
    const Grid g = po.g;
    const int ow = g.out_w(), oh = g.out_h();
    const bool along_x = g.x1 - g.x0 >= g.y1 - g.y0;
    const int L = (along_x ? g.x1 - g.x0 : g.y1 - g.y0) + 1;   // sensor px along the cut side
    // strip rectangles: even starts and lengths, so a strip's pixels are the whole frame's at either scale
    // a strip starts on the coarsest pyramid level's grid of the whole frame (and so on the depth volume's): its
    // pyramid then holds the whole frame's samples, and C away from the strip's edges is the unstripped C exactly.
    // The starts are rounded down to that grid, so the overlap is at least two of its steps
    const int full_levels = n_levels(ow, oh);
    const int step = (int)std::lround((1 << (full_levels - 1)) / g.scale);
    int ov = std::max(((int)so.strip_overlap_px + 1) & ~1, 2 * step);
    int len = ((L + (n - 1) * ov + n - 1) / n + 1) & ~1;
    if (len < 2 * ov + 64 || len >= L) {
        logf("strips: %d strips with %d px of overlap do not fit %d px", n, ov, L);
        return 5;   // not a count this rectangle can take: run_stack stops adding strips, or reports the option
    }
    std::vector<int> a(n), b(n);
    for (int i = 0; i < n; ++i) {
        a[i] = (int)((double)i * (L - len) / (n - 1)) / step * step;
        b[i] = i == n - 1 ? L - 1 : a[i] + len - 1;
    }
    // the starts were rounded down to the grid, which can leave an overlap short of the one asked for (392 px of 512
    // with 256 px steps): a strip then runs on to the full overlap with the next
    for (int i = 0; i + 1 < n; ++i) b[i] = std::max(b[i], std::min(L - 1, a[i + 1] + ov - 1));
    for (int i = 0; i + 1 < n; ++i)
        if (b[i] - a[i + 1] + 1 < ov / 2) { logf("strips: %d strips of %d px leave too little overlap on the %d px grid", n, len, step); return 5; }

    // the frames: listed once, resident once for every strip
    std::vector<fs::path> paths = po.paths;
    std::vector<Similarity> T = po.T;
    if (paths.empty() && !read_list(po.list, paths, T)) return 1;
    if (paths.empty() || T.size() != paths.size()) { logf("no frames to stack\n"); return 1; }
    std::unique_ptr<FrameCache> own_cache;
    struct Unlink { FrameCache*& p; bool mine; ~Unlink() { if (mine) p = nullptr; } } unlink{po.opt.cache, false};
    if (so.cache_frames && !po.opt.cache) { own_cache = std::make_unique<FrameCache>(); po.opt.cache = own_cache.get(); unlink.mine = true; }
    if (so.cache_frames && po.opt.cache) ready_cache(po.opt.cache, paths, po.opt.devices);
    const int K = (int)paths.size();
    const int levels = n_levels(ow, oh);

    // the whole frame's noise, as an unstripped render measures it; failing that the first strip's serves them all
    NoiseEstimate ne;
    try {
        if (std::string gp = gpu_problem(po.opt.devices); !gp.empty()) { logf("%s", gp.c_str()); return 1; }
        TiffInfo t0 = tiff_probe(paths[0]);
        if (!t0.ok()) { logf("%s: %s\n", paths[0].string().c_str(), t0.error.c_str()); return 1; }
        Pass1 P;
        P.g = g; P.g.W = (int)t0.width; P.g.H = (int)t0.height;
        P.paths = paths; P.T = T;
        P.fp.levels = levels;
        P.vp = volume_params(po, P.g);
        CK(cudaSetDevice(po.opt.devices.empty() ? 0 : po.opt.devices[0]));
        cudaStream_t s;
        CK(cudaStreamCreate(&s));
        struct Close { cudaStream_t s; ~Close() { cudaStreamDestroy(s); } } close{s};
        if (hooks.progress) hooks.progress("noise floor of the whole frame", 0.0);
        if (measure_noise(so, P, s, ne)) {   // no room, or a frame that cannot be read (the first strip will say which)
            cudaGetLastError();
            logf("strips: the noise could not be measured on the whole frame; the first strip's estimate is used");
            ne = NoiseEstimate{};
        }
    } catch (const std::exception& e) {
        if (!strstr(e.what(), "memory")) throw;
        logf("strips: no room to measure the noise on the whole frame; the first strip's estimate is used");
        ne = NoiseEstimate{};
    }
    if (hooks.cancelled()) return 3;
    trim_pools(so.po.opt.devices);

    // the flat mask's thresholds are ranks over the rectangle (the flattest fifth, the 90th percentile of texture), so
    // a strip's own would follow its content: one fold of the whole frame, without the volume, gives the frame's
    float flat_lo = -1.f, flat_hi = -1.f;
    if (po.flat_blur_px > 0) {
        // When the whole frame's fold does not fit either (a 13920 x 9280 frame's state alone is 7 GB), it is folded
        // at half size: the guide is reduced to the depth grid anyway, and the thresholds from the half-size fold
        // were 2.4 % and 1.1 % under the full-size ones on the 261-frame stack, where strips ranking their own
        // differed by a fifth (FINDINGS 55)
        int r = 4;
        for (double sc : {g.scale, 0.5}) {
            if (r == 0 || (sc == 0.5 && g.scale != 1.0)) break;
            r = 4;
            try {
                Pass1Options pp = po;
                pp.paths = paths; pp.T = T;
                pp.g.scale = sc;
                pp.volume = false; pp.with_a = false; pp.host_state = false; pp.stop_if_short = false; pp.levels = 0;
                pp.hooks = hooks;
                Pass1 P;
                struct FreeState { FoldState& s; ~FreeState() { s.free(); } } free_state{P.dst};
                r = run_pass1(pp, P);
                if (r == 0 && !hooks.cancelled()) {
                    if (hooks.progress) hooks.progress("flat regions of the whole frame", 0.0);
                    cudaStream_t s = nullptr;
                    struct Close { cudaStream_t& s; ~Close() { if (s) cudaStreamDestroy(s); } } close{s};
                    if (cudaSetDevice(P.devs[0]) != cudaSuccess || cudaStreamCreate(&s) != cudaSuccess) r = 1;
                    else r = flat_limits(P.dst, po, volume_params(po, P.g), P.g.out_w(), P.g.out_h(), s, flat_lo, flat_hi);
                }
            } catch (const std::exception& e) {
                if (!strstr(e.what(), "memory")) throw;
                r = 4;
            }
            if (hooks.cancelled()) return 3;
            cudaGetLastError();
            trim_pools(so.po.opt.devices);
            if (r == 0 && sc != g.scale) logf("strips: the whole frame folded at half size for the flat regions' thresholds");
        }
        if (r != 0) {
            flat_lo = flat_hi = -1.f;
            logf("strips: the whole frame could not be folded for the flat regions' thresholds; every strip ranks its own");
        }
    }
    logf("strips: %d %s strips of %d px with %d px of overlap (%d px %s); %d pyramid levels; flat thresholds %.4g .. %.4g", n,
         along_x ? "vertical" : "horizontal", len, ov, L, along_x ? "wide" : "high", levels, flat_lo, flat_hi);

    // the feather: strip i's weight rises across the middle third of its overlap with strip i - 1 and falls across
    // the middle third of its overlap with strip i + 1 (output px along the cut side)
    const int No = along_x ? ow : oh;
    std::vector<int> off(n), ext(n);
    auto rise = [&](int i, int t) -> float {   // the weight strip i has gained over strip i - 1 at output position t
        if (i <= 0) return 1.f;
        if (i >= n) return 0.f;
        double s0 = off[i], e0 = off[i - 1] + ext[i - 1], c = 0.5 * (s0 + e0), f = std::max(1.0, (e0 - s0) / 6);
        return (float)std::clamp((t + 0.5 - (c - f)) / (2 * f), 0.0, 1.0);
    };

    std::vector<std::string> names;
    std::vector<std::vector<float>> acc;
    std::vector<float> depth((size_t)ow * oh, 0.f), conf((size_t)ow * oh, 0.f), noise_mad;
    std::vector<double> energy(K, 0.0);
    std::vector<uint8_t> icc;
    double area = 0;
    StackOptions::Result sum;   // the stages' seconds over the strips
    for (int i = 0; i < n; ++i) {
        if (hooks.cancelled()) return 3;
        StackOptions ss = so;
        ss.po.paths = paths; ss.po.T = T;
        ss.po.levels = levels;
        ss.po.flat_lo = flat_lo; ss.po.flat_hi = flat_hi;
        (along_x ? ss.po.g.x0 : ss.po.g.y0) = (along_x ? g.x0 : g.y0) + a[i];
        (along_x ? ss.po.g.x1 : ss.po.g.y1) = (along_x ? g.x0 : g.y0) + b[i];
        ss.out.clear();
        ss.note.clear();
        StackOptions::Result res;
        ss.result = &res;
        // a strip's previews and frame views are on its own grid: the front end gets the stage names and, after
        // each strip, the render so far
        ss.hooks.preview = nullptr;
        ss.hooks.frame = nullptr;
        ss.hooks.view_window = nullptr;
        if (hooks.progress) {
            std::string pre = "strip " + std::to_string(i + 1) + " of " + std::to_string(n) + ": ";
            ss.hooks.progress = [pre, hooks](const std::string& stage, double f) { hooks.progress(stage == "done" ? pre + "done" : pre + stage, stage == "done" ? 0.0 : f); };
        }
        logf("strip %d of %d: %s %d..%d", i + 1, n, along_x ? "x" : "y", (along_x ? g.x0 : g.y0) + a[i], (along_x ? g.x0 : g.y0) + b[i]);
        if (int r = run_stack_once(ss, &ne)) return r;
        if (hooks.cancelled()) return 3;
        trim_pools(so.po.opt.devices);

        off[i] = (int)std::lround(a[i] * g.scale);
        ext[i] = along_x ? res.w : res.h;
        if (off[i] + ext[i] > No || (along_x ? res.h != oh : res.w != ow)) { logf("strips: strip %d is %d x %d at %d, outside the %d x %d render", i + 1, res.w, res.h, off[i], ow, oh); return 1; }
        if (i == 0) {
            names = res.names;
            acc.resize(names.size());
            for (auto& im : acc) im.assign((size_t)ow * oh * 3, 0.f);
            noise_mad = res.noise_mad;
            icc = res.icc;
        } else if (res.names != names) { logf("strips: strip %d returned other outputs than the first", i + 1); return 1; }
        // strip i's finished weights need strip i + 1's start, which its rectangle already gives
        if (i + 1 < n) { off[i + 1] = (int)std::lround(a[i + 1] * g.scale); }
        std::vector<float> w(ext[i]);
        for (int t = 0; t < ext[i]; ++t) {
            float up = rise(i, off[i] + t);
            float down = 1.f;
            if (i + 1 < n) {   // the fall towards strip i + 1, from the rectangles (strip i + 1 is not rendered yet)
                double s0 = off[i + 1], e0 = off[i] + ext[i], c = 0.5 * (s0 + e0), f = std::max(1.0, (e0 - s0) / 6);
                down = 1.f - (float)std::clamp((off[i] + t + 0.5 - (c - f)) / (2 * f), 0.0, 1.0);
            }
            w[t] = up * down;
        }
        const int sw = res.w, sh = res.h;
        auto add = [&](const std::vector<float>& src, std::vector<float>& dst, int ch) {
            std::vector<std::thread> th;
            const int nt = 8;
            for (int q = 0; q < nt; ++q)
                th.emplace_back([&, q] {
                    for (int y = sh * q / nt; y < sh * (q + 1) / nt; ++y) {
                        const float* sp = src.data() + (size_t)y * sw * ch;
                        float* dp = dst.data() + ((size_t)(along_x ? y : y + off[i]) * ow + (along_x ? off[i] : 0)) * ch;
                        const float wy = along_x ? 0.f : w[y];
                        for (int x = 0; x < sw; ++x) {
                            const float wt = along_x ? w[x] : wy;
                            if (wt <= 0.f) continue;
                            for (int c = 0; c < ch; ++c) dp[(size_t)x * ch + c] += wt * sp[(size_t)x * ch + c];
                        }
                    }
                });
            for (auto& t : th) t.join();
        };
        for (size_t j = 0; j < names.size(); ++j) add(res.images[j], acc[j], 3);
        if (res.depth.size() == (size_t)sw * sh) add(res.depth, depth, 1);
        if (res.conf.size() == (size_t)sw * sh) add(res.conf, conf, 1);
        for (int k = 0; k < K && k < (int)res.frame_energy.size(); ++k) energy[k] += res.frame_energy[k] * (double)sw * sh;
        area += (double)sw * sh;
        sum.pass1_seconds += res.pass1_seconds; sum.depth_seconds += res.depth_seconds; sum.refine_seconds += res.refine_seconds;
        sum.compose_seconds += res.compose_seconds; sum.finalize_seconds += res.finalize_seconds;
        sum.frames_per_second += res.frames_per_second / n;
        if (hooks.preview) {   // the render so far
            size_t jb = std::find(names.begin(), names.end(), "B") - names.begin();
            if (jb >= names.size()) jb = 0;
            if (!names.empty()) { std::vector<float> small; int pw, ph; downsample_preview(acc[jb], ow, oh, hooks.preview_down, small, pw, ph); hooks.preview(names[jb], small.data(), pw, ph); }
        }
    }
    for (double& e : energy) e /= std::max(area, 1.0);
    if (hooks.preview)
        for (size_t j = 0; j < names.size(); ++j) { std::vector<float> small; int pw, ph; downsample_preview(acc[j], ow, oh, hooks.preview_down, small, pw, ph); hooks.preview(names[j], small.data(), pw, ph); }

    FrameSharpness sharp = frame_sharpness(energy, so.soft_cutoff);
    if (sharp.threshold > 0) {
        std::string soft;
        for (int k = 0; k < K; ++k)
            if (sharp.soft[k]) { char bf[64]; snprintf(bf, sizeof bf, " %s (%.2f)", paths[k].filename().string().c_str(), sharp.ratio[k]); soft += bf; }
        logf("sharpness: %d soft frame%s (focus energy under %.2f of the neighbours')%s%s", sharp.n_soft, sharp.n_soft == 1 ? "" : "s",
             sharp.threshold, soft.empty() ? "" : ":", soft.c_str());
    }

    // ---- write, as one render
    auto t5 = std::chrono::steady_clock::now();
    bool wrote = true;   // false: the outputs exist (the result takes them) but are not in the folder
    if (!so.out.empty() && !(wrote = make_out_dir(so.out))) {}
    else if (!so.out.empty()) {
        fs::path od(so.out);
        std::vector<const std::vector<float>*> imgs;
        for (auto& a : acc) imgs.push_back(&a);
        wrote = write_outputs(od, names, imgs, ow, oh, icc) == names.size();
        if (so.diagnostics) {   // the stitched maps; the smoothed depth and the flat mask exist per strip only
            write_raw(od / "depth.f32", depth.data(), depth.size());
            write_raw(od / "conf.f32", conf.data(), conf.size());
        }
        FILE* mf = fopen((od / "stack.txt").string().c_str(), "w");
        if (mf) {
            fprintf(mf, "w %d\nh %d\nframes %d\nlevels %d\nscale %g\nrect %d %d %d %d\nnoise_mad", ow, oh, K, levels, g.scale, g.x0, g.y0, g.x1, g.y1);
            for (float v : noise_mad) fprintf(mf, " %.9g", v);
            fprintf(mf, "\nlevel_shapes");
            for (auto& sh : pyramid_shapes(ow, oh, levels)) fprintf(mf, " %d %d", sh.w, sh.h);
            fprintf(mf, "\nframe_energy");
            for (double v : energy) fprintf(mf, " %.9g", v);
            fprintf(mf, "\nsoft_threshold %.9g\nsoft", sharp.threshold);
            for (int k = 0; k < K; ++k) if (sharp.soft[k]) fprintf(mf, " %d", k);
            fprintf(mf, "\noutputs");
            for (auto& nme : names) fprintf(mf, " %s", nme.c_str());
            fprintf(mf, "\nstrips %d %s overlap %d\n", n, along_x ? "x" : "y", ov);
            if (!so.note.empty()) fprintf(mf, "options %s\n", so.note.c_str());
            fclose(mf);
        }
        if (wrote) logf("wrote %zu outputs (%dx%d) in %.2f s -> %s", names.size(), ow, oh, since(t5), so.out.c_str());
    }
    if (so.result) {
        auto& R = *so.result;
        R = sum;
        R.w = ow; R.h = oh;
        R.names = names;
        R.images = std::move(acc);
        R.depth = std::move(depth); R.conf = std::move(conf);
        R.noise_mad = noise_mad; R.icc = icc;
        R.frame_energy = energy;
        R.write_seconds = since(t5);
        R.seconds = since(t_all);
    }
    logf("fusion total %.2f s in %d strips", since(t_all), n);
    if (hooks.progress) hooks.progress("done", 1.0);
    return wrote ? 0 : 6;   // 6: rendered, not written (the result holds the images)
}

static int run_stack_once(StackOptions& so, NoiseEstimate* noise_cache) {
    Pass1Options& po = so.po;
    po.volume = true;
    po.stop_if_short = so.strips != 1;
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
        ready_cache(po.opt.cache, paths, po.opt.devices);
        if (own_cache) logf("frames resident in host memory: %d of %zu, budget %.1f GB\n", po.opt.cache->count(), paths.size(), po.opt.cache->budget() / 1e9);
    }

    // ---- pass 1 + volume
    auto wants = [&](const char* name) { return so.outputs.empty() || std::find(so.outputs.begin(), so.outputs.end(), name) != so.outputs.end(); };
    po.with_a = wants("A");
    Pass1 P;
    struct FreeState { FoldState& s; ~FreeState() { s.free(); } } free_state{P.dst};   // the merged state lives on devs[0] until the end
    if (int r = run_pass1(po, P)) return r;
    if (hooks.cancelled()) return 3;
    logf("pass 1: %.2f s setup, %.2f s stream + %.2f s merge + %.2f s volume\n", P.setup_seconds, P.st.seconds, P.merge_seconds, P.vol_seconds);
    if (so.result) {
        so.result->pass1_seconds = since(t_all);
        so.result->frames_per_second = P.st.seconds > 0 ? P.st.frames / P.st.seconds : 0;
    }

    return fuse_from_pass1(so, P, true, t_all, noise_cache);
}

// The fusion from a finished pass 1: the depth map, the refinement, the composition, the finalize and the
// outputs. run_stack_once calls it once; run_sweep calls it once per parameter set over one pass 1, keeping the
// pinned host volume (release_volume = false) and the two-frame noise estimate (noise_cache) between variants.
// The noise of the frames from two adjacent middle frames (their difference is noise plus a sliver of focus
// change): the per-level MAD of the Laplacian pyramid for the finalize (C's smoothing gate), and the floor of
// the depth volume's focus measure for the confidence ramp (FINDINGS 38). Returns a load error code, 0 if fine.
static int measure_noise(const StackOptions& so, const Pass1& P, cudaStream_t s, NoiseEstimate& ne) {
    const int K = (int)P.paths.size();
    const int ow = P.g.out_w(), oh = P.g.out_h();
    const size_t no = (size_t)ow * oh;
    int mid = std::max(0, K / 2 - 1);
    float* d_pair[2] = {nullptr, nullptr};
    float *d_rgb = nullptr, *d_scr = nullptr;
    VolumeScratch sc;
    bool sc_on = false;
    struct Tidy { float** pair; float*& d_rgb; float*& d_scr; VolumeScratch& sc; bool& sc_on;   // on every way out
                  ~Tidy() { cudaFree(pair[0]); cudaFree(pair[1]); cudaFree(d_rgb); cudaFree(d_scr); if (sc_on) sc.free(); } } tidy{d_pair, d_rgb, d_scr, sc, sc_on};
    {   // the two warped frames, one frame as read beside them, and the pyramids of the difference: not past the
        // card's free memory (FINDINGS 55)
        size_t fr = 0, tot = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        const size_t need = (size_t)P.g.W * P.g.H * 16 + (P.g.scale != 1.0 ? (size_t)P.g.work_w() * P.g.work_h() * 12 : 0) + no * 60 + ((size_t)512 << 20);
        if (fr < need) { logf("GPU memory ran out (the noise of a %d x %d rectangle needs %.2f GB, %.2f GB free)", ow, oh, need / 1e9, fr / 1e9); return 4; }
    }
    for (int j = 0; j < 2; ++j) {
        TiffInfo info;
        if (int r = load_frame_to_device(P.paths[std::min(K - 1, mid + j)], info, d_rgb, so.po.opt.cache, so.po.dust.get())) return r;
        if (P.g.scale != 1.0 && !d_scr) CK(cudaMalloc(&d_scr, (size_t)P.g.work_w() * P.g.work_h() * 3 * sizeof(float)));
        CK(cudaMalloc(&d_pair[j], no * 3 * sizeof(float)));
        load_aligned(d_rgb, P.g, P.T[std::min(K - 1, mid + j)], d_scr, d_pair[j], false, s);
        CK(cudaStreamSynchronize(s));
        cudaFree(d_rgb);
        d_rgb = nullptr;
    }
    cudaFree(d_scr);
    d_scr = nullptr;
    ne.mad = noise_from_pair(d_pair[0], d_pair[1], ow, oh, P.fp.levels, s);
    if (P.vp.dw > 0 && P.vp.dh > 0) {
        sc.alloc(ow, oh, P.vp);
        sc_on = true;
        ne.depth_floor = depth_noise_floor(d_pair[0], d_pair[1], P.vp, sc, s);
        CK(cudaStreamSynchronize(s));
    }
    return 0;
}

static int fuse_from_pass1(StackOptions& so, Pass1& P, bool release_volume, std::chrono::steady_clock::time_point t_all,
                           NoiseEstimate* noise_cache) {
    Pass1Options& po = so.po;
    const Hooks& hooks = so.hooks;
    const std::string& out = so.out;
    const int refine_r = so.refine_r, streams = so.streams;
    const double refine_fm_px = so.refine_fm_px, soft_px = so.soft_px;
    auto since = [](std::chrono::steady_clock::time_point t) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count(); };
    auto stage = [&](const char* name, double f) { if (hooks.progress) hooks.progress(name, f); };
    const int K = (int)P.paths.size();
    const int ow = P.g.out_w(), oh = P.g.out_h();
    const size_t no = (size_t)ow * oh, nd = (size_t)P.vp.dw * P.vp.dh;
    const int prim = P.devs[0];
    // ---- depth (SGM) on the primary GPU
    stage("depth map", 0.0);
    auto t1 = std::chrono::steady_clock::now();
    CK(cudaSetDevice(prim));
    cudaStream_t s = nullptr;
    CK(cudaStreamCreate(&s));
    // the noise estimate first: the confidence ramp needs the focus measure's floor, the finalize the pyramid MADs
    NoiseEstimate ne;
    if (noise_cache && !noise_cache->mad.empty()) ne = *noise_cache;   // the sweep: the same two frames every variant
    else {
        stage("depth map: noise floor", 0.03);
        if (int r = measure_noise(so, P, s, ne)) { cudaStreamDestroy(s); return r; }
        if (noise_cache) *noise_cache = ne;
    }
    // an estimate from another grid (the whole frame's, for a strip): one value per pyramid level of this one
    if (!ne.mad.empty() && (int)ne.mad.size() != P.fp.levels - 1) ne.mad.resize(P.fp.levels - 1, ne.mad.back());
    DepthResult R;
    float* d_guide = nullptr;
    // the guide and the stream go on every way out of the fusion, a throw from a kernel wrapper included
    struct Top { float*& guide; cudaStream_t& s; int prim; ~Top() { cudaSetDevice(prim); cudaFree(guide); if (s) cudaStreamDestroy(s); } } top{d_guide, s, prim};
    {
        float *d_V, *d_ga;
        struct DevFree { std::vector<void*> p; ~DevFree() { for (void* q : p) cudaFree(q); } } dfree;   // freed on any exit, a throw included
        cudaError_t ev = cudaMalloc(&d_V, nd * K * sizeof(float));
        if (ev == cudaSuccess) dfree.p.push_back(d_V);
        if (ev != cudaSuccess) {
            cudaGetLastError();
            logf("depth: not enough GPU memory for the %.1f GB volume", nd * K * 4.0 / 1e9);
            return 4;   // run_stack retries at 1/4 resolution
        }
        // Windows does not refuse an allocation past the card: it pages GPU memory to the host and the kernels
        // crawl (FINDINGS 46: the cost volume took 29 s instead of 2 s). Say so when the headroom is gone.
        {
            size_t fr = 0, tot = 0;
            if (cudaMemGetInfo(&fr, &tot) == cudaSuccess) {
                logf("depth: %.2f of %.2f GB free on the GPU beside the %.1f GB volume\n", fr / 1e9, tot / 1e9, nd * K * 4.0 / 1e9);
                if (fr < ((size_t)2 << 30)) logf("depth: under 2 GB of GPU memory left: the driver may page and the depth stage run slowly\n");
            }
        }
        CK_MEM(cudaMalloc(&d_guide, no * sizeof(float)));   // outlives this block: the smoothing and the finalize use it
        CK_MEM(cudaMalloc(&d_ga, nd * sizeof(float)));
        dfree.p.push_back(d_ga);
        CK(cudaMemcpy(d_guide, P.guide.data(), no * sizeof(float), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_ga, P.ga.data(), nd * sizeof(float), cudaMemcpyHostToDevice));
        // plane-major host volumes -> k-fastest device volumes, a few planes at a time through a staging buffer
        // (a full plane-major copy on the device doubled the volume's footprint: 4.9 GB more at 1/2 resolution)
        stage("depth map: volume to the GPU", 0.08);
        {
            const int kc = std::max(1, std::min(K, (int)(((size_t)256 << 20) / (nd * sizeof(float)))));
            float* d_stage;
            CK_MEM(cudaMalloc(&d_stage, (size_t)kc * nd * sizeof(float)));
            const double total_gb = nd * K * 4.0 / 1e9;   // the float32 focus volume; the luma volume streams from the host inside depth_from_volume
            auto tu = std::chrono::steady_clock::now();
            for (int k0 = 0; k0 < K; k0 += kc) {
                int c = std::min(kc, K - k0);
                CK(cudaMemcpyAsync(d_stage, P.V_all.data() + (size_t)k0 * nd, (size_t)c * nd * sizeof(float), cudaMemcpyHostToDevice, s));
                planes_to_kfastest_f32_chunk(d_stage, d_V, P.vp.dw, P.vp.dh, K, k0, c, s);
                CK(cudaStreamSynchronize(s));   // the staging buffer is reused
                // the bar moves through the upload with the gigabytes done and the rate
                double done_gb = nd * (double)(k0 + c) * 4.0 / 1e9, secs = since(tu);
                char msg[128];
                snprintf(msg, sizeof msg, "depth map: volume to the GPU, %.1f of %.1f GB at %.1f GB/s", done_gb, total_gb, secs > 0 ? done_gb / secs : 0.0);
                stage(msg, 0.08 + 0.07 * (k0 + c) / K);
            }
            logf("depth: %.1f GB volume to the GPU in %.2f s\n", total_gb, since(tu));
            cudaFree(d_stage);
        }
        stage("depth map: cost volume", 0.15);
        if (release_volume) { P.V_all.release_async(); P.fm_all.release_async(); }   // lum_all after the depth: it streams from it
        DepthParams dp;
        dp.halo_prior = (float)so.halo_prior;
        dp.dark_prior = (float)so.dark_prior;
        dp.dark_reach_px = (float)so.dark_reach_px;
        dp.end_vote_px = (float)so.end_vote_px;
        dp.depth_full_fraction = (float)po.depth_fraction;   // the spill-zone radii are in sensor px
        if (so.sgm_p1 > 0) dp.p1 = (float)so.sgm_p1;
        if (so.sgm_p2 > 0) dp.p2 = (float)so.sgm_p2;
        if (so.conf_lo > 0) dp.c_lo = (float)so.conf_lo;
        if (so.conf_hi > 0) dp.c_hi = (float)so.conf_hi;
        if (so.sgm_scale && po.depth_fraction != 0.25) {   // the same smoothness per micron on a finer grid
            float k = (float)(po.depth_fraction / 0.25);
            dp.p1 *= k;
            dp.p2 *= k;
        }
        // the noise-adaptive ramp (FINDINGS 38): the thresholds were set on frames whose focus-measure noise floor
        // was conf_noise_ref; a noisier stack gets a lower ramp by (ref / floor)^gamma, at least half. A cleaner
        // stack keeps the ramp: raising it would hand detail that the calibration stack keeps to the priors
        float conf_scale = 1.f;
        if (so.conf_noise_gamma > 0 && so.conf_noise_ref > 0 && ne.depth_floor > 0) {
            conf_scale = (float)std::pow(so.conf_noise_ref / ne.depth_floor, so.conf_noise_gamma);
            conf_scale = std::min(1.f, std::max(0.5f, conf_scale));
            if (std::fabs(conf_scale - 1.f) < 0.05f) conf_scale = 1.f;   // the calibration stack itself: bit-identical
            if (conf_scale < 1.f) {   // (the ramp as given, whatever its values, when the scale is 1)
                dp.c_lo *= conf_scale;
                dp.c_hi = std::min(0.95f, dp.c_hi * conf_scale);
            }
        }
        dp.end_frames = so.sgm_end_frames;
        dp.progress = [&](const char* what, double f) { stage(what, f); };   // the bar moves through the depth stage
        logf("depth: SGM P1 %.3f P2 %.3f, confidence ramp %.2f..%.2f (noise floor %.4f, ramp x%.2f), dark prior %.2f (reach %.0f px), halo prior %.2f, volume %dx%d\n",
             dp.p1, dp.p2, dp.c_lo, dp.c_hi, ne.depth_floor, conf_scale, dp.dark_prior, dp.dark_reach_px, dp.halo_prior, P.vp.dw, P.vp.dh);
        float* d_flat = nullptr;
        if (!P.flat.empty() && po.flat_blur_px > 0) {   // the flat-region mask from the volume stage
            CK(cudaMalloc(&d_flat, P.flat.size() * sizeof(float)));
            dfree.p.push_back(d_flat);
            CK(cudaMemcpyAsync(d_flat, P.flat.data(), P.flat.size() * sizeof(float), cudaMemcpyHostToDevice, s));
            dp.flat = d_flat;
            dp.flat_blur_px = (float)po.flat_blur_px;
            dp.flat_min = (float)po.flat_min;
            dp.flat_band_px = (float)so.flat_band_px;
            dp.flat_keep = (float)so.flat_keep;
            dp.flat_end = so.flat_end;
            dp.flat_end_soft = (float)so.flat_end_soft;
        }
        R = depth_from_volume(d_V, P.lum_all.data(), d_guide, d_ga, P.vp.dw, P.vp.dh, K, ow, oh, dp, s);
        if (release_volume) P.lum_all.release_async();
        // d_V and d_ga are freed here by dfree going out of scope
    }
    logf("depth: %.2f s (cost %.0f ms, SGM %.0f ms%s, post %.0f ms); end-peaked %.1f%%\n", since(t1), R.ms_cost, R.ms_sgm,
         R.sgm_bands > 1 ? (" in " + std::to_string(R.sgm_bands) + " bands").c_str() : "", R.ms_post, 100 * R.end_fraction);
    if (so.result) so.result->depth_seconds = since(t1);
    auto preview_depth = [&](const char* name) {
        if (!hooks.preview) return;
        std::vector<float> rgb(no * 3);
        for (size_t i = 0; i < no; ++i) { float v = R.depth_s[i] / (float)std::max(K - 1, 1); rgb[i * 3] = rgb[i * 3 + 1] = rgb[i * 3 + 2] = v; }
        std::vector<float> small; int pw, ph;
        downsample_preview(rgb, ow, oh, hooks.preview_down, small, pw, ph);
        hooks.preview(name, small.data(), pw, ph);
    };
    preview_depth("depth map");
    if (hooks.cancelled()) return 3;   // (	op frees the guide and the stream)

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
        // on a way out that is not the end of the pass (a card out of memory part-way through these allocations, a
        // frame that could not be read) the cards' buffers go, so that the attempt that follows finds them free
        struct Tidy {
            std::vector<RefineState>& rst; std::vector<std::vector<RefineScratch>>& rsc; std::vector<float*>& d_depth;
            const std::vector<int>& devs; int prim; bool done = false;
            ~Tidy() {
                if (done) return;
                for (int d : devs) {
                    if (cudaSetDevice(d) != cudaSuccess) continue;
                    for (auto& sc : rsc[d]) sc.free();
                    cudaFree(d_depth[d]);
                    rst[d].free();
                }
                cudaSetDevice(prim);
                cudaGetLastError();
            }
        } tidy{rst, rsc, d_depth, P.devs, prim};
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
        const StreamerStats st2 = run_pass(back_paths, back_k, P.T, P.g, po, P.devs, streams, po.opt.readers, po.opt.cache, ctx,
                 [&](int dev, PassSlot& S, int, cudaStream_t ss) { refine_measure(S.d_warp, rsc[dev][S.index], ss); },
                 [&](int dev, PassSlot& S, int k, cudaStream_t fs) {
                     refine_accumulate(rst[dev], k, rsc[dev][S.index].fm, d_depth[dev], refine_r, fs);
                     if (hooks.preview && dev == P.devs[0] && (++refined % std::max(1, hooks.preview_every) == 0) && ctx[dev].bp.tryFlush(hooks) &&
                         ctx[dev].bp.due(hooks.preview_min_ms))
                         ctx[dev].bp.enqueue(fs, -1, "depth (building)", [&, dev](float* d, cudaStream_t ps, int& pw, int& ph) {
                             refine_preview_kernel(rst[dev], K, hooks.preview_down, d, pw, ph, ps);
                         });
                 },
                 "pass 2: fine focus", hooks);
        // a frame that was not read has not entered the refinement: no result from it (it used to be logged only,
        // and the render went on to "Done" without the frame)
        if (int r = hooks.cancelled() ? 0 : stream_fail(st2)) return r;
        for (size_t i = 1; i < P.devs.size(); ++i) merge_device(rst[prim], rst[P.devs[i]]);
        for (int d : P.devs) {
            CK(cudaSetDevice(d));
            for (auto& sc : rsc[d]) sc.free();
            cudaFree(d_depth[d]);
        }
        CK(cudaSetDevice(prim));
        refine_depth(R.depth_s, R.conf, R.spill, P.vp.dw, P.vp.dh, rst[prim], ow, oh, P.g.scale, s);
        for (int d : P.devs) rst[d].free();   // leaves the last device selected
        tidy.done = true;
        CK(cudaSetDevice(prim));
    }
    std::vector<float> Ds = smooth_depth(R.depth_s, d_guide, ow, oh, P.g.scale, K, s);
    logf("refine: %.2f s\n", since(t2));
    if (so.result) so.result->refine_seconds = since(t2);
    preview_depth("depth map (refined)");
    if (hooks.cancelled()) return 3;   // (	op frees the guide and the stream)

    // ---- pass 3: composition over the frames that carry a label
    auto t3 = std::chrono::steady_clock::now();
    std::vector<float> b_img;
    {
        // the smooth-depth blend widened in flat regions (FINDINGS 41): a per-pixel width map from the flat mask
        const bool wide = so.flat_blend > 1.0 && !P.flat.empty() && po.flat_blur_px > 0;
        std::vector<int> ks = compose_frames(R.depth_s, Ds, K, wide ? (float)so.flat_blend : 1.f);
        std::vector<fs::path> sub;
        for (int k : ks) sub.push_back(P.paths[k]);
        int ndev = 0;
        CK(cudaGetDeviceCount(&ndev));
        std::vector<ComposeState> cst(ndev);
        std::vector<std::vector<ComposeScratch>> csc(ndev);
        std::vector<float*> d_D(ndev, nullptr), d_Ds(ndev, nullptr), d_W(ndev, nullptr);
        float s1 = (float)std::max(0.8, soft_px * P.g.scale);
        float* d_fs = nullptr;   // the flat mask on the card, while its width map is made
        struct Tidy {   // as in pass 2: the cards' buffers go on a way out that is not the end of the pass
            std::vector<ComposeState>& cst; std::vector<std::vector<ComposeScratch>>& csc;
            std::vector<float*>& d_D; std::vector<float*>& d_Ds; std::vector<float*>& d_W; float*& d_fs;
            const std::vector<int>& devs; int prim; bool done = false;
            ~Tidy() {
                if (done) return;
                cudaFree(d_fs);   // (on the device that was selected when the way out was taken)
                for (int d : devs) {
                    if (cudaSetDevice(d) != cudaSuccess) continue;
                    for (auto& sc : csc[d]) sc.free();
                    cudaFree(d_D[d]); cudaFree(d_Ds[d]); cudaFree(d_W[d]);
                    cst[d].free();
                }
                cudaSetDevice(prim);
                cudaGetLastError();
            }
        } tidy{cst, csc, d_D, d_Ds, d_W, d_fs, P.devs, prim};
        for (int d : P.devs) {
            CK(cudaSetDevice(d));
            cst[d].alloc(d, ow, oh);
            csc[d].resize(streams);
            for (auto& sc : csc[d]) sc.alloc(ow, oh, s1);
            CK(cudaMalloc(&d_D[d], no * sizeof(float)));
            CK(cudaMalloc(&d_Ds[d], no * sizeof(float)));
            CK(cudaMemcpy(d_D[d], R.depth_s.data(), no * sizeof(float), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_Ds[d], Ds.data(), no * sizeof(float), cudaMemcpyHostToDevice));
            if (wide) {
                CK(cudaMalloc(&d_fs, P.flat.size() * sizeof(float)));
                CK(cudaMalloc(&d_W[d], no * sizeof(float)));
                CK(cudaMemcpy(d_fs, P.flat.data(), P.flat.size() * sizeof(float), cudaMemcpyHostToDevice));
                flat_width_map(d_fs, P.vp.dw, P.vp.dh, d_W[d], ow, oh, (float)so.flat_blend, 0);
                CK(cudaDeviceSynchronize());
                cudaFree(d_fs);
                d_fs = nullptr;
            }
        }
        if (wide) logf("compose: the smooth-depth blend spans up to %.0f frames in flat regions; %zu frames carry weight\n", so.flat_blend, ks.size());
        std::vector<PassDev> ctx;
        std::atomic<int> composed{0};
        const StreamerStats st3 = run_pass(sub, ks, P.T, P.g, po, P.devs, streams, po.opt.readers, po.opt.cache, ctx,
                 [&](int, PassSlot&, int, cudaStream_t) {},
                 [&](int dev, PassSlot& S, int k, cudaStream_t fs) {
                     compose_frame(cst[dev], k, S.d_warp, d_D[dev], d_Ds[dev], csc[dev][S.index], fs, d_W[dev]);
                     // live view: the first GPU's composition so far, every preview_every frames
                     if (hooks.preview && dev == P.devs[0] && (++composed % std::max(1, hooks.preview_every) == 0) && ctx[dev].bp.tryFlush(hooks) &&
                         ctx[dev].bp.due(hooks.preview_min_ms))
                         ctx[dev].bp.enqueue(fs, -1, "B (building)", [&, dev](float* d, cudaStream_t ps, int& pw, int& ph) {
                             compose_preview_kernel(cst[dev], hooks.preview_down, d, pw, ph, ps);
                         });
                 },
                 "pass 3: composition", hooks);
        if (int r = hooks.cancelled() ? 0 : stream_fail(st3)) return r;   // a frame missing from the composition: no result
        for (size_t i = 1; i < P.devs.size(); ++i) merge_device(cst[prim], cst[P.devs[i]]);
        for (int d : P.devs) {
            CK(cudaSetDevice(d));
            for (auto& sc : csc[d]) sc.free();
            cudaFree(d_D[d]); cudaFree(d_Ds[d]); cudaFree(d_W[d]);
        }
        CK(cudaSetDevice(prim));
        b_img = compose_mix(cst[prim], R.conf, ow, oh, P.g.scale, s);
        for (int d : P.devs) cst[d].free();   // leaves the last device selected
        tidy.done = true;
        CK(cudaSetDevice(prim));
        logf("compose: %zu of %d frames carry a label; %.2f s\n", ks.size(), K, since(t3));
    }
    if (so.result) so.result->compose_seconds = since(t3);
    if (hooks.preview) { std::vector<float> small; int pw, ph; downsample_preview(b_img, ow, oh, hooks.preview_down, small, pw, ph); hooks.preview("B", small.data(), pw, ph); }
    if (hooks.cancelled()) return 3;   // (	op frees the guide and the stream)
    stage("finalize", 0.0);

    // ---- finalize (the noise estimate was taken before the depth stage)
    auto t4 = std::chrono::steady_clock::now();
    const std::vector<float>& noise_mad = ne.mad;
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
            if (wanted(F.names[i])) { std::vector<float> small; int pw, ph; downsample_preview(F.images[i], ow, oh, hooks.preview_down, small, pw, ph); hooks.preview(F.names[i], small.data(), pw, ph); }
    std::vector<uint8_t> icc = tiff_probe(P.paths[0]).icc;   // the frames' colour profile goes into every output
    if (so.result) {
        auto& Rr = *so.result;
        Rr.w = ow; Rr.h = oh;
        Rr.names.clear(); Rr.images.clear();
        for (size_t i = 0; i < F.names.size(); ++i)
            if (wanted(F.names[i])) { Rr.names.push_back(F.names[i]); Rr.images.push_back(std::move(F.images[i])); }
        Rr.depth = R.depth_s; Rr.conf = R.conf; Rr.noise_mad = noise_mad; Rr.icc = icc;
        Rr.frame_energy = P.frame_energy;
    }
    // the soft-frame flag (FINDINGS 49): the frames whose focus energy fell under their neighbours'
    FrameSharpness sharp = frame_sharpness(P.frame_energy, so.soft_cutoff);
    if (sharp.threshold > 0) {
        std::string soft;
        for (int k = 0; k < K; ++k)
            if (sharp.soft[k]) { char b[64]; snprintf(b, sizeof b, " %s (%.2f)", P.paths[k].filename().string().c_str(), sharp.ratio[k]); soft += b; }
        logf("sharpness: %d soft frame%s (focus energy under %.2f of the neighbours')%s%s", sharp.n_soft, sharp.n_soft == 1 ? "" : "s",
             sharp.threshold, soft.empty() ? "" : ":", soft.c_str());
    }

    // ---- write
    auto t5 = std::chrono::steady_clock::now();
    bool wrote = true;   // false: the outputs exist (the result holds them) but are not in the folder
    if (!out.empty() && !(wrote = make_out_dir(out))) {}
    else if (!out.empty()) {
    fs::path od(out);
    std::vector<std::string> wnames;
    std::vector<const std::vector<float>*> wimgs;
    for (size_t i = 0; i < F.names.size(); ++i) {
        if (!wanted(F.names[i])) continue;
        const std::vector<float>* img = &F.images[i];
        if (so.result) {   // the image moved into the result
            size_t j = std::find(so.result->names.begin(), so.result->names.end(), F.names[i]) - so.result->names.begin();
            img = &so.result->images[j];
        }
        wnames.push_back(F.names[i]);
        wimgs.push_back(img);
    }
    const size_t written = write_outputs(od, wnames, wimgs, ow, oh, icc);
    wrote = written == wnames.size();
    if (so.diagnostics) {   // the CLI's raw maps for analysis scripts and hash checks; the GUI writes none
        write_raw(od / "depth.f32", R.depth_s.data(), R.depth_s.size());
        if (!P.flat.empty()) write_raw(od / "flat.f32", P.flat.data(), P.flat.size());   // depth-resolution grid (stack.txt: w, h x depth fraction)
        write_raw(od / "depth_smooth.f32", Ds.data(), Ds.size());
        write_raw(od / "conf.f32", R.conf.data(), R.conf.size());
        if (getenv("STACKER_C_DEBUG")) {   // C's selection per pyramid level: the chosen frame (int16) and its energy (f32), level sizes in stack.txt
            CK(cudaSetDevice(prim));
            for (int l = 0; l < P.fp.levels - 1; ++l) {
                size_t m = (size_t)P.dst.shapes[l].w * P.dst.shapes[l].h;
                std::vector<int16_t> idx(m);
                std::vector<float> sal(m);
                CK(cudaMemcpy(idx.data(), P.dst.c_idx[l], m * sizeof(int16_t), cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(sal.data(), P.dst.c_sal[l], m * sizeof(float), cudaMemcpyDeviceToHost));
                write_raw(od / ("c_idx_L" + std::to_string(l) + ".i16"), idx.data(), m);
                write_raw(od / ("c_sal_L" + std::to_string(l) + ".f32"), sal.data(), m);
            }
        }
    }
    FILE* mf = fopen((od / "stack.txt").string().c_str(), "w");
    if (mf) {
        fprintf(mf, "w %d\nh %d\nframes %d\nlevels %d\nscale %g\nrect %d %d %d %d\nnoise_mad", ow, oh, K, P.fp.levels, P.g.scale, P.g.x0, P.g.y0, P.g.x1, P.g.y1);
        for (float v : noise_mad) fprintf(mf, " %.9g", v);
        fprintf(mf, "\nlevel_shapes");
        for (auto& sh : P.dst.shapes) fprintf(mf, " %d %d", sh.w, sh.h);
        fprintf(mf, "\nframe_energy");
        for (double v : P.frame_energy) fprintf(mf, " %.9g", v);
        fprintf(mf, "\nsoft_threshold %.9g\nsoft", sharp.threshold);
        for (int k = 0; k < K; ++k) if (sharp.soft[k]) fprintf(mf, " %d", k);
        fprintf(mf, "\noutputs");
        for (auto& nme : F.names) fprintf(mf, " %s", nme.c_str());
        fprintf(mf, "\n");
        if (!so.note.empty()) fprintf(mf, "options %s\n", so.note.c_str());
        fclose(mf);
    }
    if (wrote) logf("wrote %zu outputs (%dx%d) in %.2f s -> %s", written, ow, oh, since(t5), out.c_str());
    if (so.result) so.result->write_seconds = since(t5);
    }
    if (so.result) so.result->seconds = since(t_all);
    logf("fusion total %.2f s", since(t_all));
    stage("done", 1.0);
    return wrote ? 0 : 6;   // 6: rendered, not written (the result holds the images)
}

// ----------------------------------------------------------------------------- sweep: many B's over one pass 1

// Everything in Pass1Options that changes pass 1's result: variants with the same key share one pass 1 + volume.
static std::string pass1_key(const Pass1Options& po) {
    char b[512];
    // the warp's filter is in the key: the frames are warped in pass 1, so two --interp variants cannot share one
    // (0.6.14 .. 0.6.18 left it out: an --interp variant silently took the group's first filter)
    snprintf(b, sizeof b, "%g|%g|%g|%g|%g|%g|%g|%g|%g|%g|%d,%d,%d,%d|%g|%d|%d", po.radius_px, po.power, po.c_radius_px, po.b_radius_px,
             po.depth_fraction, po.depth_fm_px, po.vol_agg_px, po.flat_blur_px, po.flat_reach_px, po.flat_min, po.g.x0, po.g.y0, po.g.x1,
             po.g.y1, po.g.scale, po.g.interp, (int)po.frames_sel.size());
    std::string k = b;
    for (int f : po.frames_sel) k += "," + std::to_string(f);
    return k;
}

static std::string json_str(const std::string& s) {
    std::string o = "\"";
    for (char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if ((unsigned char)c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; }
        else o += c;
    }
    return o + "\"";
}

int run_sweep(const StackOptions& base, const std::vector<SweepVariant>& variants, const std::string& out_dir) {
    if (variants.empty()) { logf("sweep: no variants"); return 2; }
    auto t_all = std::chrono::steady_clock::now();
    auto since = [](std::chrono::steady_clock::time_point t) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count(); };
    // each variant's options on top of the base; group by what pass 1 sees
    struct Prepared { StackOptions so; std::string key, note; };
    std::vector<Prepared> prep;
    for (auto& v : variants) {
        Prepared p{base, "", ""};
        for (auto& [a, val] : v.options) {
            if (!parse_stack_option(a, val, p.so)) { logf("sweep: variant %s: unknown option %s", v.name.c_str(), a.c_str()); return 2; }
            p.note += (p.note.empty() ? "" : " ") + a + " " + val;
        }
        p.key = pass1_key(p.so.po);
        p.so.note = p.note;
        p.so.out = (fs::path(out_dir) / v.name).string();
        p.so.result = nullptr;
        prep.push_back(std::move(p));
    }
    std::vector<std::string> groups;   // pass-1 keys in order of first appearance
    for (auto& p : prep) if (std::find(groups.begin(), groups.end(), p.key) == groups.end()) groups.push_back(p.key);
    logf("sweep: %zu variants in %zu pass-1 group(s) -> %s", variants.size(), groups.size(), out_dir.c_str());

    // one frame cache for the whole sweep: the frames are read once and stay pinned across groups
    std::unique_ptr<FrameCache> own_cache;
    if (base.cache_frames && !base.po.opt.cache) own_cache = std::make_unique<FrameCache>();
    FrameCache* cache = base.cache_frames ? (base.po.opt.cache ? base.po.opt.cache : own_cache.get()) : nullptr;
    if (cache) {
        std::vector<fs::path> paths = base.po.paths;
        std::vector<Similarity> T;
        if (paths.empty() && !base.po.list.empty()) read_list(base.po.list, paths, T);
        ready_cache(cache, paths, base.po.opt.devices);
    }

    struct Done { std::string name, dir, note; int group; int rc; double seconds; double change; };   // change: % of pixels whose B luma differs > 2 % from the reference variant
    std::vector<Done> done;
    // the reference B (the variant named "base", else the first one): each variant's visible change is measured against it
    std::vector<float> ref_b;
    int ref_w = 0, ref_h = 0;
    auto luma_change = [&](const std::vector<float>& img, int w, int h) -> double {
        if (ref_b.empty() || w != ref_w || h != ref_h) return -1.0;
        size_t n = (size_t)w * h, hits = 0;
        for (size_t i = 0; i < n; ++i) {
            float a = 0.2126f * img[i * 3] + 0.7152f * img[i * 3 + 1] + 0.0722f * img[i * 3 + 2];
            float b = 0.2126f * ref_b[i * 3] + 0.7152f * ref_b[i * 3 + 1] + 0.0722f * ref_b[i * 3 + 2];
            if (std::fabs(a - b) > 0.02f) ++hits;
        }
        return 100.0 * hits / (double)n;
    };
    auto write_manifest = [&] {
        fs::create_directories(out_dir);
        FILE* f = fopen((fs::path(out_dir) / "sweep.json").string().c_str(), "w");
        if (!f) return;
        fprintf(f, "{\n  \"list\": %s,\n  \"base\": %s,\n  \"variants\": [\n", json_str(base.po.list.generic_string()).c_str(), json_str(base.note).c_str());
        for (size_t i = 0; i < done.size(); ++i) {
            auto& d = done[i];
            fprintf(f, "    {\"name\": %s, \"dir\": %s, \"options\": %s, \"pass1_group\": %d, \"rc\": %d, \"seconds\": %.2f, \"change_pct\": %.2f}%s\n",
                    json_str(d.name).c_str(), json_str(d.dir).c_str(), json_str(d.note).c_str(), d.group, d.rc, d.seconds, d.change,
                    i + 1 < done.size() ? "," : "");
        }
        fprintf(f, "  ]\n}\n");
        fclose(f);
    };

    int failed = 0;
    // progress for a front end: "variant 3 of 20, soft-px-2: pass 2: fine focus", the bar over the whole sweep
    // (a sweep gave no indication of how far it was); a group's pass 1 counts as its first variant
    const size_t nv = prep.size();
    size_t vi = 0;   // variants finished so far, over all groups
    auto sweep_progress = [&](size_t idx, const std::string& label) -> ProgressFn {
        if (!base.hooks.progress) return {};
        return [&, idx, label](const std::string& name, double f) {
            base.hooks.progress("variant " + std::to_string(idx + 1) + " of " + std::to_string(nv) + ", " + label + ": " + name,
                                ((double)idx + std::min(1.0, std::max(0.0, f))) / (double)nv);
        };
    };
    for (size_t gi = 0; gi < groups.size(); ++gi) {
        // pass 1 + volume once for the group, from the first variant's pass-1 options
        size_t first = 0;
        while (prep[first].key != groups[gi]) ++first;
        Pass1Options po = prep[first].so.po;
        po.volume = true;
        po.hooks = base.hooks;
        po.hooks.progress = sweep_progress(vi, "group " + std::to_string(gi + 1) + " of " + std::to_string(groups.size()));
        po.opt.cache = cache;
        po.with_a = false;   // A is never a sweep output
        auto t_g = std::chrono::steady_clock::now();
        Pass1 P;
        struct FreeState { FoldState& s; ~FreeState() { s.free(); } } free_state{P.dst};
        if (int r = run_pass1(po, P)) {
            logf("sweep: pass 1 of group %zu failed (rc %d)%s", gi + 1, r, r == 4 ? "; out of GPU memory: put --depth-fraction 0.25 in those variants" : "");
            for (size_t i = 0; i < prep.size(); ++i)
                if (prep[i].key == groups[gi]) { done.push_back({variants[i].name, prep[i].so.out, prep[i].note, (int)gi + 1, r, 0.0, -1.0}); ++failed; ++vi; }
            write_manifest();
            continue;
        }
        logf("sweep: group %zu pass 1 + volume %.2f s", gi + 1, since(t_g));
        NoiseEstimate noise;
        for (size_t i = 0; i < prep.size(); ++i) {
            if (prep[i].key != groups[gi]) continue;
            if (base.hooks.cancelled()) return 3;
            auto t_v = std::chrono::steady_clock::now();
            StackOptions so = prep[i].so;
            so.po.volume = true;
            so.po.hooks = base.hooks;
            so.po.opt.cache = cache;
            so.hooks.progress = sweep_progress(vi, variants[i].name);
            so.po.hooks.progress = so.hooks.progress;
            logf("sweep: %s  [%s]", variants[i].name.c_str(), prep[i].note.empty() ? "base" : prep[i].note.c_str());
            StackOptions::Result res;   // the B stays in memory for the change measure (the TIFFs are written as usual)
            so.result = &res;
            int rc = fuse_from_pass1(so, P, false, t_v, &noise);
            if (rc) ++failed;
            double change = -1.0;
            if (rc == 0) {
                auto it = std::find(res.names.begin(), res.names.end(), "B");
                if (it != res.names.end()) {
                    auto& b = res.images[it - res.names.begin()];
                    if (ref_b.empty() && (variants[i].name == "base" || !std::any_of(variants.begin(), variants.end(), [](const SweepVariant& v) { return v.name == "base"; }))) {
                        ref_b = b; ref_w = res.w; ref_h = res.h;
                        change = 0.0;
                    } else change = luma_change(b, res.w, res.h);
                }
            }
            if (change >= 0 && !ref_b.empty() && variants[i].name != "base") logf("sweep: %s changes %.2f %% of B's pixels by more than 2 %%", variants[i].name.c_str(), change);
            done.push_back({variants[i].name, so.out, prep[i].note, (int)gi + 1, rc, since(t_v), change});
            write_manifest();
            ++vi;
        }
    }
    logf("sweep: %zu variants, %d failed, %.1f s total -> %s", done.size(), failed, since(t_all), (fs::path(out_dir) / "sweep.json").string().c_str());
    return failed ? 1 : 0;
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
    else if (a == "--rs-bands") ao.rs_bands = atoi(v.c_str());
    else if (a == "--band-cycles") { auto d = parse_list(v); if (d.size() == 2) { ao.band_lo = d[0]; ao.band_hi = d[1]; } }
    else return false;
    return true;
}


// ---- rolling shutter and LED banding (FINDINGS 51)

// Each frame's row-mean luma against the stack's median row profile. The LED's flicker, read out line by line, is
// a ripple across the rows of a few cycles per frame, the same frequency in every frame at a random phase; the
// frame's own content drifting in and out of focus changes the profile too, but slowly (a cycle or two) or at the
// texture's fine scale. The fit projects each frame's ratio-to-median residual on the Fourier modes band_lo..band_hi
// cycles per frame height and keeps only those: band_pct is their peak-to-peak, row_gain the inverse.
static void banding_fit(const std::vector<float>& rows_all, int N, int H, const AlignOptions& ao, AlignOut& A) {
    A.band_pct.assign(N, 0.f);
    A.row_gain.assign((size_t)N * H, 1.f);
    A.band_peak_cycles = 0; A.band_peak_power = 0; A.band_median_pct = 0;
    if (N < 3 || H < 64) return;
    const double kPi = 3.14159265358979323846;
    // the reference profile of frame k: the median over the frames around it (the flicker's phase is random from
    // frame to frame, so the median drops it; the content's own profile, which drifts with the focus along the
    // stack, stays). The stack-wide median was tried first: a frame far from the middle differs from it by the
    // content's drift alone, 14 % against the 3 % of the flicker on the 210-frame stack
    const int half = std::min(8, (N - 1) / 2);
    std::vector<float> med((size_t)N * H), col(2 * half + 1);
    for (int k = 0; k < N; ++k) {
        int k0 = std::clamp(k - half, 0, N - 1 - 2 * half), k1 = k0 + 2 * half;
        for (int y = 0; y < H; ++y) {
            for (int q = k0; q <= k1; ++q) col[q - k0] = rows_all[(size_t)q * H + y];
            std::nth_element(col.begin(), col.begin() + half, col.end());
            med[(size_t)k * H + y] = col[half];
        }
    }
    const int cmax = std::min(40, H / 4);
    const int lo = std::clamp(std::min(ao.band_lo, ao.band_hi), 1, cmax), hi = std::clamp(std::max(ao.band_lo, ao.band_hi), lo, cmax);
    std::vector<double> cs((size_t)(cmax + 1) * H), sn((size_t)(cmax + 1) * H), power(cmax + 1, 0.0), r(H), pct(N);
    for (int c = 1; c <= cmax; ++c)
        for (int y = 0; y < H; ++y) { double a = 2 * kPi * c * (y + 0.5) / H; cs[(size_t)c * H + y] = std::cos(a); sn[(size_t)c * H + y] = std::sin(a); }
    for (int k = 0; k < N; ++k) {
        double mean = 0;
        for (int y = 0; y < H; ++y) { float m = med[(size_t)k * H + y]; r[y] = m > 1e-6f ? rows_all[(size_t)k * H + y] / m - 1.0 : 0.0; mean += r[y]; }
        mean /= H;
        for (int y = 0; y < H; ++y) r[y] -= mean;
        std::vector<double> a(cmax + 1, 0.0), b(cmax + 1, 0.0);
        for (int c = 1; c <= cmax; ++c) {
            double sa = 0, sb = 0;
            const double *C = &cs[(size_t)c * H], *S = &sn[(size_t)c * H];
            for (int y = 0; y < H; ++y) { sa += r[y] * C[y]; sb += r[y] * S[y]; }
            a[c] = 2 * sa / H; b[c] = 2 * sb / H;
            power[c] += (a[c] * a[c] + b[c] * b[c]) / N;
        }
        double fmin = 1e300, fmax = -1e300;
        float* gain = &A.row_gain[(size_t)k * H];
        for (int y = 0; y < H; ++y) {
            double f = 0;
            for (int c = lo; c <= hi; ++c) f += a[c] * cs[(size_t)c * H + y] + b[c] * sn[(size_t)c * H + y];
            fmin = std::min(fmin, f); fmax = std::max(fmax, f);
            gain[y] = (float)(1.0 / std::clamp(1.0 + f, 0.5, 2.0));
        }
        pct[k] = 100 * (fmax - fmin);
        A.band_pct[k] = (float)pct[k];
    }
    int pk = std::min(2, cmax);   // the strongest mode past the frame-long trend (1 cycle = the content's drift, not flicker)
    for (int c = pk + 1; c <= cmax; ++c) if (power[c] > power[pk]) pk = c;
    A.band_peak_cycles = pk;
    A.band_peak_power = power[pk];
    std::vector<double> sp = pct;
    std::nth_element(sp.begin(), sp.begin() + N / 2, sp.end());
    A.band_median_pct = sp[N / 2];
    int worst = (int)(std::max_element(pct.begin(), pct.end()) - pct.begin());
    logf("  banding: strongest row frequency %d cycles per frame (period %d rows); flicker in cycles %d..%d: median %.2f %% peak-to-peak, largest %.2f %% (%s)\n",
         pk, H / pk, lo, hi, A.band_median_pct, pct[worst], A.paths[worst].filename().string().c_str());
}

// The rolling shutter's warping: for the baseline-1 and -2 pairs, frame j is warped onto frame i by the pair
// similarity from the global solve and each of rs_bands row bands gets its own translation (band_shifts). The band
// shifts of all pairs are chained per band into one offset per frame (chain_offsets, the stack's mean held at 0):
// the frame's displacement at that band of rows relative to the rigid model, in full-res px.
static int rolling_shutter_pass(const std::vector<float>& luma_stack, int N, int w, int h, double fx, double fy,
                                const std::vector<int>& devs, const AlignOptions& ao, AlignOut& A) {
    const int NB = ao.rs_bands;
    std::vector<std::pair<int, int>> pairs;
    for (int i = 0; i + 1 < N; ++i) pairs.push_back({i, i + 1});
    for (int i = 0; i + 2 < N; ++i) pairs.push_back({i, i + 2});
    std::sort(pairs.begin(), pairs.end());   // a device's chunk walks the stack in order: its level cache serves neighbouring pairs
    std::vector<BandShift> res(pairs.size());
    const int nd = (int)devs.size();
    std::vector<std::thread> workers;
    std::vector<std::string> errs(nd);
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < nd; ++r)
        workers.emplace_back([&, r] {
            const int d = devs[r];
            try {
                CK_THROW(cudaSetDevice(d));
                cudaStream_t s;
                CK_THROW(cudaStreamCreate(&s));
                AlignWork wk;
                wk.alloc(w, h);
                float* d_luma;
                CK_THROW(cudaMalloc(&d_luma, (size_t)w * h * sizeof(float)));
                std::map<int, AlignPyramid::Level> cache;
                std::list<int> lru;
                auto get = [&](int k) -> const AlignPyramid::Level& {
                    auto it = cache.find(k);
                    if (it != cache.end()) { lru.remove(k); lru.push_back(k); return it->second; }
                    while (cache.size() >= 6) { int v = lru.front(); lru.pop_front(); level0_free(cache[v]); cache.erase(v); }
                    CK_THROW(cudaMemcpyAsync(d_luma, luma_stack.data() + (size_t)k * w * h, (size_t)w * h * sizeof(float), cudaMemcpyHostToDevice, s));
                    AlignPyramid::Level L;
                    level0_build(L, d_luma, w, h, s);
                    CK_THROW(cudaStreamSynchronize(s));
                    lru.push_back(k);
                    return cache[k] = L;
                };
                const size_t e0 = pairs.size() * r / nd, e1 = pairs.size() * (r + 1) / nd;
                for (size_t e = e0; e < e1; ++e) {
                    if (ao.hooks.cancelled()) break;
                    auto [i, j] = pairs[e];
                    std::complex<double> zij = A.G.z[j] / A.G.z[i], tij = A.G.t[j] - zij * A.G.t[i];
                    const AlignPyramid::Level& Li = get(i);
                    const AlignPyramid::Level& Lj = get(j);
                    res[e] = band_shifts(Li, Lj, zij, tij, NB, wk, s);
                }
                for (auto& kv : cache) level0_free(kv.second);
                wk.free();
                cudaFree(d_luma);
                cudaStreamDestroy(s);
            } catch (const std::exception& ex) { errs[r] = ex.what(); }
        });
    for (auto& t : workers) t.join();
    for (int r = 0; r < nd; ++r) if (!errs[r].empty()) { logf("GPU %d: rolling shutter: %s\n", devs[r], errs[r].c_str()); return 1; }
    if (ao.hooks.cancelled()) return 3;
    A.rs_bands = NB;
    A.rs_dx.assign((size_t)N * NB, 0.f);
    A.rs_dy.assign((size_t)N * NB, 0.f);
    A.rs_px.assign(N, 0.f);
    const double band_px = (double)w * h / NB;
    int used = 0;
    for (int b = 0; b < NB; ++b) {
        std::vector<std::complex<double>> m(pairs.size());
        std::vector<double> wgt(pairs.size());
        for (size_t e = 0; e < pairs.size(); ++e) {
            const BandShift& B = res[e];
            // a band is usable when it correlates, covers most of its pixels and did not run away
            bool ok = (int)B.dx.size() == NB && B.ncc[b] > 0.3 && B.n[b] > 0.3 * band_px && std::hypot(B.dx[b], B.dy[b]) < 0.05 * h;
            m[e] = {ok ? B.dx[b] : 0.0, ok ? B.dy[b] : 0.0};
            wgt[e] = ok ? (pairs[e].second - pairs[e].first == 1 ? 1.0 : 0.5) : 0.0;
            if (ok) ++used;
        }
        auto dk = chain_offsets(N, pairs, m, wgt);
        // the chain also integrates what changes slowly along the stack (the specimen's relief read as a band-wise
        // drift, the lens' distortion changing with focus: a frame-to-frame bias of a tenth of a pixel adds up to
        // 20 px over 200 frames); the rolling shutter is the frame's own departure from its neighbours, so the
        // median of the six frames around it (itself left out: a frame that is the median of its window would
        // otherwise read 0) is taken out
        const int hw = std::min(3, (N - 1) / 2);
        std::vector<std::complex<double>> trend(N);
        std::vector<double> re, im;
        for (int k = 0; k < N; ++k) {
            int k0 = std::clamp(k - hw, 0, N - 1 - 2 * hw);
            re.clear(); im.clear();
            for (int q = k0; q <= k0 + 2 * hw; ++q) if (q != k) { re.push_back(dk[q].real()); im.push_back(dk[q].imag()); }
            if (re.empty()) { trend[k] = dk[k]; continue; }
            auto med_of = [](std::vector<double>& v) { std::sort(v.begin(), v.end()); size_t n = v.size(); return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]); };
            trend[k] = {med_of(re), med_of(im)};
        }
        for (int k = 0; k < N; ++k) {
            A.rs_dx[(size_t)k * NB + b] = (float)((dk[k].real() - trend[k].real()) / fx);
            A.rs_dy[(size_t)k * NB + b] = (float)((dk[k].imag() - trend[k].imag()) / fy);
        }
    }
    std::vector<double> sp(N);
    for (int k = 0; k < N; ++k) {
        float x0 = 1e30f, x1 = -1e30f, y0 = 1e30f, y1 = -1e30f;
        for (int b = 0; b < NB; ++b) {
            float x = A.rs_dx[(size_t)k * NB + b], y = A.rs_dy[(size_t)k * NB + b];
            x0 = std::min(x0, x); x1 = std::max(x1, x); y0 = std::min(y0, y); y1 = std::max(y1, y);
        }
        A.rs_px[k] = std::max(x1 - x0, y1 - y0);
        sp[k] = A.rs_px[k];
    }
    int worst = (int)(std::max_element(sp.begin(), sp.end()) - sp.begin());
    std::nth_element(sp.begin(), sp.begin() + N / 2, sp.end());
    A.rs_median_px = sp[N / 2];
    logf("  rolling shutter: %zu pairs x %d bands (%d usable) in %.2f s; band-to-band displacement median %.2f px, largest %.2f px (%s)\n",
         pairs.size(), NB, used, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), A.rs_median_px,
         A.rs_px[worst], A.paths[worst].filename().string().c_str());
    return 0;
}

// run_job: the alignment's per-frame row corrections into the fusion's warps when asked (FINDINGS 51): the gain
// table as measured; the band offsets interpolated linearly between the band centres onto every sensor row
static void apply_row_fixes(StackOptions& so, const AlignOut& A) {
    const int N = (int)A.paths.size(), H = A.H;
    so.po.row_gain.clear(); so.po.rs_dx.clear(); so.po.rs_dy.clear();
    if (so.fix_banding && (int)A.row_gain.size() == N * H) {
        so.po.row_gain = A.row_gain;
        logf("  banding correction: each frame's row gain goes into its warps\n");
    }
    if (so.fix_rs && A.rs_bands > 0 && (int)A.rs_dx.size() == N * A.rs_bands) {
        const int NB = A.rs_bands;
        so.po.rs_dx.assign((size_t)N * H, 0.f);
        so.po.rs_dy.assign((size_t)N * H, 0.f);
        for (int k = 0; k < N; ++k)
            for (int y = 0; y < H; ++y) {
                double p = (y + 0.5) * NB / H - 0.5;   // band coordinate: the band centres sit at 0, 1, ..., NB-1
                int b0 = (int)std::floor(p);
                double f = p - b0;
                int ba = std::clamp(b0, 0, NB - 1), bb = std::clamp(b0 + 1, 0, NB - 1);
                so.po.rs_dx[(size_t)k * H + y] = (float)((1 - f) * A.rs_dx[(size_t)k * NB + ba] + f * A.rs_dx[(size_t)k * NB + bb]);
                so.po.rs_dy[(size_t)k * H + y] = (float)((1 - f) * A.rs_dy[(size_t)k * NB + ba] + f * A.rs_dy[(size_t)k * NB + bb]);
            }
        logf("  rolling shutter correction: each frame's row offsets go into its warps\n");
    }
}

int run_align(AlignOptions& ao, AlignOut& A) {
    ao.readers = std::clamp(ao.readers, 1, 256);
    ao.pair_threads = std::clamp(ao.pair_threads, 1, 64);
    ao.rs_bands = std::clamp(ao.rs_bands, 0, 64);   // row bands of the rolling-shutter probe: 100000 took two minutes
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
            std::string low = name;
            for (char& ch : low) ch = (char)tolower((unsigned char)ch);
            // a dust map (a photograph of a white background) kept beside the frames is not one of them
            const bool dust_file = low.find("dust map") != std::string::npos || low.find("dustmap") != std::string::npos || low.find("dust_map") != std::string::npos || low.find("dust-map") != std::string::npos;
            if (dust_file) { logf("  skipping %s (a dust map by its name)", name.c_str()); continue; }
            if (e.is_regular_file() && is_frame_file(e.path()) && name.rfind(prefix, 0) == 0) paths.push_back(e.path());
        }
        // by name as a person reads it: digits as numbers and without regard to case, so frame9 comes before frame10
        // and IMG_2 beside img_3 (plain code order stacked unpadded numbering as 1, 10, 100, 11, ...). Names that are
        // equal that way fall back to the plain order, so the result does not depend on the directory's listing.
        auto natural_less = [](const fs::path& pa, const fs::path& pb) {
            const std::wstring a = pa.filename().wstring(), b = pb.filename().wstring();
            size_t i = 0, j = 0;
            while (i < a.size() && j < b.size()) {
                if (iswdigit(a[i]) && iswdigit(b[j])) {
                    size_t i0 = i, j0 = j;
                    while (i < a.size() && a[i] == L'0') ++i;
                    while (j < b.size() && b[j] == L'0') ++j;
                    size_t i1 = i, j1 = j;
                    while (i < a.size() && iswdigit(a[i])) ++i;
                    while (j < b.size() && iswdigit(b[j])) ++j;
                    if (i - i1 != j - j1) return i - i1 < j - j1;   // more digits = larger
                    int c = a.compare(i1, i - i1, b, j1, j - j1);
                    if (c) return c < 0;
                    if (i - i0 != j - j0) return i - i0 > j - j0;   // 007 before 7
                    continue;
                }
                wint_t ca = towlower(a[i]), cb = towlower(b[j]);
                if (ca != cb) return ca < cb;
                ++i; ++j;
            }
            if (a.size() - i != b.size() - j) return a.size() - i < b.size() - j;
            return pa < pb;
        };
        std::sort(paths.begin(), paths.end(), natural_less);
    }
    // frames are the files of the most common size; anything else in the folder (a previous stack's
    // outputs, say) is left out and reported. Between two sizes with as many files each (the raw files and
    // their exports side by side) the developed files win: 2 points per file, 1 more when it is not raw.
    {
        std::vector<TiffInfo> infos;
        std::map<std::pair<unsigned, unsigned>, int> sizes;
        for (auto& p : paths) {
            infos.push_back(tiff_probe(p));
            if (infos.back().ok()) sizes[{infos.back().width, infos.back().height}] += infos.back().raw ? 2 : 3;
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
    if (ao.max_frames > 0 && N > ao.max_frames) {   // said before any work: the depth stage refused only after alignment and pass 1
        logf("the stack has %d frames: at most %d can be fused in one render. Leave frames out (every second one, or one end) and render again.", N, ao.max_frames);
        note_fail(Fail::TooManyFrames, std::to_string(N));
        return 1;
    }
    if (std::string gp = gpu_problem(devs, true); !gp.empty()) { logf("%s", gp.c_str()); note_fail(Fail::GpuFault, gp); return 1; }
    int ndev = 0;
    CK(cudaGetDeviceCount(&ndev));
    if (devs.empty()) for (int i = 0; i < ndev; ++i) devs.push_back(i);
    TiffInfo t0 = tiff_probe(paths[0]);
    if (!t0.ok()) { logf("%s: %s\n", paths[0].string().c_str(), t0.error.c_str()); return 1; }
    const int W = (int)t0.width, H = (int)t0.height;
    if (W < 64 || H < 64) { logf("the frames are %d x %d px: alignment needs at least 64 x 64\n", W, H); return 1; }
    const int w = std::max(16, (int)std::nearbyint(W * align_scale)), h = std::max(16, (int)std::nearbyint(H * align_scale));
    const double fx = (double)w / W, fy = (double)h / H;
    auto t_all = std::chrono::steady_clock::now();

    // ---- pass 0: luma at the align scale, for every frame, to the host
    std::vector<float> luma_stack((size_t)N * w * h);
    std::vector<float> rows_all((size_t)N * H);   // each frame's row-mean luma at full resolution (banding, FINDINGS 51)
    {
        struct Slot { float *d_rgb = nullptr, *d_y = nullptr, *d_small = nullptr, *d_rows = nullptr, *d_dust = nullptr; AreaResizer rz; AsyncPreview fp; };
        std::vector<std::vector<Slot>> slots(ndev);
        // the dust map (dust.h): this pass measures how deep each speck's shadow is in every frame; the correction made
        // from that goes into the passes that follow. The alignment itself reads the frames as they are.
        std::vector<DevDustProbe> dd(ndev);
        const size_t dn = ao.dust && ao.dust->w == W && ao.dust->h == H ? ao.dust->px.size() : 0;
        std::vector<float> dust_ratios(dn * (size_t)N, 1.f);
        StreamerOptions opt;
        opt.devices = devs;
        opt.streams_per_gpu = 3;
        opt.readers = readers;
        opt.cancel = hooks.cancel;
        opt.cache = ao.cache;
        if (ao.cache) {
            ready_cache(ao.cache, paths, devs);
            logf("frames resident in host memory: %d of %d, budget %.1f GB\n", ao.cache->count(), N, ao.cache->budget() / 1e9);
        }
        // the stage's name tells the front end whether this pass reads the disk or only converts resident frames
        // (0.6.25: with the frames read while the folder's thumbnails loaded, "reading" looked like a second read)
        const char* stage_name = ao.cache && ao.cache->count() >= N ? "frames from memory" : "reading frames";
        std::atomic<int> done{0};
        // the slots go on every way out before the end of the pass (a card out of memory part-way through them)
        struct Tidy {
            std::vector<std::vector<Slot>>& slots; const std::vector<int>& devs; std::vector<DevDustProbe>& dd; bool done = false;
            ~Tidy() {
                if (done) return;
                for (int d : devs) {
                    if (cudaSetDevice(d) != cudaSuccess) continue;
                    for (auto& S : slots[d]) { cudaFree(S.d_rgb); cudaFree(S.d_y); cudaFree(S.d_small); cudaFree(S.d_rows); cudaFree(S.d_dust); S.rz.free(); S.fp.free(); }
                    dd[d].free();
                }
                cudaGetLastError();
            }
        } tidy{slots, devs, dd};
        // slots by the memory each card has free: a frame as float RGB and its luma per slot (16 bytes a pixel) beside
        // the streamer's copy of the file (up to 6). Three were taken whatever the card and the frame size, which a
        // 4 to 8 GB card with 60 to 100 Mpx frames does not hold.
        opt.streams_by_device.assign(devs.size(), opt.streams_per_gpu);
        for (size_t i = 0; i < devs.size(); ++i) {
            size_t fr = 0, tot = 0;
            CK(cudaSetDevice(devs[i]));
            CK(cudaMemGetInfo(&fr, &tot));
            const size_t per = (size_t)W * H * 24 + ((size_t)64 << 20), keep = (size_t)512 << 20;
            const int fit = fr > keep ? (int)std::min<size_t>((size_t)opt.streams_per_gpu, (fr - keep) / per) : 0;
            if (fit < 1) {
                logf("GPU %d memory ran out (reading a %d x %d frame needs %.2f GB, %.2f GB free)", devs[i], W, H, per / 1e9, fr / 1e9);
                return 4;
            }
            if (fit < opt.streams_per_gpu) logf("  GPU %d reads with %d stream%s (%.2f GB free)", devs[i], fit, fit == 1 ? "" : "s", fr / 1e9);
            opt.streams_by_device[i] = fit;
        }
        for (size_t di = 0; di < devs.size(); ++di) {
            const int d = devs[di];
            CK(cudaSetDevice(d));
            dd[d].upload(ao.dust.get(), W, H);
            slots[d].resize(opt.streams_by_device[di]);
            for (auto& S : slots[d]) {
                CK(cudaMalloc(&S.d_rgb, (size_t)W * H * 3 * sizeof(float)));
                CK(cudaMalloc(&S.d_y, (size_t)W * H * sizeof(float)));
                CK(cudaMalloc(&S.d_small, (size_t)w * h * sizeof(float)));
                CK(cudaMalloc(&S.d_rows, (size_t)H * sizeof(float)));
                if (dn) CK(cudaMalloc(&S.d_dust, dn * sizeof(float)));
                S.rz.build(W, H, w, h);
                if (hooks.frame) S.fp.alloc_view(hooks, W, H);
            }
        }
        auto on_frame = [&](const DeviceFrame& f) {
            auto& S = slots[f.dev][f.slot];
            const bool view = view_due(hooks, S.fp);
            S.fp.guard(f.stream);
            convert_to_rgb_f32(f.d_pixels(), *f.info, S.d_rgb, nullptr, f.stream);
            if (dn) {   // each speck's shadow in this frame, against its surroundings
                dd[f.dev].measure(S.d_rgb, S.d_dust, f.stream);
                cudaMemcpyAsync(dust_ratios.data() + (size_t)f.k * dn, S.d_dust, dn * sizeof(float), cudaMemcpyDeviceToHost, f.stream);
            }
            if (view) enqueue_view(hooks, S.fp, f.stream, f.k, S.d_rgb, W, H);   // the frame as read (unaligned); delivered at the slot's next turn
            luma(S.d_rgb, S.d_y, (size_t)W * H, f.stream);
            S.rz.run(S.d_y, S.d_small, 1, f.stream);
            cudaMemcpyAsync(luma_stack.data() + (size_t)f.k * w * h, S.d_small, (size_t)w * h * sizeof(float), cudaMemcpyDeviceToHost, f.stream);
            row_mean(S.d_y, W, H, S.d_rows, f.stream);
            cudaMemcpyAsync(rows_all.data() + (size_t)f.k * H, S.d_rows, (size_t)H * sizeof(float), cudaMemcpyDeviceToHost, f.stream);
            if (hooks.progress) hooks.progress(stage_name, (double)(++done) / N);
        };
        StreamerStats st = stream_frames(paths, opt, on_frame);
        for (auto& e : st.errors) logf("  error: %s\n", e.c_str());
        for (int d : devs) {
            CK(cudaSetDevice(d));
            CK(cudaDeviceSynchronize());
            for (auto& S : slots[d]) { S.fp.flush(hooks); cudaFree(S.d_rgb); cudaFree(S.d_y); cudaFree(S.d_small); cudaFree(S.d_rows); cudaFree(S.d_dust); S.rz.free(); S.fp.free(); }
            dd[d].free();
        }
        tidy.done = true;
        // a frame that could not be read has no luma: the alignment of zeros ended much later as "no usable pair",
        // or went through on garbage. The pass's errors end the job here, with the file named.
        if (!hooks.cancelled()) if (int r = stream_fail(st)) return r;
        if (dn && !hooks.cancelled()) {
            A.dust = calibrate_dust(*ao.dust, dust_ratios, N);
            logf("dust map: measured on the %d frames: %zu pixels brightened, %zu filled in; the deepest shadow keeps %.0f %% of the light (%.0f %% in the map)\n",
                 N, A.dust->idx.size(), A.dust->heal.size(), 100.0 * A.dust->deepest, 100.0 * A.dust->map_deepest);
        }
        logf("align: %d frames %dx%d -> luma %dx%d in %.2f s%s\n", N, W, H, w, h, st.seconds, source_note(st).c_str());
        if (getenv("STACKER_MEM_DEBUG"))
            for (int d : devs) { size_t fr, tot; cudaSetDevice(d); cudaMemGetInfo(&fr, &tot); logf("  [mem] GPU %d after pass 0: %.2f GB free", d, fr / 1e9); }
        if (hooks.cancelled()) return 3;
    }

    banding_fit(rows_all, N, H, ao, A);

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
    // ... as far as the card's free memory goes (a pyramid is about 24 bytes per pixel of the align grid): the cache
    // only saves rebuilding, so a smaller one changes the time, not the transforms. Each thread holds two at once.
    for (int d : devs) {
        size_t fr = 0, tot = 0;
        if (cudaSetDevice(d) != cudaSuccess || cudaMemGetInfo(&fr, &tot) != cudaSuccess) { cudaGetLastError(); continue; }
        const size_t per = (size_t)w * h * 24, keep = ((size_t)768 << 20) + (size_t)T * w * h * 64;
        const int fit = fr > keep ? (int)std::min<size_t>(1000, (fr - keep) / std::max<size_t>(per, 1)) : 0;
        const int cap = std::max(2 * T + 2, std::min(caches[d].cap, fit));
        if (cap < caches[d].cap) logf("  GPU %d keeps %d pyramids (%.2f GB free)", d, cap, fr / 1e9);
        caches[d].cap = cap;
    }
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
    if (ao.rs_bands > 0 && N >= 3) {
        if (int r = rolling_shutter_pass(luma_stack, N, w, h, fx, fy, devs, ao, A)) return r;
        if (hooks.cancelled()) return 3;
    }
    if (ao.um_per_px < 0) ao.um_per_px = -ao.um_per_px * 1000.0 / W;   // from --fov-mm
    A.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_all).count();
    logf("  solve: %d/%d pairs usable, %d outliers, ncc min %.4f median %.4f; ref #%d; native crop %dx%d (L%d R%d T%d B%d); scale spread %.3f%%%s; %.2f s\n",
           A.G.pairs_used, A.G.pairs_total, A.G.outliers, A.G.ncc_min, A.G.ncc_median, A.ref, A.rect[2] - A.rect[0] + 1, A.rect[3] - A.rect[1] + 1,
           A.rect[0], W - 1 - A.rect[2], A.rect[1], H - 1 - A.rect[3], 100 * (A.smax / A.smin - 1), ao.lock_scale ? " (scale locked)" : "", A.seconds);
    return 0;
}

std::string report_json(const AlignOptions& ao, const AlignOut& A) {
    // an alignment that failed early (no GPU, say) has its paths but no transforms: then no per-frame rows
    const int N = (int)std::min(A.paths.size(), A.T.size()), W = A.W, H = A.H;
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
    const bool have_band = (int)A.band_pct.size() == N, have_rs = A.rs_bands > 0 && (int)A.rs_px.size() == N;
    for (int k = 0; k < N; ++k) {
        std::complex<double> z(A.T[k].zr, A.T[k].zi);
        add("  {\"i\": %d, \"file\": \"%s\", \"s\": %.17g, \"theta_deg\": %.17g, \"tx\": %.17g, \"ty\": %.17g", k,
            A.paths[k].filename().string().c_str(), std::abs(z), std::arg(z) * 180.0 / 3.14159265358979323846, A.T[k].tx, A.T[k].ty);
        if (have_band) add(", \"band_pct\": %.4g", A.band_pct[k]);
        if (have_rs) {   // FINDINGS 51: the frame's displacement per row band (full-res px, top to bottom) and its spread
            add(", \"rs_px\": %.4g, \"rs_dx\": [", A.rs_px[k]);
            for (int b = 0; b < A.rs_bands; ++b) add("%s%.3f", b ? ", " : "", A.rs_dx[(size_t)k * A.rs_bands + b]);
            add("%s", "], \"rs_dy\": [");
            for (int b = 0; b < A.rs_bands; ++b) add("%s%.3f", b ? ", " : "", A.rs_dy[(size_t)k * A.rs_bands + b]);
            add("%s", "]");
        }
        add("}%s\n", k + 1 < N ? "," : "");
    }
    s += " ],\n";
    if (have_band || have_rs)
        add(" \"rolling_shutter\": {\"bands\": %d, \"median_px\": %.4g, \"banding_cycles\": [%d, %d], \"banding_peak_cycles\": %d, \"banding_median_pct\": %.4g},\n",
            A.rs_bands, A.rs_median_px, ao.band_lo, ao.band_hi, A.band_peak_cycles, A.band_median_pct);
    s += " \"transform_convention\": \"reference-grid centred px -> frame centred px: p_k = s e^{i theta} p + t\"\n}\n";
    return s;
}

void write_report(const AlignOptions& ao, const AlignOut& A, const fs::path& od) {
    const int N = (int)A.paths.size();
    const int* rect = A.rect;
    if (!make_out_dir(od)) return;   // (logged and noted)
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

// The job's dust map, when it fits the frames: a map of another size (another camera, a crop) corrects nothing and
// would put its specks' positions in the wrong places, so it is left out and that is said.
static std::shared_ptr<const DustFix> use_dust(const JobOptions& jo, const AlignOptions& ao) {
    if (!jo.dust || jo.dust->px.empty()) return nullptr;
    fs::path first = !ao.paths.empty() ? ao.paths.front() : fs::path();
    if (first.empty()) {   // a folder: its first frame file
        std::error_code ec;
        for (auto& e : fs::directory_iterator(ao.dir, ec)) if (e.is_regular_file() && is_frame_file(e.path())) { first = e.path(); break; }
    }
    TiffInfo t = first.empty() ? TiffInfo() : tiff_probe(first);
    if (t.ok() && ((int)t.width != jo.dust->w || (int)t.height != jo.dust->h)) {
        logf("dust map %s is %d x %d, the frames are %u x %u: it is not applied", jo.dust->name.c_str(), jo.dust->w, jo.dust->h, t.width, t.height);
        return nullptr;
    }
    logf("dust map %s: %d spot%s, %zu pixels in and around them", jo.dust->name.c_str(), jo.dust->spots,
         jo.dust->spots == 1 ? "" : "s", jo.dust->px.size());
    return jo.dust;
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
    // an output folder that cannot be made is found now, not after the alignment and the whole fusion
    if (so.write_outputs && !make_out_dir(out)) return 1;
    ao.max_frames = kMaxFrames;
    ao.dust = so.po.dust = use_dust(jo, ao);
    if (int r = run_align(ao, A)) return r;
    if (A.dust) so.po.dust = A.dust;   // the dust map with its depths measured on these frames
    write_report(ao, A, out);
    so.po.paths = A.paths;
    so.po.T = A.T;
    apply_row_fixes(so, A);
    so.po.g.x0 = A.rect[0]; so.po.g.y0 = A.rect[1]; so.po.g.x1 = A.rect[2]; so.po.g.y1 = A.rect[3];
    so.out = so.write_outputs ? out : std::string();   // the benchmark computes everything but writes nothing
    if (int r = run_stack(so)) return r;
    logf("run: %zu frames aligned and stacked in %.2f s -> %s", A.paths.size(),
         std::chrono::duration<double>(std::chrono::steady_clock::now() - t_all).count(), out.c_str());
    return 0;
}

int run_sweep_job(JobOptions& jo, const std::vector<SweepVariant>& variants, const std::string& out_dir, AlignOut& A, Hooks hooks) {
    AlignOptions& ao = jo.align;
    StackOptions& so = jo.stack;
    if (so.po.opt.devices.empty()) so.po.opt.devices = ao.devs; else if (ao.devs.empty()) ao.devs = so.po.opt.devices;
    so.po.opt.readers = ao.readers;
    ao.out = out_dir;
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
    if (!make_out_dir(out_dir)) return 1;
    ao.max_frames = kMaxFrames;
    ao.dust = so.po.dust = use_dust(jo, ao);
    if (int r = run_align(ao, A)) return r;
    if (A.dust) so.po.dust = A.dust;
    write_report(ao, A, out_dir);   // report.json + list.txt beside the variants: `stackcli sweep` can continue from them
    so.po.paths = A.paths;
    so.po.T = A.T;
    apply_row_fixes(so, A);
    so.po.g.x0 = A.rect[0]; so.po.g.y0 = A.rect[1]; so.po.g.x1 = A.rect[2]; so.po.g.y1 = A.rect[3];
    if (so.outputs.empty()) so.outputs = {"B"};
    so.out.clear();
    so.result = nullptr;
    int rc = run_sweep(so, variants, out_dir);
    logf("sweep: %zu frames aligned, %zu variants in %.1f s -> %s", A.paths.size(), variants.size(),
         std::chrono::duration<double>(std::chrono::steady_clock::now() - t_all).count(), out_dir.c_str());
    return rc;
}

}  // namespace stackcore

