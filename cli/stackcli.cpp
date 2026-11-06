// stackcli: headless runner (DESIGN.md repo layout). Step 1 commands:
//
//   stackcli probe  FILE.tif                       TIFF layout as the reader sees it
//   stackcli dump   FILE.tif OUT.f32 [--gpu N]     read -> GPU -> float RGB -> raw file (H x W x 3 float32)
//   stackcli stream DIR [--prefix IMG_] [--gpus 0,1] [--streams 3] [--readers 16] [--pool N] [--sums OUT.txt]
//                                                  stream every frame through the GPUs: throughput + per-frame checksums
//   stackcli warp   FILE.tif OUT.f32 --z ZR,ZI --t TX,TY --rect X0,Y0,X1,Y1 [--scale 1|0.5] [--quantize] [--gpu N]
//                                                  read -> GPU -> fusion.load_aligned equivalent -> raw float RGB
//   stackcli fold   LIST.txt --rect X0,Y0,X1,Y1 --out DIR [--scale 1|0.5] [--radius-px 6] [--power 4]
//                   [--frames 0,5,9] [--gpus 0,1] [--streams 2] [--readers 16]
//                   [--volume] [--depth-fraction 0.25] [--depth-fm-px 6] [--vol-agg-px 32]
//                                                  pass 1 (A, B0, C pyramid) over the frames in LIST (lines:
//                                                  path zr zi tx ty), one partial state per GPU, merged on the host,
//                                                  written as raw arrays for tools/validate/check_fold.py. With
//                                                  --volume also the depth-map input: per-frame focus/brightness
//                                                  slices at the depth resolution and the guided-filter aggregated
//                                                  focus volume (tools/validate/check_volume.py)
//   stackcli depth  VOLDIR --out DIR [--crop X0,Y0,W,H] [--gpu N] [--dirs 8] ...
//                                                  SGM depth from a `fold --volume` output (tools/validate/check_depth.py)
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "stackcore/align.h"
#include "stackcore/compose.h"

#include <atomic>
#include <list>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "stackcore/convert.h"
#include "stackcore/depth.h"
#include "stackcore/fold.h"
#include "stackcore/streamer.h"
#include "stackcore/tiff.h"
#include "stackcore/volume.h"
#include "stackcore/warp.h"

namespace fs = std::filesystem;
using namespace stackcore;

#define CK(x)                                                                                   \
    do {                                                                                        \
        cudaError_t e_ = (x);                                                                   \
        if (e_ != cudaSuccess) {                                                                \
            fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
            return 1;                                                                           \
        }                                                                                       \
    } while (0)
#define CK_THROW(x)                                                                             \
    do {                                                                                        \
        cudaError_t e_ = (x);                                                                   \
        if (e_ != cudaSuccess) throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(e_)); \
    } while (0)

static int usage() {
    fprintf(stderr,
            "usage:\n"
            "  stackcli run DIR [--out DIR] [--prefix IMG_] [--scale 1|0.5] [--outputs B,C,Cn] [--ref middle] [--lock-scale 1]\n"
            "               [--fov-mm X | --um-per-px X] [--gpus 0,1] [--streams 2] [--align-scale 0.25] [--baselines 1,2,4,8]\n"
            "               align the frames in DIR and fuse them: report.json + 16-bit TIFFs in DIR_stacker (or --out)\n"
            "  developer commands:\n"
            "  stackcli probe FILE.tif\n"
            "  stackcli dump FILE.tif OUT.f32 [--gpu N]\n"
            "  stackcli stream DIR [--prefix IMG_] [--gpus 0,1] [--streams 3] [--readers 16] [--pool N] [--sums OUT.txt]\n"
            "  stackcli warp FILE.tif OUT.f32 --z ZR,ZI --t TX,TY --rect X0,Y0,X1,Y1 [--scale 1|0.5] [--quantize] [--gpu N]\n"
            "  stackcli fold LIST.txt --rect X0,Y0,X1,Y1 --out DIR [--scale 1|0.5] [--radius-px 6] [--power 4]\n"
            "                [--frames 0,5,9] [--gpus 0,1] [--streams 2] [--readers 16] [--volume]\n"
            "  stackcli depth VOLDIR --out DIR [--crop X0,Y0,W,H] [--gpu N] [--dirs 8]\n"
            "  stackcli stack LIST.txt --rect X0,Y0,X1,Y1 --out DIR [--scale 1|0.5] [--gpus 0,1] [--streams 2]\n"
            "                 [--refine-r 12] [--refine-fm-px 9] [--soft-px 3]      the whole fusion -> 16-bit TIFFs\n"
            "  stackcli align DIR --out DIR [--prefix IMG_] [--align-scale 0.25] [--baselines 1,2,4,8] [--ref middle]\n"
            "                 [--gpus 0,1] [--pairs i:j,...]      alignment -> report.json + list.txt (+ per-pair dump)\n");
    return 2;
}

static void print_info(const fs::path& p, const TiffInfo& t);

static std::vector<double> parse_doubles(const std::string& s) {
    std::vector<double> v;
    size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find(',', i);
        if (j == std::string::npos) j = s.size();
        if (j > i) v.push_back(atof(s.substr(i, j - i).c_str()));
        i = j + 1;
    }
    return v;
}

// read FILE into a device float RGB frame; returns 0 and fills info/d_rgb on success
static int load_frame_to_device(const fs::path& in, TiffInfo& info, float*& d_rgb) {
    info = tiff_probe(in);
    print_info(in, info);
    if (!info.ok()) return 1;
    size_t cap = tiff_buffer_capacity(info.pixel_bytes());
    uint8_t* h = nullptr;
    CK(cudaHostAlloc((void**)&h, cap, cudaHostAllocPortable));
    TiffRead rd = tiff_read_pixels(in, info, h, cap);
    if (!rd.ok()) { fprintf(stderr, "read failed: %s\n", rd.error.c_str()); return 1; }
    printf("  read %zu bytes in %.1f ms (%s), pixels at +%zu\n", rd.bytes, rd.ms, rd.unbuffered ? "unbuffered" : "buffered",
           rd.data_offset_in_buf);
    uint8_t* d_src;
    size_t npx = (size_t)info.width * info.height;
    CK(cudaMalloc(&d_src, rd.bytes));
    CK(cudaMalloc(&d_rgb, npx * 3 * sizeof(float)));
    CK(cudaMemcpy(d_src, h + rd.data_offset_in_buf, rd.bytes, cudaMemcpyHostToDevice));
    convert_to_rgb_f32(d_src, info, d_rgb, nullptr, 0);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    CK(cudaFree(d_src));
    CK(cudaFreeHost(h));
    return 0;
}

static int write_f32(const fs::path& out, const float* d_buf, size_t n, int w, int h) {
    std::vector<float> buf(n);
    CK(cudaMemcpy(buf.data(), d_buf, n * sizeof(float), cudaMemcpyDeviceToHost));
    FILE* f = fopen(out.string().c_str(), "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", out.string().c_str()); return 1; }
    fwrite(buf.data(), sizeof(float), n, f);
    fclose(f);
    printf("  wrote %s (%d x %d x 3 float32)\n", out.string().c_str(), h, w);
    return 0;
}

static int cmd_warp(int argc, char** argv) {
    if (argc < 4) return usage();
    fs::path in = argv[2], out = argv[3];
    Similarity T;
    Grid g;
    bool quantize = false, have_rect = false;
    int gpu = 0;
    for (int i = 4; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--quantize") { quantize = true; continue; }
        if (i + 1 >= argc) return usage();
        std::string v = argv[++i];
        if (a == "--z") { auto d = parse_doubles(v); if (d.size() != 2) return usage(); T.zr = d[0]; T.zi = d[1]; }
        else if (a == "--t") { auto d = parse_doubles(v); if (d.size() != 2) return usage(); T.tx = d[0]; T.ty = d[1]; }
        else if (a == "--rect") { auto d = parse_doubles(v); if (d.size() != 4) return usage(); g.x0 = (int)d[0]; g.y0 = (int)d[1]; g.x1 = (int)d[2]; g.y1 = (int)d[3]; have_rect = true; }
        else if (a == "--scale") g.scale = atof(v.c_str());
        else if (a == "--gpu") gpu = atoi(v.c_str());
        else return usage();
    }
    if (g.scale != 1.0 && g.scale != 0.5) { fprintf(stderr, "--scale must be 1 or 0.5\n"); return 2; }
    CK(cudaSetDevice(gpu));
    TiffInfo info;
    float* d_rgb = nullptr;
    if (int r = load_frame_to_device(in, info, d_rgb)) return r;
    g.W = (int)info.width;
    g.H = (int)info.height;
    if (!have_rect) { g.x0 = 0; g.y0 = 0; g.x1 = g.W - 1; g.y1 = g.H - 1; }
    float *d_scratch = nullptr, *d_out = nullptr;
    if (g.scale != 1.0) CK(cudaMalloc(&d_scratch, (size_t)g.work_w() * g.work_h() * 3 * sizeof(float)));
    size_t n_out = (size_t)g.out_w() * g.out_h() * 3;
    CK(cudaMalloc(&d_out, n_out * sizeof(float)));
    double M[6];
    affine_for(T, g, M);
    printf("  grid %d x %d at scale %g -> %d x %d; M = [%.9f %.9f %.4f; %.9f %.9f %.4f]; %s positions\n", g.W, g.H, g.scale,
           g.out_w(), g.out_h(), M[0], M[1], M[2], M[3], M[4], M[5], quantize ? "1/32 px (OpenCV)" : "exact");
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    load_aligned(d_rgb, g, T, d_scratch, d_out, quantize, 0);   // warm-up
    CK(cudaEventRecord(e0));
    for (int i = 0; i < 10; ++i) load_aligned(d_rgb, g, T, d_scratch, d_out, quantize, 0);
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    CK(cudaGetLastError());
    float ms;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    printf("  load_aligned: %.2f ms per frame (median-ish of 10)\n", ms / 10);
    return write_f32(out, d_out, n_out, g.out_w(), g.out_h());
}

static void print_info(const fs::path& p, const TiffInfo& t) {
    printf("%s\n", p.filename().string().c_str());
    if (!t.ok()) { printf("  unusable: %s\n", t.error.c_str()); return; }
    printf("  %u x %u, %u-bit x %u samples, photometric %u, %s, %s\n", t.width, t.height, t.bits, t.samples,
           t.photometric, t.big_endian ? "big-endian" : "little-endian", t.bigtiff ? "BigTIFF" : "classic");
    printf("  %zu strips, %s, pixel data at %llu, %llu bytes (file %llu)\n", t.strip_offsets.size(),
           t.contiguous ? "contiguous" : "with gaps", (unsigned long long)t.data_offset,
           (unsigned long long)t.data_bytes, (unsigned long long)t.file_size);
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

static int cmd_probe(int argc, char** argv) {
    if (argc < 3) return usage();
    print_info(argv[2], tiff_probe(argv[2]));
    return 0;
}

static int cmd_dump(int argc, char** argv) {
    if (argc < 4) return usage();
    fs::path in = argv[2], out = argv[3];
    int gpu = 0;
    for (int i = 4; i + 1 < argc; i += 2)
        if (std::string(argv[i]) == "--gpu") gpu = atoi(argv[i + 1]);
    TiffInfo info = tiff_probe(in);
    print_info(in, info);
    if (!info.ok()) return 1;
    CK(cudaSetDevice(gpu));
    size_t cap = tiff_buffer_capacity(info.pixel_bytes());
    uint8_t* h = nullptr;
    CK(cudaHostAlloc((void**)&h, cap, cudaHostAllocPortable));
    TiffRead rd = tiff_read_pixels(in, info, h, cap);
    if (!rd.ok()) { fprintf(stderr, "read failed: %s\n", rd.error.c_str()); return 1; }
    printf("  read %zu bytes in %.1f ms (%s), pixels at +%zu\n", rd.bytes, rd.ms, rd.unbuffered ? "unbuffered" : "buffered",
           rd.data_offset_in_buf);
    uint8_t* d_src;
    float* d_rgb;
    double* d_sum;
    size_t npx = (size_t)info.width * info.height;
    CK(cudaMalloc(&d_src, rd.bytes));
    CK(cudaMalloc(&d_rgb, npx * 3 * sizeof(float)));
    CK(cudaMalloc(&d_sum, sizeof(double)));
    CK(cudaMemset(d_sum, 0, sizeof(double)));
    CK(cudaMemcpy(d_src, h + rd.data_offset_in_buf, rd.bytes, cudaMemcpyHostToDevice));
    convert_to_rgb_f32(d_src, info, d_rgb, d_sum, 0);
    CK(cudaGetLastError());
    std::vector<float> rgb(npx * 3);
    double sum = 0;
    CK(cudaMemcpy(rgb.data(), d_rgb, rgb.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost));
    FILE* f = fopen(out.string().c_str(), "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", out.string().c_str()); return 1; }
    fwrite(rgb.data(), sizeof(float), rgb.size(), f);
    fclose(f);
    printf("  wrote %s (%u x %u x 3 float32), checksum %.6f\n", out.string().c_str(), info.height, info.width, sum);
    return 0;
}

static int cmd_stream(int argc, char** argv) {
    if (argc < 3) return usage();
    fs::path dir = argv[2];
    std::string prefix = "IMG_", sums_path;
    StreamerOptions opt;
    for (int i = 3; i + 1 < argc; i += 2) {
        std::string a = argv[i], v = argv[i + 1];
        if (a == "--prefix") prefix = v;
        else if (a == "--gpus") opt.devices = parse_list(v);
        else if (a == "--streams") opt.streams_per_gpu = atoi(v.c_str());
        else if (a == "--readers") opt.readers = atoi(v.c_str());
        else if (a == "--pool") opt.pool = atoi(v.c_str());
        else if (a == "--sums") sums_path = v;
        else return usage();
    }
    std::vector<fs::path> paths;
    for (auto& e : fs::directory_iterator(dir)) {
        auto name = e.path().filename().string();
        auto ext = e.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (e.is_regular_file() && (ext == ".tif" || ext == ".tiff") && name.rfind(prefix, 0) == 0) paths.push_back(e.path());
    }
    std::sort(paths.begin(), paths.end());
    if (paths.empty()) { fprintf(stderr, "no %s*.tif in %s\n", prefix.c_str(), dir.string().c_str()); return 1; }
    const int N = (int)paths.size();

    int ndev = 0;
    CK(cudaGetDeviceCount(&ndev));
    std::vector<int> devs = opt.devices;
    if (devs.empty()) for (int i = 0; i < ndev; ++i) devs.push_back(i);
    // largest frame in the set sizes the per-slot float buffers (headers only, cheap)
    size_t slot_px = 0;
    for (auto& p : paths) {
        TiffInfo t = tiff_probe(p);
        if (t.ok()) slot_px = std::max(slot_px, (size_t)t.width * t.height);
    }
    if (!slot_px) { fprintf(stderr, "no readable frames\n"); return 1; }
    // per device: one float frame per stream slot (reused), a per-frame checksum array
    std::vector<std::vector<float*>> d_rgb(ndev);
    std::vector<double*> d_sums(ndev, nullptr);
    for (int d : devs) {
        CK(cudaSetDevice(d));
        d_rgb[d].resize(opt.streams_per_gpu);
        for (auto& p : d_rgb[d]) CK(cudaMalloc(&p, slot_px * 3 * sizeof(float)));
        CK(cudaMalloc(&d_sums[d], N * sizeof(double)));
        CK(cudaMemset(d_sums[d], 0, N * sizeof(double)));
    }

    auto on_frame = [&](const DeviceFrame& f) {
        size_t npx = (size_t)f.info->width * f.info->height;
        if (npx > slot_px) { fprintf(stderr, "frame %d larger than the slot buffers\n", f.k); return; }
        convert_to_rgb_f32(f.d_pixels(), *f.info, d_rgb[f.dev][f.slot], d_sums[f.dev] + f.k, f.stream);
    };

    printf("streaming %d frames from %s on %zu GPU(s), %d streams/GPU, %d readers\n", N, dir.string().c_str(), devs.size(),
           opt.streams_per_gpu, opt.readers);
    StreamerStats st = stream_frames(paths, opt, on_frame);
    for (auto& e : st.errors) fprintf(stderr, "  error: %s\n", e.c_str());
    printf("%d frames, %.2f GB in %.3f s: %.1f frames/s, %.2f GB/s; %d unbuffered; mean read %.1f ms\n", st.frames,
           st.bytes / 1e9, st.seconds, st.frames / st.seconds, st.bytes / st.seconds / 1e9, st.unbuffered,
           st.frames ? st.read_ms_sum / st.frames : 0.0);
    for (size_t i = 0; i < devs.size(); ++i) printf("  GPU %d: %d frames\n", devs[i], st.frames_per_device[i]);

    // merge the per-frame checksums (each frame ran on exactly one device)
    std::vector<double> sums(N, 0.0), part(N);
    for (int d : devs) {
        CK(cudaSetDevice(d));
        CK(cudaMemcpy(part.data(), d_sums[d], N * sizeof(double), cudaMemcpyDeviceToHost));
        for (int k = 0; k < N; ++k) sums[k] += part[k];
        for (auto p : d_rgb[d]) CK(cudaFree(p));
        CK(cudaFree(d_sums[d]));
    }
    if (!sums_path.empty()) {
        FILE* f = fopen(sums_path.c_str(), "w");
        if (f) {
            for (int k = 0; k < N; ++k) fprintf(f, "%s %.6f\n", paths[k].filename().string().c_str(), sums[k]);
            fclose(f);
            printf("  checksums -> %s\n", sums_path.c_str());
        }
    }
    return st.errors.empty() ? 0 : 1;
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

// Everything the pass-1 stream produces, for `fold` to write out and for `stack` to carry on with.
struct Pass1 {
    Grid g;
    FoldParams fp;
    VolumeParams vp;
    std::vector<fs::path> paths;
    std::vector<Similarity> T;
    std::vector<int> devs;
    HostFoldState merged;
    std::vector<float> guide, ga;            // output res, depth res
    std::vector<uint16_t> fm_all, lum_all;   // plane-major (K, dh, dw) float16
    std::vector<float> V_all;                // plane-major, aggregated
    int slices = 0;
    StreamerStats st;
    double vol_seconds = 0;
};

struct Pass1Options {
    fs::path list;                        // frame list file, or...
    std::vector<fs::path> paths;          // ...the frames and transforms already in memory
    std::vector<Similarity> T;
    Grid g;
    double radius_px = 6.0, power = 4.0;
    std::vector<int> frames_sel;
    StreamerOptions opt;
    bool volume = false;
    double depth_fraction = 0.25, depth_fm_px = 6.0, vol_agg_px = 32.0;
};

static bool read_list(const fs::path& list, std::vector<fs::path>& paths, std::vector<Similarity>& T) {
    // path zr zi tx ty per line (paths may contain spaces: the last four fields are numbers)
    FILE* f = fopen(list.string().c_str(), "r");
    if (!f) { fprintf(stderr, "cannot read %s\n", list.string().c_str()); return false; }
    char line[4096];
    while (fgets(line, sizeof line, f)) {
        std::string s(line);
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) s.pop_back();
        if (s.empty() || s[0] == '#') continue;
        double v[4];
        size_t end = s.size();
        for (int j = 3; j >= 0; --j) {
            size_t sp = s.rfind(' ', end - 1);
            if (sp == std::string::npos) { fprintf(stderr, "bad list line: %s\n", s.c_str()); fclose(f); return false; }
            v[j] = atof(s.substr(sp + 1, end - sp - 1).c_str());
            end = sp;
        }
        paths.push_back(s.substr(0, end));
        T.push_back({v[0], v[1], v[2], v[3]});
    }
    fclose(f);
    return !paths.empty();
}

static int run_pass1(const Pass1Options& po, Pass1& P);

static int cmd_fold(int argc, char** argv) {
    if (argc < 3) return usage();
    Pass1Options po;
    po.list = argv[2];
    std::string out;
    po.opt.streams_per_gpu = 2;
    bool have_rect = false;
    Grid& g = po.g;
    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--volume") { po.volume = true; continue; }
        if (i + 1 >= argc) return usage();
        std::string v = argv[++i];
        if (a == "--rect") { auto d = parse_doubles(v); if (d.size() != 4) return usage(); g.x0 = (int)d[0]; g.y0 = (int)d[1]; g.x1 = (int)d[2]; g.y1 = (int)d[3]; have_rect = true; }
        else if (a == "--out") out = v;
        else if (a == "--scale") g.scale = atof(v.c_str());
        else if (a == "--radius-px") po.radius_px = atof(v.c_str());
        else if (a == "--power") po.power = atof(v.c_str());
        else if (a == "--frames") po.frames_sel = parse_list(v);
        else if (a == "--gpus") po.opt.devices = parse_list(v);
        else if (a == "--streams") po.opt.streams_per_gpu = atoi(v.c_str());
        else if (a == "--readers") po.opt.readers = atoi(v.c_str());
        else if (a == "--depth-fraction") po.depth_fraction = atof(v.c_str());
        else if (a == "--depth-fm-px") po.depth_fm_px = atof(v.c_str());
        else if (a == "--vol-agg-px") po.vol_agg_px = atof(v.c_str());
        else return usage();
    }
    if (out.empty() || !have_rect) return usage();
    Pass1 P;
    if (int r = run_pass1(po, P)) return r;
    if (!write_state(P.merged, out)) { fprintf(stderr, "cannot write %s\n", out.c_str()); return 1; }
    printf("  merged state (%d frames) -> %s\n", P.merged.count, out.c_str());
    if (po.volume) {
        fs::path od(out);
        const auto& vp = P.vp;
        bool okw = write_raw(od / "guide.f32", P.guide.data(), P.guide.size()) && write_raw(od / "guide_d.f32", P.ga.data(), P.ga.size()) &&
                   write_raw(od / "vol_fm.f16", P.fm_all.data(), P.fm_all.size()) && write_raw(od / "vol_lum.f16", P.lum_all.data(), P.lum_all.size()) &&
                   write_raw(od / "vol_V.f32", P.V_all.data(), P.V_all.size());
        FILE* mf = fopen((od / "volume.txt").string().c_str(), "w");
        if (mf) {
            fprintf(mf, "dw %d\ndh %d\nK %d\nslices %d\nfm_radius %.6g\nagg_r %d\nagg_eps %.6g\nlayout plane-major (K, dh, dw)\n", vp.dw, vp.dh,
                    (int)P.paths.size(), P.slices, vp.fm_radius, vp.agg_r, vp.agg_eps);
            fclose(mf);
        }
        if (!okw) { fprintf(stderr, "cannot write the volume files\n"); return 1; }
        printf("  volume: %d slices of %dx%d, fm sigma %.2f, aggregation r=%d; guide + aggregation %.2f s -> %s\n", P.slices, vp.dw, vp.dh,
               vp.fm_radius, vp.agg_r, P.vol_seconds, out.c_str());
    }
    return P.st.errors.empty() ? 0 : 1;
}

static int run_pass1(const Pass1Options& po, Pass1& P) {
    Grid g = po.g;
    StreamerOptions opt = po.opt;
    const bool volume = po.volume;
    if (g.scale != 1.0 && g.scale != 0.5) { fprintf(stderr, "--scale must be 1 or 0.5\n"); return 2; }
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
    if (!t0.ok()) { fprintf(stderr, "%s: %s\n", paths[ks[0]].string().c_str(), t0.error.c_str()); return 1; }
    g.W = (int)t0.width;
    g.H = (int)t0.height;
    int ow = g.out_w(), oh = g.out_h();
    FoldParams fp;
    fp.levels = n_levels(ow, oh);
    fp.radius = (float)std::max(0.8, po.radius_px * g.scale);
    fp.power = (float)po.power;
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
    for (int d : devs) {
        auto& D = ctx[d];
        D.dev = d;
        CK(cudaSetDevice(d));
        CK(cudaStreamCreateWithFlags(&D.fold_s, cudaStreamNonBlocking));
        D.st.alloc(d, ow, oh, fp.levels);
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
        size_t margin = 512u << 20;
        int fit = fr > margin ? (int)((fr - margin) / per_slot) : 0;
        int S = std::max(1, std::min(opt.streams_per_gpu, fit));
        if (fit < 1) fprintf(stderr, "GPU %d: not enough memory for the state plus one slot (%.2f GB free)\n", d, fr / 1e9);
        opt.streams_by_device.push_back(S);
        D.slots.resize(S);
        for (auto& S : D.slots) {
            CK(cudaMalloc(&S.d_rgb, npx_full * 3 * sizeof(float)));
            if (g.scale != 1.0) CK(cudaMalloc(&S.d_scratch, (size_t)g.work_w() * g.work_h() * 3 * sizeof(float)));
            CK(cudaMalloc(&S.d_warp, npx_out * 3 * sizeof(float)));
            S.sc.alloc(ow, oh, fp.levels, fp.radius);
            if (volume) S.vsc.alloc(ow, oh, vp);
            CK(cudaEventCreateWithFlags(&S.ready, cudaEventDisableTiming));
            CK(cudaEventCreateWithFlags(&S.done, cudaEventDisableTiming));
        }
        CK(cudaMemGetInfo(&fr, &tot));
        printf("GPU %d: state %.2f GB + %d slot(s) x %.2f GB (%d requested, %d fit); %.2f of %.2f GB free after allocation\n", d,
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
            cudaMallocAsync(&D.fm16[k], nb, f.stream);
            cudaMallocAsync(&D.lum16[k], nb, f.stream);
            volume_slice(S.d_warp, vp, S.vsc, D.fm16[k], D.lum16[k], f.stream);
        }
        cudaEventRecord(S.ready, f.stream);
        cudaStreamWaitEvent(D.fold_s, S.ready, 0);
        fold_accumulate(D.st, k, S.d_warp, fp, S.sc, D.fold_s);
        cudaEventRecord(S.done, D.fold_s);
        S.used = true;
    };

    printf("fold: %zu frames, grid %dx%d -> %dx%d at scale %g, %d levels, blur sigma %.2f px, %zu GPU(s)\n",
           run_paths.size(), g.W, g.H, ow, oh, g.scale, fp.levels, fp.radius, devs.size());
    StreamerStats st = stream_frames(run_paths, opt, on_frame);
    for (auto& e : st.errors) fprintf(stderr, "  error: %s\n", e.c_str());
    for (int d : devs) {
        CK(cudaSetDevice(d));
        CK(cudaStreamSynchronize(ctx[d].fold_s));
        CK(cudaGetLastError());
    }
    printf("%d frames in %.3f s: %.1f frames/s, %.2f GB/s read\n", st.frames, st.seconds, st.frames / st.seconds,
           st.bytes / st.seconds / 1e9);
    for (size_t i = 0; i < devs.size(); ++i) printf("  GPU %d: %d frames\n", devs[i], st.frames_per_device[i]);

    // host merge in device order (any order gives the same B / C; A to double rounding)
    HostFoldState& merged = P.merged;
    bool first = true;
    for (int d : devs) {
        HostFoldState h = download(ctx[d].st);
        if (first) { merged = std::move(h); first = false; }
        else merge_into(merged, h);
    }
    P.st = st;

    if (volume) {
        auto t0 = std::chrono::steady_clock::now();
        // guide: luma of the collapsed merged C, clipped to [0, 1], on the first device; then at the depth res
        int d0 = devs[0];
        CK(cudaSetDevice(d0));
        cudaStream_t s0 = ctx[d0].fold_s;
        std::vector<const float*> d_coef;
        std::vector<float*> owned;
        for (int l = 0; l < fp.levels - 1; ++l) {
            float* p;
            CK(cudaMalloc(&p, merged.c_coef[l].size() * sizeof(float)));
            CK(cudaMemcpy(p, merged.c_coef[l].data(), merged.c_coef[l].size() * sizeof(float), cudaMemcpyHostToDevice));
            d_coef.push_back(p);
            owned.push_back(p);
        }
        std::vector<float> base_f(merged.c_base.size());
        for (size_t i = 0; i < base_f.size(); ++i) base_f[i] = (float)(merged.c_base[i] / std::max(merged.count, 1));
        float *d_base, *d_coll, *d_t1, *d_t2, *d_guide, *d_ga;
        CK(cudaMalloc(&d_base, base_f.size() * sizeof(float)));
        CK(cudaMemcpy(d_base, base_f.data(), base_f.size() * sizeof(float), cudaMemcpyHostToDevice));
        CK(cudaMalloc(&d_coll, npx_out * 3 * sizeof(float)));
        CK(cudaMalloc(&d_t1, npx_out * 3 * sizeof(float)));
        CK(cudaMalloc(&d_t2, npx_out * 3 * sizeof(float)));
        CK(cudaMalloc(&d_guide, npx_out * sizeof(float)));
        size_t nd = (size_t)vp.dw * vp.dh;
        CK(cudaMalloc(&d_ga, nd * sizeof(float)));
        collapse_pyramid(d_coef, d_base, merged.shapes, d_coll, d_t1, d_t2, s0);
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
        for (auto p : owned) cudaFree(p);
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
            for (int k = 0; k < K; ++k) {
                if (!D.fm16[k]) continue;
                if (vp.agg_r > 0) aggregate_slice(d_g, gs, D.fm16[k], d_V, vp.agg_eps, s);
                else { f16_to_f32(D.fm16[k], d_V, nd, s); relu_sqrt(d_V, nd, s); }
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

// ----------------------------------------------------------------------------- depth (SGM)

static std::map<std::string, std::string> read_kv(const fs::path& p) {
    std::map<std::string, std::string> m;
    FILE* f = fopen(p.string().c_str(), "r");
    if (!f) return m;
    char line[512];
    while (fgets(line, sizeof line, f)) {
        std::string s(line);
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
        size_t sp = s.find(' ');
        if (sp != std::string::npos) m[s.substr(0, sp)] = s.substr(sp + 1);
    }
    fclose(f);
    return m;
}

template <class T>
static bool read_raw(const fs::path& p, std::vector<T>& v, size_t n) {
    v.resize(n);
    FILE* f = fopen(p.string().c_str(), "rb");
    if (!f) return false;
    size_t m = fread(v.data(), sizeof(T), n, f);
    fclose(f);
    return m == n;
}

static int cmd_depth(int argc, char** argv) {
    if (argc < 3) return usage();
    fs::path vd = argv[2];
    std::string out;
    int gpu = 0;
    std::vector<int> crop;
    DepthParams dp;
    for (int i = 3; i + 1 < argc; i += 2) {
        std::string a = argv[i], v = argv[i + 1];
        if (a == "--out") out = v;
        else if (a == "--gpu") gpu = atoi(v.c_str());
        else if (a == "--crop") crop = parse_list(v);
        else if (a == "--dirs") dp.dirs = atoi(v.c_str());
        else return usage();
    }
    if (out.empty()) return usage();
    auto vm = read_kv(vd / "volume.txt"), mm = read_kv(vd / "manifest.txt");
    if (vm.empty() || mm.empty()) { fprintf(stderr, "no volume.txt / manifest.txt in %s\n", vd.string().c_str()); return 1; }
    int dw = atoi(vm["dw"].c_str()), dh = atoi(vm["dh"].c_str()), K = atoi(vm["K"].c_str());
    int ow = atoi(mm["w"].c_str()), oh = atoi(mm["h"].c_str());
    size_t nd = (size_t)dw * dh, no = (size_t)ow * oh;
    std::vector<float> V, guide, ga;
    std::vector<uint16_t> Lv;
    if (!read_raw(vd / "vol_V.f32", V, nd * K) || !read_raw(vd / "vol_lum.f16", Lv, nd * K) || !read_raw(vd / "guide.f32", guide, no) ||
        !read_raw(vd / "guide_d.f32", ga, nd)) { fprintf(stderr, "cannot read the volume files\n"); return 1; }
    if (!crop.empty()) {   // crop at the depth res; the output grid is an integer multiple
        if (crop.size() != 4) return usage();
        if (ow % dw != 0 || oh % dh != 0) { fprintf(stderr, "--crop needs an integer output/depth ratio (%d/%d, %d/%d)\n", ow, dw, oh, dh); return 1; }
        int sx = ow / dw, sy = oh / dh, x0 = crop[0], y0 = crop[1], cw = crop[2], ch = crop[3];
        std::vector<float> V2((size_t)cw * ch * K), ga2((size_t)cw * ch), g2((size_t)cw * sx * ch * sy);
        std::vector<uint16_t> L2((size_t)cw * ch * K);
        for (int k = 0; k < K; ++k)
            for (int y = 0; y < ch; ++y) {
                memcpy(&V2[((size_t)k * ch + y) * cw], &V[((size_t)k * dh + y0 + y) * dw + x0], cw * sizeof(float));
                memcpy(&L2[((size_t)k * ch + y) * cw], &Lv[((size_t)k * dh + y0 + y) * dw + x0], cw * sizeof(uint16_t));
            }
        for (int y = 0; y < ch; ++y) memcpy(&ga2[(size_t)y * cw], &ga[(size_t)(y0 + y) * dw + x0], cw * sizeof(float));
        for (int y = 0; y < ch * sy; ++y) memcpy(&g2[(size_t)y * cw * sx], &guide[(size_t)(y0 * sy + y) * ow + x0 * sx], cw * sx * sizeof(float));
        V.swap(V2); Lv.swap(L2); ga.swap(ga2); guide.swap(g2);
        dw = cw; dh = ch; ow = cw * sx; oh = ch * sy;
        nd = (size_t)dw * dh; no = (size_t)ow * oh;
    }
    CK(cudaSetDevice(gpu));
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    float *d_planes, *d_V, *d_guide, *d_ga;
    void *d_L16p, *d_L16;
    CK(cudaMalloc(&d_planes, nd * K * sizeof(float)));
    CK(cudaMalloc(&d_V, nd * K * sizeof(float)));
    CK(cudaMalloc(&d_L16p, nd * K * 2));
    CK(cudaMalloc(&d_L16, nd * K * 2));
    CK(cudaMalloc(&d_guide, no * sizeof(float)));
    CK(cudaMalloc(&d_ga, nd * sizeof(float)));
    auto t0 = std::chrono::steady_clock::now();
    CK(cudaMemcpy(d_planes, V.data(), nd * K * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_L16p, Lv.data(), nd * K * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_guide, guide.data(), no * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_ga, ga.data(), nd * sizeof(float), cudaMemcpyHostToDevice));
    planes_to_kfastest_f32(d_planes, d_V, dw, dh, K, s);
    planes_to_kfastest_f16(d_L16p, d_L16, dw, dh, K, s);
    CK(cudaStreamSynchronize(s));
    cudaFree(d_planes);
    cudaFree(d_L16p);
    double ms_up = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    printf("depth: volume %dx%dx%d -> output %dx%d on GPU %d (upload + layout %.0f ms)\n", dw, dh, K, ow, oh, gpu, ms_up);
    DepthResult R = depth_from_volume(d_V, d_L16, d_guide, d_ga, dw, dh, K, ow, oh, dp, s);
    printf("  cost %.0f ms, SGM (%d dirs) %.0f ms, upsample + guided filter + median %.0f ms; end-peaked %.1f%% of pixels\n",
           R.ms_cost, dp.dirs, R.ms_sgm, R.ms_post, 100 * R.end_fraction);
    fs::create_directories(out);
    fs::path od(out);
    bool okw = write_raw(od / "depth_s.f32", R.depth_s.data(), R.depth_s.size()) && write_raw(od / "conf.f32", R.conf.data(), R.conf.size()) &&
               write_raw(od / "spill.f32", R.spill.data(), R.spill.size()) && write_raw(od / "depth_d.f32", R.depth_d.data(), R.depth_d.size());
    FILE* mf = fopen((od / "depth.txt").string().c_str(), "w");
    if (mf) {
        fprintf(mf, "dw %d\ndh %d\nK %d\now %d\noh %d\ndirs %d\nend_fraction %.6f\nms_cost %.1f\nms_sgm %.1f\nms_post %.1f\n", dw, dh, K, ow, oh,
                dp.dirs, R.end_fraction, R.ms_cost, R.ms_sgm, R.ms_post);
        fclose(mf);
    }
    cudaFree(d_V); cudaFree(d_L16); cudaFree(d_guide); cudaFree(d_ga);
    cudaStreamDestroy(s);
    if (!okw) { fprintf(stderr, "cannot write %s\n", out.c_str()); return 1; }
    printf("  -> %s\n", out.c_str());
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
};
struct PassDev {
    int dev = -1;
    cudaStream_t fold_s = nullptr;
    std::vector<PassSlot> slots;
};

template <class SlotStage, class AccStage>
static StreamerStats run_pass(const std::vector<fs::path>& run_paths, const std::vector<int>& ks, const std::vector<Similarity>& T,
                              const Grid& g, const std::vector<int>& devs, int streams, int readers, std::vector<PassDev>& ctx,
                              SlotStage slot_stage, AccStage acc_stage, const char* what) {
    int ndev = 0;
    CK_THROW(cudaGetDeviceCount(&ndev));
    ctx.assign(ndev, PassDev{});
    size_t npx_full = (size_t)g.W * g.H, npx_out = (size_t)g.out_w() * g.out_h();
    StreamerOptions opt;
    opt.devices = devs;
    opt.streams_per_gpu = streams;
    opt.readers = readers;
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
    };
    StreamerStats st = stream_frames(run_paths, opt, on_frame);
    for (auto& e : st.errors) fprintf(stderr, "  error: %s\n", e.c_str());
    for (int d : devs) {
        CK_THROW(cudaSetDevice(d));
        CK_THROW(cudaStreamSynchronize(ctx[d].fold_s));
        CK_THROW(cudaGetLastError());
    }
    printf("  %s: %d frames in %.2f s (%.1f frames/s)", what, st.frames, st.seconds, st.frames / st.seconds);
    for (size_t i = 0; i < devs.size(); ++i) printf("%s GPU %d %d", i ? "," : ";", devs[i], st.frames_per_device[i]);
    printf("\n");
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

struct StackOptions {
    Pass1Options po;                  // frames, transforms, grid, GPUs, readers (po.volume is forced on)
    std::string out;
    int streams = 2, refine_r = 12;
    double refine_fm_px = 9.0, soft_px = 3.0;
    std::vector<std::string> outputs; // which images to write; empty = all
};

// parses the options shared by `stack` and `run`; returns false on an unknown option
static bool parse_stack_option(const std::string& a, const std::string& v, StackOptions& so) {
    Grid& g = so.po.g;
    if (a == "--scale") g.scale = atof(v.c_str());
    else if (a == "--gpus") so.po.opt.devices = parse_list(v);
    else if (a == "--streams") { so.streams = atoi(v.c_str()); so.po.opt.streams_per_gpu = so.streams; }
    else if (a == "--readers") so.po.opt.readers = atoi(v.c_str());
    else if (a == "--refine-r") so.refine_r = atoi(v.c_str());
    else if (a == "--refine-fm-px") so.refine_fm_px = atof(v.c_str());
    else if (a == "--soft-px") so.soft_px = atof(v.c_str());
    else if (a == "--outputs") {
        so.outputs.clear();
        size_t i = 0;
        while (i < v.size()) { size_t j = v.find(',', i); if (j == std::string::npos) j = v.size(); if (j > i) so.outputs.push_back(v.substr(i, j - i)); i = j + 1; }
    } else return false;
    return true;
}

static int run_stack(StackOptions& so);

static int cmd_stack(int argc, char** argv) {
    if (argc < 3) return usage();
    StackOptions so;
    so.po.list = argv[2];
    so.po.opt.streams_per_gpu = 2;
    bool have_rect = false;
    Grid& g = so.po.g;
    for (int i = 3; i + 1 < argc; i += 2) {
        std::string a = argv[i], v = argv[i + 1];
        if (a == "--rect") { auto d = parse_doubles(v); if (d.size() != 4) return usage(); g.x0 = (int)d[0]; g.y0 = (int)d[1]; g.x1 = (int)d[2]; g.y1 = (int)d[3]; have_rect = true; }
        else if (a == "--out") so.out = v;
        else if (!parse_stack_option(a, v, so)) return usage();
    }
    if (so.out.empty() || !have_rect) return usage();
    return run_stack(so);
}

static int run_stack(StackOptions& so) {
    Pass1Options& po = so.po;
    po.volume = true;
    const std::string& out = so.out;
    const int refine_r = so.refine_r, streams = so.streams;
    const double refine_fm_px = so.refine_fm_px, soft_px = so.soft_px;
    auto t_all = std::chrono::steady_clock::now();
    auto since = [](std::chrono::steady_clock::time_point t) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count(); };

    // ---- pass 1 + volume
    Pass1 P;
    if (int r = run_pass1(po, P)) return r;
    const int K = (int)P.paths.size();
    const int ow = P.g.out_w(), oh = P.g.out_h();
    const size_t no = (size_t)ow * oh, nd = (size_t)P.vp.dw * P.vp.dh;
    const int prim = P.devs[0];
    printf("pass 1: %.2f s stream + %.2f s volume\n", P.st.seconds, P.vol_seconds);

    // ---- depth (SGM) on the primary GPU
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
        R = depth_from_volume(d_V, d_L16, d_guide, d_ga, P.vp.dw, P.vp.dh, K, ow, oh, dp, s);
        cudaFree(d_V); cudaFree(d_L16); cudaFree(d_ga);
    }
    printf("depth: %.2f s (cost %.0f ms, SGM %.0f ms, post %.0f ms); end-peaked %.1f%%\n", since(t1), R.ms_cost, R.ms_sgm, R.ms_post,
           100 * R.end_fraction);

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
        std::vector<PassDev> ctx;
        run_pass(P.paths, all_k, P.T, P.g, P.devs, streams, po.opt.readers, ctx,
                 [&](int dev, PassSlot& S, int, cudaStream_t ss) { refine_measure(S.d_warp, rsc[dev][S.index], ss); },
                 [&](int dev, PassSlot& S, int k, cudaStream_t fs) { refine_accumulate(rst[dev], k, rsc[dev][S.index].fm, d_depth[dev], refine_r, fs); },
                 "refine pass");
        HostRefine hr;
        bool first = true;
        for (int d : P.devs) {
            HostRefine h = download(rst[d]);
            if (first) { hr = std::move(h); first = false; } else merge_into(hr, h);
            rst[d].free();
            CK(cudaSetDevice(d));
            for (auto& sc : rsc[d]) sc.free();
            cudaFree(d_depth[d]);
        }
        CK(cudaSetDevice(prim));
        refine_depth(R.depth_s, R.conf, R.spill, P.vp.dw, P.vp.dh, hr, ow, oh, P.g.scale, s);
    }
    std::vector<float> Ds = smooth_depth(R.depth_s, d_guide, ow, oh, P.g.scale, K, s);
    printf("refine: %.2f s\n", since(t2));

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
        run_pass(sub, ks, P.T, P.g, P.devs, streams, po.opt.readers, ctx,
                 [&](int, PassSlot&, int, cudaStream_t) {},
                 [&](int dev, PassSlot& S, int k, cudaStream_t fs) { compose_frame(cst[dev], k, S.d_warp, d_D[dev], d_Ds[dev], csc[dev][S.index], fs); },
                 "compose pass");
        HostCompose hc;
        bool first = true;
        for (int d : P.devs) {
            HostCompose h = download(cst[d]);
            if (first) { hc = std::move(h); first = false; } else merge_into(hc, h);
            cst[d].free();
            CK(cudaSetDevice(d));
            for (auto& sc : csc[d]) sc.free();
            cudaFree(d_D[d]); cudaFree(d_Ds[d]);
        }
        CK(cudaSetDevice(prim));
        b_img = compose_mix(hc, R.conf, ow, oh, P.g.scale, s);
        printf("compose: %zu of %d frames carry a label; %.2f s\n", ks.size(), K, since(t3));
    }

    // ---- noise from two adjacent middle frames, then finalize
    auto t4 = std::chrono::steady_clock::now();
    std::vector<float> noise_mad;
    {
        int mid = std::max(0, K / 2 - 1);
        float* d_pair[2];
        for (int j = 0; j < 2; ++j) {
            TiffInfo info;
            float* d_rgb = nullptr;
            if (int r = load_frame_to_device(P.paths[mid + j], info, d_rgb)) return r;
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
    FinalOutputs F = finalize(P.merged, noise_mad, R.depth_s, b_img, fpar, s);
    printf("finalize: %.2f s; noise sigma per level (x1e-4):", since(t4));
    for (float v : noise_mad) printf(" %.2f", v * 1e4);
    printf("\n");

    // ---- write
    auto t5 = std::chrono::steady_clock::now();
    fs::create_directories(out);
    fs::path od(out);
    size_t written = 0;
    for (size_t i = 0; i < F.names.size(); ++i) {
        if (!so.outputs.empty() && std::find(so.outputs.begin(), so.outputs.end(), F.names[i]) == so.outputs.end()) continue;
        if (!write_tiff_rgb16((od / (F.names[i] + ".tif")).string(), F.images[i], ow, oh)) { fprintf(stderr, "cannot write %s\n", F.names[i].c_str()); return 1; }
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
    cudaFree(d_guide);
    cudaStreamDestroy(s);
    printf("wrote %zu outputs (%dx%d) in %.2f s -> %s\nfusion total %.2f s\n", written, ow, oh, since(t5), out.c_str(), since(t_all));
    return 0;
}

// ----------------------------------------------------------------------------- align

struct AlignOptions {
    fs::path dir;
    std::string out, prefix = "IMG_", ref_rule = "middle", pairs_sel;
    double align_scale = 0.25;
    std::vector<int> baselines = {1, 2, 4, 8}, devs;
    int readers = 16;
    bool lock_scale = false;
    double um_per_px = 0;             // 0 = unknown (from --um-per-px or --fov-mm)
};

static bool parse_align_option(const std::string& a, const std::string& v, AlignOptions& ao) {
    if (a == "--prefix") ao.prefix = v;
    else if (a == "--align-scale") ao.align_scale = atof(v.c_str());
    else if (a == "--baselines") ao.baselines = parse_list(v);
    else if (a == "--ref") ao.ref_rule = v;
    else if (a == "--gpus") ao.devs = parse_list(v);
    else if (a == "--pairs") ao.pairs_sel = v;
    else if (a == "--readers") ao.readers = atoi(v.c_str());
    else if (a == "--lock-scale") ao.lock_scale = v != "0" && v != "false";
    else if (a == "--um-per-px") ao.um_per_px = atof(v.c_str());
    else if (a == "--fov-mm") ao.um_per_px = -atof(v.c_str());   // resolved once the width is known
    else return false;
    return true;
}

struct AlignOut {
    std::vector<fs::path> paths;
    std::vector<Similarity> T;        // relative to the reference, full-res px (scale locked if asked)
    int W = 0, H = 0, ref = 0, rect[4] = {0, 0, 0, 0};
    GlobalSolve G;
    double smin = 1, smax = 1, seconds = 0;
};

static void write_report(const AlignOptions& ao, const AlignOut& A, const fs::path& od);

static int run_align(AlignOptions& ao, AlignOut& A) {
    const fs::path& dir = ao.dir;
    const std::string &out = ao.out, &prefix = ao.prefix, &ref_rule = ao.ref_rule, &pairs_sel = ao.pairs_sel;
    const double align_scale = ao.align_scale;
    const std::vector<int>& baselines = ao.baselines;
    std::vector<int> devs = ao.devs;
    const int readers = ao.readers;
    std::vector<fs::path>& paths = A.paths;
    for (auto& e : fs::directory_iterator(dir)) {
        auto name = e.path().filename().string();
        auto ext = e.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (e.is_regular_file() && (ext == ".tif" || ext == ".tiff") && name.rfind(prefix, 0) == 0) paths.push_back(e.path());
    }
    std::sort(paths.begin(), paths.end());
    const int N = (int)paths.size();
    if (N < 3) { fprintf(stderr, "need >= 3 frames, found %d\n", N); return 1; }
    int ndev = 0;
    CK(cudaGetDeviceCount(&ndev));
    if (devs.empty()) for (int i = 0; i < ndev; ++i) devs.push_back(i);
    TiffInfo t0 = tiff_probe(paths[0]);
    if (!t0.ok()) { fprintf(stderr, "%s: %s\n", paths[0].string().c_str(), t0.error.c_str()); return 1; }
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
        };
        StreamerStats st = stream_frames(paths, opt, on_frame);
        for (auto& e : st.errors) fprintf(stderr, "  error: %s\n", e.c_str());
        for (int d : devs) {
            CK(cudaSetDevice(d));
            CK(cudaDeviceSynchronize());
            for (auto& S : slots[d]) { cudaFree(S.d_rgb); cudaFree(S.d_y); cudaFree(S.d_small); S.rz.free(); }
        }
        printf("align: %d frames %dx%d -> luma %dx%d in %.2f s\n", N, W, H, w, h, st.seconds);
        if (getenv("STACKER_MEM_DEBUG"))
            for (int d : devs) { size_t fr, tot; cudaSetDevice(d); cudaMemGetInfo(&fr, &tot); printf("  [mem] GPU %d after pass 0: %.2f GB free\n", d, fr / 1e9); }
    }

    // ---- pairs, dynamically shared by the GPUs, each with an LRU cache of pyramids
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
    std::atomic<int> next{0};
    auto t_pairs = std::chrono::steady_clock::now();
    std::vector<std::thread> workers;
    std::vector<std::string> worker_err(ndev);
    std::vector<int> per_dev(ndev, 0), builds(ndev, 0);
    for (int d : devs)
        workers.emplace_back([&, d] {
            try {
                CK_THROW(cudaSetDevice(d));
                cudaStream_t s;
                CK_THROW(cudaStreamCreate(&s));
                AlignWork wk;
                wk.alloc(w, h);
                std::unordered_map<int, AlignPyramid> cache;
                std::list<int> lru;
                float* d_luma;
                CK_THROW(cudaMalloc(&d_luma, (size_t)w * h * sizeof(float)));
                auto get = [&](int k) -> AlignPyramid& {
                    auto it = cache.find(k);
                    if (it != cache.end()) { lru.remove(k); lru.push_back(k); return it->second; }
                    if (cache.size() >= 24) { int old = lru.front(); lru.pop_front(); cache[old].free(); cache.erase(old); }
                    CK_THROW(cudaMemcpyAsync(d_luma, luma_stack.data() + (size_t)k * w * h, (size_t)w * h * sizeof(float), cudaMemcpyHostToDevice, s));
                    AlignPyramid& P = cache[k];
                    P.build(d_luma, w, h, s);
                    lru.push_back(k);
                    ++builds[d];
                    return P;
                };
                AlignParams ap;
                for (;;) {
                    int e = next++;
                    if (e >= (int)pairs.size()) break;
                    auto [i, j] = pairs[e];
                    AlignPyramid& Pi = get(i);
                    AlignPyramid& Pj = get(j);
                    results[e] = align_pair(Pi, Pj, i, j, wk, ap, s);
                    ++per_dev[d];
                }
                for (auto& kv : cache) kv.second.free();
                wk.free();
                cudaFree(d_luma);
                cudaStreamDestroy(s);
            } catch (const std::exception& ex) { worker_err[d] = ex.what(); }
        });
    for (auto& t : workers) t.join();
    for (int d : devs) if (!worker_err[d].empty()) { fprintf(stderr, "GPU %d: %s\n", d, worker_err[d].c_str()); return 1; }
    if (getenv("STACKER_MEM_DEBUG"))
        for (int d : devs) { size_t fr, tot; cudaSetDevice(d); cudaMemGetInfo(&fr, &tot); printf("  [mem] GPU %d after pairs: %.2f GB free, %d pyramid builds\n", d, fr / 1e9, builds[d]); }
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_pairs).count();
    printf("  %zu pairs in %.2f s (%.1f ms/pair)", pairs.size(), secs, 1000 * secs / pairs.size());
    for (int d : devs) printf("%s GPU %d %d", d == devs[0] ? ";" : ",", d, per_dev[d]);
    printf("\n");
    fs::create_directories(out);
    fs::path od(out);
    if (!pairs_sel.empty()) {   // per-pair dump for validation
        FILE* f = fopen((od / "pairs.txt").string().c_str(), "w");
        for (auto& r : results) {
            printf("  pair %d:%d  z %.9f%+.9fi  t %.5f %.5f  gain %.5f  ncc %.5f  n %d  iters %d  var %.3e %.3e %.3e %.3e  %s\n", r.i, r.j,
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
    printf("  solve: %d/%d pairs usable, %d outliers, ncc min %.4f median %.4f; ref #%d; native crop %dx%d (L%d R%d T%d B%d); scale spread %.3f%%%s; %.2f s\n",
           A.G.pairs_used, A.G.pairs_total, A.G.outliers, A.G.ncc_min, A.G.ncc_median, A.ref, A.rect[2] - A.rect[0] + 1, A.rect[3] - A.rect[1] + 1,
           A.rect[0], W - 1 - A.rect[2], A.rect[1], H - 1 - A.rect[3], 100 * (A.smax / A.smin - 1), ao.lock_scale ? " (scale locked)" : "", A.seconds);
    return 0;
}

static void write_report(const AlignOptions& ao, const AlignOut& A, const fs::path& od) {
    const int N = (int)A.paths.size(), W = A.W, H = A.H;
    const int* rect = A.rect;
    fs::create_directories(od);
    FILE* f = fopen((od / "report.json").string().c_str(), "w");
    if (!f) { fprintf(stderr, "cannot write %s\n", (od / "report.json").string().c_str()); return; }
    fprintf(f, "{\n \"stack\": \"%s\",\n \"alignment\": {\"align_scale\": %g, \"baselines\": [", ao.dir.filename().string().c_str(), ao.align_scale);
    for (size_t i = 0; i < ao.baselines.size(); ++i) fprintf(f, "%s%d", i ? ", " : "", ao.baselines[i]);
    fprintf(f, "], \"scale_locked\": %s, \"pairs_used\": %d, \"pairs_total\": %d, \"outliers\": %d, \"ncc_min\": %.17g, \"ncc_median\": %.17g},\n",
            ao.lock_scale ? "true" : "false", A.G.pairs_used, A.G.pairs_total, A.G.outliers, A.G.ncc_min, A.G.ncc_median);
    int cw = rect[2] - rect[0] + 1, ch = rect[3] - rect[1] + 1;
    fprintf(f, " \"geometry\": {\n  \"input\": {\"width\": %d, \"height\": %d, \"frames\": %d, \"um_per_px_calibrated\": ", W, H, N);
    if (ao.um_per_px > 0) fprintf(f, "%.17g},\n", ao.um_per_px); else fprintf(f, "null},\n");
    fprintf(f, "  \"reference\": {\"index\": %d, \"rule\": \"%s\"},\n  \"selected_policy\": \"native-crop\",\n  \"policies\": {\"native-crop\": {\"width\": %d, \"height\": %d, "
               "\"crop_px\": {\"left\": %d, \"right\": %d, \"top\": %d, \"bottom\": %d}, \"pixels_kept_pct\": %.17g",
            A.ref, ao.ref_rule.c_str(), cw, ch, rect[0], W - 1 - rect[2], rect[1], H - 1 - rect[3], 100.0 * cw * ch / ((double)W * H));
    if (ao.um_per_px > 0)
        fprintf(f, ", \"um_per_px\": %.17g, \"um_per_px_range_over_depth\": [%.17g, %.17g], \"range_pct\": [%.17g, %.17g], \"fov_mm\": [%.17g, %.17g]",
                ao.um_per_px, ao.um_per_px * A.smin, ao.um_per_px * A.smax, 100 * (A.smin - 1), 100 * (A.smax - 1), cw * ao.um_per_px / 1000.0,
                ch * ao.um_per_px / 1000.0);
    fprintf(f, "}},\n  \"scale_spread\": {\"s_min\": %.17g, \"s_max\": %.17g, \"total_pct\": %.17g, \"corner_shift_px\": %.17g}\n },\n \"per_frame\": [\n",
            A.smin, A.smax, 100 * (A.smax / A.smin - 1), (A.smax / A.smin - 1) * std::hypot((double)W, (double)H) / 2.0);
    for (int k = 0; k < N; ++k) {
        std::complex<double> z(A.T[k].zr, A.T[k].zi);
        fprintf(f, "  {\"i\": %d, \"file\": \"%s\", \"s\": %.17g, \"theta_deg\": %.17g, \"tx\": %.17g, \"ty\": %.17g}%s\n", k,
                A.paths[k].filename().string().c_str(), std::abs(z), std::arg(z) * 180.0 / 3.14159265358979323846, A.T[k].tx, A.T[k].ty,
                k + 1 < N ? "," : "");
    }
    fprintf(f, " ],\n \"transform_convention\": \"reference-grid centred px -> frame centred px: p_k = s e^{i theta} p + t\"\n}\n");
    fclose(f);
    FILE* lf = fopen((od / "list.txt").string().c_str(), "w");
    if (lf) {
        for (int k = 0; k < N; ++k) fprintf(lf, "%s %.17g %.17g %.17g %.17g\n", A.paths[k].string().c_str(), A.T[k].zr, A.T[k].zi, A.T[k].tx, A.T[k].ty);
        fclose(lf);
    }
    printf("  report.json + list.txt -> %s (rect %d,%d,%d,%d)\n", od.string().c_str(), rect[0], rect[1], rect[2], rect[3]);
}

static int cmd_align(int argc, char** argv) {
    if (argc < 3) return usage();
    AlignOptions ao;
    ao.dir = argv[2];
    for (int i = 3; i + 1 < argc; i += 2) {
        std::string a = argv[i], v = argv[i + 1];
        if (a == "--out") ao.out = v;
        else if (!parse_align_option(a, v, ao)) return usage();
    }
    if (ao.out.empty()) return usage();
    AlignOut A;
    if (int r = run_align(ao, A)) return r;
    if (!ao.pairs_sel.empty()) return 0;
    write_report(ao, A, ao.out);
    return 0;
}

// ----------------------------------------------------------------------------- run: folder in, stack out

static int cmd_run(int argc, char** argv) {
    if (argc < 3) return usage();
    AlignOptions ao;
    ao.dir = argv[2];
    StackOptions so;
    so.po.opt.streams_per_gpu = 2;
    std::string out;
    for (int i = 3; i + 1 < argc; i += 2) {
        std::string a = argv[i], v = argv[i + 1];
        if (a == "--out") out = v;
        else if (parse_align_option(a, v, ao)) continue;
        else if (parse_stack_option(a, v, so)) continue;
        else return usage();
    }
    if (out.empty()) out = (ao.dir.parent_path() / (ao.dir.filename().string() + "_stacker")).string();
    if (so.po.opt.devices.empty()) so.po.opt.devices = ao.devs; else if (ao.devs.empty()) ao.devs = so.po.opt.devices;
    so.po.opt.readers = ao.readers;
    ao.out = out;
    ao.pairs_sel.clear();
    auto t_all = std::chrono::steady_clock::now();
    AlignOut A;
    if (int r = run_align(ao, A)) return r;
    write_report(ao, A, out);
    so.po.paths = A.paths;
    so.po.T = A.T;
    so.po.g.x0 = A.rect[0]; so.po.g.y0 = A.rect[1]; so.po.g.x1 = A.rect[2]; so.po.g.y1 = A.rect[3];
    so.out = out;
    if (int r = run_stack(so)) return r;
    printf("run: %zu frames aligned and stacked in %.2f s -> %s\n", A.paths.size(),
           std::chrono::duration<double>(std::chrono::steady_clock::now() - t_all).count(), out.c_str());
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) return usage();
    std::string cmd = argv[1];
    if (cmd == "probe") return cmd_probe(argc, argv);
    if (cmd == "dump") return cmd_dump(argc, argv);
    if (cmd == "stream") return cmd_stream(argc, argv);
    if (cmd == "warp") return cmd_warp(argc, argv);
    if (cmd == "fold") return cmd_fold(argc, argv);
    if (cmd == "depth") return cmd_depth(argc, argv);
    if (cmd == "stack") {
        try { return cmd_stack(argc, argv); }
        catch (const std::exception& e) { fprintf(stderr, "stack failed: %s\n", e.what()); return 1; }
    }
    if (cmd == "align") {
        try { return cmd_align(argc, argv); }
        catch (const std::exception& e) { fprintf(stderr, "align failed: %s\n", e.what()); return 1; }
    }
    if (cmd == "run") {
        try { return cmd_run(argc, argv); }
        catch (const std::exception& e) { fprintf(stderr, "run failed: %s\n", e.what()); return 1; }
    }
    return usage();
}
