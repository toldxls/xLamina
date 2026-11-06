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
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "stackcore/convert.h"
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

static int usage() {
    fprintf(stderr,
            "usage:\n"
            "  stackcli probe FILE.tif\n"
            "  stackcli dump FILE.tif OUT.f32 [--gpu N]\n"
            "  stackcli stream DIR [--prefix IMG_] [--gpus 0,1] [--streams 3] [--readers 16] [--pool N] [--sums OUT.txt]\n"
            "  stackcli warp FILE.tif OUT.f32 --z ZR,ZI --t TX,TY --rect X0,Y0,X1,Y1 [--scale 1|0.5] [--quantize] [--gpu N]\n"
            "  stackcli fold LIST.txt --rect X0,Y0,X1,Y1 --out DIR [--scale 1|0.5] [--radius-px 6] [--power 4]\n"
            "                [--frames 0,5,9] [--gpus 0,1] [--streams 2] [--readers 16]\n");
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

static int cmd_fold(int argc, char** argv) {
    if (argc < 3) return usage();
    fs::path list = argv[2];
    Grid g;
    std::string out;
    double radius_px = 6.0, power = 4.0;
    std::vector<int> frames_sel;
    StreamerOptions opt;
    opt.streams_per_gpu = 2;
    bool have_rect = false, volume = false;
    double depth_fraction = 0.25, depth_fm_px = 6.0, vol_agg_px = 32.0;
    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--volume") { volume = true; continue; }
        if (i + 1 >= argc) return usage();
        std::string v = argv[++i];
        if (a == "--rect") { auto d = parse_doubles(v); if (d.size() != 4) return usage(); g.x0 = (int)d[0]; g.y0 = (int)d[1]; g.x1 = (int)d[2]; g.y1 = (int)d[3]; have_rect = true; }
        else if (a == "--out") out = v;
        else if (a == "--scale") g.scale = atof(v.c_str());
        else if (a == "--radius-px") radius_px = atof(v.c_str());
        else if (a == "--power") power = atof(v.c_str());
        else if (a == "--frames") frames_sel = parse_list(v);
        else if (a == "--gpus") opt.devices = parse_list(v);
        else if (a == "--streams") opt.streams_per_gpu = atoi(v.c_str());
        else if (a == "--readers") opt.readers = atoi(v.c_str());
        else if (a == "--depth-fraction") depth_fraction = atof(v.c_str());
        else if (a == "--depth-fm-px") depth_fm_px = atof(v.c_str());
        else if (a == "--vol-agg-px") vol_agg_px = atof(v.c_str());
        else return usage();
    }
    if (out.empty() || !have_rect) return usage();
    if (g.scale != 1.0 && g.scale != 0.5) { fprintf(stderr, "--scale must be 1 or 0.5\n"); return 2; }

    // the list: path zr zi tx ty per line (paths may contain spaces: the last four fields are numbers)
    std::vector<fs::path> paths;
    std::vector<Similarity> T;
    {
        FILE* f = fopen(list.string().c_str(), "r");
        if (!f) { fprintf(stderr, "cannot read %s\n", list.string().c_str()); return 1; }
        char line[4096];
        while (fgets(line, sizeof line, f)) {
            std::string s(line);
            while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) s.pop_back();
            if (s.empty() || s[0] == '#') continue;
            double v[4];
            size_t end = s.size();
            for (int j = 3; j >= 0; --j) {
                size_t sp = s.rfind(' ', end - 1);
                if (sp == std::string::npos) { fprintf(stderr, "bad list line: %s\n", s.c_str()); fclose(f); return 1; }
                v[j] = atof(s.substr(sp + 1, end - sp - 1).c_str());
                end = sp;
            }
            paths.push_back(s.substr(0, end));
            T.push_back({v[0], v[1], v[2], v[3]});
        }
        fclose(f);
    }
    if (paths.empty()) { fprintf(stderr, "empty list\n"); return 1; }
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
    fp.radius = (float)std::max(0.8, radius_px * g.scale);
    fp.power = (float)power;
    VolumeParams vp;
    {
        double ds = depth_fraction / g.scale;   // depth res = fixed fraction of the sensor
        vp.dw = std::max(16, (int)std::nearbyint(ow * ds));
        vp.dh = std::max(16, (int)std::nearbyint(oh * ds));
        vp.fm_radius = (float)std::max(0.8, depth_fm_px * depth_fraction);
        vp.agg_r = (int)std::nearbyint(vol_agg_px * depth_fraction);
        if (vol_agg_px * depth_fraction < 1.0) vp.agg_r = 0;
    }
    const int K = (int)paths.size();

    int ndev = 0;
    CK(cudaGetDeviceCount(&ndev));
    std::vector<int> devs = opt.devices;
    if (devs.empty()) for (int i = 0; i < ndev; ++i) devs.push_back(i);
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
    HostFoldState merged;
    bool first = true;
    for (int d : devs) {
        HostFoldState h = download(ctx[d].st);
        if (first) { merged = std::move(h); first = false; }
        else merge_into(merged, h);
    }
    if (!write_state(merged, out)) { fprintf(stderr, "cannot write %s\n", out.c_str()); return 1; }
    printf("  merged state (%d frames) -> %s\n", merged.count, out.c_str());

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
        std::vector<uint16_t> fm_all(nd * K, 0), lum_all(nd * K, 0);
        std::vector<float> V_all(nd * K, 0.f);
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
        double vs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        fs::path od(out);
        bool okw = write_raw(od / "guide.f32", guide_h.data(), guide_h.size()) && write_raw(od / "guide_d.f32", ga_h.data(), ga_h.size()) &&
                   write_raw(od / "vol_fm.f16", fm_all.data(), fm_all.size()) && write_raw(od / "vol_lum.f16", lum_all.data(), lum_all.size()) &&
                   write_raw(od / "vol_V.f32", V_all.data(), V_all.size());
        FILE* mf = fopen((od / "volume.txt").string().c_str(), "w");
        if (mf) {
            fprintf(mf, "dw %d\ndh %d\nK %d\nslices %d\nfm_radius %.6g\nagg_r %d\nagg_eps %.6g\nlayout plane-major (K, dh, dw)\n", vp.dw, vp.dh, K,
                    done, vp.fm_radius, vp.agg_r, vp.agg_eps);
            fclose(mf);
        }
        if (!okw) { fprintf(stderr, "cannot write the volume files\n"); return 1; }
        printf("  volume: %d slices of %dx%d, fm sigma %.2f, aggregation r=%d; guide + aggregation %.2f s -> %s\n", done, vp.dw, vp.dh,
               vp.fm_radius, vp.agg_r, vs, out.c_str());
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

int main(int argc, char** argv) {
    if (argc < 2) return usage();
    std::string cmd = argv[1];
    if (cmd == "probe") return cmd_probe(argc, argv);
    if (cmd == "dump") return cmd_dump(argc, argv);
    if (cmd == "stream") return cmd_stream(argc, argv);
    if (cmd == "warp") return cmd_warp(argc, argv);
    if (cmd == "fold") return cmd_fold(argc, argv);
    return usage();
}
