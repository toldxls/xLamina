// stackcli: headless runner (DESIGN.md repo layout). Step 1 commands:
//
//   stackcli probe  FILE.tif                       TIFF layout as the reader sees it
//   stackcli dump   FILE.tif OUT.f32 [--gpu N]     read -> GPU -> float RGB -> raw file (H x W x 3 float32)
//   stackcli stream DIR [--prefix IMG_] [--gpus 0,1] [--streams 3] [--readers 16] [--pool N] [--sums OUT.txt]
//                                                  stream every frame through the GPUs: throughput + per-frame checksums
//   stackcli warp   FILE.tif OUT.f32 --z ZR,ZI --t TX,TY --rect X0,Y0,X1,Y1 [--scale 1|0.5] [--quantize] [--gpu N]
//                                                  read -> GPU -> fusion.load_aligned equivalent -> raw float RGB
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "stackcore/convert.h"
#include "stackcore/streamer.h"
#include "stackcore/tiff.h"
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
            "  stackcli warp FILE.tif OUT.f32 --z ZR,ZI --t TX,TY --rect X0,Y0,X1,Y1 [--scale 1|0.5] [--quantize] [--gpu N]\n");
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

int main(int argc, char** argv) {
    if (argc < 2) return usage();
    std::string cmd = argv[1];
    if (cmd == "probe") return cmd_probe(argc, argv);
    if (cmd == "dump") return cmd_dump(argc, argv);
    if (cmd == "stream") return cmd_stream(argc, argv);
    if (cmd == "warp") return cmd_warp(argc, argv);
    return usage();
}
