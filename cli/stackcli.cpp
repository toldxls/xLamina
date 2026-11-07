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

#include "stackcore/convert.h"
#include "stackcore/depth.h"
#include "stackcore/pipeline.h"
#include "stackcore/tiff.h"
#include "version.h"

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
            "xLamina " XLAMINA_VERSION " - multi-GPU photomicroscopy stacker\n"
            "usage:\n"
            "  stackcli run DIR [--out DIR] [--prefix IMG_] [--scale 1|0.5] [--outputs B,C,Cn] [--ref middle] [--lock-scale 1]\n"
            "               [--fov-mm X | --um-per-px X] [--gpus 0,1] [--streams 2] [--align-scale 0.25] [--baselines 1,2,4,8] [--pair-threads 3]\n"
            "               [--smoothing 0] [--c-radius 0]   align the TIFF frames in DIR (all of them, or those starting with --prefix) and fuse them:\n"
            "               report.json + 16-bit TIFFs in DIR_stacker (or --out). Outputs: A B B0 C CB Cn CBn Cd Cg Cg2 H\n"
            "               --smoothing S: C takes detail weaker than S x the noise from B (0 = raw selection, 2 ~ Helicon's 1, 3 = Cn)\n"
            "               --c-radius R: C aggregates each frame's pyramid energy over R sensor px before choosing, so a strong\n"
            "                             edge's frame wins the band beside it instead of its defocus rims (0 = off, 32 is a good start)\n"
            "               --b-radius R: the same for B's focus volume: each frame's focus slice is blurred over R sensor px\n"
            "                             (isotropic, after the edge-aware aggregation) before the depth map is solved (0 = off)\n"
            "               --halo-prior H (2) / --dark-prior D (1): B's preference for darker frames beside bright objects / in\n"
            "                             featureless pixels; raise H to push the band beside a bright plate to its own frame\n"
            "               --cache 0: read every pass from disk instead of keeping the frames pinned in host memory\n"
            "               depth volume (the GUI's Advanced panel): --radius-px 6 --power 4 --depth-fraction 0.25 --depth-fm-px 6 --vol-agg-px 32\n"
            "               refinement / composition: --refine-r 12 --refine-fm-px 9 --soft-px 3\n"
            "  developer commands:\n"
            "  stackcli probe FILE.tif\n"
            "  stackcli dump FILE.tif OUT.f32 [--gpu N]\n"
            "  stackcli stream DIR [--prefix IMG_] [--gpus 0,1] [--streams 3] [--readers 16] [--pool N] [--sums OUT.txt]\n"
            "  stackcli warp FILE.tif OUT.f32 --z ZR,ZI --t TX,TY --rect X0,Y0,X1,Y1 [--scale 1|0.5] [--quantize] [--gpu N]\n"
            "  stackcli fold LIST.txt --rect X0,Y0,X1,Y1 --out DIR [--scale 1|0.5] [--radius-px 6] [--power 4] [--c-radius 0]\n"
            "                [--frames 0,5,9] [--gpus 0,1] [--streams 2] [--readers 16] [--volume]\n"
            "  stackcli depth VOLDIR --out DIR [--crop X0,Y0,W,H] [--gpu N] [--dirs 8]\n"
            "  stackcli stack LIST.txt --rect X0,Y0,X1,Y1 --out DIR [--scale 1|0.5] [--gpus 0,1] [--streams 2]\n"
            "                 [--refine-r 12] [--refine-fm-px 9] [--soft-px 3] [--smoothing 0] [--c-radius 0]   the whole fusion -> 16-bit TIFFs\n"
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
        else if (a == "--c-radius") po.c_radius_px = atof(v.c_str());
        else if (a == "--b-radius") po.b_radius_px = atof(v.c_str());
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
    po.host_state = true;   // the validation reads the merged state from disk
    Pass1 P;
    if (int r = run_pass1(po, P)) return r;
    P.dst.free();
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
    JobOptions jo;
    jo.align.dir = argv[2];
    jo.stack.po.opt.streams_per_gpu = 2;
    for (int i = 3; i + 1 < argc; i += 2) {
        std::string a = argv[i], v = argv[i + 1];
        if (a == "--out") jo.out = v;
        else if (parse_align_option(a, v, jo.align)) continue;
        else if (parse_stack_option(a, v, jo.stack)) continue;
        else return usage();
    }
    AlignOut A;
    return run_job(jo, A, Hooks{});
}

int main(int argc, char** argv) {
    if (argc < 2) return usage();
    std::string cmd = argv[1];
    if (cmd == "--version" || cmd == "-v") { printf("xLamina %s\n", XLAMINA_VERSION); return 0; }
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
