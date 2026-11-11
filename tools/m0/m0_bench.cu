// Phase B, M0: measure before optimising (docs/HANDOFF.md §4, docs/DESIGN.md "Milestones").
//
//   m0_bench [--frames DIR] [--ext .tif] [--threads N] [--reps N] [--json FILE]
//
// Per GPU: name, SMs, memory, PCIe link as negotiated; H2D / D2H bandwidth
// (pinned and pageable, one frame = 6960 x 4640 x 16-bit RGB); throughput of
// the core per-frame kernels (uint16 -> luma, similarity warp via texture,
// focus measure, pyramid level). All GPUs at once: concurrent H2D, to see
// what the host side sustains. Host: memcpy bandwidth, and the read rate of
// the frame folder (unbuffered and cached), a RAM disk or a fast SSD.
//
// Kernels here are throughput stand-ins with the right memory traffic, not
// the reference implementations: those come in later milestones, checked
// against proto/stackproto.

#include <cuda_runtime.h>
#include <nvml.h>
#include <windows.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using clk = std::chrono::steady_clock;

#define CK(x)                                                                              \
    do {                                                                                   \
        cudaError_t e_ = (x);                                                              \
        if (e_ != cudaSuccess) {                                                           \
            fprintf(stderr, "CUDA error: %s\n  at %s:%d: %s\n", cudaGetErrorString(e_),    \
                    "m0_bench.cu", __LINE__, #x);                                               \
            exit(1);                                                                       \
        }                                                                                  \
    } while (0)

static const int FW = 6960, FH = 4640;                     // the test stack's frame size
static const size_t NPX = (size_t)FW * FH;
static const size_t FRAME_BYTES = NPX * 3 * sizeof(uint16_t);  // 193.8 MB, 16-bit RGB

// ----------------------------------------------------------------------------- kernels

__constant__ float c_kern[32];

__global__ void k_u16_to_luma(const uint16_t* __restrict__ rgb, float* __restrict__ y, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint16_t* p = rgb + i * 3;
    y[i] = (0.2126f * p[0] + 0.7152f * p[1] + 0.0722f * p[2]) * (1.0f / 65535.0f);
}

// output(x, y) = frame(a x + b y + c, d x + e y + f): similarity, hardware bilinear fetch
__global__ void k_warp(cudaTextureObject_t tex, float* __restrict__ out, int W, int H,
                       float a, float b, float c, float d, float e, float f) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    float u = a * x + b * y + c + 0.5f, v = d * x + e * y + f + 0.5f;
    out[(size_t)y * W + x] = tex2D<float>(tex, u, v);
}

__device__ __forceinline__ float at(const float* __restrict__ p, int W, int H, int x, int y) {
    x = min(max(x, 0), W - 1);
    y = min(max(y, 0), H - 1);
    return p[(size_t)y * W + x];
}

template <int R>
__global__ void k_blur_h(const float* __restrict__ in, float* __restrict__ out, int W, int H) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    float s = 0.f;
#pragma unroll
    for (int k = -R; k <= R; ++k) s += c_kern[k + R] * at(in, W, H, x + k, y);
    out[(size_t)y * W + x] = s;
}

template <int R>
__global__ void k_blur_v(const float* __restrict__ in, float* __restrict__ out, int W, int H) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    float s = 0.f;
#pragma unroll
    for (int k = -R; k <= R; ++k) s += c_kern[k + R] * at(in, W, H, x, y + k);
    out[(size_t)y * W + x] = s;
}

// 3x3 Laplacian, squared (the focus measure's core)
__global__ void k_lap_sq(const float* __restrict__ in, float* __restrict__ out, int W, int H) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    float l = at(in, W, H, x - 1, y) + at(in, W, H, x + 1, y) + at(in, W, H, x, y - 1) +
              at(in, W, H, x, y + 1) - 4.f * at(in, W, H, x, y);
    out[(size_t)y * W + x] = l * l;
}

// Burt-Adelson 5-tap pyrDown (one Laplacian pyramid level's reduce step)
__global__ void k_pyrdown(const float* __restrict__ in, float* __restrict__ out, int W, int H,
                          int w2, int h2) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w2 || y >= h2) return;
    const float k5[5] = {1 / 16.f, 4 / 16.f, 6 / 16.f, 4 / 16.f, 1 / 16.f};
    float s = 0.f;
#pragma unroll
    for (int dy = -2; dy <= 2; ++dy)
#pragma unroll
        for (int dx = -2; dx <= 2; ++dx)
            s += k5[dy + 2] * k5[dx + 2] * at(in, W, H, 2 * x + dx, 2 * y + dy);
    out[(size_t)y * w2 + x] = s;
}

// ----------------------------------------------------------------------------- helpers

static void set_gauss(float sigma, int R) {
    float k[32], sum = 0.f;
    for (int i = -R; i <= R; ++i) sum += k[i + R] = expf(-0.5f * i * i / (sigma * sigma));
    for (int i = 0; i < 2 * R + 1; ++i) k[i] /= sum;
    CK(cudaMemcpyToSymbol(c_kern, k, sizeof(float) * (2 * R + 1)));
}

template <class F>
static float median_ms(F&& f, int reps, cudaStream_t s) {
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    f();
    CK(cudaStreamSynchronize(s));
    std::vector<float> t;
    for (int i = 0; i < reps; ++i) {
        CK(cudaEventRecord(a, s));
        f();
        CK(cudaEventRecord(b, s));
        CK(cudaEventSynchronize(b));
        float ms;
        CK(cudaEventElapsedTime(&ms, a, b));
        t.push_back(ms);
    }
    CK(cudaGetLastError());
    std::sort(t.begin(), t.end());
    CK(cudaEventDestroy(a));
    CK(cudaEventDestroy(b));
    return t[t.size() / 2];
}

static double gbps(size_t bytes, double ms) { return bytes / (ms * 1e-3) / 1e9; }

struct GpuInfo {
    int dev;
    std::string name, bus;
    int sms, gen_cur, gen_max, width_cur, width_max;
    size_t mem;
    double power_w;
    // results (ms per frame)
    float h2d_pin, h2d_page, d2h_pin, luma, warp, fm, pyr;
};

static void nvml_fill(GpuInfo& g) {
    g.gen_cur = g.gen_max = g.width_cur = g.width_max = 0;
    g.power_w = 0;
    nvmlDevice_t h;
    if (nvmlDeviceGetHandleByPciBusId_v2(g.bus.c_str(), &h) != NVML_SUCCESS) return;
    unsigned v;
    if (nvmlDeviceGetCurrPcieLinkGeneration(h, &v) == NVML_SUCCESS) g.gen_cur = v;
    if (nvmlDeviceGetMaxPcieLinkGeneration(h, &v) == NVML_SUCCESS) g.gen_max = v;
    if (nvmlDeviceGetCurrPcieLinkWidth(h, &v) == NVML_SUCCESS) g.width_cur = v;
    if (nvmlDeviceGetMaxPcieLinkWidth(h, &v) == NVML_SUCCESS) g.width_max = v;
    if (nvmlDeviceGetPowerManagementLimit(h, &v) == NVML_SUCCESS) g.power_w = v / 1000.0;
}

// ----------------------------------------------------------------------------- per-GPU bench

struct DevBufs {
    cudaStream_t s;
    uint16_t* d_rgb;
    uint16_t* h_pin;   // portable pinned, one frame
};

static void bench_gpu(GpuInfo& g, DevBufs& b, uint16_t* h_page, int reps) {
    CK(cudaSetDevice(g.dev));
    CK(cudaStreamCreate(&b.s));
    CK(cudaMalloc(&b.d_rgb, FRAME_BYTES));
    CK(cudaHostAlloc(&b.h_pin, FRAME_BYTES, cudaHostAllocPortable));
    memset(b.h_pin, 0x55, FRAME_BYTES);
    float *d_y, *d_t1, *d_t2, *d_half;
    int w2 = (FW + 1) / 2, h2 = (FH + 1) / 2;
    CK(cudaMalloc(&d_y, NPX * 4));
    CK(cudaMalloc(&d_t1, NPX * 4));
    CK(cudaMalloc(&d_t2, NPX * 4));
    CK(cudaMalloc(&d_half, (size_t)w2 * h2 * 4));
    cudaStream_t s = b.s;

    g.h2d_pin = median_ms([&] { CK(cudaMemcpyAsync(b.d_rgb, b.h_pin, FRAME_BYTES, cudaMemcpyHostToDevice, s)); }, reps, s);
    g.h2d_page = median_ms([&] { CK(cudaMemcpyAsync(b.d_rgb, h_page, FRAME_BYTES, cudaMemcpyHostToDevice, s)); }, reps, s);
    g.d2h_pin = median_ms([&] { CK(cudaMemcpyAsync(b.h_pin, b.d_rgb, FRAME_BYTES, cudaMemcpyDeviceToHost, s)); }, reps, s);

    dim3 blk(32, 8), grd((FW + 31) / 32, (FH + 7) / 8);
    g.luma = median_ms([&] { k_u16_to_luma<<<(unsigned)((NPX + 255) / 256), 256, 0, s>>>(b.d_rgb, d_y, NPX); }, reps, s);

    cudaResourceDesc rd{};
    rd.resType = cudaResourceTypePitch2D;
    rd.res.pitch2D.devPtr = d_y;
    rd.res.pitch2D.desc = cudaCreateChannelDesc<float>();
    rd.res.pitch2D.width = FW;
    rd.res.pitch2D.height = FH;
    rd.res.pitch2D.pitchInBytes = FW * sizeof(float);
    cudaTextureDesc td{};
    td.addressMode[0] = td.addressMode[1] = cudaAddressModeMirror;
    td.filterMode = cudaFilterModeLinear;
    td.readMode = cudaReadModeElementType;
    cudaTextureObject_t tex;
    CK(cudaCreateTextureObject(&tex, &rd, &td, nullptr));
    // similarity like the real stack's extremes: scale 1.0125, rotation 0.3 deg, shift (30, 70)
    float sc = 1.0125f, th = 0.3f * 3.14159265f / 180.f, cx = FW / 2.f, cy = FH / 2.f;
    float a = sc * cosf(th), bb = -sc * sinf(th), d = sc * sinf(th), e = sc * cosf(th);
    float c = cx - a * cx - bb * cy + 30.f, f = cy - d * cx - e * cy + 70.f;
    g.warp = median_ms([&] { k_warp<<<grd, blk, 0, s>>>(tex, d_t1, FW, FH, a, bb, c, d, e, f); }, reps, s);

    // focus measure: blur(0.8) -> Laplacian^2 -> blur(3.0); OpenCV float kernels are 4 sigma: radius 3 / 12
    g.fm = median_ms([&] {
        set_gauss(0.8f, 3);
        k_blur_h<3><<<grd, blk, 0, s>>>(d_t1, d_t2, FW, FH);
        k_blur_v<3><<<grd, blk, 0, s>>>(d_t2, d_y, FW, FH);
        k_lap_sq<<<grd, blk, 0, s>>>(d_y, d_t2, FW, FH);
        set_gauss(3.0f, 12);
        k_blur_h<12><<<grd, blk, 0, s>>>(d_t2, d_y, FW, FH);
        k_blur_v<12><<<grd, blk, 0, s>>>(d_y, d_t2, FW, FH);
    }, reps, s);

    dim3 grd2((w2 + 31) / 32, (h2 + 7) / 8);
    g.pyr = median_ms([&] { k_pyrdown<<<grd2, blk, 0, s>>>(d_t1, d_half, FW, FH, w2, h2); }, reps, s);

    CK(cudaDestroyTextureObject(tex));
    CK(cudaFree(d_y));
    CK(cudaFree(d_t1));
    CK(cudaFree(d_t2));
    CK(cudaFree(d_half));
}

// all GPUs upload a frame at the same time (split mode): wall-clock median
static double bench_concurrent_h2d(std::vector<GpuInfo>& gs, std::vector<DevBufs>& bs, int reps) {
    std::vector<double> t;
    for (int r = 0; r < reps + 1; ++r) {
        auto t0 = clk::now();
        for (size_t i = 0; i < gs.size(); ++i) {
            CK(cudaSetDevice(gs[i].dev));
            CK(cudaMemcpyAsync(bs[i].d_rgb, bs[i].h_pin, FRAME_BYTES, cudaMemcpyHostToDevice, bs[i].s));
        }
        for (size_t i = 0; i < gs.size(); ++i) {
            CK(cudaSetDevice(gs[i].dev));
            CK(cudaStreamSynchronize(bs[i].s));
        }
        if (r) t.push_back(std::chrono::duration<double, std::milli>(clk::now() - t0).count());
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

// ----------------------------------------------------------------------------- host side

static double bench_memcpy(int threads, int reps) {
    // each byte of a frame crosses host RAM ~3x (file cache or RAM disk -> pinned -> DMA); this is one crossing
    std::vector<std::vector<uint8_t>> src(threads), dst(threads);
    for (int i = 0; i < threads; ++i) {
        src[i].assign(FRAME_BYTES, (uint8_t)i);
        dst[i].assign(FRAME_BYTES, 0);
    }
    std::vector<double> t;
    for (int r = 0; r < reps + 1; ++r) {
        auto t0 = clk::now();
        std::vector<std::thread> th;
        for (int i = 0; i < threads; ++i)
            th.emplace_back([&, i] { memcpy(dst[i].data(), src[i].data(), FRAME_BYTES); });
        for (auto& x : th) x.join();
        if (r) t.push_back(std::chrono::duration<double, std::milli>(clk::now() - t0).count());
    }
    std::sort(t.begin(), t.end());
    return gbps(FRAME_BYTES * (size_t)threads, t[t.size() / 2]);
}

struct ReadResult {
    double gbps = 0;
    size_t bytes = 0;
    int files = 0;
    double ms = 0;
};

static ReadResult bench_read(const std::vector<fs::path>& files, int threads, bool unbuffered) {
    ReadResult r;
    r.files = (int)files.size();
    std::atomic<size_t> next{0}, total{0};
    // one whole frame per read so the destination is never L3-resident (a
    // 16 MB chunk reused per thread sat in the 7950X3D's cache and read 20 GB/s)
    const size_t CH = 256u << 20;
    auto t0 = clk::now();
    std::vector<std::thread> th;
    for (int i = 0; i < threads; ++i)
        th.emplace_back([&] {
            void* buf = VirtualAlloc(nullptr, CH, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
            for (;;) {
                size_t k = next++;
                if (k >= files.size()) break;
                DWORD flags = FILE_FLAG_SEQUENTIAL_SCAN | (unbuffered ? FILE_FLAG_NO_BUFFERING : 0);
                HANDLE h = CreateFileW(files[k].c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                                       OPEN_EXISTING, flags, nullptr);
                if (h == INVALID_HANDLE_VALUE) continue;
                DWORD got;
                while (ReadFile(h, buf, (DWORD)CH, &got, nullptr) && got) total += got;
                CloseHandle(h);
            }
            VirtualFree(buf, 0, MEM_RELEASE);
        });
    for (auto& x : th) x.join();
    r.ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    r.bytes = total;
    r.gbps = gbps(r.bytes, r.ms);
    return r;
}

// ----------------------------------------------------------------------------- main

int main(int argc, char** argv) {
    std::string frames, ext = ".tif", json;
    int threads = 8, reps = 10;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto val = [&] { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "--frames") frames = val();
        else if (a == "--ext") ext = val();
        else if (a == "--threads") threads = atoi(val().c_str());
        else if (a == "--reps") reps = atoi(val().c_str());
        else if (a == "--json") json = val();
        else { fprintf(stderr, "usage: m0_bench [--frames DIR] [--ext .tif] [--threads N] [--reps N] [--json FILE]\n"); return 2; }
    }
    if (frames.empty()) {
        const char* e = getenv("STACKER_FRAMES");
        if (e) frames = e;
    }

    int n = 0;
    CK(cudaGetDeviceCount(&n));
    bool nvml_ok = nvmlInit_v2() == NVML_SUCCESS;
    std::vector<GpuInfo> gs(n);
    for (int i = 0; i < n; ++i) {
        cudaDeviceProp p;
        CK(cudaGetDeviceProperties(&p, i));
        char bus[32];
        CK(cudaDeviceGetPCIBusId(bus, sizeof bus, i));
        gs[i].dev = i;
        gs[i].name = p.name;
        gs[i].bus = bus;
        gs[i].sms = p.multiProcessorCount;
        gs[i].mem = p.totalGlobalMem;
        if (nvml_ok) nvml_fill(gs[i]);
    }
    int drv = 0, rt = 0;
    CK(cudaDriverGetVersion(&drv));
    CK(cudaRuntimeGetVersion(&rt));
    printf("CUDA driver %d.%d, runtime %d.%d, %d GPU(s)\n\n", drv / 1000, drv % 1000 / 10, rt / 1000, rt % 1000 / 10, n);
    printf("%-4s %-28s %-14s %4s %7s  %-14s %6s\n", "dev", "name", "bus", "SMs", "mem GB", "PCIe (max)", "W");
    for (auto& g : gs)
        printf("%-4d %-28s %-14s %4d %7.1f  Gen%d x%-2d (Gen%d x%d) %6.0f\n", g.dev, g.name.c_str(), g.bus.c_str(),
               g.sms, g.mem / 1e9, g.gen_cur, g.width_cur, g.gen_max, g.width_max, g.power_w);

    uint16_t* h_page = (uint16_t*)malloc(FRAME_BYTES);
    memset(h_page, 0x33, FRAME_BYTES);
    std::vector<DevBufs> bs(n);
    printf("\nPer GPU, one 6960x4640 frame (%.1f MB as 16-bit RGB), median of %d:\n", FRAME_BYTES / 1e6, reps);
    printf("%-4s %10s %10s %10s | %8s %8s %8s %8s | %9s\n", "dev", "H2D pin", "H2D page", "D2H pin", "luma", "warp", "focus", "pyrDown", "kern fps");
    printf("%-4s %10s %10s %10s | %8s %8s %8s %8s |\n", "", "GB/s", "GB/s", "GB/s", "ms", "ms", "ms", "ms");
    for (int i = 0; i < n; ++i) {
        bench_gpu(gs[i], bs[i], h_page, reps);
        auto& g = gs[i];
        float kern = g.luma + g.warp + g.fm + g.pyr;
        printf("%-4d %10.2f %10.2f %10.2f | %8.2f %8.2f %8.2f %8.2f | %9.1f\n", g.dev, gbps(FRAME_BYTES, g.h2d_pin),
               gbps(FRAME_BYTES, g.h2d_page), gbps(FRAME_BYTES, g.d2h_pin), g.luma, g.warp, g.fm, g.pyr, 1000.f / kern);
    }
    double conc_ms = 0;
    if (n > 1) {
        conc_ms = bench_concurrent_h2d(gs, bs, reps);
        printf("\nAll %d GPUs uploading a frame at once: %.2f ms, %.2f GB/s aggregate\n", n, conc_ms,
               gbps(FRAME_BYTES * (size_t)n, conc_ms));
    }

    printf("\nHost memcpy, one frame per thread: ");
    double mc1 = bench_memcpy(1, reps), mc8 = bench_memcpy(threads, reps);
    printf("1 thread %.1f GB/s, %d threads %.1f GB/s aggregate\n", mc1, threads, mc8);

    ReadResult rd_un, rd_buf;
    if (!frames.empty() && fs::is_directory(frames)) {
        std::vector<fs::path> files;
        for (auto& e : fs::directory_iterator(frames))
            if (e.is_regular_file() && e.path().extension() == ext && e.path().filename().wstring().rfind(L"IMG_", 0) == 0)
                files.push_back(e.path());
        std::sort(files.begin(), files.end());
        if (files.empty()) printf("\nNo IMG_*%s files in %s\n", ext.c_str(), frames.c_str());
        else {
            rd_un = bench_read(files, threads, true);
            bench_read(files, threads, false);          // first buffered pass fills the file cache
            rd_buf = bench_read(files, threads, false); // second one measures the cache itself
            printf("\nRead %d files (%.1f GB) from %s with %d threads: unbuffered %.2f GB/s (%.0f ms), from the file cache %.2f GB/s (%.0f ms)\n",
                   rd_un.files, rd_un.bytes / 1e9, frames.c_str(), threads, rd_un.gbps, rd_un.ms, rd_buf.gbps, rd_buf.ms);
            if (rd_un.bytes == 0) printf("  (unbuffered reads failed on this volume: the driver may not support FILE_FLAG_NO_BUFFERING)\n");
        }
    } else
        printf("\nNo frame folder (--frames or STACKER_FRAMES): skipped the read-rate test\n");

    if (!json.empty()) {
        FILE* f = fopen(json.c_str(), "w");
        if (f) {
            fprintf(f, "{\n \"frame_bytes\": %zu, \"reps\": %d,\n \"gpus\": [\n", FRAME_BYTES, reps);
            for (int i = 0; i < n; ++i) {
                auto& g = gs[i];
                fprintf(f, "  {\"dev\": %d, \"name\": \"%s\", \"bus\": \"%s\", \"sms\": %d, \"mem_bytes\": %zu, \"pcie_gen\": %d, \"pcie_width\": %d, "
                           "\"pcie_gen_max\": %d, \"pcie_width_max\": %d, \"power_limit_w\": %.0f, \"h2d_pinned_gbps\": %.3f, \"h2d_pageable_gbps\": %.3f, "
                           "\"d2h_pinned_gbps\": %.3f, \"ms\": {\"luma\": %.3f, \"warp\": %.3f, \"focus\": %.3f, \"pyrdown\": %.3f}}%s\n",
                        g.dev, g.name.c_str(), g.bus.c_str(), g.sms, g.mem, g.gen_cur, g.width_cur, g.gen_max, g.width_max, g.power_w,
                        gbps(FRAME_BYTES, g.h2d_pin), gbps(FRAME_BYTES, g.h2d_page), gbps(FRAME_BYTES, g.d2h_pin), g.luma, g.warp, g.fm, g.pyr,
                        i + 1 < n ? "," : "");
            }
            fprintf(f, " ],\n \"concurrent_h2d_ms\": %.3f,\n \"host_memcpy_gbps\": {\"1\": %.2f, \"%d\": %.2f},\n", conc_ms, mc1, threads, mc8);
            fprintf(f, " \"read\": {\"dir\": \"%s\", \"files\": %d, \"bytes\": %zu, \"threads\": %d, \"unbuffered_gbps\": %.3f, \"buffered_gbps\": %.3f}\n}\n",
                    frames.c_str(), rd_un.files, rd_un.bytes, threads, rd_un.gbps, rd_buf.gbps);
            fclose(f);
            printf("\nwrote %s\n", json.c_str());
        }
    }
    if (nvml_ok) nvmlShutdown();
    return 0;
}
