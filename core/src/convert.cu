#include "stackcore/convert.h"
#include <stdexcept>
#include <string>

namespace stackcore {
namespace {

template <int BITS, int SAMPLES, bool BE>
__global__ void k_convert(const uint8_t* __restrict__ src, size_t npx, float* __restrict__ dst, double* sum) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    float v[3] = {0.f, 0.f, 0.f};
    if (i < npx) {
        const uint8_t* p = src + i * SAMPLES * (BITS / 8);
#pragma unroll
        for (int c = 0; c < 3; ++c) {
            int s = SAMPLES == 1 ? 0 : c;
            unsigned u;
            if (BITS == 16) u = BE ? ((unsigned)p[2 * s] << 8) | p[2 * s + 1] : ((unsigned)p[2 * s + 1] << 8) | p[2 * s];
            else u = p[s];
            v[c] = (float)u / (BITS == 16 ? 65535.0f : 255.0f);
        }
        float* o = dst + i * 3;
        o[0] = v[0];
        o[1] = v[1];
        o[2] = v[2];
    }
    if (sum) {
        double part = (double)v[0] + (double)v[1] + (double)v[2];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) part += __shfl_down_sync(0xffffffffu, part, off);
        __shared__ double warp_sum[32];
        int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
        if (lane == 0) warp_sum[warp] = part;
        __syncthreads();
        if (warp == 0) {
            int nw = (blockDim.x + 31) >> 5;
            part = lane < nw ? warp_sum[lane] : 0.0;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) part += __shfl_down_sync(0xffffffffu, part, off);
            if (lane == 0) atomicAdd(sum, part);
        }
    }
}

template <int BITS, int SAMPLES>
void launch(const uint8_t* src, const TiffInfo& info, float* dst, double* sum, cudaStream_t s) {
    size_t npx = (size_t)info.width * info.height;
    unsigned blocks = (unsigned)((npx + 255) / 256);
    if (info.big_endian) k_convert<BITS, SAMPLES, true><<<blocks, 256, 0, s>>>(src, npx, dst, sum);
    else k_convert<BITS, SAMPLES, false><<<blocks, 256, 0, s>>>(src, npx, dst, sum);
}

}  // namespace

void convert_to_rgb_f32(const uint8_t* d_src, const TiffInfo& info, float* d_dst, double* d_sum, cudaStream_t s) {
    // tiff_probe admits 1, 3 and 4 samples; anything else would read past the frame buffer at the wrong stride
    if (info.samples != 1 && info.samples != 3 && info.samples != 4)
        throw std::runtime_error("convert: unsupported sample count " + std::to_string(info.samples));
    if (info.bits == 16) {
        if (info.samples == 1) launch<16, 1>(d_src, info, d_dst, d_sum, s);
        else if (info.samples == 3) launch<16, 3>(d_src, info, d_dst, d_sum, s);
        else launch<16, 4>(d_src, info, d_dst, d_sum, s);
    } else {
        if (info.samples == 1) launch<8, 1>(d_src, info, d_dst, d_sum, s);
        else if (info.samples == 3) launch<8, 3>(d_src, info, d_dst, d_sum, s);
        else launch<8, 4>(d_src, info, d_dst, d_sum, s);
    }
}

}  // namespace stackcore
