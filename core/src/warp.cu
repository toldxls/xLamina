#include "stackcore/warp.h"

#include <cmath>

namespace stackcore {

// ----------------------------------------------------------------------------- geometry (host)

static int round_half_even(double v) { return (int)std::nearbyint(v); }   // Python round()

int Grid::work_w() const { return scale == 1.0 ? W : round_half_even(W * scale); }
int Grid::work_h() const { return scale == 1.0 ? H : round_half_even(H * scale); }
int Grid::out_w() const { return round_half_even((x1 - x0 + 1) * scale); }
int Grid::out_h() const { return round_half_even((y1 - y0 + 1) * scale); }

void affine_for(const Similarity& T, const Grid& g, double M[6]) {
    int w = g.work_w(), h = g.work_h();
    double fx = (double)w / g.W, fy = (double)h / g.H;
    double tx = T.tx * fx, ty = T.ty * fy;
    double cx = (w - 1) / 2.0, cy = (h - 1) / 2.0;          // sim.centre
    double zr = T.zr, zi = T.zi;
    double ox = cx + tx - zr * cx + zi * cy;                 // sim.to_affine
    double oy = cy + ty - zi * cx - zr * cy;
    double x0 = g.x0 * fx, y0 = g.y0 * fy;                  // output (u, v) = work (x0 + u, y0 + v)
    M[0] = zr; M[1] = -zi; M[2] = ox + zr * x0 + (-zi) * y0;
    M[3] = zi; M[4] = zr;  M[5] = oy + zi * x0 + zr * y0;
}

// ----------------------------------------------------------------------------- kernels

namespace {

// OpenCV resizeAreaFast_ for 3-channel float: ((p00 + p01) + p10) + p11, times 0.25
__global__ void k_box_down2(const float* __restrict__ src, int w, int h, float* __restrict__ dst, int w2, int h2) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w2 || y >= h2) return;
    const float* p0 = src + ((size_t)(2 * y) * w + 2 * x) * 3;
    const float* p1 = p0 + (size_t)w * 3;
    float* o = dst + ((size_t)y * w2 + x) * 3;
#pragma unroll
    for (int c = 0; c < 3; ++c) {
        float s = __fadd_rn(__fadd_rn(__fadd_rn(p0[c], p0[3 + c]), p1[c]), p1[3 + c]);
        o[c] = __fmul_rn(s, 0.25f);
    }
}

// OpenCV interpolateCubic, A = -0.75, float arithmetic without FMA contraction
__device__ __forceinline__ void cubic_weights(float x, float* c) {
    const float A = -0.75f;
    float x1 = x + 1.f;
    c[0] = __fsub_rn(__fmul_rn(__fadd_rn(__fmul_rn(__fsub_rn(__fmul_rn(A, x1), 5 * A), x1), 8 * A), x1), 4 * A);
    c[1] = __fadd_rn(__fmul_rn(__fmul_rn(__fsub_rn(__fmul_rn(A + 2, x), A + 3), x), x), 1.f);
    float y = 1.f - x;
    c[2] = __fadd_rn(__fmul_rn(__fmul_rn(__fsub_rn(__fmul_rn(A + 2, y), A + 3), y), y), 1.f);
    c[3] = __fsub_rn(__fsub_rn(__fsub_rn(1.f, c[0]), c[1]), c[2]);
}

__device__ __forceinline__ int reflect(int p, int len) {   // cv::borderInterpolate, BORDER_REFLECT
    if (len == 1) return 0;
    while ((unsigned)p >= (unsigned)len) p = p < 0 ? -p - 1 : 2 * len - 1 - p;
    return p;
}

template <bool QUANT>
__global__ void k_warp_bicubic(const float* __restrict__ src, int sw, int sh, float* __restrict__ dst, int ow, int oh,
                               double m0, double m1, double m2, double m3, double m4, double m5) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= ow || y >= oh) return;
    int sx, sy;
    float fx, fy;
    if (QUANT) {
        // cv::WarpAffineInvoker: AB_BITS 10, INTER_BITS 5, round_delta 16, saturate_cast = round-to-nearest-even
        const int AB_SCALE = 1 << 10;
        int X0 = __double2int_rn((m1 * y + m2) * AB_SCALE) + 16;
        int Y0 = __double2int_rn((m4 * y + m5) * AB_SCALE) + 16;
        int X = (X0 + __double2int_rn(m0 * x * AB_SCALE)) >> 5;
        int Y = (Y0 + __double2int_rn(m3 * x * AB_SCALE)) >> 5;
        sx = X >> 5;
        sy = Y >> 5;
        fx = (float)(X & 31) * (1.f / 32);
        fy = (float)(Y & 31) * (1.f / 32);
    } else {
        // exact positions in double (float32 positions were tried: same 1e-6 rms vs OpenCV 5)
        double u = m0 * x + m1 * y + m2, v = m3 * x + m4 * y + m5;
        double fu = floor(u), fv = floor(v);
        sx = (int)fu;
        sy = (int)fv;
        fx = (float)(u - fu);
        fy = (float)(v - fv);
    }
    float wx[4], wy[4];
    cubic_weights(fx, wx);
    cubic_weights(fy, wy);
    float w[16];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) w[i * 4 + j] = __fmul_rn(wy[i], wx[j]);   // initInterTab2D: tab[y][x] products
    int bx = sx - 1, by = sy - 1;
    float* o = dst + ((size_t)y * ow + x) * 3;
    if ((unsigned)bx < (unsigned)(sw - 3) && (unsigned)by < (unsigned)(sh - 3)) {
        // remapBicubic inner path: per row ((a + b) + c) + d, rows accumulated in order
#pragma unroll
        for (int c = 0; c < 3; ++c) {
            float sum = 0.f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float* S = src + ((size_t)(by + i) * sw + bx) * 3 + c;
                float r = __fmul_rn(S[0], w[i * 4]);
                r = __fadd_rn(r, __fmul_rn(S[3], w[i * 4 + 1]));
                r = __fadd_rn(r, __fmul_rn(S[6], w[i * 4 + 2]));
                r = __fadd_rn(r, __fmul_rn(S[9], w[i * 4 + 3]));
                sum = i == 0 ? r : __fadd_rn(sum, r);
            }
            o[c] = sum;
        }
    } else {
        // border path: every term added in turn to a running sum
        int xs[4], ys[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            xs[i] = reflect(bx + i, sw);
            ys[i] = reflect(by + i, sh);
        }
#pragma unroll
        for (int c = 0; c < 3; ++c) {
            float sum = 0.f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float* S = src + (size_t)ys[i] * sw * 3 + c;
#pragma unroll
                for (int j = 0; j < 4; ++j) sum = __fadd_rn(sum, __fmul_rn(S[xs[j] * 3], w[i * 4 + j]));
            }
            o[c] = sum;
        }
    }
}

}  // namespace

void box_down2_rgb(const float* d_src, int w, int h, float* d_dst, cudaStream_t s) {
    int w2 = w / 2, h2 = h / 2;
    dim3 blk(32, 8), grd((w2 + 31) / 32, (h2 + 7) / 8);
    k_box_down2<<<grd, blk, 0, s>>>(d_src, w, h, d_dst, w2, h2);
}

void warp_bicubic_rgb(const float* d_src, int sw, int sh, const double M[6], float* d_dst, int ow, int oh,
                      bool quantize, cudaStream_t s) {
    dim3 blk(32, 8), grd((ow + 31) / 32, (oh + 7) / 8);
    if (quantize)
        k_warp_bicubic<true><<<grd, blk, 0, s>>>(d_src, sw, sh, d_dst, ow, oh, M[0], M[1], M[2], M[3], M[4], M[5]);
    else
        k_warp_bicubic<false><<<grd, blk, 0, s>>>(d_src, sw, sh, d_dst, ow, oh, M[0], M[1], M[2], M[3], M[4], M[5]);
}

void load_aligned(const float* d_frame, const Grid& g, const Similarity& T, float* d_scratch, float* d_out,
                  bool quantize, cudaStream_t s) {
    double M[6];
    affine_for(T, g, M);
    const float* src = d_frame;
    int sw = g.W, sh = g.H;
    if (g.scale != 1.0) {
        box_down2_rgb(d_frame, g.W, g.H, d_scratch, s);
        src = d_scratch;
        sw = g.work_w();
        sh = g.work_h();
    }
    warp_bicubic_rgb(src, sw, sh, M, d_out, g.out_w(), g.out_h(), quantize, s);
}

}  // namespace stackcore
