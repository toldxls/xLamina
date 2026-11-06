// Image primitives shared by the fusion stages, written to follow OpenCV's
// float paths (reflect-101 borders, getGaussianKernel coefficients, pyrDown /
// pyrUp 5-tap with pyrUp's 7/8 far-edge weights, normalised box filter,
// INTER_AREA resize). Single-channel float unless the name says rgb (x3).
#pragma once

#include <cuda_runtime.h>

#include <vector>

namespace stackcore {

// A Gaussian kernel resident on the current device (cv::getGaussianKernel,
// ksize = round(8 sigma + 1) | 1 as cv::GaussianBlur picks for float images).
struct DevGaussian {
    int R = 0;
    float* d_k = nullptr;
    void build(double sigma);
    void free();
};

// out = GaussianBlur(in), tmp is a w*h scratch
void gaussian_blur(const float* d_in, float* d_out, float* d_tmp, int w, int h, const DevGaussian& g, cudaStream_t s);
// out = boxFilter(in, (2r+1)^2, normalised), tmp is a w*h scratch
void box_filter(const float* d_in, float* d_out, float* d_tmp, int w, int h, int r, cudaStream_t s);
// y = rgb @ (0.2126, 0.7152, 0.0722)
void luma(const float* d_rgb, float* d_y, size_t n, cudaStream_t s);
// out = Laplacian(in, ksize 3)^2 (OpenCV's {2,0,2,0,-8,0,2,0,2})
void laplacian3_sq(const float* d_in, float* d_out, int w, int h, cudaStream_t s);
// out = |Laplacian(in, ksize 3)|
void laplacian3_abs(const float* d_in, float* d_out, int w, int h, cudaStream_t s);
// cv::pyrDown / pyrUp, 3 channels; pyrUp to an explicit size (2w or 2w-1)
void pyr_down_rgb(const float* d_in, int w, int h, float* d_out, int w2, int h2, cudaStream_t s);
void pyr_up_rgb(const float* d_in, int sw, int sh, float* d_out, int dw, int dh, cudaStream_t s);
// Laplacian level: lap = g - pyrUp(gn) (fused, no scratch)
void lap_level_rgb(const float* d_g, const float* d_gn, int sw, int sh, float* d_lap, int dw, int dh, cudaStream_t s);
// out = a * b, out = max(a, 0), out = sqrt(a), out = a * scale, out = clip(a, 0, 1): elementwise helpers
void mul(const float* a, const float* b, float* out, size_t n, cudaStream_t s);
void add(const float* a, const float* b, float* out, size_t n, cudaStream_t s);
void f16_to_f32(const void* a, float* out, size_t n, cudaStream_t s);
void f32_to_f16(const float* a, void* out, size_t n, cudaStream_t s);   // round to nearest even, like numpy
void relu_sqrt(float* a, size_t n, cudaStream_t s);
void clip01(float* a, size_t n, cudaStream_t s);

// cv::resize INTER_AREA from (sw, sh) to (dw, dh), `ch` interleaved channels.
// Integer factors take OpenCV's fast path (plain cell sums), anything else its
// weighted-overlap path, with the same float summation order.
class AreaResizer {
public:
    void build(int sw, int sh, int dw, int dh);   // on the current device
    void free();
    void run(const float* d_src, float* d_dst, int ch, cudaStream_t s) const;
    int sw = 0, sh = 0, dw = 0, dh = 0;

private:
    bool fast_ = false;
    int fx_ = 0, fy_ = 0;
    int *d_xoff_ = nullptr, *d_xidx_ = nullptr, *d_yoff_ = nullptr, *d_yidx_ = nullptr;
    float *d_xw_ = nullptr, *d_yw_ = nullptr;
};

// cv::getGaussianKernel for CV_32F; cv::GaussianBlur's ksize/2 for float images
std::vector<float> gaussian_kernel(int n, double sigma);
int blur_radius(double sigma);

}  // namespace stackcore
