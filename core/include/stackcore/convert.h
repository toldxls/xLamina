// Raw TIFF bytes on the device -> float32 RGB in [0, 1], the same values as
// proto/stackproto/frames.read_rgb (uint16 / 65535, uint8 / 255; grey
// replicated to 3 channels; a 4th sample dropped).
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "stackcore/tiff.h"

namespace stackcore {

// dst: height * width * 3 floats, interleaved RGB. If sum != nullptr, the
// double at *sum is atomically increased by the sum of every output value
// (a cheap whole-frame checksum for validation); zero it beforehand.
void convert_to_rgb_f32(const uint8_t* d_src, const TiffInfo& info, float* d_dst, double* d_sum, cudaStream_t s);

}  // namespace stackcore
