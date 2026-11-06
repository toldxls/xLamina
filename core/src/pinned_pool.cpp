#include "stackcore/pinned_pool.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace stackcore {

PinnedPool::PinnedPool(size_t bytes_per_buffer, int count) : bytes_(bytes_per_buffer) {
    for (int i = 0; i < count; ++i) {
        void* p = nullptr;
        cudaError_t e = cudaHostAlloc(&p, bytes_, cudaHostAllocPortable);
        if (e != cudaSuccess)
            throw std::runtime_error("cudaHostAlloc(" + std::to_string(bytes_) + " x " + std::to_string(i + 1) + "): " +
                                     cudaGetErrorString(e));
        all_.push_back((uint8_t*)p);
        free_.push_back((uint8_t*)p);
    }
}

PinnedPool::~PinnedPool() {
    for (auto p : all_) cudaFreeHost(p);
}

uint8_t* PinnedPool::acquire() {
    std::unique_lock<std::mutex> lk(m_);
    cv_.wait(lk, [&] { return !free_.empty(); });
    uint8_t* p = free_.back();
    free_.pop_back();
    return p;
}

void PinnedPool::release(uint8_t* p) {
    {
        std::lock_guard<std::mutex> lk(m_);
        free_.push_back(p);
    }
    cv_.notify_one();
}

}  // namespace stackcore
