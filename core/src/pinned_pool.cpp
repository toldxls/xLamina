#include "stackcore/pinned_pool.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace stackcore {

PinnedPool::PinnedPool(size_t bytes_per_buffer, int count) : bytes_(bytes_per_buffer), count_(count) {}

PinnedPool::~PinnedPool() {
    for (auto p : all_) cudaFreeHost(p);
}

uint8_t* PinnedPool::acquire() {
    std::unique_lock<std::mutex> lk(m_);
    if (free_.empty() && (int)all_.size() + allocating_ < count_) {   // pin one more, up to the ring's size
        ++allocating_;
        lk.unlock();
        void* p = nullptr;
        cudaError_t e = cudaHostAlloc(&p, bytes_, cudaHostAllocPortable);
        lk.lock();
        --allocating_;
        if (e != cudaSuccess) {
            count_ = (int)all_.size();   // no more pinning; callers wait for what exists
            cv_.notify_all();
            if (all_.empty()) throw std::runtime_error("cudaHostAlloc(" + std::to_string(bytes_) + "): " + cudaGetErrorString(e));
        } else {
            all_.push_back((uint8_t*)p);
            return (uint8_t*)p;
        }
    }
    cv_.wait(lk, [&] { return !free_.empty(); });
    uint8_t* p = free_.back();
    free_.pop_back();
    return p;
}

void PinnedPool::grow(int count) {
    std::lock_guard<std::mutex> lk(m_);
    if (count > count_) count_ = count;
}

bool PinnedPool::idle() {
    std::lock_guard<std::mutex> lk(m_);
    return allocating_ == 0 && free_.size() == all_.size();
}

void PinnedPool::release(uint8_t* p) {
    {
        std::lock_guard<std::mutex> lk(m_);
        free_.push_back(p);
    }
    cv_.notify_one();
}

}  // namespace stackcore
