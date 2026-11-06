// A fixed set of page-locked host buffers that readers fill and GPU uploads
// drain (DESIGN.md: "the RAM disk is the cache; the app streams files into a
// small ring of pinned buffers").
#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

namespace stackcore {

class PinnedPool {
public:
    PinnedPool(size_t bytes_per_buffer, int count);   // cudaHostAllocPortable: usable by every device
    ~PinnedPool();
    PinnedPool(const PinnedPool&) = delete;
    PinnedPool& operator=(const PinnedPool&) = delete;

    uint8_t* acquire();          // blocks until a buffer is free
    void release(uint8_t* p);
    size_t bytes() const { return bytes_; }
    int count() const { return (int)all_.size(); }

private:
    size_t bytes_;
    std::vector<uint8_t*> all_, free_;
    std::mutex m_;
    std::condition_variable cv_;
};

}  // namespace stackcore
