// A fixed set of page-locked host buffers that readers fill and GPU uploads
// drain: the streaming ring for frames that are not resident (frame_cache.h).
#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

namespace stackcore {

class PinnedPool {
public:
    PinnedPool(size_t bytes_per_buffer, int count);   // cudaHostAllocPortable: usable by every device; pinned on first use
    ~PinnedPool();
    PinnedPool(const PinnedPool&) = delete;
    PinnedPool& operator=(const PinnedPool&) = delete;

    uint8_t* acquire();          // blocks until a buffer is free
    void release(uint8_t* p);
    void grow(int count);        // a later pass with more streams: the ring may pin up to `count` buffers
    bool idle();                 // every buffer handed out has come back
    size_t bytes() const { return bytes_; }
    int count() const { return count_; }

private:
    size_t bytes_;
    int count_;
    int allocating_ = 0;   // buffers being pinned outside the lock
    std::vector<uint8_t*> all_, free_;
    std::mutex m_;
    std::condition_variable cv_;
};

}  // namespace stackcore
