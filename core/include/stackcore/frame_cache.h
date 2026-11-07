// Resident frames: the stack held in page-locked host memory across the
// streaming passes (and, in the GUI, across jobs). The first pass that meets a
// frame reads it from disk into a pinned buffer the cache keeps; every later
// pass uploads straight from that buffer, so the GPUs pull at their PCIe link
// rate instead of the RAM disk's read rate (FINDINGS 26). Frames that do not
// fit the memory budget go through the streaming ring as before.
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "stackcore/tiff.h"

namespace stackcore {

class FrameCache {
public:
    struct Entry {
        uint8_t* buf = nullptr;   // pinned (cudaHostAllocPortable), capacity `cap`
        size_t cap = 0;
        size_t offset = 0;        // pixel data starts here inside buf
        size_t bytes = 0;         // pixel bytes
        uint64_t file_size = 0;   // identity of the file the bytes came from
        int64_t mtime = 0;
    };

    FrameCache() = default;
    ~FrameCache();
    FrameCache(const FrameCache&) = delete;
    FrameCache& operator=(const FrameCache&) = delete;

    // Keeps the entries for `paths` whose file is unchanged, frees the rest, and sets the byte
    // budget for this job from the machine's free memory (or `budget_bytes` when > 0).
    void prepare(const std::vector<std::filesystem::path>& paths, size_t budget_bytes = 0);
    void clear();

    const Entry* find(const std::filesystem::path& p);   // nullptr when the frame is not resident
    // A pinned buffer of `cap` bytes for a frame about to be read, or nullptr when the budget is
    // spent (or pinning failed): the caller then uses the streaming ring.
    uint8_t* reserve(size_t cap);
    // Adds the frame; returns the entry to upload from. If the path was inserted meanwhile by another
    // reader (a list with a repeated frame), `buf` is freed and the surviving entry is returned.
    const Entry* insert(const std::filesystem::path& p, uint8_t* buf, size_t cap, size_t offset, size_t bytes);
    void unreserve(uint8_t* buf, size_t cap);              // the read failed: give the buffer back

    size_t bytes() const { return resident_; }
    int count() const { return (int)entries_.size(); }
    size_t budget() const { return budget_; }

private:
    std::map<std::wstring, Entry> entries_;
    std::mutex m_;
    size_t resident_ = 0;   // bytes pinned: resident entries plus reserved buffers
    size_t budget_ = 0;
};

}  // namespace stackcore
