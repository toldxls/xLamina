#include "stackcore/frame_cache.h"

#include <cuda_runtime.h>
#include <windows.h>

#include <algorithm>
#include <set>

namespace stackcore {

namespace fs = std::filesystem;

static void file_identity(const fs::path& p, uint64_t& size, int64_t& mtime) {
    std::error_code ec;
    size = (uint64_t)fs::file_size(p, ec);
    if (ec) size = 0;
    auto t = fs::last_write_time(p, ec);
    mtime = ec ? 0 : (int64_t)t.time_since_epoch().count();
}

FrameCache::~FrameCache() { clear(); }

void FrameCache::clear() {
    std::lock_guard<std::mutex> lk(m_);
    for (auto& [k, e] : entries_) cudaFreeHost(e.buf);
    entries_.clear();
    resident_ = 0;
}

void FrameCache::prepare(const std::vector<fs::path>& paths, size_t budget_bytes) {
    std::lock_guard<std::mutex> lk(m_);
    std::set<std::wstring> keep;
    for (auto& p : paths) keep.insert(p.native());
    for (auto it = entries_.begin(); it != entries_.end();) {
        bool stale = !keep.count(it->first);
        if (!stale) {
            uint64_t sz; int64_t mt;
            file_identity(fs::path(it->first), sz, mt);
            stale = sz != it->second.file_size || mt != it->second.mtime;
        }
        if (stale) { cudaFreeHost(it->second.buf); resident_ -= it->second.cap; it = entries_.erase(it); }
        else ++it;
    }
    if (budget_bytes > 0) { budget_ = budget_bytes; return; }
    // pinned memory is never paged, so stay well inside what is free now and inside a fixed share
    // of the machine, whatever the RAM disk and the other programs hold
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof ms;
    GlobalMemoryStatusEx(&ms);
    const size_t reserve = (size_t)8 << 30;
    size_t by_free = ms.ullAvailPhys > reserve ? (size_t)ms.ullAvailPhys - reserve + resident_ : resident_;
    size_t by_total = (size_t)(ms.ullTotalPhys * 0.6);
    budget_ = std::min(by_free, by_total);
}

const FrameCache::Entry* FrameCache::find(const fs::path& p) {
    std::lock_guard<std::mutex> lk(m_);
    auto it = entries_.find(p.native());
    return it == entries_.end() ? nullptr : &it->second;
}

bool FrameCache::with(const fs::path& p, const std::function<void(const Entry&)>& f) {
    std::lock_guard<std::mutex> lk(m_);
    auto it = entries_.find(p.native());
    if (it == entries_.end()) return false;
    f(it->second);
    return true;
}

uint8_t* FrameCache::reserve(size_t cap) {
    {
        std::lock_guard<std::mutex> lk(m_);
        if (resident_ + cap > budget_) return nullptr;
        resident_ += cap;
    }
    void* p = nullptr;
    if (cudaHostAlloc(&p, cap, cudaHostAllocPortable) != cudaSuccess) {
        std::lock_guard<std::mutex> lk(m_);
        resident_ -= cap;
        budget_ = resident_;   // pinning failed: take no more this job
        return nullptr;
    }
    return (uint8_t*)p;
}

void FrameCache::unreserve(uint8_t* buf, size_t cap) {
    cudaFreeHost(buf);
    std::lock_guard<std::mutex> lk(m_);
    resident_ -= cap;
}

const FrameCache::Entry* FrameCache::insert(const fs::path& p, uint8_t* buf, size_t cap, size_t offset, size_t bytes) {
    Entry e;
    e.buf = buf; e.cap = cap; e.offset = offset; e.bytes = bytes;
    file_identity(p, e.file_size, e.mtime);
    std::lock_guard<std::mutex> lk(m_);
    auto [it, fresh] = entries_.emplace(p.native(), e);
    if (!fresh) {   // the same path read twice in one pass: keep the first copy
        cudaFreeHost(buf);
        resident_ -= cap;
    }
    return &it->second;
}

}  // namespace stackcore
