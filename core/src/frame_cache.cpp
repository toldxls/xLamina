#include "stackcore/frame_cache.h"

#include <cuda_runtime.h>
#include <windows.h>

#include <algorithm>
#include <atomic>
#include <set>
#include <thread>

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
    // the buffers are taken out under the lock and unpinned outside it: unpinning tens of GB takes seconds, and the
    // window's memory readout asks for bytes() meanwhile
    std::vector<uint8_t*> go;
    {
        std::lock_guard<std::mutex> lk(m_);
        for (auto& [k, e] : entries_) go.push_back(e.buf);
        for (auto& s : spare_) go.push_back(s.first);
        entries_.clear();
        spare_.clear();
        resident_ = 0;
    }
    for (uint8_t* b : go) cudaFreeHost(b);
}

size_t FrameCache::bytes() {
    // never waits: a pass that is unpinning under the lock gets the last figure instead of holding the window up
    std::unique_lock<std::mutex> lk(m_, std::try_to_lock);
    if (!lk.owns_lock()) return last_bytes_.load();
    size_t b = resident_;
    for (auto& s : spare_) b += s.second;
    last_bytes_ = b;
    return b;
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
        if (stale) { spare_.push_back({it->second.buf, it->second.cap}); resident_ -= it->second.cap; it = entries_.erase(it); }
        else ++it;
    }
    // no more spare buffers are kept than this stack has frames to read: the rest of a larger stack's go back now
    // (a folder of 85 GB opened before one of 47 GB left 85 GB pinned beside it)
    {
        size_t missing = 0;
        for (auto& k : keep) if (!entries_.count(k)) ++missing;
        while (spare_.size() > missing) { cudaFreeHost(spare_.back().first); spare_.pop_back(); }
    }
    if (budget_bytes > 0) { budget_ = budget_bytes; return; }
    // pinned memory is never paged, so stay well inside what is free now and inside a fixed share
    // of the machine, whatever the other programs (or a RAM disk) hold
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof ms;
    GlobalMemoryStatusEx(&ms);
    // ... and leave the job its own memory: the depth volume on the host is the large piece, about half the frames'
    // bytes for 8-bit files at the default depth resolution (a quarter for 16-bit), beside the alignment's luma and
    // the outputs. A fixed 8 GB left a 32 GB PC paging and a 16 GB one without room for the volume at all.
    uint64_t frames_bytes = 0;
    for (auto& k : keep) { uint64_t sz = 0; int64_t mt = 0; file_identity(fs::path(k), sz, mt); frames_bytes += sz; }
    const size_t reserve = std::max<size_t>((size_t)8 << 30, (size_t)(frames_bytes / 2) + ((size_t)4 << 30));
    // what this cache holds already is not "free" to Windows but is this job's to use: the resident frames and the
    // spare buffers kept for it (left out, a stack reopened after an edit to its files got a budget short by them)
    size_t ours = resident_;
    for (auto& s : spare_) ours += s.second;
    size_t by_free = ms.ullAvailPhys > reserve ? (size_t)ms.ullAvailPhys - reserve + ours : ours;
    size_t by_total = (size_t)(ms.ullTotalPhys * 0.6);
    budget_ = std::min(by_free, by_total);
}

void FrameCache::prepin(const std::vector<fs::path>& paths, size_t cap) {
    size_t want = 0;
    {
        std::lock_guard<std::mutex> lk(m_);
        // spares of another size are no use to this stack: their memory goes back
        size_t have = 0;
        for (size_t i = 0; i < spare_.size();) {
            if (spare_[i].second == cap) { ++have; ++i; }
            else { cudaFreeHost(spare_[i].first); spare_[i] = spare_.back(); spare_.pop_back(); }
        }
        std::set<std::wstring> seen;
        size_t missing = 0;
        for (auto& p : paths) if (!entries_.count(p.native()) && seen.insert(p.native()).second) ++missing;
        const size_t room = budget_ > resident_ ? (budget_ - resident_) / cap : 0;
        missing = std::min(missing, room);
        want = missing > have ? missing - have : 0;
        while (have > missing) {   // more spares than this stack can use
            cudaFreeHost(spare_.back().first);
            spare_.pop_back();
            --have;
        }
    }
    if (!want) return;
    std::atomic<size_t> next{0};
    std::atomic<bool> stop{false};
    std::vector<std::thread> th;
    for (int t = 0; t < 16; ++t)
        th.emplace_back([&] {
            while (!stop.load() && next++ < want) {
                void* p = nullptr;
                if (cudaHostAlloc(&p, cap, cudaHostAllocPortable) != cudaSuccess) { stop = true; break; }
                std::lock_guard<std::mutex> lk(m_);
                spare_.push_back({(uint8_t*)p, cap});
            }
        });
    for (auto& t : th) t.join();
    cudaGetLastError();
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
    {   // a buffer pinned earlier (prepin, or a frame that left the cache)
        std::lock_guard<std::mutex> lk(m_);
        for (size_t i = 0; i < spare_.size(); ++i)
            if (spare_[i].second == cap) {
                uint8_t* b = spare_[i].first;
                spare_[i] = spare_.back();
                spare_.pop_back();
                return b;
            }
        // none of this size: the spares are another stack's frame size and of no use to this one
        for (auto& s : spare_) cudaFreeHost(s.first);
        spare_.clear();
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
    std::lock_guard<std::mutex> lk(m_);
    spare_.push_back({buf, cap});
    resident_ -= cap;
}

const FrameCache::Entry* FrameCache::insert(const fs::path& p, uint8_t* buf, size_t cap, size_t offset, size_t bytes) {
    Entry e;
    e.buf = buf; e.cap = cap; e.offset = offset; e.bytes = bytes;
    file_identity(p, e.file_size, e.mtime);
    std::lock_guard<std::mutex> lk(m_);
    auto [it, fresh] = entries_.emplace(p.native(), e);
    if (!fresh) {   // the same path read twice in one pass: keep the first copy
        spare_.push_back({buf, cap});
        resident_ -= cap;
    }
    return &it->second;
}

}  // namespace stackcore
