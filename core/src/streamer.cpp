#include "stackcore/streamer.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

#include "stackcore/pinned_pool.h"

namespace stackcore {
namespace {

#define CK(x)                                                                                        \
    do {                                                                                             \
        cudaError_t e_ = (x);                                                                        \
        if (e_ != cudaSuccess) throw std::runtime_error(std::string(#x) + ": " + cudaGetErrorString(e_)); \
    } while (0)

struct Ready {
    int k;
    uint8_t* buf;
    size_t offset;   // pixel data offset inside buf
    size_t bytes;
    bool pooled;     // buf belongs to the ring (release after the upload); else to the cache
};

struct Queue {
    std::deque<Ready> items;
    std::mutex m;
    std::condition_variable cv;
    bool done = false;

    void push(Ready r) {
        { std::lock_guard<std::mutex> lk(m); items.push_back(r); }
        cv.notify_one();
    }
    bool pop(Ready& r) {
        std::unique_lock<std::mutex> lk(m);
        cv.wait(lk, [&] { return !items.empty() || done; });
        if (items.empty()) return false;
        r = items.front();
        items.pop_front();
        return true;
    }
    void finish() {
        { std::lock_guard<std::mutex> lk(m); done = true; }
        cv.notify_all();
    }
};

}  // namespace

StreamerStats stream_frames(const std::vector<std::filesystem::path>& paths, const StreamerOptions& opt,
                            const FrameFn& on_frame) {
    StreamerStats st;
    const int N = (int)paths.size();
    std::vector<TiffInfo> infos(N);
    size_t max_px = 0;
    for (int k = 0; k < N; ++k) {
        infos[k] = tiff_probe(paths[k]);
        if (!infos[k].ok()) st.errors.push_back(paths[k].filename().string() + ": " + infos[k].error);
        else max_px = std::max(max_px, infos[k].pixel_bytes());
    }
    if (max_px == 0) return st;
    // one size per pass: the callers allocate their device frames for the first frame's size, so a frame of
    // another size (a list written by hand, a file replaced since the folder was scanned) is left out here
    for (int k = 0, k0 = -1; k < N; ++k) {
        if (!infos[k].ok()) continue;
        if (k0 < 0) { k0 = k; continue; }
        if (infos[k].width != infos[k0].width || infos[k].height != infos[k0].height) {
            infos[k].error = "not the size of the first frame";
            st.errors.push_back(paths[k].filename().string() + ": " + infos[k].error);
        }
    }

    std::vector<int> devs = opt.devices;
    if (devs.empty()) {
        int n = 0;
        CK(cudaGetDeviceCount(&n));
        for (int i = 0; i < n; ++i) devs.push_back(i);
    }
    std::vector<int> S_dev(devs.size(), std::max(1, opt.streams_per_gpu));
    if (opt.streams_by_device.size() == devs.size())
        for (size_t i = 0; i < devs.size(); ++i) S_dev[i] = std::max(1, opt.streams_by_device[i]);
    int S_sum = 0;
    for (int s : S_dev) S_sum += s;
    const int R = std::max(1, opt.readers);
    const int P = opt.pool > 0 ? opt.pool : R + 2 * S_sum;
    const size_t cap = tiff_buffer_capacity(max_px);
    PinnedPool pool(cap, P);
    st.frames_per_device.assign(devs.size(), 0);

    Queue q;
    std::atomic<int> next{0};
    std::mutex stat_m;
    auto t0 = std::chrono::steady_clock::now();

    // readers
    std::vector<std::thread> readers;
    std::atomic<bool> readers_failed{false};   // a reader hit a fatal error (pinning): the pass ends early, reported in errors
    for (int r = 0; r < R; ++r)
        readers.emplace_back([&] {
          try {
            for (;;) {
                if ((opt.cancel && opt.cancel->load()) || readers_failed.load()) break;
                int k = next++;
                if (k >= N) break;
                if (!infos[k].ok()) continue;
                if (opt.cache) {
                    if (const FrameCache::Entry* e = opt.cache->find(paths[k])) {   // resident: no read
                        { std::lock_guard<std::mutex> lk(stat_m); ++st.from_cache; st.bytes += e->bytes; }
                        q.push({k, e->buf, e->offset, e->bytes, false});
                        continue;
                    }
                }
                uint8_t* buf = opt.cache ? opt.cache->reserve(cap) : nullptr;
                const bool pooled = buf == nullptr;
                if (pooled) buf = pool.acquire();
                TiffRead rd = tiff_read_pixels(paths[k], infos[k], buf, cap);
                if (!rd.ok()) {
                    std::lock_guard<std::mutex> lk(stat_m);
                    st.errors.push_back(paths[k].filename().string() + ": " + rd.error);
                    if (pooled) pool.release(buf); else opt.cache->unreserve(buf, cap);
                    continue;
                }
                Ready ready{k, buf, rd.data_offset_in_buf, rd.bytes, pooled};
                if (!pooled) {   // the cache may hand back an earlier copy of the same path (and free ours)
                    const FrameCache::Entry* e = opt.cache->insert(paths[k], buf, cap, rd.data_offset_in_buf, rd.bytes);
                    ready.buf = e->buf; ready.offset = e->offset; ready.bytes = e->bytes;
                }
                {
                    std::lock_guard<std::mutex> lk(stat_m);
                    st.read_ms_sum += rd.ms;
                    st.unbuffered += rd.unbuffered;
                    st.bytes += rd.bytes;
                    if (!pooled) ++st.to_cache;
                }
                q.push(ready);
            }
          } catch (const std::exception& e) {   // e.g. the ring could not pin a buffer: stop reading, keep the workers draining
            std::lock_guard<std::mutex> lk(stat_m);
            st.errors.push_back(std::string("reader: ") + e.what());
            readers_failed = true;
          }
        });

    // one worker thread per GPU, S streams each
    std::vector<std::thread> workers;
    std::vector<std::string> worker_errors(devs.size());
    std::atomic<int> live_workers{(int)devs.size()};
    for (size_t di = 0; di < devs.size(); ++di)
        workers.emplace_back([&, di] {
            const int S = S_dev[di];
            try {
                CK(cudaSetDevice(devs[di]));
                std::vector<cudaStream_t> streams(S);
                std::vector<cudaEvent_t> uploaded(S);
                std::vector<uint8_t*> dbuf(S);
                std::vector<uint8_t*> pending(S, nullptr);
                for (int j = 0; j < S; ++j) {
                    CK(cudaStreamCreateWithFlags(&streams[j], cudaStreamNonBlocking));
                    CK(cudaEventCreateWithFlags(&uploaded[j], cudaEventDisableTiming));
                    CK(cudaMalloc(&dbuf[j], max_px));
                }
                int n = 0;
                const int cap = opt.max_frames_by_device.size() == devs.size() ? opt.max_frames_by_device[di] : 0;
                Ready r;
                while ((cap <= 0 || n < cap) && q.pop(r)) {   // a capped card leaves the rest to the others
                    int j = n++ % S;
                    if (n > S) {                    // the upload from S frames ago must be done (S frames in flight)
                        CK(cudaEventSynchronize(uploaded[j]));
                        if (pending[j]) pool.release(pending[j]);
                        pending[j] = nullptr;
                    }
                    // the stream orders this copy after the previous frame's kernels on dbuf[j]
                    CK(cudaMemcpyAsync(dbuf[j], r.buf + r.offset, r.bytes, cudaMemcpyHostToDevice, streams[j]));
                    CK(cudaEventRecord(uploaded[j], streams[j]));
                    pending[j] = r.pooled ? r.buf : nullptr;   // a cache buffer stays where it is
                    DeviceFrame f{r.k, devs[di], j, streams[j], dbuf[j], &infos[r.k]};
                    on_frame(f);
                    CK(cudaGetLastError());
                    ++st.frames_per_device[di];
                }
                for (int j = 0; j < S; ++j) {
                    CK(cudaStreamSynchronize(streams[j]));
                    if (pending[j]) pool.release(pending[j]);
                    CK(cudaFree(dbuf[j]));
                    CK(cudaEventDestroy(uploaded[j]));
                    CK(cudaStreamDestroy(streams[j]));
                }
            } catch (const std::exception& e) {
                worker_errors[di] = e.what();
                // other GPUs take over this one's frames; only the last worker standing
                // has to drain the queue so readers never block on a full pool
                if (--live_workers == 0) {
                    Ready r;
                    while (q.pop(r)) if (r.pooled) pool.release(r.buf);
                }
                return;
            }
            if (--live_workers == 0) {   // the last worker (a capped one, the others gone) drains what is left
                Ready r;
                while (q.pop(r)) if (r.pooled) pool.release(r.buf);
            }
        });

    for (auto& t : readers) t.join();
    q.finish();
    for (auto& t : workers) t.join();
    st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    for (size_t di = 0; di < devs.size(); ++di) {
        st.frames += st.frames_per_device[di];
        if (!worker_errors[di].empty()) st.errors.push_back("device " + std::to_string(devs[di]) + ": " + worker_errors[di]);
    }
    return st;
}

}  // namespace stackcore
