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

    std::vector<int> devs = opt.devices;
    if (devs.empty()) {
        int n = 0;
        CK(cudaGetDeviceCount(&n));
        for (int i = 0; i < n; ++i) devs.push_back(i);
    }
    const int S = std::max(1, opt.streams_per_gpu), R = std::max(1, opt.readers);
    const int P = opt.pool > 0 ? opt.pool : R + 2 * S * (int)devs.size();
    const size_t cap = tiff_buffer_capacity(max_px);
    PinnedPool pool(cap, P);
    st.frames_per_device.assign(devs.size(), 0);

    Queue q;
    std::atomic<int> next{0};
    std::mutex stat_m;
    auto t0 = std::chrono::steady_clock::now();

    // readers
    std::vector<std::thread> readers;
    for (int r = 0; r < R; ++r)
        readers.emplace_back([&] {
            for (;;) {
                int k = next++;
                if (k >= N) break;
                if (!infos[k].ok()) continue;
                uint8_t* buf = pool.acquire();
                TiffRead rd = tiff_read_pixels(paths[k], infos[k], buf, cap);
                if (!rd.ok()) {
                    std::lock_guard<std::mutex> lk(stat_m);
                    st.errors.push_back(paths[k].filename().string() + ": " + rd.error);
                    pool.release(buf);
                    continue;
                }
                {
                    std::lock_guard<std::mutex> lk(stat_m);
                    st.read_ms_sum += rd.ms;
                    st.unbuffered += rd.unbuffered;
                    st.bytes += rd.bytes;
                }
                q.push({k, buf, rd.data_offset_in_buf, rd.bytes});
            }
        });

    // one worker thread per GPU, S streams each
    std::vector<std::thread> workers;
    std::vector<std::string> worker_errors(devs.size());
    for (size_t di = 0; di < devs.size(); ++di)
        workers.emplace_back([&, di] {
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
                Ready r;
                while (q.pop(r)) {
                    int j = n++ % S;
                    if (pending[j]) {               // the upload from S frames ago must be done
                        CK(cudaEventSynchronize(uploaded[j]));
                        pool.release(pending[j]);
                        pending[j] = nullptr;
                    }
                    // the stream orders this copy after the previous frame's kernels on dbuf[j]
                    CK(cudaMemcpyAsync(dbuf[j], r.buf + r.offset, r.bytes, cudaMemcpyHostToDevice, streams[j]));
                    CK(cudaEventRecord(uploaded[j], streams[j]));
                    pending[j] = r.buf;
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
                // keep draining so readers never block forever on a full pool
                Ready r;
                while (q.pop(r)) pool.release(r.buf);
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
