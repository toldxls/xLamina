// Frame streaming: TIFF files -> reader thread pool -> pinned ring -> per-GPU
// upload streams -> a per-frame callback that enqueues the consumer's kernels
// on that stream. GPUs pull frames from one queue, so a faster card simply
// takes more of them (DESIGN.md "Multi-GPU scheduler", split mode).
#pragma once

#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <vector>

#include "stackcore/frame_cache.h"
#include "stackcore/tiff.h"

namespace stackcore {

struct DeviceFrame {
    int k;                     // frame index in the input list
    int dev;                   // CUDA device the frame landed on
    int slot;                  // stream slot on that device (0..streams-1)
    cudaStream_t stream;       // the callback enqueues its work here
    const uint8_t* d_bytes;    // device copy of the file's pixel data (raw TIFF bytes)
    const TiffInfo* info;      // geometry, bit depth, byte order
    const uint8_t* d_pixels() const { return d_bytes; }
};

// Called on the device's worker thread, with cudaSetDevice(dev) in effect.
// Must only enqueue asynchronous work on frame.stream (no sync inside).
using FrameFn = std::function<void(const DeviceFrame&)>;

struct StreamerOptions {
    std::vector<int> devices;      // empty = all
    int streams_per_gpu = 3;
    std::vector<int> streams_by_device;   // optional, parallel to `devices`: per-GPU stream count (memory)
    std::vector<int> max_frames_by_device;   // optional, parallel to `devices`: a GPU stops taking frames after this many (0 = no cap)
    int readers = 16;
    int pool = 0;                  // pinned buffers; 0 = readers + 2 * streams * gpus
    std::atomic<bool>* cancel = nullptr;   // when set, readers stop taking frames and the pass ends early
    FrameCache* cache = nullptr;           // resident frames: uploaded from here; frames read this pass join it while its budget lasts
};

struct StreamerStats {
    double seconds = 0;            // wall time, first read to last kernel done
    uint64_t bytes = 0;            // pixel bytes read
    int frames = 0;
    std::vector<int> frames_per_device;
    double read_ms_sum = 0;        // summed over frames (reader-thread time)
    int unbuffered = 0;            // frames read with FILE_FLAG_NO_BUFFERING
    int from_cache = 0;            // frames uploaded from resident memory (no read)
    int to_cache = 0;              // frames read into the cache this pass
    std::vector<std::string> errors;
};

// Probes every file first (header only); frames whose probe fails are
// reported in stats.errors and skipped. Returns when every frame's callback
// work has completed on its stream.
StreamerStats stream_frames(const std::vector<std::filesystem::path>& paths, const StreamerOptions& opt,
                            const FrameFn& on_frame);

// The ring of page-locked read buffers is kept from one pass to the next and from job to job: with two GPUs
// loaded the driver counts everything a process has ever pinned against a ceiling, freed or not (FINDINGS 55),
// so nothing calls this in a render. It frees the ring for a caller that is done with the GPUs.
void release_stream_ring();

}  // namespace stackcore
