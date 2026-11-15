// Runs a stacking job on a background thread and reports through signals.
#pragma once

#include <QImage>
#include <QObject>
#include <QThread>

#include <atomic>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include "liveplayer.h"
#include "stackcore/frame_cache.h"
#include "stackcore/pipeline.h"

// An image file that is not a frame (a comparison image: PNG, a grey TIFF, ...) as 8-bit RGB, decoded by Windows
// (stackcore::decode_rgb8). A null image, with the reason in err, when Windows cannot read it.
QImage readImageViaWindows(const QString& path, QString* err = nullptr);

// A building map or an output's preview as the engine hands it over: float RGB, interleaved, [0, 1] (FloatImage,
// liveplayer.h). The 8-bit conversion happens off the engine's threads, and the buffers come from a small pool handed
// back after converting (a fresh 10 MB vector per image cost ~3 ms of page faults on the GPU's worker thread,
// FINDINGS 30). The frame views arrive as 8-bit pixels and are copied into pooled images (ViewPool).

struct JobResultData;

// Regions of a finished render to render again (the two-up viewer's Refine; stackcore::run_region): the tiles go in
// with their rectangles and variants and come back with a B per variant.
struct RegionJobData {
    std::shared_ptr<JobResultData> render;      // the render the regions are parts of
    std::vector<stackcore::RegionTile> tiles;
};

// Finished images of one size stacked again as if they were frames (the two-up viewer's Make composite): no
// alignment, they lie on each other already. The outputs come back in JobResultData::result; nothing is written.
struct CompositeJobData {
    std::vector<std::filesystem::path> paths;   // the images, each a file the engine reads
    QString label;                              // what they are: "2B1C"
    double fovMm = 0;                           // the width the composite stands for, mm (its reference image's); 0 = not known
    QStringList outputs;                        // the methods to stack them with ("B", "C", "CBn"); empty = the Render panel's
    QString sources;                            // their names in the list, for the note
    QString dir;                                // where the composite's TIFFs go
    QString temp;                               // a folder of copies made for the job (images that had no such file), removed after it
};

struct JobResultData {
    stackcore::StackOptions::Result result;
    stackcore::AlignOut align;
    // the fusion's options as the render ran them (the frames, their transforms and row corrections, the dust map,
    // the rectangle): what a region of it is rendered again with. Null for a sweep, a region job and a failed render
    std::shared_ptr<stackcore::StackOptions> stack;
    std::shared_ptr<RegionJobData> regions;     // set for a region job: its tiles; `result` is empty
    std::shared_ptr<CompositeJobData> composite;   // set for a composite: `result` holds its outputs
    QString reportJson;
    QString outDir;
    int rc = 0;           // 0 done, 1 failed, 2 a bad option, 3 cancelled, 4 the GPU memory ran out, 6 rendered but not written
    stackcore::FailInfo fail;   // the cause the engine noted (pipeline.h), for the message
    QString stage;        // the stage the job was in when it ended
    QString error;        // the first error line of the job's log
    double seconds = 0;   // wall time of the whole job (alignment + fusion + writing)
    bool sweep = false;   // a sweep (outDir holds sweep.json and the variants), not a render: `result` is empty
    int variants = 0;
    QString settingsTag;  // the render's finish time, settings and output FOV for the Save dialog's file names
                          // ("2026-11-08_14-15-27_bicubic_s2_c32_b0_1.49mm"), set by the main window
};

// The frame views' pixel buffers, used again and again (a fresh buffer of several MB per view costs its page faults
// every time, FINDINGS 30). An image made here hands its buffer back when its last copy goes, on whichever thread
// that is and however late (the viewer holds the last one), so the images keep the pool alive.
class ViewPool : public std::enable_shared_from_this<ViewPool> {
public:
    QImage image(const uint32_t* px, int w, int h);   // a copy of w x h pixels (0xffRRGGBB) as an RGB32 image
private:
    struct Buf { std::unique_ptr<uint32_t[]> p; size_t cap = 0; };
    struct Back { std::shared_ptr<ViewPool> pool; Buf buf; };
    std::mutex m_;
    std::vector<Buf> free_;
};

class Worker : public QObject {
    Q_OBJECT
public:
    explicit Worker(QObject* parent = nullptr);
    ~Worker() override;
    bool running() const { return running_; }
    // live views at 1/liveDown of the output; quiet = no previews or frame views at all (the benchmark)
    void start(const stackcore::JobOptions& jo, int liveDown = 6, bool quiet = false);
    // a sweep: align as a render would, then a B per variant into outDir (run_sweep_job); finished() carries sweep = true
    void startSweep(const stackcore::JobOptions& jo, const std::vector<stackcore::SweepVariant>& variants, const QString& outDir, int liveDown = 6);
    // regions of job->render rendered again (run_region), without live views; finished() carries `regions` = job
    void startRegions(std::shared_ptr<RegionJobData> job);
    // job->paths stacked with jo's fusion settings and outputs (run_stack, the frames as they lie); finished() carries `composite` = job
    void startComposite(const stackcore::JobOptions& jo, std::shared_ptr<CompositeJobData> job);
    void cancel() { cancel_.store(true); }
    void releaseCache();                                 // frees the resident frames (when the cache is turned off)
    bool resident(const QStringList& files);             // true when every file is pinned in the cache already
    // the folder's frames become the cache's set and the budget is taken from the machine's free memory, so the
    // thumbnail loader can fill it (0.6.24: the frames are read once, while the thumbnails load)
    void prepareCache(const QStringList& files);
    stackcore::FrameCache* cache() { return &cache_; }
    // Frame views and building previews go to the LivePlayer, which paces and orders them and converts them on its
    // own thread (0.6.32 / 0.6.34); the engine never waits for the screen
    LivePlayer* player() { return player_; }
    // the live view's window (0.6.33): asked by the engine's threads for each view, when the live view is "Full"
    void setViewWindowProvider(std::function<stackcore::ViewWindow(int, int)> f) { viewWindow_ = std::move(f); }
    void setViewMaxPx(size_t px) { viewMaxPx_ = px; }   // the largest such view, set before a run

signals:
    void progress(QString stage, double fraction);
    void logLine(QString line);
    void finished(std::shared_ptr<JobResultData> data);

private:
    QThread thread_;
    // the common part of start and startSweep: the hooks, the thread, the finished signal; `run` does the job
    void launch(std::shared_ptr<stackcore::JobOptions> jo, std::shared_ptr<JobResultData> data, int liveDown, bool quiet,
                std::function<int(stackcore::JobOptions&, JobResultData&, stackcore::Hooks&)> run);
    FloatImage take(const float* rgb, size_t n);   // a pooled buffer holding a copy of rgb[0..n)
    void recycle(FloatImage img);
    std::atomic<bool> cancel_{false};
    LivePlayer* player_ = nullptr;
    std::function<stackcore::ViewWindow(int, int)> viewWindow_;
    size_t viewMaxPx_ = 4000000;
    std::mutex poolMutex_;
    std::vector<FloatImage> pool_;
    std::shared_ptr<ViewPool> viewPool_ = std::make_shared<ViewPool>();
    bool running_ = false;
    stackcore::FrameCache cache_;   // the stack stays pinned in host memory between runs (FINDINGS 26)
};

// Reads the frames one by one on a thread and emits a thumbnail of each. With a frame cache set, each frame is
// read into a pinned buffer the cache keeps (within its budget), so the stack is resident by the time Render is
// pressed and the render starts at the alignment; a frame already resident is sampled from the cache, not read.
class ThumbLoader : public QObject {
    Q_OBJECT
public:
    explicit ThumbLoader(QObject* parent = nullptr);
    ~ThumbLoader() override;
    void setCache(stackcore::FrameCache* cache) { cache_ = cache; }   // nullptr: plain reads (the default)
    // emits thumbnail(i, image) per file, or thumbnail(indices[i], image) when indices are given (a partial list)
    void start(const QStringList& files, int width, const QList<int>& indices = {});
    void stop();
    void shutdown();   // stop and wait for the threads: before the frame cache they fill goes (the window's end)

signals:
    void thumbnail(int index, QImage image);   // a null image: the file could not be read

private:
    QThread thread_;
    std::atomic<bool> stop_{false};
    std::atomic<int> generation_{0};
    stackcore::FrameCache* cache_ = nullptr;
};
