// Runs a stacking job on a background thread and reports through signals.
#pragma once

#include <QImage>
#include <QObject>
#include <QThread>

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include "liveplayer.h"
#include "stackcore/frame_cache.h"
#include "stackcore/pipeline.h"

// A live-view image as the engine hands it over: float RGB, interleaved, [0, 1] (FloatImage, liveplayer.h). The
// 8-bit conversion happens off the engine's threads, and the buffers come from a small pool handed back after
// converting (a fresh 10 MB vector per frame cost ~3 ms of page faults on the GPU's worker thread, FINDINGS 30).

struct JobResultData {
    stackcore::StackOptions::Result result;
    stackcore::AlignOut align;
    QString reportJson;
    QString outDir;
    int rc = 0;
    double seconds = 0;   // wall time of the whole job (alignment + fusion + writing)
    bool sweep = false;   // a sweep (outDir holds sweep.json and the variants), not a render: `result` is empty
    int variants = 0;
    QString settingsTag;  // the render's finish time, settings and output FOV for the Save dialog's file names
                          // ("2026-11-08_14-15-27_bicubic_s2_c32_b0_1.49mm"), set by the main window
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
    void cancel() { cancel_.store(true); }
    void releaseCache();                                 // frees the resident frames (when the cache is turned off)
    bool resident(const QStringList& files);             // true when every file is pinned in the cache already
    // the folder's frames become the cache's set and the budget is taken from the machine's free memory, so the
    // thumbnail loader can fill it (0.6.24: the frames are read once, while the thumbnails load)
    void prepareCache(const QStringList& files);
    stackcore::FrameCache* cache() { return &cache_; }
    // Building previews are dropped while two are still waiting for the GUI thread, so a slow display never
    // queues up images and the engine never waits for the screen. The GUI returns each buffer once converted.
    // Frame views go to the LivePlayer, which paces and orders them on its own thread (0.6.32).
    void previewShown(FloatImage img) { --previewsInFlight_; recycle(std::move(img)); }
    LivePlayer* player() { return player_; }
    // the live view's window (0.6.33): asked by the engine's threads for each view, when the live view is "Full"
    void setViewWindowProvider(std::function<stackcore::ViewWindow(int, int)> f) { viewWindow_ = std::move(f); }

signals:
    void progress(QString stage, double fraction);
    void preview(QString name, FloatImage rgb, int w, int h);
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
    std::atomic<int> previewsInFlight_{0};
    LivePlayer* player_ = nullptr;
    std::function<stackcore::ViewWindow(int, int)> viewWindow_;
    std::mutex poolMutex_;
    std::vector<FloatImage> pool_;
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

signals:
    void thumbnail(int index, QImage image);

private:
    QThread thread_;
    std::atomic<bool> stop_{false};
    std::atomic<int> generation_{0};
    stackcore::FrameCache* cache_ = nullptr;
};
