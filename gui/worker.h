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

#include "stackcore/frame_cache.h"
#include "stackcore/pipeline.h"

// A live-view image as the engine hands it over: float RGB, interleaved, [0, 1]. The 8-bit conversion
// happens on the GUI thread, and the buffers come from a small pool the GUI hands back after converting
// (a fresh 10 MB vector per frame cost ~3 ms of page faults on the GPU's worker thread, FINDINGS 30).
using FloatImage = std::shared_ptr<std::vector<float>>;

struct JobResultData {
    stackcore::StackOptions::Result result;
    stackcore::AlignOut align;
    QString reportJson;
    QString outDir;
    int rc = 0;
    double seconds = 0;   // wall time of the whole job (alignment + fusion + writing)
    bool sweep = false;   // a sweep (outDir holds sweep.json and the variants), not a render: `result` is empty
    int variants = 0;
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
    // Live views are dropped while two are still waiting for the GUI thread, so a slow display never
    // queues up frames (at Full, 365 MB each) and the engine never waits for the screen. The GUI returns
    // each buffer once it has converted it.
    void viewShown(FloatImage img) { --viewsInFlight_; recycle(std::move(img)); }
    void previewShown(FloatImage img) { --previewsInFlight_; recycle(std::move(img)); }

signals:
    void progress(QString stage, double fraction);
    void preview(QString name, FloatImage rgb, int w, int h);
    void frame(int k, FloatImage rgb, int w, int h);
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
    std::atomic<int> viewsInFlight_{0}, previewsInFlight_{0};
    std::mutex poolMutex_;
    std::vector<FloatImage> pool_;
    bool running_ = false;
    stackcore::FrameCache cache_;   // the stack stays pinned in host memory between runs (FINDINGS 26)
};

// Reads the frames one by one on a thread and emits a thumbnail of each.
class ThumbLoader : public QObject {
    Q_OBJECT
public:
    explicit ThumbLoader(QObject* parent = nullptr);
    ~ThumbLoader() override;
    // emits thumbnail(i, image) per file, or thumbnail(indices[i], image) when indices are given (a partial list)
    void start(const QStringList& files, int width, const QList<int>& indices = {});
    void stop();

signals:
    void thumbnail(int index, QImage image);

private:
    QThread thread_;
    std::atomic<bool> stop_{false};
    std::atomic<int> generation_{0};
};
