#include "worker.h"

#include <QMetaObject>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <cstdlib>
#include <cstring>

#include "imageview.h"
#include "stackcore/tiff.h"

using namespace stackcore;

Worker::Worker(QObject* parent) : QObject(parent) {}

Worker::~Worker() {
    cancel();
    thread_.quit();
    thread_.wait();
}

void Worker::releaseCache() {
    if (!running_) cache_.clear();
}

bool Worker::resident(const QStringList& files) {
    if (running_ || files.isEmpty()) return false;
    for (const QString& f : files) if (!cache_.find(f.toStdWString())) return false;
    return true;
}

void Worker::prepareCache(const QStringList& files) {
    if (running_) return;
    std::vector<std::filesystem::path> paths;
    for (const QString& f : files) paths.emplace_back(f.toStdWString());
    cache_.prepare(paths);   // keeps what is resident and unchanged, frees the rest, sets the budget
}

FloatImage Worker::take(const float* rgb, size_t n) {
    FloatImage img;
    {
        std::lock_guard<std::mutex> lk(poolMutex_);
        if (!pool_.empty()) { img = std::move(pool_.back()); pool_.pop_back(); }
    }
    if (!img) img = std::make_shared<std::vector<float>>();
    img->resize(n);   // no reallocation once the pool has warmed up
    memcpy(img->data(), rgb, n * sizeof(float));
    return img;
}

void Worker::recycle(FloatImage img) {
    if (!img) return;
    std::lock_guard<std::mutex> lk(poolMutex_);
    if (pool_.size() < 8) pool_.push_back(std::move(img));
}

void Worker::start(const JobOptions& jo_in, int liveDown, bool quiet) {
    if (running_) return;
    running_ = true;
    cancel_.store(false);
    auto jo = std::make_shared<JobOptions>(jo_in);
    auto data = std::make_shared<JobResultData>();
    jo->stack.result = &data->result;
    jo->cache = jo->stack.cache_frames ? &cache_ : nullptr;   // with the cache off the resident frames are left alone, just not used
    launch(jo, data, liveDown, quiet, [](JobOptions& j, JobResultData& d, Hooks& hooks) {
        int rc = run_job(j, d.align, hooks);
        d.reportJson = QString::fromStdString(report_json(j.align, d.align));
        d.outDir = QString::fromStdString(j.stack.out.empty() ? j.out : j.stack.out);
        return rc;
    });
}

void Worker::startSweep(const JobOptions& jo_in, const std::vector<SweepVariant>& variants, const QString& outDir, int liveDown) {
    if (running_) return;
    running_ = true;
    cancel_.store(false);
    auto jo = std::make_shared<JobOptions>(jo_in);
    auto data = std::make_shared<JobResultData>();
    data->sweep = true;
    data->variants = (int)variants.size();
    data->outDir = outDir;
    jo->stack.result = nullptr;
    jo->cache = jo->stack.cache_frames ? &cache_ : nullptr;
    std::string out = outDir.toLocal8Bit().toStdString();
    launch(jo, data, liveDown, false, [variants, out](JobOptions& j, JobResultData& d, Hooks& hooks) {
        return run_sweep_job(j, variants, out, d.align, hooks);
    });
}

void Worker::launch(std::shared_ptr<JobOptions> jo, std::shared_ptr<JobResultData> data, int liveDown, bool quiet,
                    std::function<int(JobOptions&, JobResultData&, Hooks&)> run) {
    QObject* worker = new QObject();
    worker->moveToThread(&thread_);
    thread_.start();
    QMetaObject::invokeMethod(worker, [this, jo, data, worker, liveDown, quiet, run] {
        set_log([this](const std::string& s) { emit logLine(QString::fromStdString(s)); });
        Hooks hooks;
        hooks.cancel = &cancel_;
        hooks.preview_down = std::max(1, liveDown);
        hooks.frame_down = std::max(1, liveDown);
        hooks.preview_every = liveDown >= 6 ? 3 : liveDown >= 3 ? 6 : 12;   // dearer previews, fewer of them
        hooks.progress = [this](const std::string& stage, double f) { emit progress(QString::fromStdString(stage), f); };
        viewsInFlight_ = previewsInFlight_ = 0;
        // timing experiments: XLAMINA_NOVIEW=1 drops the per-frame view, XLAMINA_NOPREVIEW=1 the building maps
        const bool noView = getenv("XLAMINA_NOVIEW"), noPreview = getenv("XLAMINA_NOPREVIEW");
        if (!quiet) {   // the floats go to the GUI thread in pooled buffers; it converts them (see FloatImage)
            hooks.preview = [this](const std::string& name, const float* rgb, int w, int h) {
                bool building = name.find("(building)") != std::string::npos;
                if (building && previewsInFlight_.load() >= 2) return;   // the screen has not caught up: skip this one
                ++previewsInFlight_;   // final previews are never skipped, but they are counted the same way
                emit preview(QString::fromStdString(name), take(rgb, (size_t)w * h * 3), w, h);
            };
            hooks.frame = [this](int k, const float* rgb, int w, int h) {
                if (viewsInFlight_.load() >= 2) return;
                ++viewsInFlight_;
                emit frame(k, take(rgb, (size_t)w * h * 3), w, h);
            };
            if (noView) hooks.frame = nullptr;
            if (noPreview) hooks.preview = nullptr;
            if (getenv("XLAMINA_NOOPHOOKS")) {   // the GPU-side preview work without the host side
                hooks.frame = [](int, const float*, int, int) {};
                hooks.preview = [](const std::string&, const float*, int, int) {};
            }
        }
        auto t0 = std::chrono::steady_clock::now();
        try {
            data->rc = run(*jo, *data, hooks);
        } catch (const std::exception& e) {
            emit logLine(QString("failed: %1").arg(e.what()));
            data->rc = 1;
        }
        data->seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        set_log(nullptr);
        running_ = false;
        emit finished(data);
        worker->deleteLater();
    }, Qt::QueuedConnection);
}

// ----------------------------------------------------------------------------- thumbnails

ThumbLoader::ThumbLoader(QObject* parent) : QObject(parent) {}

ThumbLoader::~ThumbLoader() {
    stop();
    thread_.quit();
    thread_.wait();
}

void ThumbLoader::stop() {
    stop_.store(true);
    ++generation_;
}

void ThumbLoader::start(const QStringList& files, int width, const QList<int>& indices) {
    stop();
    stop_.store(false);
    int gen = generation_.load();
    QObject* job = new QObject();
    job->moveToThread(&thread_);
    thread_.start();
    stackcore::FrameCache* cache = cache_;
    QMetaObject::invokeMethod(job, [this, files, width, gen, job, indices, cache] {
        std::vector<uint8_t> buf;
        for (int i = 0; i < files.size() && !stop_.load() && generation_.load() == gen; ++i) {
            const std::filesystem::path path(files[i].toStdWString());
            TiffInfo info = tiff_probe(path.native());
            if (!info.ok() || info.samples < 3) continue;
            int down = std::max(1, (int)info.width / width);
            int pw = info.width / down, ph = info.height / down;
            QImage img(pw, ph, QImage::Format_RGB32);
            size_t bpp = info.samples * (info.bits / 8);
            auto sample = [&](const uint8_t* px) {   // every down-th pixel of every down-th row, 8 bits per channel
                for (int y = 0; y < ph; ++y) {
                    QRgb* out = reinterpret_cast<QRgb*>(img.scanLine(y));
                    const uint8_t* srow = px + (size_t)(y * down) * info.width * bpp;
                    for (int x = 0; x < pw; ++x) {
                        const uint8_t* p = srow + (size_t)(x * down) * bpp;
                        int c[3];
                        for (int k = 0; k < 3; ++k) {
                            if (info.bits == 16) { unsigned v = info.big_endian ? (p[2 * k] << 8) | p[2 * k + 1] : (p[2 * k + 1] << 8) | p[2 * k]; c[k] = v >> 8; }
                            else c[k] = p[k];
                        }
                        out[x] = qRgb(c[0], c[1], c[2]);
                    }
                }
            };
            // resident already (the folder was opened before, or a render read it): sample it under the cache's lock
            if (cache && cache->with(path, [&](const stackcore::FrameCache::Entry& e) { sample(e.buf + e.offset); })) {
                emit thumbnail(indices.isEmpty() ? i : indices.value(i, i), img);
                continue;
            }
            const size_t cap = tiff_buffer_capacity(info.pixel_bytes());
            uint8_t* pinned = cache ? cache->reserve(cap) : nullptr;   // nullptr: no cache, or its budget is spent
            if (pinned) {
                TiffRead rd = tiff_read_pixels(path, info, pinned, cap);
                if (!rd.ok()) { cache->unreserve(pinned, cap); continue; }
                sample(pinned + rd.data_offset_in_buf);
                cache->insert(path, pinned, cap, rd.data_offset_in_buf, rd.bytes);   // the stack warms up while the list fills
            } else {
                size_t need = cap + kReadAlign;
                if (buf.size() < need) buf.resize(need);
                uint8_t* base = buf.data() + (kReadAlign - ((uintptr_t)buf.data() % kReadAlign)) % kReadAlign;
                TiffRead rd = tiff_read_pixels(path, info, base, buf.size() - (base - buf.data()));
                if (!rd.ok()) continue;
                sample(base + rd.data_offset_in_buf);
            }
            emit thumbnail(indices.isEmpty() ? i : indices.value(i, i), img);
        }
        job->deleteLater();
    }, Qt::QueuedConnection);
}
