#include "worker.h"

#include <QMetaObject>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <thread>
#include <cstdlib>
#include <cstring>

#include "imageview.h"
#include "stackcore/tiff.h"

using namespace stackcore;

Worker::Worker(QObject* parent) : QObject(parent) {
    player_ = new LivePlayer;   // its own thread; owned here
    player_->setRecycler([this](FloatImage img) { recycle(std::move(img)); });
}

Worker::~Worker() {
    cancel();
    thread_.quit();
    thread_.wait();
    delete player_;
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
        // the building maps go to the right pane, which never shows more than ~1/4 of the output: at Full (1) they were
        // 31 Mpx floats converted on the GUI thread several times a second, the choppiness Travis saw (FINDINGS 47)
        hooks.preview_down = std::max(4, liveDown);
        hooks.frame_down = std::max(1, liveDown);
        // the building maps grow on a time basis (every 50 ms at most), not every n-th frame: no steps (0.6.32)
        hooks.preview_every = 1;
        hooks.preview_min_ms = 50;
        hooks.frame_min_ms = 33;
        hooks.progress = [this](const std::string& stage, double f) {
            // the live view's sweep: each pass through the stack restarts it; pass 2 (the fine focus) runs backwards
            int pass = stage.find("reading") != std::string::npos || stage.find("frames from") != std::string::npos ? 0
                     : stage.find("pass 1") != std::string::npos ? 1 : stage.find("pass 2") != std::string::npos ? 2
                     : stage.find("pass 3") != std::string::npos ? 3 : -1;
            if (pass >= 0) player_->setPass(pass, pass == 2 ? -1 : 1);
            emit progress(QString::fromStdString(stage), f);
        };
        player_->begin();
        // timing experiments: XLAMINA_NOVIEW=1 drops the per-frame view, XLAMINA_NOPREVIEW=1 the building maps
        const bool noView = getenv("XLAMINA_NOVIEW"), noPreview = getenv("XLAMINA_NOPREVIEW");
        if (!quiet) {   // the floats go off the engine's threads in pooled buffers (see FloatImage)
            hooks.preview = [this](const std::string& name, const float* rgb, int w, int h) {
                player_->pushPreview(QString::fromStdString(name), take(rgb, (size_t)w * h * 3), w, h);   // converted off the GUI thread
            };
            hooks.frame = [this](int k, const float* rgb, int w, int h, const stackcore::ViewWindow& win) {
                player_->push(k, take(rgb, (size_t)w * h * 3), w, h, QRect(win.x0, win.y0, win.w, win.h), QSize(win.fw, win.fh));
            };
            // "Full" renders what the view shows (the visible window at the screen's resolution) rather than 31 Mpx
            if (liveDown == 1 && viewWindow_) hooks.view_window = viewWindow_;
            if (noView) hooks.frame = nullptr;
            if (noPreview) hooks.preview = nullptr;
            if (getenv("XLAMINA_NOOPHOOKS")) {   // the GPU-side preview work without the host side
                hooks.frame = [](int, const float*, int, int, const stackcore::ViewWindow&) {};
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
        player_->end();
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
        // one file; run by a few threads at once (a JPEG thumbnail decodes in ~50 ms, a folder of 200 took 10 s on one)
        auto one = [&](int i, std::vector<uint8_t>& buf) {
            const std::filesystem::path path(files[i].toStdWString());
            TiffInfo info = tiff_probe(path.native());
            if (!info.ok() || info.samples < 3) return;
            if (info.jpeg && width < (int)info.width / 2) {   // a thumbnail: decoded small by the codec itself; no cache fill (the render's 16 readers decode in parallel). The full-size view below decodes whole
                std::vector<uint8_t> small;
                int sw = 0, sh = 0;
                if (!jpeg_thumbnail(path, width, small, sw, sh)) return;
                QImage img(sw, sh, QImage::Format_RGB32);
                for (int y = 0; y < sh; ++y) {
                    QRgb* out = reinterpret_cast<QRgb*>(img.scanLine(y));
                    const uint8_t* p = small.data() + (size_t)y * sw * 3;
                    for (int x = 0; x < sw; ++x, p += 3) out[x] = qRgb(p[0], p[1], p[2]);
                }
                emit thumbnail(indices.isEmpty() ? i : indices.value(i, i), img);
                return;
            }
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
                return;
            }
            const size_t cap = tiff_buffer_capacity(info.pixel_bytes());
            uint8_t* pinned = cache ? cache->reserve(cap) : nullptr;   // nullptr: no cache, or its budget is spent
            if (pinned) {
                TiffRead rd = tiff_read_pixels(path, info, pinned, cap);
                if (!rd.ok()) { cache->unreserve(pinned, cap); return; }
                sample(pinned + rd.data_offset_in_buf);
                cache->insert(path, pinned, cap, rd.data_offset_in_buf, rd.bytes);   // the stack warms up while the list fills
            } else {
                size_t need = cap + kReadAlign;
                if (buf.size() < need) buf.resize(need);
                uint8_t* base = buf.data() + (kReadAlign - ((uintptr_t)buf.data() % kReadAlign)) % kReadAlign;
                TiffRead rd = tiff_read_pixels(path, info, base, buf.size() - (base - buf.data()));
                if (!rd.ok()) return;
                sample(base + rd.data_offset_in_buf);
            }
            emit thumbnail(indices.isEmpty() ? i : indices.value(i, i), img);
        };
        std::atomic<int> next{0};
        std::vector<std::thread> pool;
        for (int t = 0; t < 6; ++t)
            pool.emplace_back([&] {
                std::vector<uint8_t> buf;
                for (;;) {
                    int i = next++;
                    if (i >= files.size() || stop_.load() || generation_.load() != gen) break;
                    one(i, buf);
                }
            });
        for (auto& t : pool) t.join();
        job->deleteLater();
    }, Qt::QueuedConnection);
}
