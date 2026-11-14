#include "worker.h"

#include "diaglog.h"

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

QImage readImageViaWindows(const QString& path, QString* err) {
    std::vector<uint8_t> rgb;
    int w = 0, h = 0;
    std::string why;
    if (!stackcore::decode_rgb8(path.toStdWString(), rgb, w, h, &why)) { if (err) *err = QString::fromStdString(why); return {}; }
    QImage img(w, h, QImage::Format_RGB32);
    if (img.isNull()) { if (err) *err = "not enough memory for the image"; return {}; }
    for (int y = 0; y < h; ++y) {
        QRgb* out = reinterpret_cast<QRgb*>(img.scanLine(y));
        const uint8_t* p = rgb.data() + (size_t)y * w * 3;
        for (int x = 0; x < w; ++x, p += 3) out[x] = qRgb(p[0], p[1], p[2]);
    }
    return img;
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
        if (rc == 0) d.reportJson = QString::fromStdString(report_json(j.align, d.align));   // a failed job has no alignment to report
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
        // every line goes to the session's file on this thread, so it is on the disk whatever happens next; the first
        // line that reports an error is kept for the failure message
        auto firstError = std::make_shared<QString>();
        auto lastStage = std::make_shared<QString>();
        auto guard = std::make_shared<std::mutex>();
        set_log([this, firstError, guard](const std::string& s) {
            const QString line = QString::fromStdString(s);
            DiagLog::write(line);
            const QString t = line.trimmed();
            if (t.startsWith("error:") || t.startsWith("failed:") || t.startsWith("CUDA error") || t.startsWith("cannot ") || t.contains("ran out")) {
                std::lock_guard<std::mutex> lk(*guard);
                if (firstError->isEmpty()) *firstError = t;
            }
            emit logLine(line);
        });
        stackcore::take_fail();   // nothing left over from an earlier job
        Hooks hooks;
        hooks.cancel = &cancel_;
        // the building maps go to the right pane, which never shows more than ~1/4 of the output: at Full (1) they were
        // 31 Mpx floats converted on the GUI thread several times a second, which made the view choppy (FINDINGS 47)
        hooks.preview_down = std::max(4, liveDown);
        hooks.frame_down = std::max(1, liveDown);
        // the building maps grow on a time basis (every 50 ms at most), not every n-th frame: no steps (0.6.32)
        hooks.preview_every = 1;
        hooks.preview_min_ms = 50;
        hooks.frame_min_ms = 33;
        hooks.progress = [this, lastStage, guard](const std::string& stage, double f) {
            {
                const QString st = QString::fromStdString(stage);
                DiagLog::runStage(st);
                std::lock_guard<std::mutex> lk(*guard);
                *lastStage = st.section(',', 0, 0);
            }
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
            if (liveDown == 1 && viewWindow_) { hooks.view_window = viewWindow_; hooks.view_max_px = viewMaxPx_; }
            if (noView) hooks.frame = nullptr;
            if (noPreview) hooks.preview = nullptr;
            if (getenv("XLAMINA_NOOPHOOKS")) {   // the GPU-side preview work without the host side
                hooks.frame = [](int, const float*, int, int, const stackcore::ViewWindow&) {};
                hooks.preview = [](const std::string&, const float*, int, int) {};
            }
        }
        auto t0 = std::chrono::steady_clock::now();
        auto say = [&](const QString& line) {
            DiagLog::write(line);
            { std::lock_guard<std::mutex> lk(*guard); if (firstError->isEmpty()) *firstError = line; }
            emit logLine(line);
        };
        try {
            data->rc = run(*jo, *data, hooks);
        } catch (const std::bad_alloc&) {   // the PC's own memory (the alignment's luma stack, a whole-frame buffer)
            stackcore::note_fail(stackcore::Fail::HostMemory, lastStage->toStdString());
            say("failed: the PC's memory (RAM) ran out");
            data->rc = 1;
        } catch (const std::exception& e) {
            say(QString("failed: %1").arg(e.what()));
            data->rc = 1;
        } catch (...) {
            say("failed: an unexpected error");
            data->rc = 1;
        }
        data->seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        data->fail = stackcore::take_fail();
        {
            std::lock_guard<std::mutex> lk(*guard);
            data->stage = *lastStage;
            data->error = *firstError;
        }
        // an alignment that found no pair for a frame says so in its exception's text
        if (data->rc != 0 && data->fail.kind == stackcore::Fail::None && data->error.contains("usable pair"))
            data->fail = {stackcore::Fail::Alignment, data->error.section("failed: ", 1).toStdString()};
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

// the thread ends when its job has (the job's own threads are joined by it); start() starts it again
void ThumbLoader::shutdown() {
    stop();
    thread_.quit();
    thread_.wait();
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
        auto one = [&](int i, std::vector<uint8_t>& buf) -> bool {
            const std::filesystem::path path(files[i].toStdWString());
            TiffInfo info = tiff_probe(path.native());
            if (!info.ok()) return false;
            if ((info.jpeg || info.raw) && width < (int)info.width / 2) {   // a thumbnail: decoded small by the codec itself (a raw file developed at half size); no cache fill (the render's 16 readers decode in parallel). The full-size view below decodes whole
                std::vector<uint8_t> small;
                int sw = 0, sh = 0;
                if (!fast_thumbnail(path, info, width, small, sw, sh)) return false;
                QImage img(sw, sh, QImage::Format_RGB32);
                for (int y = 0; y < sh; ++y) {
                    QRgb* out = reinterpret_cast<QRgb*>(img.scanLine(y));
                    const uint8_t* p = small.data() + (size_t)y * sw * 3;
                    for (int x = 0; x < sw; ++x, p += 3) out[x] = qRgb(p[0], p[1], p[2]);
                }
                emit thumbnail(indices.isEmpty() ? i : indices.value(i, i), img);
                return true;
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
                            const int s = info.samples >= 3 ? k : 0;   // a grey frame: its one sample three times
                            if (info.bits == 16) { unsigned v = info.big_endian ? (p[2 * s] << 8) | p[2 * s + 1] : (p[2 * s + 1] << 8) | p[2 * s]; c[k] = v >> 8; }
                            else c[k] = p[s];
                        }
                        out[x] = qRgb(c[0], c[1], c[2]);
                    }
                }
            };
            // resident already (the folder was opened before, or a render read it): sample it under the cache's lock
            if (cache && cache->with(path, [&](const stackcore::FrameCache::Entry& e) { sample(e.buf + e.offset); })) {
                emit thumbnail(indices.isEmpty() ? i : indices.value(i, i), img);
                return true;
            }
            const size_t cap = tiff_buffer_capacity(info.pixel_bytes());
            uint8_t* pinned = cache ? cache->reserve(cap) : nullptr;   // nullptr: no cache, or its budget is spent
            if (pinned) {
                TiffRead rd = tiff_read_pixels(path, info, pinned, cap);
                if (!rd.ok()) { cache->unreserve(pinned, cap); return false; }
                sample(pinned + rd.data_offset_in_buf);
                cache->insert(path, pinned, cap, rd.data_offset_in_buf, rd.bytes);   // the stack warms up while the list fills
            } else {
                size_t need = cap + kReadAlign;
                if (buf.size() < need) buf.resize(need);
                uint8_t* base = buf.data() + (kReadAlign - ((uintptr_t)buf.data() % kReadAlign)) % kReadAlign;
                TiffRead rd = tiff_read_pixels(path, info, base, buf.size() - (base - buf.data()));
                if (!rd.ok()) return false;
                sample(base + rd.data_offset_in_buf);
            }
            emit thumbnail(indices.isEmpty() ? i : indices.value(i, i), img);
            return true;
        };
        std::atomic<int> next{0};
        std::vector<std::thread> pool;
        for (int t = 0; t < 6; ++t)
            pool.emplace_back([&] {
                std::vector<uint8_t> buf;
                for (;;) {
                    int i = next++;
                    if (i >= files.size() || stop_.load() || generation_.load() != gen) break;
                    bool ok = false;
                    try { ok = one(i, buf); } catch (const std::exception&) {}   // a file whose header asks for more memory than there is: no thumbnail, not a crash
                    // a file that could not be read says so with a null image (the views that wait for it stopped waiting never)
                    if (!ok && !stop_.load() && generation_.load() == gen) emit thumbnail(indices.isEmpty() ? i : indices.value(i, i), QImage());
                }
            });
        for (auto& t : pool) t.join();
        job->deleteLater();
    }, Qt::QueuedConnection);
}
