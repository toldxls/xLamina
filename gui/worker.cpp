#include "worker.h"

#include <QMetaObject>

#include <algorithm>

#include "imageview.h"
#include "stackcore/tiff.h"

using namespace stackcore;

Worker::Worker(QObject* parent) : QObject(parent) {}

Worker::~Worker() {
    cancel();
    thread_.quit();
    thread_.wait();
}

void Worker::start(const JobOptions& jo_in, int liveDown) {
    if (running_) return;
    running_ = true;
    cancel_.store(false);
    auto jo = std::make_shared<JobOptions>(jo_in);
    auto data = std::make_shared<JobResultData>();
    jo->stack.result = &data->result;
    jo->cache = &cache_;
    QObject* worker = new QObject();
    worker->moveToThread(&thread_);
    thread_.start();
    QMetaObject::invokeMethod(worker, [this, jo, data, worker, liveDown] {
        set_log([this](const std::string& s) { emit logLine(QString::fromStdString(s)); });
        Hooks hooks;
        hooks.cancel = &cancel_;
        hooks.preview_down = std::max(1, liveDown);
        hooks.frame_down = std::max(1, liveDown);
        hooks.preview_every = liveDown >= 6 ? 3 : liveDown >= 3 ? 6 : 12;   // dearer previews, fewer of them
        hooks.progress = [this](const std::string& stage, double f) { emit progress(QString::fromStdString(stage), f); };
        hooks.preview = [this](const std::string& name, const std::vector<float>& rgb, int w, int h) {
            float gain = 1.f;
            if (name.rfind("focus", 0) == 0) {   // the focus measure has no fixed scale: show it up to its 99th percentile
                std::vector<float> v;
                v.reserve((size_t)w * h / 4 + 1);
                for (size_t i = 0; i < (size_t)w * h; i += 4) v.push_back(rgb[i * 3]);
                if (!v.empty()) {
                    size_t q = v.size() * 99 / 100;
                    std::nth_element(v.begin(), v.begin() + q, v.end());
                    gain = v[q] > 1e-9f ? 1.f / v[q] : 1.f;
                }
            }
            emit preview(QString::fromStdString(name), toQImage(rgb.data(), w, h, gain));
        };
        hooks.frame = [this](int k, const std::vector<float>& rgb, int w, int h) { emit frame(k, toQImage(rgb.data(), w, h)); };
        try {
            data->rc = run_job(*jo, data->align, hooks);
            data->reportJson = QString::fromStdString(report_json(jo->align, data->align));
            data->outDir = QString::fromStdString(jo->stack.out);
        } catch (const std::exception& e) {
            emit logLine(QString("failed: %1").arg(e.what()));
            data->rc = 1;
        }
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

void ThumbLoader::start(const QStringList& files, int width) {
    stop();
    stop_.store(false);
    int gen = generation_.load();
    QObject* job = new QObject();
    job->moveToThread(&thread_);
    thread_.start();
    QMetaObject::invokeMethod(job, [this, files, width, gen, job] {
        std::vector<uint8_t> buf;
        for (int i = 0; i < files.size() && !stop_.load() && generation_.load() == gen; ++i) {
            TiffInfo info = tiff_probe(files[i].toStdWString());
            if (!info.ok() || info.samples < 3) continue;
            size_t need = tiff_buffer_capacity(info.pixel_bytes()) + kReadAlign;
            if (buf.size() < need) buf.resize(need);
            uint8_t* base = buf.data() + (kReadAlign - ((uintptr_t)buf.data() % kReadAlign)) % kReadAlign;
            TiffRead rd = tiff_read_pixels(files[i].toStdWString(), info, base, buf.size() - (base - buf.data()));
            if (!rd.ok()) continue;
            const uint8_t* px = base + rd.data_offset_in_buf;
            int down = std::max(1, (int)info.width / width);
            int pw = info.width / down, ph = info.height / down;
            QImage img(pw, ph, QImage::Format_RGB32);
            size_t bpp = info.samples * (info.bits / 8);
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
            emit thumbnail(i, img);
        }
        job->deleteLater();
    }, Qt::QueuedConnection);
}
