#include "liveplayer.h"

#include <QMetaObject>
#include <QTimer>

#include <algorithm>
#include <chrono>

#include "imageview.h"   // toQImage

namespace {
double nowMs() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
}

LivePlayer::LivePlayer(QObject* parent) : QObject(parent) {
    moveToThread(&thread_);
    QTimer* t = new QTimer;
    t->moveToThread(&thread_);
    t->setInterval(4);   // the clock the sweep is timed on; a tick with nothing due costs nothing
    t->setTimerType(Qt::PreciseTimer);
    connect(t, &QTimer::timeout, this, &LivePlayer::tick);
    connect(&thread_, &QThread::started, t, qOverload<>(&QTimer::start));
    connect(&thread_, &QThread::finished, t, &QObject::deleteLater);
    thread_.start();
    previewThread_ = std::thread([this] { previewLoop(); });
}

LivePlayer::~LivePlayer() {
    {
        std::lock_guard<std::mutex> lk(m_);
        quit_ = true;
    }
    previewCv_.notify_all();
    if (previewThread_.joinable()) previewThread_.join();
    thread_.quit();
    thread_.wait();
    end();
}

void LivePlayer::setColorSpace(const QColorSpace& cs) {
    std::lock_guard<std::mutex> lk(m_);
    cs_ = cs;
}

QString LivePlayer::restart() {
    QString stats;
    if (pushed_ > 0)
        stats = QString("live view: pass %1: %2 views arrived, %3 shown, %4 too late, %5 skipped, %6 left; waiting target %7; conversion %8 ms a view")
                    .arg(pass_.load()).arg(pushed_).arg(shown_).arg(late_).arg(skipped_).arg(ring_.size()).arg(target_).arg(shown_ ? convertMs_ / shown_ : 0.0, 0, 'f', 1);
    pushed_ = late_ = skipped_ = shown_ = 0;
    convertMs_ = 0;
    ring_.clear();
    arrivals_.clear();
    cursor_ = -1;
    playing_ = false;
    return stats;
}

void LivePlayer::push(int k, QImage img, const QRect& win, const QSize& full) {
    const double now = nowMs();
    std::lock_guard<std::mutex> lk(m_);
    const int dir = dir_.load();
    ++pushed_;
    arrivals_.push_back(now);
    while (arrivals_.size() > kArrivals) arrivals_.pop_front();
    if (cursor_ >= 0 && (k - cursor_) * dir <= 0) {   // behind what is on screen: the sweep never steps back
        ++late_;
        target_ = std::min(target_ + 1, kMaxTarget);   // (the views are more out of order than was allowed for)
        return;
    }
    if (ring_.empty() && !playing_) firstArrival_ = now;
    ring_[k] = Item{std::move(img), win, full};   // (a repeated frame: the newer copy stays)
    // the screen is not keeping up: the views next in line are skipped, so what is shown stays near the engine
    while (ring_.size() > kRing) { ring_.erase(dir > 0 ? ring_.begin() : std::prev(ring_.end())); ++skipped_; }
}

void LivePlayer::pushPreview(const QString& name, FloatImage rgb, int w, int h) {
    FloatImage drop;
    {
        std::lock_guard<std::mutex> lk(m_);
        auto it = previews_.find(name);
        if (it != previews_.end()) drop = std::move(it->second.rgb);   // the screen had not caught up: the newer one replaces it
        previews_[name] = Preview{std::move(rgb), w, h, ++seq_};
    }
    previewCv_.notify_one();
    if (drop && recycle_) recycle_(std::move(drop));
}

void LivePlayer::setPass(int pass, int dir) {
    dir_.store(dir);
    if (pass_.load() != pass) {
        QString stats;
        {
            std::lock_guard<std::mutex> lk(m_);
            stats = restart();   // a new pass runs through the whole stack again: the previous pass's leftovers go
            pass_.store(pass);
        }
        if (!stats.isEmpty()) emit passStats(stats);
    }
}

void LivePlayer::begin() {
    end();
    std::lock_guard<std::mutex> lk(m_);
    target_ = 4;
}

void LivePlayer::end() {
    QString stats;
    {
        std::lock_guard<std::mutex> lk(m_);
        stats = restart();
        // (the previews stay: the finished outputs' previews arrive at the very end and must reach the screen)
        pass_.store(-1);
    }
    if (!stats.isEmpty()) emit passStats(stats);
}

// The building maps and the outputs' previews, oldest first, one at a time.
void LivePlayer::previewLoop() {
    for (;;) {
        Preview pv;
        QString name;
        QColorSpace cs;
        {
            std::unique_lock<std::mutex> lk(m_);
            previewCv_.wait(lk, [this] { return quit_ || !previews_.empty(); });
            if (quit_) return;
            auto it = previews_.begin();
            for (auto j = previews_.begin(); j != previews_.end(); ++j) if (j->second.seq < it->second.seq) it = j;
            name = it->first;
            pv = std::move(it->second);
            previews_.erase(it);
            cs = cs_;
        }
        if (!pv.rgb) continue;
        float gain = 1.f;
        if (name.startsWith("focus")) {   // the focus measure has no fixed scale: show it up to its 99th percentile
            std::vector<float> v;
            v.reserve((size_t)pv.w * pv.h / 4 + 1);
            for (size_t i = 0; i < (size_t)pv.w * pv.h; i += 4) v.push_back((*pv.rgb)[i * 3]);
            if (!v.empty()) {
                size_t q = v.size() * 99 / 100;
                std::nth_element(v.begin(), v.begin() + q, v.end());
                gain = v[q] > 1e-9f ? 1.f / v[q] : 1.f;
            }
        }
        QImage img = toQImage(pv.rgb->data(), pv.w, pv.h, gain);
        if (recycle_) recycle_(std::move(pv.rgb));
        const bool grey = name.startsWith("depth") || name.startsWith("focus");
        if (!grey && cs.isValid()) {
            img.setColorSpace(cs);
            img.convertToColorSpace(QColorSpace::SRgb);
        }
        emit previewReady(name, img);
    }
}

// One step of the sweep when one is due: the next frame in stack order of those waiting.
void LivePlayer::tick() {
    const double now = nowMs();
    Item item;
    int k = -1;
    QColorSpace cs;
    {
        std::lock_guard<std::mutex> lk(m_);
        if (ring_.empty()) return;
        if (!playing_) {   // a pass begins: a few views first, so that a late one still finds its place in the order
            if ((int)ring_.size() < target_ && now - firstArrival_ < 250) return;
            playing_ = true;
            nextDue_ = now;
        }
        if (now < nextDue_) return;
        auto it = dir_.load() > 0 ? ring_.begin() : std::prev(ring_.end());
        k = it->first;
        item = std::move(it->second);
        ring_.erase(it);
        cursor_ = k;
        ++shown_;
        cs = cs_;
        // the engine's pace, from the arrivals; the views waiting steer it: more than the target a little faster,
        // fewer a little slower (an empty ring twice in a row is a stutter, a full one is delay)
        double interval = arrivals_.size() >= 4 ? (arrivals_.back() - arrivals_.front()) / (double)(arrivals_.size() - 1) : 33.0;
        interval = std::clamp(interval, 14.0, 250.0);
        const double steer = std::clamp(1.0 + 0.6 * (target_ - (double)ring_.size()) / target_, 0.5, 1.6);
        // (late by more than a step, after the ring ran empty: the pace starts again from now, not in a burst)
        nextDue_ = (now - nextDue_ > interval ? now : nextDue_) + std::max(12.0, interval * steer);
    }
    if (item.img.isNull()) return;
    if (cs.isValid()) {
        item.img.setColorSpace(cs);
        item.img.convertToColorSpace(QColorSpace::SRgb);
    }
    const double took = nowMs() - now;
    {
        std::lock_guard<std::mutex> lk(m_);
        convertMs_ += took;
    }
    emit view(k, item.img, item.win, item.full);
}
