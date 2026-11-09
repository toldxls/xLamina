#include "liveplayer.h"

#include <QMetaObject>
#include <QTimer>

#include "imageview.h"   // toQImage

LivePlayer::LivePlayer(QObject* parent) : QObject(parent) {
    moveToThread(&thread_);
    QTimer* t = new QTimer;
    t->moveToThread(&thread_);
    t->setInterval(33);   // 30 views a second: the sweep's pace
    t->setTimerType(Qt::PreciseTimer);
    connect(t, &QTimer::timeout, this, &LivePlayer::tick);
    connect(&thread_, &QThread::started, t, qOverload<>(&QTimer::start));
    connect(&thread_, &QThread::finished, t, &QObject::deleteLater);
    thread_.start();
}

LivePlayer::~LivePlayer() {
    thread_.quit();
    thread_.wait();
    end();
}

void LivePlayer::setColorSpace(const QColorSpace& cs) {
    std::lock_guard<std::mutex> lk(m_);
    cs_ = cs;
}

void LivePlayer::push(int k, FloatImage rgb, int w, int h, const QRect& win, const QSize& full) {
    std::vector<FloatImage> drop;
    {
        std::lock_guard<std::mutex> lk(m_);
        auto it = ring_.find(k);
        if (it != ring_.end()) drop.push_back(std::move(it->second.rgb));   // a repeated frame: the newer copy stays
        ring_[k] = Item{std::move(rgb), w, h, ++seq_, win, full};
        newest_ = k;
        while (ring_.size() > kRing) {   // the ring holds the recent views: the oldest arrival goes
            auto oldest = ring_.begin();
            for (auto j = ring_.begin(); j != ring_.end(); ++j) if (j->second.seq < oldest->second.seq) oldest = j;
            drop.push_back(std::move(oldest->second.rgb));
            ring_.erase(oldest);
        }
    }
    if (recycle_) for (auto& d : drop) recycle_(std::move(d));
}

void LivePlayer::setPass(int pass, int dir) {
    dir_.store(dir);
    if (pass_.exchange(pass) != pass) {
        std::vector<FloatImage> drop;
        {
            std::lock_guard<std::mutex> lk(m_);
            cursor_ = -1;   // a new pass runs through the whole stack again: start where its first frames are
            for (auto& [k, it] : ring_) drop.push_back(std::move(it.rgb));   // the previous pass's leftovers
            ring_.clear();
        }
        if (recycle_) for (auto& d : drop) recycle_(std::move(d));
    }
}

void LivePlayer::begin() { end(); }

void LivePlayer::end() {
    std::vector<FloatImage> drop;
    {
        std::lock_guard<std::mutex> lk(m_);
        for (auto& [k, it] : ring_) drop.push_back(std::move(it.rgb));
        ring_.clear();
        cursor_ = -1;
        newest_ = -1;
    }
    pass_.store(-1);
    if (recycle_) for (auto& d : drop) recycle_(std::move(d));
}

// one step of the sweep: the next frame in stack order that has arrived, or none this tick
void LivePlayer::tick() {
    Item item;
    int k = -1;
    QColorSpace cs;
    std::vector<FloatImage> drop;
    {
        std::lock_guard<std::mutex> lk(m_);
        if (ring_.empty()) return;
        const int dir = dir_.load();
        auto pick = [&](int from) -> std::map<int, Item>::iterator {   // the nearest key past `from` in the sweep's direction
            if (dir > 0) return from < 0 ? ring_.begin() : ring_.upper_bound(from);
            if (from < 0) return std::prev(ring_.end());
            auto it = ring_.lower_bound(from);   // first key >= from
            return it == ring_.begin() ? ring_.end() : std::prev(it);
        };
        // the engine is far ahead: jump, so the view keeps up and the ring does not run dry behind us
        if (cursor_ >= 0 && newest_ >= 0 && (newest_ - cursor_) * dir > kLead) cursor_ = newest_ - (kLead - 4) * dir;
        auto it = pick(cursor_);
        if (it == ring_.end()) return;
        k = it->first;
        item = std::move(it->second);
        ring_.erase(it);
        cursor_ = k;
        cs = cs_;
        // what lies behind the cursor will never be shown: give the buffers back
        for (auto j = ring_.begin(); j != ring_.end();) {
            if ((j->first - cursor_) * dir < 0) { drop.push_back(std::move(j->second.rgb)); j = ring_.erase(j); }
            else ++j;
        }
    }
    if (recycle_) for (auto& d : drop) recycle_(std::move(d));
    if (!item.rgb) return;
    QImage img = toQImage(item.rgb->data(), item.w, item.h);
    if (recycle_) recycle_(std::move(item.rgb));
    if (cs.isValid()) {
        img.setColorSpace(cs);
        img.convertToColorSpace(QColorSpace::SRgb);
    }
    emit view(k, img, item.win, item.full);
}
