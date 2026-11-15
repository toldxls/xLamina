// The live view's player (0.6.32, FINDINGS 47). The engine hands over a view of each frame it processes, from
// several GPU streams at once and so out of order; the player keeps the recent ones and, on its own thread, shows
// them in stack order, forward for the read, pass 1 and pass 3, backward for pass 2, never stepping back. The
// colour-space conversion happens here, off the GUI thread, so the window keeps painting whatever the view's size.
//
// The pace is the engine's own: the views arrive in bursts (each stream hands its view over when its next frame comes
// round), so the player measures the rate they arrive at and shows one per that interval, a few frames behind the
// newest, a little faster when more are waiting and a little slower when fewer. (Before: a fixed 30 a second tick
// that showed whatever had arrived and dropped what came late: 12 to 18 views a second at uneven steps.)
//
// The building maps and the outputs' previews are converted on a second thread: a 2 Mpx float map takes longer than
// the gap between two frames of the sweep.
#pragma once

#include <QColorSpace>
#include <QImage>
#include <QObject>
#include <QThread>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

using FloatImage = std::shared_ptr<std::vector<float>>;

class LivePlayer : public QObject {
    Q_OBJECT
public:
    explicit LivePlayer(QObject* parent = nullptr);
    ~LivePlayer() override;
    void setRecycler(std::function<void(FloatImage)> f) { recycle_ = std::move(f); }   // buffers go back to the worker's pool
    void setColorSpace(const QColorSpace& cs);          // the frames' profile; views are converted to sRGB for the screen
    // from the engine's threads: a view of frame k (8 bits a channel, the frames' colour space, not shared with
    // anyone: it is converted in place) showing the crop `win` of a `full`-sized frame
    void push(int k, QImage img, const QRect& win, const QSize& full);
    // a building map or a finished output's preview (the right pane / the output tabs): the latest per name is kept
    // and converted here, off the GUI thread; "focus" maps are scaled to their 99th percentile, depth and focus stay grey
    void pushPreview(const QString& name, FloatImage rgb, int w, int h);
    // the pass the engine is in (any id; a change restarts the sweep, since every pass runs through the whole stack
    // again) and its direction, +1 forward or -1 backward
    void setPass(int pass, int dir);
    void begin();                                       // a job starts: empty the ring, forget the cursor
    void end();                                         // a job ended: drop what is left

signals:
    void view(int k, QImage image, QRect win, QSize full);   // the next frame of the sweep, display-ready (queued to the GUI thread)
    void passStats(QString line);                            // developer aid: a pass's views counted, when it has ended
    void previewReady(QString name, QImage image);           // a building map or output preview, display-ready

private:
    void tick();
    void previewLoop();
    QString restart();                // (m_ held) nothing waiting, nothing shown yet; the ended pass's counts, if any
    struct Item { QImage img; QRect win; QSize full; };
    struct Preview { FloatImage rgb; int w = 0, h = 0; long long seq = 0; };
    QThread thread_;
    std::thread previewThread_;
    std::mutex m_;
    std::condition_variable previewCv_;
    bool quit_ = false;
    std::map<int, Item> ring_;           // the views waiting, by frame: all of them ahead of the cursor
    std::map<QString, Preview> previews_;   // the latest of each name
    long long seq_ = 0;
    std::atomic<int> dir_{1};
    std::atomic<int> pass_{-1};
    int cursor_ = -1;                 // the k on screen, -1 = none yet
    std::deque<double> arrivals_;     // when the last views arrived (ms): the pace
    bool playing_ = false;            // the sweep has started (enough views waited, or long enough)
    double firstArrival_ = 0, nextDue_ = 0;
    QColorSpace cs_;
    std::function<void(FloatImage)> recycle_;
    // The views kept waiting, so that one arriving out of order still finds its place: the cards hand their views
    // over at their own pace, and how far out of order that leaves them depends on the PC. Each view that arrives too
    // late to be shown raises it by one for the rest of the job.
    int target_ = 4;
    int pushed_ = 0, late_ = 0, skipped_ = 0, shown_ = 0;   // this pass's views: arrived, too late, skipped, shown
    double convertMs_ = 0;
    static constexpr size_t kRing = 32;    // more than this waiting: the screen is not keeping up, and the next ones are skipped
    static constexpr int kMaxTarget = 8;
    static constexpr size_t kArrivals = 24;
};
