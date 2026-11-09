// The live view's player (0.6.32, FINDINGS 47). The engine hands over a view of each frame it processes, from
// several GPU streams at once and so out of order; the player keeps the recent ones in a ring and, on its own
// thread at a steady 30 frames a second, shows them in stack order, forward for the read, pass 1 and pass 3,
// backward for pass 2, never stepping back, skipping ahead only when the engine is far in front. The float to
// 8-bit and colour-space conversions happen here, off the GUI thread, so the window keeps painting whatever the
// view's size. Before, each view went straight to the GUI thread as it landed: frames from two cards interleaved,
// a Full view cost that thread 300 ms, and three independent throttles made the cadence irregular.
#pragma once

#include <QColorSpace>
#include <QImage>
#include <QObject>
#include <QThread>

#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

using FloatImage = std::shared_ptr<std::vector<float>>;

class LivePlayer : public QObject {
    Q_OBJECT
public:
    explicit LivePlayer(QObject* parent = nullptr);
    ~LivePlayer() override;
    void setRecycler(std::function<void(FloatImage)> f) { recycle_ = std::move(f); }   // buffers go back to the worker's pool
    void setColorSpace(const QColorSpace& cs);          // the frames' profile; views are converted to sRGB for the screen
    // from the engine's threads: a view of frame k, w x h pixels showing the crop `win` of a `full`-sized frame
    void push(int k, FloatImage rgb, int w, int h, const QRect& win, const QSize& full);
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
    void previewReady(QString name, QImage image);           // a building map or output preview, display-ready

private:
    void tick();
    struct Item { FloatImage rgb; int w = 0, h = 0; long long seq = 0; QRect win; QSize full; };
    QThread thread_;
    std::mutex m_;
    std::map<int, Item> ring_;
    std::map<QString, Item> previews_;   // the latest of each name
    long long seq_ = 0;
    int newest_ = -1;                 // the k that arrived last
    std::atomic<int> dir_{1};
    std::atomic<int> pass_{-1};
    int cursor_ = -1;                 // the k on screen, -1 = none yet
    QColorSpace cs_;
    std::function<void(FloatImage)> recycle_;
    static constexpr size_t kRing = 24;
    static constexpr int kLead = 10;  // when the engine is this far ahead the cursor jumps to keep up
};
