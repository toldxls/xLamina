// The depth map as a picture (developer controls): the frame index per pixel through a palette, over a chosen
// range of frames, optionally in bands that repeat every N frames (contour lines: a step of one or two frames
// shows). Display only: the render and its files are untouched. DepthBar is the row of controls, used under the
// main view and in the two-up viewer.
#pragma once

#include <QCheckBox>
#include <QComboBox>
#include <QImage>
#include <QLabel>
#include <QSettings>
#include <QSlider>
#include <QSpinBox>
#include <QWidget>

#include <filesystem>
#include <vector>

struct DepthStyle {
    int palette = 2;       // index into depthPaletteNames()
    bool invert = false;
    int lo = 0, hi = -1;   // the frames the palette spans (0-based); hi < 0 = the last frame
    int bands = 0;         // > 0: the palette repeats every `bands` frames
    // -100 .. 100 each: the position in the palette is shifted (brightness) and stretched about its middle
    // (contrast, x 4 at 100), so a few frames' difference can be spread over many colours
    int brightness = 0, contrast = 0;
    int overlay = 0;       // two-up viewer: the map over the other images, % opacity (0 = off)
    bool popup = true;     // the frame under the cursor in a popup beside it
    int within = 0;        // the popup's "within N px": the patch around the cursor whose frames it names, in image pixels; 0 = about 8 pixels of the screen, whatever the zoom
    int radiusAt(double zoom) const;   // that patch's half-width in image pixels at a zoom
    bool highlight = true; // main view: every pixel of the frame under the cursor is marked
    bool sameImage(const DepthStyle& o) const {   // the two draw the same picture
        return palette == o.palette && invert == o.invert && lo == o.lo && hi == o.hi && bands == o.bands &&
               brightness == o.brightness && contrast == o.contrast;
    }
    bool operator==(const DepthStyle& o) const { return sameImage(o) && overlay == o.overlay && popup == o.popup && highlight == o.highlight && within == o.within; }
    // the position in the palette (0 .. 1) after brightness and contrast
    double adjust(double t) const;
    // kept between sessions, all but the range, which belongs to a stack
    static DepthStyle load(QSettings& s);
    void save(QSettings& s) const;
};

QStringList depthPaletteNames();
// palette colour at t in [0, 1]
QRgb depthColor(int palette, double t);
// depth: w*h frame indices (0 .. K-1, fractional between frames)
QImage depthImage(const float* depth, int w, int h, int K, const DepthStyle& st);
// the frames holding the 1st and the 99th percentile of the map: the range that spends the palette on the subject
void depthRange(const float* depth, size_t n, int K, int& lo, int& hi);
// what the map holds at (x, y), for the popup: the frame (or the two it lies between) by number and file name, and
// the span of frames within `radius` pixels when there is more than one. Empty outside the map
QString depthFrameText(const float* depth, int w, int h, const std::vector<std::filesystem::path>& paths, int x, int y, int radius);

// One frame marked across a part of the map: an overlay for the rectangle `vis` of the image, one overlay pixel per
// s x s image pixels (a block is marked when any of its pixels is frame k's), the frame's pixels in a colour the
// palette does not hold and the rest dimmed. idx = the map rounded to frames, w pixels wide.
QImage depthLayerMark(const uint16_t* idx, int w, int k, const QRect& vis, int s, int palette);

class DepthBar : public QWidget {
    Q_OBJECT
public:
    explicit DepthBar(bool withOverlay, QWidget* parent = nullptr);
    DepthStyle style() const { return st_; }
    void setStyle(const DepthStyle& st);   // no signal
    void setFrames(int K);                 // the stack's frame count: the limits of the range

signals:
    void changed();
    void fitRequested();                   // "Fit": the owner answers with setStyle(depthRange(...))

private:
    void read();
    void legend();
    DepthStyle st_;
    int K_ = 2;
    QComboBox* palette_ = nullptr;
    QCheckBox* invert_ = nullptr;
    QSpinBox* lo_ = nullptr;
    QSpinBox* hi_ = nullptr;
    QSpinBox* bands_ = nullptr;
    QSlider* overlay_ = nullptr;
    QSlider* brightness_ = nullptr;
    QSlider* contrast_ = nullptr;
    QCheckBox* popup_ = nullptr;
    QSpinBox* within_ = nullptr;
    QCheckBox* highlight_ = nullptr;   // the main view's bar only
    QLabel* legend_ = nullptr;
    QLabel* ends_[2] = {nullptr, nullptr};
};
