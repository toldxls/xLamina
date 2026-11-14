#include "depthview.h"

#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QPushButton>

#include <algorithm>
#include <cmath>

namespace {
// eight equally spaced anchors of each matplotlib map, interpolated linearly
const int kAnchors = 8;
const unsigned kViridis[kAnchors] = {0x440154, 0x46337E, 0x365C8D, 0x277F8E, 0x1FA187, 0x4AC16D, 0x9FDA3A, 0xFDE725};
const unsigned kInferno[kAnchors] = {0x000004, 0x280B54, 0x65156E, 0x9F2A63, 0xD44842, 0xF57D15, 0xFAC127, 0xFCFFA4};
const unsigned kMagma[kAnchors] = {0x000004, 0x221150, 0x5F187F, 0x982D80, 0xD3436E, 0xF8765C, 0xFEBA80, 0xFCFDBF};

QRgb anchored(const unsigned* a, double t) {
    double x = t * (kAnchors - 1);
    int i = std::min((int)x, kAnchors - 2);
    double f = x - i;
    auto ch = [&](int shift) {
        int c0 = (a[i] >> shift) & 255, c1 = (a[i + 1] >> shift) & 255;
        return (int)std::lround(c0 + (c1 - c0) * f);
    };
    return qRgb(ch(16), ch(8), ch(0));
}

// Turbo (Mikhailov 2019), the polynomial fit
QRgb turbo(double x) {
    double r = 0.13572138 + x * (4.61539260 + x * (-42.66032258 + x * (132.13108234 + x * (-152.94239396 + x * 59.28637943))));
    double g = 0.09140261 + x * (2.19418839 + x * (4.84296658 + x * (-14.18503333 + x * (4.27729857 + x * 2.82956604))));
    double b = 0.10667330 + x * (12.64194608 + x * (-60.58204836 + x * (110.36276771 + x * (-89.90310912 + x * 27.34824973))));
    auto c = [](double v) { return (int)std::lround(std::clamp(v, 0.0, 1.0) * 255.0); };
    return qRgb(c(r), c(g), c(b));
}

const int kLut = 1024;
}  // namespace

QStringList depthPaletteNames() { return {"Grey", "Viridis", "Inferno", "Magma", "Turbo"}; }

QRgb depthColor(int palette, double t) {
    t = t > 0 ? std::min(t, 1.0) : 0.0;   // a NaN lands on 0
    switch (palette) {
    case 1: return anchored(kViridis, t);
    case 2: return anchored(kInferno, t);
    case 3: return anchored(kMagma, t);
    case 4: return turbo(t);
    default: { int v = (int)std::lround(t * 255.0); return qRgb(v, v, v); }
    }
}

DepthStyle DepthStyle::load(QSettings& s) {
    DepthStyle st;
    st.palette = std::clamp(s.value("depthView/palette", st.palette).toInt(), 0, (int)depthPaletteNames().size() - 1);
    st.invert = s.value("depthView/invert", false).toBool();
    st.bands = std::max(0, s.value("depthView/bands", 0).toInt());
    st.brightness = std::clamp(s.value("depthView/brightness", 0).toInt(), -100, 100);
    st.contrast = std::clamp(s.value("depthView/contrast", 0).toInt(), -100, 100);
    st.popup = s.value("depthView/popup", true).toBool();
    st.within = std::clamp(s.value("depthView/within", 0).toInt(), 0, 500);
    st.highlight = s.value("depthView/highlight", true).toBool();
    return st;
}

void DepthStyle::save(QSettings& s) const {
    s.setValue("depthView/palette", palette);
    s.setValue("depthView/invert", invert);
    s.setValue("depthView/bands", bands);
    s.setValue("depthView/brightness", brightness);
    s.setValue("depthView/contrast", contrast);
    s.setValue("depthView/popup", popup);
    s.setValue("depthView/within", within);
    s.setValue("depthView/highlight", highlight);
}

double DepthStyle::adjust(double t) const {
    if (!brightness && !contrast) return t;
    return std::clamp((t - 0.5 + brightness / 200.0) * std::pow(4.0, contrast / 100.0) + 0.5, 0.0, 1.0);
}

int DepthStyle::radiusAt(double zoom) const {
    if (within > 0) return within;
    return std::clamp((int)std::lround(8.0 / std::max(zoom, 1e-3)), 1, 64);   // about 8 screen pixels
}

QString depthFrameText(const float* depth, int w, int h, const std::vector<std::filesystem::path>& paths, int x, int y, int radius) {
    const int K = (int)paths.size();
    if (x < 0 || y < 0 || x >= w || y >= h || K < 1) return {};
    auto name = [&](int k) { return QString::fromStdWString(paths[std::clamp(k, 0, K - 1)].filename().wstring()); };
    const float d = std::clamp(depth[(size_t)y * w + x], 0.f, (float)(K - 1));
    const int k = (int)std::lround(d);
    QString s;
    if (std::fabs(d - k) < 0.05f) s = QString("Frame %1 of %2: %3").arg(k + 1).arg(K).arg(name(k));
    else {   // between two frames: the composition blends them
        const int a = (int)std::floor(d);
        s = QString("Frame %1 of %2: %3 + %4").arg(d + 1, 0, 'f', 1).arg(K).arg(name(a), name(a + 1));
    }
    float lo = d, hi = d;
    for (int yy = std::max(0, y - radius); yy <= std::min(h - 1, y + radius); ++yy)
        for (int xx = std::max(0, x - radius); xx <= std::min(w - 1, x + radius); ++xx) {
            const float v = depth[(size_t)yy * w + xx];
            lo = std::min(lo, v);
            hi = std::max(hi, v);
        }
    const int klo = std::clamp((int)std::lround(lo), 0, K - 1), khi = std::clamp((int)std::lround(hi), 0, K - 1);
    if (khi > klo) s += QString("\nWithin %1 px: frames %2 to %3 (%4 to %5)").arg(radius).arg(klo + 1).arg(khi + 1).arg(name(klo), name(khi));
    return s;
}

QImage depthLayerMark(const uint16_t* idx, int w, int k, const QRect& vis, int s, int palette) {
    const int ow = (vis.width() + s - 1) / s, oh = (vis.height() + s - 1) / s;
    // a colour the palette does not hold: white on Turbo, cyan on the others
    const QRgb on = palette == 4 ? qRgba(255, 255, 255, 255) : qRgba(0, 255, 255, 255);
    const QRgb off = qRgba(0, 0, 0, 120);
    const uint16_t kk = (uint16_t)k;
    QImage ov(ow, oh, QImage::Format_ARGB32_Premultiplied);
    std::vector<uint8_t> hit(ow);
    for (int oy = 0; oy < oh; ++oy) {
        std::fill(hit.begin(), hit.end(), (uint8_t)0);
        const int y1 = std::min(vis.y() + (oy + 1) * s, vis.y() + vis.height());
        for (int y = vis.y() + oy * s; y < y1; ++y) {
            const uint16_t* p = idx + (size_t)y * w + vis.x();
            for (int ox = 0; ox < ow; ++ox) {
                const int x1 = std::min((ox + 1) * s, vis.width());
                for (int x = ox * s; x < x1; ++x) hit[ox] |= (uint8_t)(p[x] == kk);
            }
        }
        QRgb* row = reinterpret_cast<QRgb*>(ov.scanLine(oy));
        for (int ox = 0; ox < ow; ++ox) row[ox] = hit[ox] ? on : off;
    }
    return ov;
}

QImage depthImage(const float* depth, int w, int h, int K, const DepthStyle& st) {
    QRgb lut[kLut];
    for (int i = 0; i < kLut; ++i) {
        double t = st.adjust(i / (double)(kLut - 1));
        lut[i] = depthColor(st.palette, st.invert ? 1.0 - t : t);
    }
    const int last = std::max(K - 1, 1);
    const float lo = (float)std::clamp(st.lo, 0, last - 1);
    const float hi = (float)(st.hi < 0 ? last : std::clamp(st.hi, (int)lo + 1, last));
    const float span = st.bands > 0 ? (float)st.bands : hi - lo;
    QImage img(w, h, QImage::Format_RGB32);
    for (int y = 0; y < h; ++y) {
        QRgb* row = reinterpret_cast<QRgb*>(img.scanLine(y));
        const float* d = depth + (size_t)y * w;
        for (int x = 0; x < w; ++x) {
            float t = (std::min(std::max(d[x], lo), hi) - lo) / span;   // frames outside the range take its ends
            if (st.bands > 0) t -= std::floor(t);
            int i = (int)(t * (kLut - 1) + 0.5f);
            row[x] = lut[i >= 0 && i < kLut ? i : 0];
        }
    }
    return img;
}

void depthRange(const float* depth, size_t n, int K, int& lo, int& hi) {
    const int last = std::max(K - 1, 1);
    std::vector<size_t> hist(last + 1, 0);
    for (size_t i = 0; i < n; ++i) {
        float v = depth[i];
        if (!(v >= 0)) v = 0;
        ++hist[std::min((int)(v + 0.5f), last)];
    }
    size_t cut = n / 100, sum = 0;
    lo = 0;
    hi = last;
    for (int k = 0; k <= last; ++k) { sum += hist[k]; if (sum > cut) { lo = k; break; } }
    sum = 0;
    for (int k = last; k >= 0; --k) { sum += hist[k]; if (sum > cut) { hi = k; break; } }
    if (hi <= lo) { lo = std::max(0, std::min(lo, last - 1)); hi = lo + 1; }
}

// ----------------------------------------------------------------------------- the controls

DepthBar::DepthBar(bool withOverlay, QWidget* parent) : QWidget(parent) {
    // two rows: what the colours span, then how they are spread and shown
    QVBoxLayout* v = new QVBoxLayout(this);
    v->setContentsMargins(6, 3, 6, 3);
    v->setSpacing(2);
    QHBoxLayout* l = new QHBoxLayout;
    l->setSpacing(6);
    v->addLayout(l);
    l->addWidget(new QLabel("Depth map"));
    palette_ = new QComboBox;
    palette_->addItems(depthPaletteNames());
    palette_->setToolTip("The colours the frames are drawn in, first frame to last.");
    l->addWidget(palette_);
    invert_ = new QCheckBox("Invert");
    invert_->setToolTip("Checked = the palette runs from the last frame to the first.");
    l->addWidget(invert_);
    l->addWidget(new QLabel(" Frames"));
    lo_ = new QSpinBox;
    hi_ = new QSpinBox;
    lo_->setToolTip("The palette starts at this frame; earlier frames take its first colour.");
    hi_->setToolTip("The palette ends at this frame; later frames take its last colour.");
    l->addWidget(lo_);
    l->addWidget(new QLabel("to"));
    l->addWidget(hi_);
    QPushButton* fit = new QPushButton("Fit");
    fit->setToolTip("The range that holds 98 % of the map's pixels.");
    QPushButton* all = new QPushButton("All");
    all->setToolTip("The whole stack.");
    l->addWidget(fit);
    l->addWidget(all);
    l->addWidget(new QLabel(" Bands"));
    bands_ = new QSpinBox;
    bands_->setRange(0, 999);
    bands_->setSpecialValueText("off");
    bands_->setSuffix(" frames");
    bands_->setToolTip("The palette repeats every this many frames, like contour lines, so a step of a frame or two shows.\noff = one pass of the palette over the range.");
    l->addWidget(bands_);
    l->addSpacing(8);
    ends_[0] = new QLabel;
    legend_ = new QLabel;
    legend_->setFixedSize(180, 14);
    ends_[1] = new QLabel;
    l->addWidget(ends_[0]);
    l->addWidget(legend_);
    l->addWidget(ends_[1]);
    l->addStretch(1);
    l = new QHBoxLayout;
    l->setSpacing(6);
    v->addLayout(l);
    auto slider = [&](const char* label, const char* tip) {
        l->addWidget(new QLabel(label));
        QSlider* s = new QSlider(Qt::Horizontal);
        s->setRange(-100, 100);
        s->setFixedWidth(90);
        s->setToolTip(tip);
        l->addWidget(s);
        connect(s, &QSlider::valueChanged, this, [this](int) { read(); });
        return s;
    };
    brightness_ = slider("Brightness", "Shifts the frames along the palette: which part of the range the contrast spreads out.");
    contrast_ = slider(" Contrast", "Stretches the palette about its middle (up to 4 x), so a few frames' difference takes many colours.");
    QPushButton* reset = new QPushButton("Reset");
    reset->setToolTip("Brightness and contrast back to the middle.");
    l->addWidget(reset);
    connect(reset, &QPushButton::clicked, this, [this] {
        DepthStyle s = st_;
        s.brightness = s.contrast = 0;
        if (s == st_) return;
        setStyle(s);
        emit changed();
    });
    popup_ = new QCheckBox("Frame popup");
    popup_->setToolTip("Checked = a note beside the cursor names the frame the map chose there, and the frames around it.\nUnchecked = the status bar only.");
    l->addWidget(popup_);
    connect(popup_, &QCheckBox::toggled, this, [this](bool) { read(); });
    l->addWidget(new QLabel(" Within"));
    within_ = new QSpinBox;
    within_->setRange(0, 500);
    within_->setSuffix(" px");
    within_->setSpecialValueText("auto");
    within_->setKeyboardTracking(false);
    within_->setToolTip("The note's second line names the frames found within this many image pixels of the cursor.\n"
                        "1 px = the cursor's own neighbours only; larger = a wider patch, so more frames.\n"
                        "auto = about 8 pixels of the screen: it widens as you zoom out.");
    l->addWidget(within_);
    connect(within_, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int) { read(); });
    if (!withOverlay) {
        highlight_ = new QCheckBox("Highlight frame");
        highlight_->setToolTip("Checked = every pixel taken from the frame under the cursor is marked across the image, the rest dimmed.\nA click holds that layer; the arrow keys or PgUp / PgDn then step through the layers, Esc lets go.");
        l->addWidget(highlight_);
        connect(highlight_, &QCheckBox::toggled, this, [this](bool) { read(); });
    }
    if (withOverlay) {
        l->addWidget(new QLabel(" Overlay"));
        overlay_ = new QSlider(Qt::Horizontal);
        overlay_->setRange(0, 100);
        overlay_->setFixedWidth(110);
        overlay_->setToolTip("The depth map over the other images, by this opacity. Left end = off.");
        l->addWidget(overlay_);
        connect(overlay_, &QSlider::valueChanged, this, [this](int) { read(); });
    }
    l->addStretch(1);
    connect(palette_, QOverload<int>::of(&QComboBox::activated), this, [this](int) { read(); });
    connect(invert_, &QCheckBox::toggled, this, [this](bool) { read(); });
    connect(lo_, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int) { read(); });
    connect(hi_, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int) { read(); });
    connect(bands_, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int) { read(); });
    connect(fit, &QPushButton::clicked, this, [this] { emit fitRequested(); });
    connect(all, &QPushButton::clicked, this, [this] { DepthStyle s = st_; s.lo = 0; s.hi = -1; setStyle(s); emit changed(); });
    for (QSpinBox* b : {lo_, hi_, bands_}) b->setKeyboardTracking(false);   // a typed 143 is one change, not three
    setFrames(2);
}

void DepthBar::setFrames(int K) {
    K_ = std::max(K, 2);
    setStyle(st_);
}

// the spin boxes count frames from 1, as the frame list does
void DepthBar::setStyle(const DepthStyle& st) {
    st_ = st;
    st_.lo = std::clamp(st_.lo, 0, K_ - 2);
    if (st_.hi >= 0) st_.hi = std::clamp(st_.hi, st_.lo + 1, K_ - 1);
    const QList<QWidget*> ws = {palette_, invert_, lo_, hi_, bands_, overlay_, brightness_, contrast_, popup_, highlight_, within_};
    for (QWidget* w : ws) if (w) w->blockSignals(true);
    brightness_->setValue(st_.brightness);
    contrast_->setValue(st_.contrast);
    popup_->setChecked(st_.popup);
    within_->setValue(st_.within);
    if (highlight_) highlight_->setChecked(st_.highlight);
    for (QWidget* w : ws) if (w) w->blockSignals(true);
    palette_->setCurrentIndex(st_.palette);
    invert_->setChecked(st_.invert);
    lo_->setRange(1, K_ - 1);
    hi_->setRange(2, K_);
    lo_->setValue(st_.lo + 1);
    hi_->setValue((st_.hi < 0 ? K_ - 1 : st_.hi) + 1);
    bands_->setValue(st_.bands);
    if (overlay_) overlay_->setValue(st_.overlay);
    for (QWidget* w : ws) if (w) w->blockSignals(false);
    legend();
}

void DepthBar::read() {
    DepthStyle s = st_;
    s.palette = palette_->currentIndex();
    s.invert = invert_->isChecked();
    s.lo = lo_->value() - 1;
    s.hi = hi_->value() - 1;
    if (s.hi <= s.lo) { if (sender() == lo_) s.hi = s.lo + 1; else s.lo = s.hi - 1; }
    if (s.lo == 0 && s.hi == K_ - 1) s.hi = -1;   // the whole stack stays the whole stack on the next one
    s.bands = bands_->value();
    s.brightness = brightness_->value();
    s.contrast = contrast_->value();
    s.popup = popup_->isChecked();
    s.within = within_->value();
    if (highlight_) s.highlight = highlight_->isChecked();
    if (overlay_) s.overlay = overlay_->value();
    if (s == st_) return;
    setStyle(s);
    emit changed();
}

// the palette as a strip between the two frames it spans (one band of it, when banded)
void DepthBar::legend() {
    QImage strip(legend_->width(), legend_->height(), QImage::Format_RGB32);
    for (int x = 0; x < strip.width(); ++x) {
        double t = st_.adjust(x / (double)(strip.width() - 1));
        QRgb c = depthColor(st_.palette, st_.invert ? 1.0 - t : t);
        for (int y = 0; y < strip.height(); ++y) strip.setPixel(x, y, c);
    }
    legend_->setPixmap(QPixmap::fromImage(strip));
    int lo = st_.lo + 1, hi = (st_.hi < 0 ? K_ - 1 : st_.hi) + 1;
    ends_[0]->setText(QString::number(lo));
    ends_[1]->setText(st_.bands > 0 ? QString("%1 (repeats)").arg(std::min(lo + st_.bands, K_)) : QString::number(hi));
}
