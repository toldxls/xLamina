#include "imageview.h"

#include <QKeyEvent>
#include <QMouseEvent>
#include <QPainter>
#include <QScrollBar>
#include <QStyleOptionGraphicsItem>
#include <QWheelEvent>

#include <algorithm>
#include <cmath>

// Draws straight from a QImage the owner keeps editing: paint() copies only the exposed rectangle, so a stroke's
// update(rect) costs the pixels of the stroke, not a 120 MB pixmap. The mask, when shown, is tinted on top.
class LiveImageItem : public QGraphicsItem {
public:
    const QImage* img = nullptr;
    const std::vector<uint8_t>* mask = nullptr;
    bool showMask = false;
    void setSource(const QImage* i, const std::vector<uint8_t>* m) { prepareGeometryChange(); img = i; mask = m; }
    QRectF boundingRect() const override { return img ? QRectF(0, 0, img->width(), img->height()) : QRectF(); }
    void paint(QPainter* p, const QStyleOptionGraphicsItem* opt, QWidget*) override {
        if (!img) return;
        QRect r = opt->exposedRect.toAlignedRect().intersected(QRect(0, 0, img->width(), img->height()));
        if (r.isEmpty()) return;
        p->drawImage(r, *img, r);
        if (showMask && mask && (int)mask->size() == img->width() * img->height()) {
            QImage tint(r.size(), QImage::Format_ARGB32);
            for (int y = 0; y < r.height(); ++y) {
                const uint8_t* m = mask->data() + (size_t)(r.y() + y) * img->width() + r.x();
                QRgb* o = reinterpret_cast<QRgb*>(tint.scanLine(y));
                for (int x = 0; x < r.width(); ++x) o[x] = m[x] ? qRgba(40, 220, 90, m[x] * 150 / 255) : 0;
            }
            p->drawImage(r, tint);
        }
    }
};

ImageView::ImageView(QWidget* parent) : QGraphicsView(parent) {
    setScene(&scene_);
    setBackgroundBrush(QColor(32, 32, 32));
    setDragMode(QGraphicsView::ScrollHandDrag);
    setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
    setResizeAnchor(QGraphicsView::AnchorViewCenter);
    setViewportUpdateMode(QGraphicsView::FullViewportUpdate);
    setRenderHints(QPainter::SmoothPixmapTransform);
    setMouseTracking(true);
    item_ = scene_.addPixmap(QPixmap());
    overlay_ = scene_.addPixmap(QPixmap());
    overlay_->setZValue(1);
    overlay_->setVisible(false);
    live_ = new LiveImageItem;
    live_->setZValue(0.5);
    live_->setVisible(false);
    scene_.addItem(live_);
    connect(this, &ImageView::zoomChanged, this, [this](double) { emit viewChanged(); });
    connect(horizontalScrollBar(), &QScrollBar::valueChanged, this, [this](int) { emit viewChanged(); });
    connect(verticalScrollBar(), &QScrollBar::valueChanged, this, [this](int) { emit viewChanged(); });
}

QRectF ImageView::visibleSceneRect() const {
    return mapToScene(viewport()->rect()).boundingRect().intersected(scene_.sceneRect());
}

void ImageView::setLiveWindow(const QImage& img, const QRect& win, const QSize& full) {
    if (img.isNull() || !full.isValid()) return;
    const bool newScene = scene_.sceneRect().size() != QSizeF(full) || (!winActive_ && image_.isNull());
    winActive_ = true;
    image_ = img;
    item_->setPixmap(QPixmap::fromImage(img));
    item_->setPos(win.topLeft());
    item_->setScale((double)win.width() / img.width());
    item_->setVisible(true);
    live_->setVisible(false);
    if (newScene) {
        scene_.setSceneRect(QRectF(QPointF(0, 0), QSizeF(full)));
        if (fitted_) fit();
    }
}

void ImageView::setOverlay(const QImage& argb, double opacity) {
    if (argb.isNull()) { overlay_->setVisible(false); overlay_->setPixmap(QPixmap()); return; }
    setOverlayAt(argb, QPointF(0, 0), 1.0, opacity);
}

void ImageView::setOverlayAt(const QImage& argb, QPointF pos, double scale, double opacity) {
    if (argb.isNull()) { setOverlay(QImage()); return; }
    overlay_->setPixmap(QPixmap::fromImage(argb));
    overlay_->setPos(pos);
    overlay_->setScale(scale);
    overlay_->setOpacity(opacity);
    overlay_->setVisible(true);
}

// ScrollHandDrag puts the hand back after every drag (mouseReleaseEvent restores the crosshair)
void ImageView::setPrecise(bool on) {
    if (precise_ == on) return;
    precise_ = on;
    if (!brush_) viewport()->setCursor(on ? Qt::CrossCursor : Qt::OpenHandCursor);
}

void ImageView::setCursorNote(const QString& text) {
    if (note_ == text) return;
    note_ = text;
    viewport()->update();
}

void ImageView::setOverlayOpacity(double opacity) { overlay_->setOpacity(opacity); }

// A new image keeps the user's zoom and position (as fractions of the image, so
// images of different sizes - a raw frame, a warped one, a full-resolution one -
// show the same area at the same on-screen size) until the view is fitted again.
void ImageView::setImage(const QImage& img, bool keepView) {
    if (winActive_) {   // a live-view crop was on show: the item goes back to covering the whole scene
        winActive_ = false;
        item_->setPos(0, 0);
        item_->setScale(1.0);
        image_ = QImage();   // (a crop is not comparable with a whole image for the keep logic)
        if (!img.isNull() && !fitted_) {   // keep the user's zoom and place: both referred to the full frame already
            image_ = img;
            item_->setPixmap(QPixmap::fromImage(img));
            scene_.setSceneRect(item_->boundingRect());
            return;
        }
    }
    bool keep = keepView && !fitted_ && !image_.isNull() && !img.isNull();
    if (keep && img.size() == image_.size()) {
        // same pixel grid (frame vs frame, output vs output): swap the pixels and leave the
        // transform and scroll position untouched, so comparing by clicking is pixel-exact
        image_ = img;
        item_->setPixmap(QPixmap::fromImage(img));
        return;
    }
    QPointF frac;
    double shownWidth = 0;
    if (keep) {
        QPointF c = mapToScene(viewport()->rect().center());
        frac = QPointF(c.x() / image_.width(), c.y() / image_.height());
        shownWidth = zoom() * image_.width();
    }
    image_ = img;
    item_->setPixmap(QPixmap::fromImage(img));
    scene_.setSceneRect(item_->boundingRect());
    if (!keep) { fit(); return; }
    double z = shownWidth / img.width();
    resetTransform();
    scale(z, z);
    centerOn(frac.x() * img.width(), frac.y() * img.height());
    emit zoomChanged(z);
}

void ImageView::setLive(const QImage* img, const std::vector<uint8_t>* mask) {
    live_->setSource(img, mask);
    live_->setVisible(img != nullptr);
    item_->setVisible(img == nullptr);
    if (img) scene_.setSceneRect(QRectF(0, 0, img->width(), img->height()));
    viewport()->update();
}

void ImageView::patch(const QRect& r) { live_->update(QRectF(r)); }

void ImageView::setMaskShown(bool on) {
    live_->showMask = on;
    live_->update();
}

void ImageView::setBrush(bool on, double radius, double hardness) {
    brush_ = on;
    brushRadius_ = radius;
    brushHardness_ = hardness;
    setDragMode(on ? QGraphicsView::NoDrag : QGraphicsView::ScrollHandDrag);
    viewport()->setCursor(on ? Qt::BlankCursor : precise_ ? Qt::CrossCursor : Qt::ArrowCursor);
    viewport()->update();
}

// ----------------------------------------------------------------------------- scale bar and measuring

bool ImageView::hasImage() const { return (!image_.isNull() || live_->img) && !scene_.sceneRect().isEmpty(); }

void ImageView::setSceneWidthUm(double um) {
    if (sceneUm_ == um) return;
    sceneUm_ = um;
    viewport()->update();
}

void ImageView::setScaleBar(bool on) {
    scaleBar_ = on;
    viewport()->update();
}

void ImageView::setMeasure(bool on) {
    if (measure_ == on) return;
    measure_ = on;
    drawing_ = panning_ = false;
    if (!brush_) {
        setDragMode(on ? QGraphicsView::NoDrag : QGraphicsView::ScrollHandDrag);
        viewport()->setCursor(on || precise_ ? Qt::CrossCursor : Qt::OpenHandCursor);
    }
    viewport()->update();
}

void ImageView::addMeasure(const QLineF& frac) {
    lines_.push_back(frac);
    emit measuresChanged();
    emit measured(lengthText(frac));
    viewport()->update();
}

void ImageView::setMeasures(const std::vector<QLineF>& lines) {
    lines_ = lines;
    viewport()->update();
}

void ImageView::clearMeasures() {
    const bool any = !lines_.empty();
    lines_.clear();
    drawing_ = false;
    if (any) emit measuresChanged();
    viewport()->update();
}

QPointF ImageView::toFrac(QPointF s) const {
    const QRectF r = scene_.sceneRect();
    return QPointF((s.x() - r.x()) / r.width(), (s.y() - r.y()) / r.height());
}

QPointF ImageView::fromFrac(QPointF f) const {
    const QRectF r = scene_.sceneRect();
    return QPointF(mapFromScene(QPointF(r.x() + f.x() * r.width(), r.y() + f.y() * r.height())));
}

// a length in the unit that reads best; in image pixels when the field of view is not known
static QString umText(double um) {
    if (um >= 1000) return QString("%1 mm").arg(um / 1000, 0, 'f', um >= 10000 ? 2 : 3);
    return QString("%1 µm").arg(um, 0, 'f', um >= 100 ? 0 : um >= 10 ? 1 : 2);
}

QString ImageView::lengthText(const QLineF& f) const {
    const QRectF r = scene_.sceneRect();
    const double px = std::hypot(f.dx() * r.width(), f.dy() * r.height());
    if (sceneUm_ > 0 && r.width() > 0) return umText(px * sceneUm_ / r.width());
    return QString("%1 px").arg(px, 0, 'f', 1);
}

void ImageView::drawMeasure(QPainter* p, const QLineF& f, bool live) const {
    const QPointF a = fromFrac(f.p1()), b = fromFrac(f.p2());
    const QColor ink = live ? QColor(255, 230, 80) : QColor(255, 255, 255);
    p->setRenderHint(QPainter::Antialiasing);
    for (int pass = 0; pass < 2; ++pass) {   // a dark edge under a bright line: plain on any image
        p->setPen(pass == 0 ? QPen(QColor(0, 0, 0, 190), 4, Qt::SolidLine, Qt::RoundCap) : QPen(ink, 1.6, Qt::SolidLine, Qt::RoundCap));
        p->drawLine(a, b);
        const QLineF l(a, b);
        if (l.length() > 1) {   // a tick across each end
            QLineF n = l.normalVector();
            n.setLength(6);
            const QPointF d = n.p2() - n.p1();
            p->drawLine(a - d, a + d);
            p->drawLine(b - d, b + d);
        }
    }
    const QString text = lengthText(f);
    QFont font = p->font();
    font.setPointSize(10);
    font.setBold(true);
    p->setFont(font);
    const QFontMetrics fm(font);
    QRectF plate(0, 0, fm.horizontalAdvance(text) + 14, fm.height() + 6);
    QPointF at = (a + b) / 2 + QPointF(10, -plate.height() - 8);
    const QRect vp = viewport()->rect();
    at.setX(std::clamp(at.x(), 4.0, std::max(4.0, vp.width() - plate.width() - 4)));
    at.setY(std::clamp(at.y(), 4.0, std::max(4.0, vp.height() - plate.height() - 4)));
    plate.moveTopLeft(at);
    p->setPen(Qt::NoPen);
    p->setBrush(QColor(0, 0, 0, 190));
    p->drawRoundedRect(plate, 4, 4);
    p->setPen(ink);
    p->drawText(plate, Qt::AlignCenter, text);
}

void ImageView::syncFrom(const ImageView* o) {
    if (!o || o == this) return;
    setTransform(o->transform());
    horizontalScrollBar()->setValue(o->horizontalScrollBar()->value());
    verticalScrollBar()->setValue(o->verticalScrollBar()->value());
    fitted_ = o->fitted_;
}

void ImageView::setLabel(const QString& text) {
    if (label_ == text) return;
    label_ = text;
    viewport()->update();
}

// the image's name in the top-left corner, in viewport coordinates, over a translucent plate; the brush circle
void ImageView::drawForeground(QPainter* p, const QRectF&) {
    p->save();
    p->resetTransform();
    if (!label_.isEmpty() && (!image_.isNull() || live_->img)) {
        QFont f = p->font();
        f.setPointSize(13);
        f.setBold(true);
        p->setFont(f);
        QFontMetrics fm(f);
        QRect r(10, 8, fm.horizontalAdvance(label_) + 20, fm.height() + 10);
        p->setPen(Qt::NoPen);
        p->setBrush(QColor(0, 0, 0, 150));
        p->drawRoundedRect(r, 6, 6);
        p->setPen(QColor(235, 235, 235));
        p->drawText(r, Qt::AlignCenter, label_);
    }
    if (!note_.isEmpty() && cursorIn_ && !brush_) {   // the note beside the cursor, kept inside the view
        QFont f = p->font();
        f.setPointSize(10);
        f.setBold(false);
        p->setFont(f);
        const QRect vp = viewport()->rect();
        QRect r = QFontMetrics(f).boundingRect(QRect(0, 0, 900, 200), Qt::AlignLeft | Qt::AlignTop, note_).adjusted(0, 0, 18, 12);
        QPoint at = cursor_.toPoint() + QPoint(18, 20);
        if (at.x() + r.width() > vp.right() - 4) at.setX(std::max(4, cursor_.toPoint().x() - 14 - r.width()));
        if (at.y() + r.height() > vp.bottom() - 4) at.setY(std::max(4, cursor_.toPoint().y() - 14 - r.height()));
        r.moveTopLeft(at);
        p->setRenderHint(QPainter::Antialiasing);
        p->setPen(QPen(QColor(255, 255, 255, 90), 1));
        p->setBrush(QColor(0, 0, 0, 205));
        p->drawRoundedRect(r, 5, 5);
        p->setPen(QColor(240, 240, 240));
        p->drawText(r.adjusted(9, 6, -9, -6), Qt::AlignLeft | Qt::AlignTop, note_);
    }
    if (hasImage()) {
        for (const QLineF& l : lines_) drawMeasure(p, l, false);
        if (drawing_) drawMeasure(p, cur_, true);
    }
    // the scale bar, bottom left: a round length (1, 2 or 5 times a power of ten) near a fifth of the view's width
    if (scaleBar_ && sceneUm_ > 0 && hasImage() && zoom() > 0) {
        const QRect vp = viewport()->rect();
        const double umPerPx = sceneUm_ / scene_.sceneRect().width() / zoom();   // per pixel of the view
        const double target = 0.22 * vp.width() * umPerPx;
        double nice = std::pow(10.0, std::floor(std::log10(target)));
        if (5 * nice <= target) nice *= 5; else if (2 * nice <= target) nice *= 2;
        const double w = nice / umPerPx;
        if (w >= 20 && vp.height() > 80) {
            const QString text = umText(nice);
            QFont f = p->font();
            f.setPointSize(11);
            f.setBold(true);
            p->setFont(f);
            const QFontMetrics fm(f);
            const double x0 = 18, y0 = vp.height() - 22;
            p->setRenderHint(QPainter::Antialiasing, false);
            p->setPen(Qt::NoPen);
            p->setBrush(QColor(0, 0, 0, 150));
            p->drawRoundedRect(QRectF(x0 - 10, y0 - fm.height() - 12, std::max(w, (double)fm.horizontalAdvance(text)) + 20, fm.height() + 26), 6, 6);
            p->setBrush(QColor(0, 0, 0));
            p->drawRect(QRectF(x0 - 1, y0 - 1, w + 2, 7));
            p->setBrush(QColor(245, 245, 245));
            p->drawRect(QRectF(x0, y0, w, 5));
            p->setPen(QColor(245, 245, 245));
            p->drawText(QRectF(x0, y0 - fm.height() - 6, std::max(w, (double)fm.horizontalAdvance(text)), fm.height()), Qt::AlignLeft | Qt::AlignVCenter, text);
        }
    }
    if (brush_ && cursorIn_) {
        double rr = brushRadius_ * zoom();
        p->setRenderHint(QPainter::Antialiasing);
        p->setBrush(Qt::NoBrush);
        p->setPen(QPen(QColor(0, 0, 0, 160), 3));
        p->drawEllipse(cursor_, rr, rr);
        p->setPen(QPen(QColor(255, 255, 255, 220), 1));
        p->drawEllipse(cursor_, rr, rr);
        if (brushHardness_ < 1.0) {   // the hard core
            double rc = rr * brushHardness_;
            p->setPen(QPen(QColor(255, 255, 255, 110), 1, Qt::DashLine));
            p->drawEllipse(cursor_, rc, rc);
        }
    }
    p->restore();
}

void ImageView::fit() {
    if (image_.isNull() && !live_->img) return;
    fitInView(scene_.sceneRect(), Qt::KeepAspectRatio);   // the scene is the image (or the live view's full frame)
    fitted_ = true;
    emit zoomChanged(zoom());
}

void ImageView::actualSize() {
    resetTransform();
    fitted_ = false;
    emit zoomChanged(1.0);
}

void ImageView::wheelEvent(QWheelEvent* e) {
    if (image_.isNull() && !live_->img) return;
    if (brush_ && (e->modifiers() & Qt::ControlModifier)) {   // Ctrl + wheel: the brush size
        double f = std::pow(1.002, e->angleDelta().y());
        brushRadius_ = std::clamp(brushRadius_ * f, 2.0, 2000.0);
        emit brushResized(brushRadius_);
        viewport()->update();
        return;
    }
    double f = std::pow(1.0015, e->angleDelta().y());
    double z = zoom() * f;
    if (z < 0.02 || z > 32) return;
    scale(f, f);
    fitted_ = false;
    emit zoomChanged(zoom());
}

void ImageView::mousePressEvent(QMouseEvent* e) {
    if (brush_) {
        if (e->button() == Qt::LeftButton) {
            painting_ = true;
            emit strokeBegin(mapToScene(e->pos()));
            return;
        }
        if (e->button() == Qt::MiddleButton || e->button() == Qt::RightButton) {
            panning_ = true;
            lastPan_ = e->pos();
            viewport()->setCursor(Qt::ClosedHandCursor);
            return;
        }
    }
    if (measure_ && !brush_) {
        if (e->button() == Qt::LeftButton && hasImage()) {
            drawing_ = true;
            const QPointF f = toFrac(mapToScene(e->pos()));
            cur_ = QLineF(f, f);
            viewport()->update();
            return;
        }
        if (e->button() == Qt::MiddleButton || e->button() == Qt::RightButton) {
            panning_ = true;
            lastPan_ = e->pos();
            viewport()->setCursor(Qt::ClosedHandCursor);
            return;
        }
    }
    if (e->button() == Qt::LeftButton && !brush_ && !measure_) { pressed_ = true; pressAt_ = e->pos(); }
    QGraphicsView::mousePressEvent(e);
}

void ImageView::mouseMoveEvent(QMouseEvent* e) {
    cursor_ = e->position();
    cursorIn_ = true;
    if (brush_) {
        // the view records the mouse position in its own handler; the wheel zoom anchors there (AnchorUnderMouse),
        // so without this call a zoom after a right-drag pan jumped to where the mouse was before the brush mode
        QGraphicsView::mouseMoveEvent(e);
        if (painting_) emit strokeTo(mapToScene(e->pos()));
        else if (panning_) {
            QPoint d = e->pos() - lastPan_;
            lastPan_ = e->pos();
            horizontalScrollBar()->setValue(horizontalScrollBar()->value() - d.x());
            verticalScrollBar()->setValue(verticalScrollBar()->value() - d.y());
        }
        viewport()->update();
    } else if (measure_) {
        QGraphicsView::mouseMoveEvent(e);
        if (drawing_) {
            QPointF end = e->position();
            if (e->modifiers() & Qt::ShiftModifier) {   // level, upright or at 45 degrees
                const QPointF a = fromFrac(cur_.p1()), d = end - a;
                const double len = std::hypot(d.x(), d.y()), step = 3.14159265358979 / 4;
                const double ang = std::round(std::atan2(d.y(), d.x()) / step) * step;
                end = a + QPointF(len * std::cos(ang), len * std::sin(ang));
            }
            cur_.setP2(toFrac(mapToScene(end.toPoint())));
        } else if (panning_) {
            QPoint d = e->pos() - lastPan_;
            lastPan_ = e->pos();
            horizontalScrollBar()->setValue(horizontalScrollBar()->value() - d.x());
            verticalScrollBar()->setValue(verticalScrollBar()->value() - d.y());
        }
        viewport()->update();
    } else {
        QGraphicsView::mouseMoveEvent(e);
        if (!note_.isEmpty()) viewport()->update();   // the note follows the cursor
    }
    QPointF p = mapToScene(e->pos());
    emit hovered((int)std::floor(p.x()), (int)std::floor(p.y()));
}

void ImageView::mouseReleaseEvent(QMouseEvent* e) {
    if (brush_) {
        if (painting_ && e->button() == Qt::LeftButton) { painting_ = false; emit strokeEnd(); return; }
        if (panning_ && (e->button() == Qt::MiddleButton || e->button() == Qt::RightButton)) { panning_ = false; viewport()->setCursor(Qt::BlankCursor); return; }
    }
    if (measure_ && !brush_) {
        if (drawing_ && e->button() == Qt::LeftButton) {
            drawing_ = false;
            const QPointF a = fromFrac(cur_.p1()), b = fromFrac(cur_.p2());
            if (std::hypot(b.x() - a.x(), b.y() - a.y()) >= 3) {   // (a click is not a line)
                lines_.push_back(cur_);
                emit measuresChanged();
                emit measured(lengthText(cur_));
            }
            viewport()->update();
            return;
        }
        if (panning_ && (e->button() == Qt::MiddleButton || e->button() == Qt::RightButton)) { panning_ = false; viewport()->setCursor(Qt::CrossCursor); return; }
    }
    QGraphicsView::mouseReleaseEvent(e);
    if ((precise_ || measure_) && !brush_) viewport()->setCursor(Qt::CrossCursor);
    if (pressed_ && e->button() == Qt::LeftButton) {
        pressed_ = false;
        if ((e->pos() - pressAt_).manhattanLength() < 4 && hasImage()) {
            const QPointF p = mapToScene(e->pos());
            emit clicked((int)std::floor(p.x()), (int)std::floor(p.y()));
        }
    }
}

void ImageView::mouseDoubleClickEvent(QMouseEvent* e) {
    if (brush_ || measure_) return;   // a double click would be a stroke, or two lines
    QGraphicsView::mouseDoubleClickEvent(e);
    fit();
}

void ImageView::keyPressEvent(QKeyEvent* e) {
    if (brush_ && (e->key() == Qt::Key_BracketLeft || e->key() == Qt::Key_BracketRight)) {
        brushRadius_ = std::clamp(brushRadius_ * (e->key() == Qt::Key_BracketRight ? 1.25 : 0.8), 2.0, 2000.0);
        emit brushResized(brushRadius_);
        viewport()->update();
        return;
    }
    if (stepKeys_ && !brush_) {
        if (e->key() == Qt::Key_Up || e->key() == Qt::Key_Left) { emit stepKey(-1); return; }
        if (e->key() == Qt::Key_Down || e->key() == Qt::Key_Right) { emit stepKey(1); return; }
        if (e->key() == Qt::Key_Escape) { emit escapePressed(); return; }
    }
    if (measure_ && !lines_.empty() && (e->key() == Qt::Key_Backspace || e->key() == Qt::Key_Delete)) {
        lines_.pop_back();
        emit measuresChanged();
        viewport()->update();
        return;
    }
    QGraphicsView::keyPressEvent(e);
}

void ImageView::leaveEvent(QEvent* e) {
    cursorIn_ = false;
    emit hovered(-1, -1);
    viewport()->update();
    QGraphicsView::leaveEvent(e);
}

void ImageView::resizeEvent(QResizeEvent* e) {
    QGraphicsView::resizeEvent(e);
    if (fitted_) fit();
    emit viewChanged();
}

QImage toQImage(const float* rgb, int w, int h, float gain) {
    QImage img(w, h, QImage::Format_RGB32);
    for (int y = 0; y < h; ++y) {
        QRgb* row = reinterpret_cast<QRgb*>(img.scanLine(y));
        const float* p = rgb + (size_t)y * w * 3;
        for (int x = 0; x < w; ++x, p += 3) {
            int r = (int)std::lround(std::clamp(p[0] * gain, 0.f, 1.f) * 255.f);
            int g = (int)std::lround(std::clamp(p[1] * gain, 0.f, 1.f) * 255.f);
            int b = (int)std::lround(std::clamp(p[2] * gain, 0.f, 1.f) * 255.f);
            row[x] = qRgb(r, g, b);
        }
    }
    return img;
}
