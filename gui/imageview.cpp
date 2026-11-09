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

void ImageView::setOverlay(const QImage& argb) {
    if (argb.isNull()) { overlay_->setVisible(false); overlay_->setPixmap(QPixmap()); return; }
    overlay_->setPixmap(QPixmap::fromImage(argb));
    overlay_->setVisible(true);
}

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
    viewport()->setCursor(on ? Qt::BlankCursor : Qt::ArrowCursor);
    viewport()->update();
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
    } else {
        QGraphicsView::mouseMoveEvent(e);
    }
    QPointF p = mapToScene(e->pos());
    emit hovered((int)p.x(), (int)p.y());
}

void ImageView::mouseReleaseEvent(QMouseEvent* e) {
    if (brush_) {
        if (painting_ && e->button() == Qt::LeftButton) { painting_ = false; emit strokeEnd(); return; }
        if (panning_ && (e->button() == Qt::MiddleButton || e->button() == Qt::RightButton)) { panning_ = false; viewport()->setCursor(Qt::BlankCursor); return; }
    }
    QGraphicsView::mouseReleaseEvent(e);
}

void ImageView::mouseDoubleClickEvent(QMouseEvent* e) {
    if (brush_) return;   // a double click would be a stroke
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
    QGraphicsView::keyPressEvent(e);
}

void ImageView::leaveEvent(QEvent* e) {
    cursorIn_ = false;
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
