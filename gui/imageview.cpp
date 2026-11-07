#include "imageview.h"

#include <QMouseEvent>
#include <QScrollBar>
#include <QWheelEvent>

#include <algorithm>
#include <cmath>

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
}

// A new image keeps the user's zoom and position (as fractions of the image, so
// images of different sizes - a raw frame, a warped one, a full-resolution one -
// show the same area at the same on-screen size) until the view is fitted again.
void ImageView::setImage(const QImage& img, bool keepView) {
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

void ImageView::fit() {
    if (image_.isNull()) return;
    fitInView(item_, Qt::KeepAspectRatio);
    fitted_ = true;
    emit zoomChanged(zoom());
}

void ImageView::actualSize() {
    resetTransform();
    fitted_ = false;
    emit zoomChanged(1.0);
}

void ImageView::wheelEvent(QWheelEvent* e) {
    if (image_.isNull()) return;
    double f = std::pow(1.0015, e->angleDelta().y());
    double z = zoom() * f;
    if (z < 0.02 || z > 32) return;
    scale(f, f);
    fitted_ = false;
    emit zoomChanged(zoom());
}

void ImageView::mouseDoubleClickEvent(QMouseEvent*) { fit(); }

void ImageView::mouseMoveEvent(QMouseEvent* e) {
    QGraphicsView::mouseMoveEvent(e);
    QPointF p = mapToScene(e->pos());
    emit hovered((int)p.x(), (int)p.y());
}

void ImageView::resizeEvent(QResizeEvent* e) {
    QGraphicsView::resizeEvent(e);
    if (fitted_) fit();
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
