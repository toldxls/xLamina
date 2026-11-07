// Zoomable, pannable image viewer: wheel zooms about the cursor, drag pans,
// double-click fits, 1:1 on request. Shows 8-bit previews of float RGB images.
#pragma once

#include <QGraphicsPixmapItem>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QImage>

class ImageView : public QGraphicsView {
    Q_OBJECT
public:
    explicit ImageView(QWidget* parent = nullptr);
    void setImage(const QImage& img, bool keepView = true);
    void fit();
    void actualSize();
    double zoom() const { return transform().m11(); }
    QImage image() const { return image_; }
    void setLabel(const QString& text);   // the image's name, drawn in the corner so the active output is never in doubt
    void syncFrom(const ImageView* o);    // take o's transform and scroll position (the two-up viewer's shared view)
    bool fitted() const { return fitted_; }

signals:
    void zoomChanged(double z);
    void hovered(int x, int y);

protected:
    void wheelEvent(QWheelEvent* e) override;
    void mouseDoubleClickEvent(QMouseEvent* e) override;
    void mouseMoveEvent(QMouseEvent* e) override;
    void resizeEvent(QResizeEvent* e) override;
    void drawForeground(QPainter* p, const QRectF& rect) override;

private:
    QGraphicsScene scene_;
    QGraphicsPixmapItem* item_ = nullptr;
    QImage image_;
    QString label_;
    bool fitted_ = true;
};

// float RGB (interleaved, [0, 1]) -> 8-bit QImage; `gain` brightens previews
QImage toQImage(const float* rgb, int w, int h, float gain = 1.0f);
