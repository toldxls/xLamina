// Zoomable, pannable image viewer: wheel zooms about the cursor, drag pans,
// double-click fits, 1:1 on request. Shows 8-bit previews of float RGB images.
// The two-up viewer's retouch mode paints on it: a live image drawn straight
// from a QImage the owner edits (patch() repaints a rectangle, no copies), a
// brush cursor, and stroke signals in image coordinates.
#pragma once

#include <QGraphicsItem>
#include <QGraphicsPixmapItem>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QImage>

#include <vector>

class LiveImageItem;

class ImageView : public QGraphicsView {
    Q_OBJECT
public:
    explicit ImageView(QWidget* parent = nullptr);
    void setImage(const QImage& img, bool keepView = true);
    // the live view's window (0.6.33): `img` shows the crop `win` of a `full`-sized frame; the scene stays the full
    // frame, so zoom and position mean the same as for the whole image, and the crop is drawn at its place and scale
    void setLiveWindow(const QImage& img, const QRect& win, const QSize& full);
    QRectF visibleSceneRect() const;      // what the viewport shows, in scene (image) coordinates
    void fit();
    void actualSize();
    double zoom() const { return transform().m11(); }
    QImage image() const { return image_; }
    void setLabel(const QString& text);   // the image's name, drawn in the corner so the active output is never in doubt
    void syncFrom(const ImageView* o);    // take o's transform and scroll position (the two-up viewer's shared view)
    bool fitted() const { return fitted_; }
    // drawn over the image on the same pixel grid (the difference highlight, the depth map); null = none
    void setOverlay(const QImage& argb, double opacity = 1.0);
    void setOverlayOpacity(double opacity);
    // retouch: the pane draws `img` (owned by the caller, same size as the image) instead of its pixmap; patch(r)
    // repaints r after the caller changed those pixels. mask (w*h, 0..255) is tinted when shown. null = back to the pixmap
    void setLive(const QImage* img, const std::vector<uint8_t>* mask);
    void patch(const QRect& r);
    void setMaskShown(bool on);
    // brush mode: the left button paints (signals below, image coordinates), the middle button pans
    void setBrush(bool on, double radius, double hardness);
    double brushRadius() const { return brushRadius_; }

signals:
    void zoomChanged(double z);
    void viewChanged();                   // zoom, scroll or size: what the viewport shows has changed
    void hovered(int x, int y);
    void strokeBegin(QPointF imagePos);
    void strokeTo(QPointF imagePos);
    void strokeEnd();
    void brushResized(double radius);   // Ctrl + wheel or [ ]

protected:
    void wheelEvent(QWheelEvent* e) override;
    void mousePressEvent(QMouseEvent* e) override;
    void mouseMoveEvent(QMouseEvent* e) override;
    void mouseReleaseEvent(QMouseEvent* e) override;
    void mouseDoubleClickEvent(QMouseEvent* e) override;
    void keyPressEvent(QKeyEvent* e) override;
    void leaveEvent(QEvent* e) override;
    void resizeEvent(QResizeEvent* e) override;
    void drawForeground(QPainter* p, const QRectF& rect) override;

private:
    QGraphicsScene scene_;
    QGraphicsPixmapItem* item_ = nullptr;
    QGraphicsPixmapItem* overlay_ = nullptr;
    LiveImageItem* live_ = nullptr;
    QImage image_;
    QString label_;
    bool fitted_ = true;
    bool winActive_ = false;              // the pixmap item shows a live-view crop, placed and scaled (setLiveWindow)
    bool brush_ = false, painting_ = false, panning_ = false;
    double brushRadius_ = 60, brushHardness_ = 0.5;
    QPoint lastPan_;
    QPointF cursor_;   // viewport position of the mouse, for the brush circle
    bool cursorIn_ = false;
};

// float RGB (interleaved, [0, 1]) -> 8-bit QImage; `gain` brightens previews
QImage toQImage(const float* rgb, int w, int h, float gain = 1.0f);
