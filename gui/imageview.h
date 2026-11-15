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
#include <QPolygonF>

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
    // The image drawn at another place and size than its own (the two-up viewer lining one image up with another of
    // another crop or scale): its top left at `pos`, each of its pixels `scale` scene units, in a scene of `scene` size
    // (the other image's). clearPlacement() puts it back; a new image keeps the placement until then.
    void setPlacement(QPointF pos, double scale, QSizeF scene);
    void clearPlacement();
    bool placed() const { return placed_; }
    void fit();
    void actualSize();
    double zoom() const { return transform().m11(); }
    QImage image() const { return image_; }
    // a faint frame round the view (the compare window's pane that is painted from); an invalid colour = none
    void setFrameTint(const QColor& c);
    void setLabel(const QString& text);   // the image's name, drawn in the corner so the active output is never in doubt
    void syncFrom(const ImageView* o);    // take o's transform and scroll position (the two-up viewer's shared view)
    bool fitted() const { return fitted_; }
    // drawn over the image on the same pixel grid (the difference highlight, the depth map); null = none
    void setOverlay(const QImage& argb, double opacity = 1.0);
    void setOverlayOpacity(double opacity);
    // an overlay for part of the image: `argb` drawn from image position `pos`, each of its pixels `scale` image pixels
    void setOverlayAt(const QImage& argb, QPointF pos, double scale, double opacity = 1.0);
    // a crosshair instead of the hand (the depth map: which pixel is under the cursor must be plain); dragging still pans
    void setPrecise(bool on);
    // a few lines drawn beside the cursor while it is over the view (the depth map's frame); empty = none
    void setCursorNote(const QString& text);
    // the arrow keys step (stepKey) instead of scrolling the image, and Esc is reported: the depth map's layers
    void setStepKeys(bool on) { stepKeys_ = on; }
    // retouch: the pane draws `img` (owned by the caller, same size as the image) instead of its pixmap; patch(r)
    // repaints r after the caller changed those pixels. mask (w*h, 0..255) is tinted when shown. null = back to the pixmap
    void setLive(const QImage* img, const std::vector<uint8_t>* mask);
    // a second mask tinted with the first when the mask is shown (the pixels a clone stroke changed); null = none.
    // setLive() drops it
    void setSecondMask(const std::vector<uint8_t>* mask);
    void patch(const QRect& r);
    void setMaskShown(bool on);
    // brush mode: the left button paints (signals below, image coordinates), the middle button pans
    void setBrush(bool on, double radius, double hardness);
    double brushRadius() const { return brushRadius_; }
    // the clone brush (the two-up viewer): with setAltPick, Alt + the left button reports a place (altClicked) instead
    // of painting. setCloneMark shows where the brush copies from: mode 1 = a fixed place `v` of the image (picked,
    // not painted from yet), 2 = the brush moved by `v` image pixels (it follows the cursor), 0 = nothing
    void setAltPick(bool on) { altPick_ = on; }
    void setCloneMark(int mode, QPointF v);
    // the width the whole image stands for, in micrometres (0 = not known): gives the scale bar its length and the
    // measured lines theirs
    void setSceneWidthUm(double um);
    void setScaleBar(bool on);
    // The unit lengths are written in, one for the measured lines and one for the scale bar: 0 automatic (µm or mm,
    // whichever reads best), 1 pixels of the image, 2 µm, 3 mm, 4 cm. While the width is not known everything is in
    // pixels, and the bar is drawn only when its unit is pixels. A click on the scale bar steps its unit (barUnitChanged)
    enum Unit { UnitAuto = 0, UnitPx, UnitUm, UnitMm, UnitCm, UnitCount };
    void setMeasureUnit(int u);
    void setBarUnit(int u);
    int measureUnit() const { return measureUnit_; }
    int barUnit() const { return barUnit_; }
    // measure mode: the left button draws a line whose length is written beside it (Shift keeps it level, upright or
    // at 45 degrees), the middle or right button pans, Backspace takes the last line away. The lines stay, at their
    // places in the image, until clearMeasures().
    void setMeasure(bool on);
    bool measuring() const { return measure_; }
    void clearMeasures();
    void addMeasure(const QLineF& frac);  // a line by its ends as fractions of the image (what a drag makes)
    int measureCount() const { return (int)lines_.size(); }
    const std::vector<QLineF>& measures() const { return lines_; }
    void setMeasures(const std::vector<QLineF>& lines);   // another view's lines (the two-up viewer): no signal
    // regions (the two-up viewer's Refine): with the tool on, the left button drags a rectangle (mode 1) or draws a
    // closed outline (mode 2), reported by regionDrawn; the middle or right button pans. The regions on show are the
    // owner's: outlines as fractions of the image, numbered from 1, drawn by their state
    struct RegionMark {
        QPolygonF frac;
        int state = 0;        // 0 marked, 1 being rendered, 2 rendered and applied, 3 rendered and switched off, 4 failed
    };
    void setRegionTool(int mode);   // 0 = off
    int regionTool() const { return region_; }
    void setRegions(const std::vector<RegionMark>& marks, int selected = -1);

signals:
    void barUnitChanged(int unit);        // the scale bar was clicked: its unit stepped to this one
    void regionDrawn(QPolygonF frac);     // an outline was finished, as fractions of the image
    void measured(QString length);        // a line was drawn: its length in words
    void measuresChanged();               // a line was drawn, removed, or all were cleared (not by setMeasures)
    void clicked(int x, int y);           // the left button pressed and released in one place (not a drag): image coordinates
    void stepKey(int delta);              // an arrow key while setStepKeys(true): -1 = up or left, +1 = down or right
    void escapePressed();
    void zoomChanged(double z);
    void viewChanged();                   // zoom, scroll or size: what the viewport shows has changed
    void hovered(int x, int y);           // image coordinates under the cursor; (-1, -1) when it leaves the view
    void altClicked(QPointF imagePos);    // Alt + the left button in brush mode, while setAltPick(true)
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
    QColor tint_;
    QString note_;
    bool precise_ = false;
    bool fitted_ = true;
    bool winActive_ = false;              // the pixmap item shows a live-view crop, placed and scaled (setLiveWindow)
    bool brush_ = false, painting_ = false, panning_ = false;
    bool altPick_ = false;
    int cloneMode_ = 0;
    QPointF cloneV_;
    double brushRadius_ = 60, brushHardness_ = 0.5;
    QPoint lastPan_;
    QPointF cursor_;   // viewport position of the mouse, for the brush circle
    bool cursorIn_ = false;
    bool stepKeys_ = false;
    QPoint pressAt_;
    bool pressed_ = false;
    double sceneUm_ = 0;
    bool scaleBar_ = true, measure_ = false, drawing_ = false;
    std::vector<QLineF> lines_;           // as fractions of the image's width and height, so they hold for any size of it
    QLineF cur_;                          // the line being drawn
    bool placed_ = false;                 // setPlacement is in effect
    QPointF placePos_;
    double placeScale_ = 1.0;
    QSizeF placeScene_;
    int region_ = 0;                      // the region tool: 0 off, 1 rectangle, 2 lasso
    bool outlining_ = false;
    QPointF regionStart_;                 // viewport position of the press
    QPolygonF curRegion_;                 // the outline being drawn, as fractions
    std::vector<RegionMark> marks_;
    int markSel_ = -1;
    void drawRegion(QPainter* p, const QPolygonF& frac, int state, int number, bool selected) const;
    bool hasImage() const;
    QPointF toFrac(QPointF scene) const;
    QPointF fromFrac(QPointF frac) const; // -> viewport
    QString lengthText(const QLineF& frac) const;
    QString unitText(double px, int unit) const;   // a length of px image pixels in a unit
    int measureUnit_ = UnitAuto, barUnit_ = UnitAuto;
    QRectF barRect_;                      // the scale bar's plate in the viewport while it is drawn (a click steps its unit)
    void drawMeasure(QPainter* p, const QLineF& frac, bool live) const;
};

// float RGB (interleaved, [0, 1]) -> 8-bit QImage; `gain` brightens previews
QImage toQImage(const float* rgb, int w, int h, float gain = 1.0f);
