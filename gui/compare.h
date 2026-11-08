// Two-up viewer: two panes with one pan and zoom, for judging B variants and other stackers' outputs at 1:1.
// Images come from the main window (its outputs and comparison images), from files, and from a sweep folder
// (stackcli sweep: sweep.json lists the variants, each with its B.tif and options). Space swaps the two panes.
// Retouch mode: the left pane becomes a result painted from the right pane's image with a soft brush, on the
// 16-bit data, saved as a 16-bit TIFF with the frames' profile.
#pragma once

#include <QColorSpace>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QImage>
#include <QLabel>
#include <QList>
#include <QMainWindow>
#include <QSplitter>
#include <QTableWidget>

#include <cstdint>
#include <vector>

#include "imageview.h"
#include "worker.h"

class CompareWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit CompareWindow(QWidget* parent = nullptr);
    ~CompareWindow() override;
    // path, when given, is the 16-bit file behind an in-memory image (the render's outputs): retouching reads it
    void addImage(const QString& name, const QImage& img, const QString& note = {}, const QString& path = {});
    void addFile(const QString& name, const QString& path, const QString& note = {}, double change = -1);   // read when first shown
    void openSweep(const QString& dir);                                                 // dir/sweep.json
    void select(int pane, const QString& name);

protected:
    void keyPressEvent(QKeyEvent* e) override;
    void keyReleaseEvent(QKeyEvent* e) override;
    void dragEnterEvent(QDragEnterEvent* e) override;
    void dropEvent(QDropEvent* e) override;
    void closeEvent(QCloseEvent* e) override;

private slots:
    void onLoaded(int index, QImage image);

private:
    struct Entry {
        QString name, path, note;
        QImage image;      // empty until loaded (files) or always set (addImage)
        bool loading = false;
        bool fileBacked = true;   // false: the image came in memory (path, if any, is only for the 16-bit data)
        double change = -1;   // a sweep variant: % of B's pixels differing > 2 % from the base (sweep.json change_pct); -1 = unknown
    };
    int indexOf(const QString& name) const;
    void addEntry(Entry e);
    void rebuildList();
    void showIn(int pane, int index);
    void refresh(int pane);
    void flip();
    void step(int pane, int delta);
    void requestLoad(int index);
    void startNextLoad();
    void trimCache();
    void openImageDialog();
    void openSweepDialog();
    void updateOverlay();        // the difference highlight over both panes, when it is on and both images are loaded

    // ---- retouch
    struct Stroke { QRect r; std::vector<uint8_t> mask; };   // the mask before the stroke, inside r
    struct Retouch {
        bool on = false;
        int w = 0, h = 0;
        int target = -1, source = -1;
        std::vector<uint16_t> tgt16, src16, res16;   // RGB interleaved
        QImage tgtDisp, srcDisp, resDisp;            // 8-bit sRGB
        std::vector<uint8_t> mask;
        std::vector<Stroke> undo;
        Stroke current;
        bool inStroke = false, dirty = false, peek = false;
        QPointF last;
        std::vector<uint8_t> icc;
    } rt_;
    bool load16(const Entry& e, std::vector<uint16_t>& out, int& w, int& h, std::vector<uint8_t>* icc);
    void retouchToggle(bool on);
    bool retouchSetSource(int index);   // false when the source does not fit the target
    void retouchBake();                 // commit the strokes: the result becomes the new target
    void dab(QPointF p);
    void composite(const QRect& r);
    void growSnapshot(const QRect& r);
    void retouchUndo();
    void retouchReset();
    void retouchSave();
    void retouchStatus();

    QList<Entry> entries_;
    QTableWidget* list_ = nullptr;
    QSplitter* split_ = nullptr;
    ImageView* view_[2] = {nullptr, nullptr};
    QComboBox* pick_[2] = {nullptr, nullptr};
    int sel_[2] = {-1, -1};
    QList<int> loadQueue_;
    int loading_ = -1;
    ThumbLoader loader_;
    QList<int> recent_;          // file-backed entries with a decoded image, most recent last
    bool syncing_ = false;
    QLabel* status_ = nullptr;
    QAction* diffAct_ = nullptr;         // "Difference": highlight where the two panes differ
    QComboBox* diffLevel_ = nullptr;     // the difference (in % of full scale) from which a pixel is highlighted
    int overlayFor_[2] = {-1, -1};       // the entries the current overlay was computed from
    QAction* flipAct_ = nullptr;
    QAction* retouchAct_ = nullptr;
    QAction* maskAct_ = nullptr;
    QAction* undoAct_ = nullptr;
    QAction* resetAct_ = nullptr;
    QAction* saveAct_ = nullptr;
    QDoubleSpinBox* brushSize_ = nullptr;
    QDoubleSpinBox* brushHard_ = nullptr;
};
