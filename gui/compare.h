// Two-up viewer: two panes with one pan and zoom, for judging B variants and other stackers' outputs at 1:1.
// Images come from the main window (its outputs and comparison images), from files, and from a sweep folder
// (stackcli sweep: sweep.json lists the variants, each with its B.tif and options). Space swaps the two panes.
#pragma once

#include <QColorSpace>
#include <QComboBox>
#include <QImage>
#include <QLabel>
#include <QList>
#include <QMainWindow>
#include <QSplitter>
#include <QTableWidget>

#include "imageview.h"
#include "worker.h"

class CompareWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit CompareWindow(QWidget* parent = nullptr);
    ~CompareWindow() override;
    void addImage(const QString& name, const QImage& img, const QString& note = {});   // already decoded (sRGB)
    void addFile(const QString& name, const QString& path, const QString& note = {}, double change = -1);   // read when first shown
    void openSweep(const QString& dir);                                                 // dir/sweep.json
    void select(int pane, const QString& name);

protected:
    void keyPressEvent(QKeyEvent* e) override;
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
};
