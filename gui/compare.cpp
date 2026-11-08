#include "compare.h"

#include <QAction>
#include <QApplication>
#include <QDir>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QFileDialog>
#include <QFileInfo>
#include <QHeaderView>
#include <QImageReader>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QKeyEvent>
#include <QMessageBox>
#include <QMimeData>
#include <QScrollBar>
#include <QSettings>
#include <QStatusBar>
#include <QToolBar>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "stackcore/compose.h"
#include "stackcore/tiff.h"

using stackcore::TiffInfo;
using stackcore::tiff_probe;

namespace {
const int kKeepDecoded = 8;   // file-backed images held in memory (about 120 MB each at full resolution)
const int kUndoDepth = 40;
}

CompareWindow::CompareWindow(QWidget* parent) : QMainWindow(parent) {
    setWindowFlag(Qt::Window);
    setWindowTitle("xLamina — two-up");
    setAcceptDrops(true);
    resize(1500, 900);

    QToolBar* tb = addToolBar("Compare");
    tb->setMovable(false);
    QAction* openSweep = tb->addAction("Open sweep…", this, &CompareWindow::openSweepDialog);
    openSweep->setStatusTip("A folder written by a sweep (holds sweep.json): its B variants join the list.");
    QAction* openImage = tb->addAction("Add image…", this, &CompareWindow::openImageDialog);
    openImage->setStatusTip("Any TIFF, JPEG or PNG, for example another stacker's output.");
    tb->addSeparator();
    flipAct_ = tb->addAction("Flip", this, &CompareWindow::flip);
    flipAct_->setShortcuts({QKeySequence(Qt::Key_Space), QKeySequence("F")});
    flipAct_->setStatusTip("Swap the two panes (Space): the image changes under your eye, at the same zoom and position.");
    QAction* fit = tb->addAction("Fit", this, [this] { view_[0]->fit(); });
    fit->setShortcut(QKeySequence("Ctrl+0"));
    QAction* one = tb->addAction("1:1", this, [this] { view_[0]->actualSize(); });
    one->setShortcut(QKeySequence("Ctrl+1"));
    tb->addSeparator();
    diffAct_ = tb->addAction("Difference");
    diffAct_->setCheckable(true);
    diffAct_->setShortcut(QKeySequence("D"));
    diffAct_->setStatusTip("Tint the pixels where the two panes differ (D): orange, the stronger the larger the difference. The status bar gives the share.");
    connect(diffAct_, &QAction::toggled, this, [this](bool) { overlayFor_[0] = overlayFor_[1] = -1; updateOverlay(); });
    diffLevel_ = new QComboBox;
    for (auto [label, pct] : std::initializer_list<std::pair<const char*, double>>{{"above 0.5 %", 0.5}, {"above 1 %", 1.0}, {"above 2 %", 2.0}, {"above 5 %", 5.0}, {"above 10 %", 10.0}})
        diffLevel_->addItem(label, pct);
    diffLevel_->setCurrentIndex(1);
    diffLevel_->setToolTip("A pixel is highlighted when its luma differs by more than this share of full scale; fully orange at four times it.");
    connect(diffLevel_, QOverload<int>::of(&QComboBox::activated), this, [this](int) { overlayFor_[0] = overlayFor_[1] = -1; updateOverlay(); });
    tb->addWidget(diffLevel_);
    tb->addSeparator();
    QAction* vertical = tb->addAction("Stack vertically");
    vertical->setCheckable(true);
    vertical->setStatusTip("Put the right pane under the left one (for wide crops).");
    connect(vertical, &QAction::toggled, this, [this](bool on) { split_->setOrientation(on ? Qt::Vertical : Qt::Horizontal); });
    QAction* next = new QAction(this);
    next->setShortcut(QKeySequence("PgDown"));
    connect(next, &QAction::triggered, this, [this] { step(1, 1); });
    addAction(next);
    QAction* prev = new QAction(this);
    prev->setShortcut(QKeySequence("PgUp"));
    connect(prev, &QAction::triggered, this, [this] { step(1, -1); });
    addAction(prev);

    // ---- retouch
    QToolBar* rb = addToolBar("Retouch");
    rb->setMovable(false);
    retouchAct_ = rb->addAction("Retouch");
    retouchAct_->setCheckable(true);
    retouchAct_->setShortcut(QKeySequence("R"));
    retouchAct_->setStatusTip("Paint the right pane's image into the left one (R). Left button paints, middle button pans, [ ] or Ctrl+wheel size the brush, Space peeks at the source.");
    connect(retouchAct_, &QAction::toggled, this, &CompareWindow::retouchToggle);
    rb->addWidget(new QLabel(" Size "));
    brushSize_ = new QDoubleSpinBox;
    brushSize_->setRange(2, 2000);
    brushSize_->setDecimals(0);
    brushSize_->setSuffix(" px");
    brushSize_->setValue(60);
    brushSize_->setToolTip("Brush radius in image pixels ([ and ], or Ctrl + wheel over the image).");
    connect(brushSize_, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { if (rt_.on) view_[0]->setBrush(true, v, brushHard_->value()); });
    rb->addWidget(brushSize_);
    rb->addWidget(new QLabel(" Hardness "));
    brushHard_ = new QDoubleSpinBox;
    brushHard_->setRange(0, 1);
    brushHard_->setSingleStep(0.1);
    brushHard_->setDecimals(1);
    brushHard_->setValue(0.5);
    brushHard_->setToolTip("Share of the radius painted at full strength; the rest fades out.");
    connect(brushHard_, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { if (rt_.on) view_[0]->setBrush(true, brushSize_->value(), v); });
    rb->addWidget(brushHard_);
    rb->addSeparator();
    maskAct_ = rb->addAction("Show mask");
    maskAct_->setCheckable(true);
    maskAct_->setShortcut(QKeySequence("M"));
    maskAct_->setStatusTip("Tint the painted pixels green (M).");
    connect(maskAct_, &QAction::toggled, this, [this](bool on) { view_[0]->setMaskShown(on); });
    undoAct_ = rb->addAction("Undo", this, &CompareWindow::retouchUndo);
    undoAct_->setShortcut(QKeySequence::Undo);
    resetAct_ = rb->addAction("Reset", this, &CompareWindow::retouchReset);
    resetAct_->setStatusTip("Drop every stroke: back to the left pane's original image.");
    saveAct_ = rb->addAction("Save result…", this, &CompareWindow::retouchSave);
    saveAct_->setShortcut(QKeySequence::Save);
    saveAct_->setStatusTip("Write the retouched image as a 16-bit TIFF with the frames' colour profile, and add it to the list.");
    for (QAction* a : {maskAct_, undoAct_, resetAct_, saveAct_}) a->setEnabled(false);
    brushSize_->setEnabled(false);
    brushHard_->setEnabled(false);

    // the list of images: name, change vs base, and the sweep options (or where it came from)
    list_ = new QTableWidget(0, 3);
    list_->setHorizontalHeaderLabels({"Image", "Change", "Options"});
    list_->horizontalHeaderItem(1)->setToolTip("Share of B's pixels that differ from the base variant by more than 2 % of full scale: how much there is to see.");
    list_->horizontalHeader()->setStretchLastSection(true);
    list_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    list_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    list_->verticalHeader()->setVisible(false);
    list_->setSelectionBehavior(QAbstractItemView::SelectRows);
    list_->setSelectionMode(QAbstractItemView::SingleSelection);
    list_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    list_->setToolTip("Click: show in the right pane. Double-click: show in the left pane. PgUp / PgDown step the right pane.");
    connect(list_, &QTableWidget::cellClicked, this, [this](int row, int) { showIn(1, row); });
    connect(list_, &QTableWidget::cellDoubleClicked, this, [this](int row, int) { showIn(0, row); });

    split_ = new QSplitter(Qt::Horizontal);
    for (int p = 0; p < 2; ++p) {
        QWidget* w = new QWidget;
        QVBoxLayout* l = new QVBoxLayout(w);
        l->setContentsMargins(0, 0, 0, 0);
        l->setSpacing(2);
        pick_[p] = new QComboBox;
        pick_[p]->setToolTip(p == 0 ? "The left pane's image (the result, when retouching)" : "The right pane's image (the source, when retouching)");
        connect(pick_[p], QOverload<int>::of(&QComboBox::activated), this, [this, p](int i) { showIn(p, i); });
        view_[p] = new ImageView;
        l->addWidget(pick_[p]);
        l->addWidget(view_[p], 1);
        split_->addWidget(w);
    }
    split_->setSizes({1, 1});
    // one pan and zoom for both: every scroll or zoom of one pane is applied to the other
    for (int a = 0; a < 2; ++a) {
        int b = 1 - a;
        connect(view_[a]->horizontalScrollBar(), &QScrollBar::valueChanged, this, [this, b](int v) {
            if (syncing_) return;
            syncing_ = true; view_[b]->horizontalScrollBar()->setValue(v); syncing_ = false;
        });
        connect(view_[a]->verticalScrollBar(), &QScrollBar::valueChanged, this, [this, b](int v) {
            if (syncing_) return;
            syncing_ = true; view_[b]->verticalScrollBar()->setValue(v); syncing_ = false;
        });
        connect(view_[a], &ImageView::zoomChanged, this, [this, a, b](double) {
            if (syncing_ || (view_[b]->image().isNull() && !rt_.on)) return;
            syncing_ = true; view_[b]->syncFrom(view_[a]); syncing_ = false;
        });
    }
    // the brush paints on the left pane
    connect(view_[0], &ImageView::strokeBegin, this, [this](QPointF p) {
        if (!rt_.on) return;
        rt_.inStroke = true;
        rt_.current = Stroke{};
        rt_.last = p;
        dab(p);
    });
    connect(view_[0], &ImageView::strokeTo, this, [this](QPointF p) {
        if (!rt_.on || !rt_.inStroke) return;
        // dabs along the way, a quarter radius apart
        double r = std::max(1.0, view_[0]->brushRadius() / 4);
        QPointF d = p - rt_.last;
        double len = std::hypot(d.x(), d.y());
        if (len < r) return;
        int n = (int)(len / r);
        for (int i = 1; i <= n; ++i) dab(rt_.last + d * (double(i) / n));
        rt_.last = p;
    });
    connect(view_[0], &ImageView::strokeEnd, this, [this] {
        if (!rt_.on || !rt_.inStroke) return;
        rt_.inStroke = false;
        if (!rt_.current.r.isEmpty()) {
            rt_.undo.push_back(std::move(rt_.current));
            if ((int)rt_.undo.size() > kUndoDepth) rt_.undo.erase(rt_.undo.begin());
            rt_.dirty = true;
        }
        rt_.current = Stroke{};
        undoAct_->setEnabled(!rt_.undo.empty());
        retouchStatus();
    });
    connect(view_[0], &ImageView::brushResized, this, [this](double r) { brushSize_->blockSignals(true); brushSize_->setValue(r); brushSize_->blockSignals(false); retouchStatus(); });

    QSplitter* outer = new QSplitter(Qt::Horizontal);
    outer->addWidget(list_);
    outer->addWidget(split_);
    outer->setStretchFactor(0, 0);
    outer->setStretchFactor(1, 1);
    outer->setSizes({340, 1100});
    setCentralWidget(outer);
    status_ = new QLabel;
    statusBar()->addWidget(status_, 1);
    statusBar()->showMessage("Open a sweep folder or add images, then pick one for each pane. Space flips them; R retouches the left one from the right.");

    connect(&loader_, &ThumbLoader::thumbnail, this, &CompareWindow::onLoaded);
    QSettings s;
    restoreGeometry(s.value("compare/geometry").toByteArray());
    if (s.contains("compare/outer")) outer->restoreState(s.value("compare/outer").toByteArray());
}

CompareWindow::~CompareWindow() { loader_.stop(); }

void CompareWindow::closeEvent(QCloseEvent* e) {
    if (rt_.on && rt_.dirty) {
        auto r = QMessageBox::question(this, "xLamina", "The retouch has unsaved strokes. Close anyway?", QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
        if (r != QMessageBox::Yes) { e->ignore(); return; }
    }
    QSettings s;
    s.setValue("compare/geometry", saveGeometry());
    if (auto* outer = qobject_cast<QSplitter*>(centralWidget())) s.setValue("compare/outer", outer->saveState());
    QMainWindow::closeEvent(e);
}

// ----------------------------------------------------------------------------- the entries

int CompareWindow::indexOf(const QString& name) const {
    for (int i = 0; i < entries_.size(); ++i) if (entries_[i].name == name) return i;
    return -1;
}

void CompareWindow::addEntry(Entry e) {
    int i = indexOf(e.name);
    if (i >= 0) {
        // the same name again: a new render's output, the same file re-opened, or a sweep opened twice -> replace.
        // The same file name from another folder (B.tif of two renders) is a different image: keep both, the
        // newcomer named by its folder (Travis, 2026-11-08)
        const bool same_source = e.path.isEmpty() || entries_[i].path.isEmpty() ||
                                 QFileInfo(e.path).absoluteFilePath() == QFileInfo(entries_[i].path).absoluteFilePath();
        if (same_source) {
            bool shown = sel_[0] == i || sel_[1] == i;
            entries_[i] = std::move(e);
            rebuildList();
            if (shown) { if (sel_[0] == i) refresh(0); if (sel_[1] == i) refresh(1); }
            return;
        }
        QFileInfo fi(e.path);
        QString named = QFileInfo(fi.absolutePath()).fileName() + "/" + e.name;
        if (indexOf(named) >= 0) named = QDir::toNativeSeparators(fi.absoluteFilePath());
        e.name = named;
        if (int j = indexOf(e.name); j >= 0) {   // that very file again
            bool shown = sel_[0] == j || sel_[1] == j;
            entries_[j] = std::move(e);
            rebuildList();
            if (shown) { if (sel_[0] == j) refresh(0); if (sel_[1] == j) refresh(1); }
            return;
        }
    }
    entries_.push_back(std::move(e));
    rebuildList();
}

void CompareWindow::addImage(const QString& name, const QImage& img, const QString& note, const QString& path) {
    if (img.isNull()) return;
    Entry e;
    e.name = name; e.note = note; e.image = img; e.path = path; e.fileBacked = false;
    addEntry(std::move(e));
}

void CompareWindow::addFile(const QString& name, const QString& path, const QString& note, double change) {
    Entry e;
    e.name = name; e.path = path; e.note = note; e.change = change;
    addEntry(std::move(e));
}

void CompareWindow::rebuildList() {
    list_->setRowCount(entries_.size());
    for (int i = 0; i < entries_.size(); ++i) {
        auto* n = new QTableWidgetItem(entries_[i].name);
        n->setToolTip(entries_[i].path.isEmpty() ? entries_[i].note : QDir::toNativeSeparators(entries_[i].path));
        list_->setItem(i, 0, n);
        auto* c = new QTableWidgetItem(entries_[i].change < 0 ? QString() : entries_[i].change == 0 ? QString("base") : QString("%1 %").arg(entries_[i].change, 0, 'f', 1));
        c->setTextAlignment(Qt::AlignRight | Qt::AlignVCenter);
        list_->setItem(i, 1, c);
        list_->setItem(i, 2, new QTableWidgetItem(entries_[i].note));
    }
    for (int p = 0; p < 2; ++p) {
        pick_[p]->blockSignals(true);
        pick_[p]->clear();
        for (auto& e : entries_) pick_[p]->addItem(e.name);
        pick_[p]->setCurrentIndex(sel_[p]);
        pick_[p]->blockSignals(false);
    }
}

// sweep.json: {"variants": [{"name", "dir", "options", "rc", "seconds", "change_pct"}, ...]}
void CompareWindow::openSweep(const QString& dir) {
    QFile f(QDir(dir).filePath("sweep.json"));
    if (!f.open(QIODevice::ReadOnly)) { QMessageBox::warning(this, "xLamina", QString("%1 holds no sweep.json.").arg(QDir::toNativeSeparators(dir))); return; }
    QJsonParseError err;
    QJsonDocument doc = QJsonDocument::fromJson(f.readAll(), &err);
    if (!doc.isObject()) { QMessageBox::warning(this, "xLamina", QString("sweep.json: %1").arg(err.errorString())); return; }
    int added = 0, failed = 0;
    // the base first, then the variants with the most to see
    struct V { QString name, b, opts; double change; };
    QList<V> vs;
    for (const QJsonValue& v : doc.object().value("variants").toArray()) {
        QJsonObject o = v.toObject();
        if (o.value("rc").toInt(1) != 0) { ++failed; continue; }
        QString vdir = o.value("dir").toString();
        if (QDir::isRelativePath(vdir)) vdir = QDir(dir).filePath(vdir);
        QString b = QDir(vdir).filePath("B.tif");
        if (!QFileInfo::exists(b)) { ++failed; continue; }
        vs.push_back({o.value("name").toString(), b, o.value("options").toString(), o.value("change_pct").toDouble(-1)});
    }
    std::stable_sort(vs.begin(), vs.end(), [](const V& a, const V& b) {
        if (a.change == 0 || b.change == 0) return a.change == 0 && b.change != 0;   // the base on top
        return a.change > b.change;
    });
    for (const V& v : vs) { addFile(v.name, v.b, v.opts.isEmpty() ? "base settings" : v.opts, v.change); ++added; }
    statusBar()->showMessage(QString("%1: %2 variants%3").arg(QDir::toNativeSeparators(dir)).arg(added).arg(failed ? QString(" (%1 without a B)").arg(failed) : ""), 8000);
    if (added && sel_[1] < 0) {   // first use: the first two variants side by side
        int first = entries_.size() - added;
        showIn(0, first);
        if (added > 1) showIn(1, first + 1);
    }
}

void CompareWindow::openSweepDialog() {
    QSettings s;
    QString d = QFileDialog::getExistingDirectory(this, "Sweep folder (holds sweep.json)", s.value("compare/lastSweep").toString());
    if (d.isEmpty()) return;
    s.setValue("compare/lastSweep", d);
    openSweep(d);
}

void CompareWindow::openImageDialog() {
    QSettings s;
    QStringList fns = QFileDialog::getOpenFileNames(this, "Images to compare", s.value("compare/lastImage").toString(), "Images (*.tif *.tiff *.jpg *.jpeg *.png);;All files (*)");
    if (fns.isEmpty()) return;
    s.setValue("compare/lastImage", QFileInfo(fns.first()).absolutePath());
    for (const QString& fn : fns) addFile(QFileInfo(fn).fileName(), fn, QDir::toNativeSeparators(QFileInfo(fn).absolutePath()));
    if (sel_[1] < 0) showIn(1, entries_.size() - 1);
}

// ----------------------------------------------------------------------------- showing

void CompareWindow::select(int pane, const QString& name) {
    int i = indexOf(name);
    if (i >= 0) showIn(pane, i);
}

void CompareWindow::showIn(int pane, int index) {
    if (index < 0 || index >= entries_.size()) return;
    if (rt_.on && pane == 0) {   // the result pane is the retouch itself
        statusBar()->showMessage("The left pane is the retouch result; leave Retouch to change it.", 4000);
        pick_[0]->blockSignals(true); pick_[0]->setCurrentIndex(sel_[0]); pick_[0]->blockSignals(false);
        return;
    }
    sel_[pane] = index;
    pick_[pane]->blockSignals(true);
    pick_[pane]->setCurrentIndex(index);
    pick_[pane]->blockSignals(false);
    refresh(pane);
}

void CompareWindow::refresh(int pane) {
    int i = sel_[pane];
    if (i < 0) return;
    Entry& e = entries_[i];
    if (e.image.isNull()) {
        view_[pane]->setLabel(e.name + " — loading…");
        requestLoad(i);
        return;
    }
    if (e.fileBacked) { recent_.removeAll(i); recent_.push_back(i); }
    syncing_ = true;   // a new image must not re-fit the other pane: it takes the other pane's view instead
    view_[pane]->setImage(e.image, true);
    int other = 1 - pane;
    if (!view_[other]->image().isNull()) view_[pane]->syncFrom(view_[other]);
    syncing_ = false;
    view_[pane]->setLabel(e.name);
    status_->setText(QString("%1  |  %2").arg(sel_[0] >= 0 ? entries_[sel_[0]].name : "—", sel_[1] >= 0 ? entries_[sel_[1]].name : "—"));
    trimCache();
    updateOverlay();
    if (rt_.on && pane == 1 && i != rt_.source) retouchSetSource(i);
}

// |luma(left) - luma(right)| as an orange tint: transparent up to the chosen level, opaque at four times it.
// Both panes get the same overlay, so the highlight sits on the pixels in question whichever image is under it.
void CompareWindow::updateOverlay() {
    if (!diffAct_->isChecked() || sel_[0] < 0 || sel_[1] < 0 || entries_[sel_[0]].image.isNull() || entries_[sel_[1]].image.isNull()) {
        view_[0]->setOverlay(QImage());
        view_[1]->setOverlay(QImage());
        overlayFor_[0] = overlayFor_[1] = -1;
        return;
    }
    if ((overlayFor_[0] == sel_[0] && overlayFor_[1] == sel_[1]) || (overlayFor_[0] == sel_[1] && overlayFor_[1] == sel_[0])) return;   // a flip keeps it
    const QImage& a = entries_[sel_[0]].image;
    const QImage& b = entries_[sel_[1]].image;
    if (a.size() != b.size()) {
        view_[0]->setOverlay(QImage());
        view_[1]->setOverlay(QImage());
        statusBar()->showMessage("Difference: the two images are not the same size.", 5000);
        return;
    }
    const double pct = diffLevel_->currentData().toDouble();
    const int t0 = std::max(1, (int)std::lround(pct / 100.0 * 255.0));   // 8-bit levels
    const int t1 = std::max(t0 + 1, 4 * t0);
    QImage ov(a.size(), QImage::Format_ARGB32);
    size_t hits = 0;
    const int w = a.width(), h = a.height();
    for (int y = 0; y < h; ++y) {
        const QRgb* pa = reinterpret_cast<const QRgb*>(a.constScanLine(y));
        const QRgb* pb = reinterpret_cast<const QRgb*>(b.constScanLine(y));
        QRgb* po = reinterpret_cast<QRgb*>(ov.scanLine(y));
        for (int x = 0; x < w; ++x) {
            // luma in 8-bit: 0.2126 R + 0.7152 G + 0.0722 B, in fixed point
            int la = (54 * qRed(pa[x]) + 183 * qGreen(pa[x]) + 19 * qBlue(pa[x])) >> 8;
            int lb = (54 * qRed(pb[x]) + 183 * qGreen(pb[x]) + 19 * qBlue(pb[x])) >> 8;
            int d = std::abs(la - lb);
            if (d <= t0) { po[x] = 0; continue; }
            ++hits;
            int alpha = std::min(230, 40 + (d - t0) * 190 / (t1 - t0));
            po[x] = qRgba(255, 140, 0, alpha);
        }
    }
    view_[0]->setOverlay(ov);
    view_[1]->setOverlay(ov);
    overlayFor_[0] = sel_[0];
    overlayFor_[1] = sel_[1];
    statusBar()->showMessage(QString("Difference: %1 % of pixels differ by more than %2 % (%3 vs %4)")
                                 .arg(100.0 * hits / ((double)w * h), 0, 'f', 2).arg(pct).arg(entries_[sel_[0]].name, entries_[sel_[1]].name));
}

void CompareWindow::flip() {
    if (rt_.on || sel_[0] < 0 || sel_[1] < 0) return;
    std::swap(sel_[0], sel_[1]);
    for (int p = 0; p < 2; ++p) {
        pick_[p]->blockSignals(true);
        pick_[p]->setCurrentIndex(sel_[p]);
        pick_[p]->blockSignals(false);
        refresh(p);
    }
}

void CompareWindow::step(int pane, int delta) {
    if (entries_.isEmpty()) return;
    int i = sel_[pane] < 0 ? 0 : sel_[pane] + delta;
    if (i < 0) i = entries_.size() - 1;
    if (i >= entries_.size()) i = 0;
    showIn(pane, i);
}

void CompareWindow::keyPressEvent(QKeyEvent* e) {
    if (e->key() == Qt::Key_Space) {
        if (rt_.on) {   // peek at the source in the result pane while Space is held
            if (!e->isAutoRepeat() && !rt_.peek) { rt_.peek = true; view_[0]->setLive(&rt_.srcDisp, nullptr); view_[0]->setLabel(entries_[rt_.source].name + "  (source)"); }
        } else flip();
        return;
    }
    QMainWindow::keyPressEvent(e);
}

void CompareWindow::keyReleaseEvent(QKeyEvent* e) {
    if (e->key() == Qt::Key_Space && rt_.on && rt_.peek && !e->isAutoRepeat()) {
        rt_.peek = false;
        view_[0]->setLive(&rt_.resDisp, &rt_.mask);
        view_[0]->setLabel(entries_[rt_.target].name + "  (retouch)");
        return;
    }
    QMainWindow::keyReleaseEvent(e);
}

// ----------------------------------------------------------------------------- loading

void CompareWindow::requestLoad(int index) {
    if (entries_[index].loading || loadQueue_.contains(index)) return;
    loadQueue_.push_back(index);
    startNextLoad();
}

void CompareWindow::startNextLoad() {
    if (loading_ >= 0 || loadQueue_.isEmpty()) return;
    loading_ = loadQueue_.takeFirst();
    Entry& e = entries_[loading_];
    e.loading = true;
    TiffInfo info = tiff_probe(e.path.toStdWString());
    if (info.ok() && info.samples >= 3) { loader_.start({e.path}, 1 << 20, {loading_}); return; }
    // not an uncompressed RGB TIFF: Qt reads it (JPEG, PNG, other TIFFs via the plugin), with its own colour space
    QImageReader rd(e.path);
    rd.setAutoTransform(true);
    QImage img = rd.read();
    if (!img.isNull()) {
        if (img.colorSpace().isValid()) img.convertToColorSpace(QColorSpace::SRgb);
        img = img.convertToFormat(QImage::Format_RGB32);
    } else {
        statusBar()->showMessage(QString("Could not read %1: %2").arg(e.name, rd.errorString()), 8000);
    }
    int idx = loading_;
    loading_ = -1;
    onLoaded(idx, img);
}

void CompareWindow::onLoaded(int index, QImage image) {
    if (index >= 0 && index < entries_.size()) {
        Entry& e = entries_[index];
        e.loading = false;
        if (!image.isNull()) {
            // the file's own ICC profile (the frames' Adobe RGB, embedded in every xLamina output) -> sRGB for the screen
            TiffInfo info = tiff_probe(e.path.toStdWString());
            if (info.ok() && !info.icc.empty()) {
                QColorSpace cs = QColorSpace::fromIccProfile(QByteArray((const char*)info.icc.data(), (int)info.icc.size()));
                if (cs.isValid()) { image.setColorSpace(cs); image.convertToColorSpace(QColorSpace::SRgb); }
            }
            e.image = image;
        } else {
            view_[sel_[0] == index ? 0 : 1]->setLabel(e.name + " — could not be read");
        }
        for (int p = 0; p < 2; ++p) if (sel_[p] == index && !e.image.isNull()) refresh(p);
    }
    if (loading_ == index) loading_ = -1;
    startNextLoad();
}

// keep the last few decoded files; the two on screen and the retouch's images always stay
void CompareWindow::trimCache() {
    while (recent_.size() > kKeepDecoded) {
        int victim = -1;
        for (int i : recent_) if (i != sel_[0] && i != sel_[1] && i != rt_.target && i != rt_.source) { victim = i; break; }
        if (victim < 0) return;
        recent_.removeAll(victim);
        entries_[victim].image = QImage();
    }
}

// ----------------------------------------------------------------------------- retouch

// the 16-bit RGB of an entry: its TIFF when it has one (and the profile for the output), else its 8-bit image x 257
bool CompareWindow::load16(const Entry& e, std::vector<uint16_t>& out, int& w, int& h, std::vector<uint8_t>* icc) {
    if (!e.path.isEmpty()) {
        TiffInfo info = tiff_probe(e.path.toStdWString());
        if (info.ok() && info.samples >= 3) {
            size_t need = stackcore::tiff_buffer_capacity(info.pixel_bytes()) + stackcore::kReadAlign;
            std::vector<uint8_t> buf(need);
            uint8_t* base = buf.data() + (stackcore::kReadAlign - ((uintptr_t)buf.data() % stackcore::kReadAlign)) % stackcore::kReadAlign;
            stackcore::TiffRead rd = stackcore::tiff_read_pixels(e.path.toStdWString(), info, base, buf.size() - (base - buf.data()));
            if (rd.ok()) {
                const uint8_t* px = base + rd.data_offset_in_buf;
                w = (int)info.width; h = (int)info.height;
                out.resize((size_t)w * h * 3);
                size_t bpp = info.samples * (info.bits / 8);
                for (size_t i = 0; i < (size_t)w * h; ++i) {
                    const uint8_t* p = px + i * bpp;
                    for (int k = 0; k < 3; ++k) {
                        if (info.bits == 16) out[i * 3 + k] = info.big_endian ? (uint16_t)((p[2 * k] << 8) | p[2 * k + 1]) : (uint16_t)((p[2 * k + 1] << 8) | p[2 * k]);
                        else out[i * 3 + k] = (uint16_t)(p[k] * 257);
                    }
                }
                if (icc) *icc = info.icc;
                return true;
            }
        }
    }
    if (e.image.isNull()) return false;
    w = e.image.width(); h = e.image.height();
    out.resize((size_t)w * h * 3);
    for (int y = 0; y < h; ++y) {
        const QRgb* row = reinterpret_cast<const QRgb*>(e.image.constScanLine(y));
        for (int x = 0; x < w; ++x) {
            size_t i = ((size_t)y * w + x) * 3;
            out[i] = (uint16_t)(qRed(row[x]) * 257); out[i + 1] = (uint16_t)(qGreen(row[x]) * 257); out[i + 2] = (uint16_t)(qBlue(row[x]) * 257);
        }
    }
    return true;
}

void CompareWindow::retouchToggle(bool on) {
    if (on) {
        if (sel_[0] < 0 || sel_[1] < 0 || entries_[sel_[0]].image.isNull() || entries_[sel_[1]].image.isNull()) {
            statusBar()->showMessage("Retouch needs an image in each pane: the left is the result, the right the source.", 5000);
            retouchAct_->blockSignals(true); retouchAct_->setChecked(false); retouchAct_->blockSignals(false);
            return;
        }
        QApplication::setOverrideCursor(Qt::WaitCursor);
        rt_ = Retouch{};
        rt_.target = sel_[0];
        int w = 0, h = 0;
        bool ok = load16(entries_[rt_.target], rt_.tgt16, w, h, &rt_.icc);
        if (ok) {
            rt_.w = w; rt_.h = h;
            rt_.tgtDisp = entries_[rt_.target].image;
            ok = rt_.tgtDisp.width() == w && rt_.tgtDisp.height() == h;
        }
        if (ok) ok = retouchSetSource(sel_[1]);
        QApplication::restoreOverrideCursor();
        if (!ok) {
            statusBar()->showMessage("Retouch: the two images must be the same size, and readable at 16 bits.", 6000);
            rt_ = Retouch{};
            retouchAct_->blockSignals(true); retouchAct_->setChecked(false); retouchAct_->blockSignals(false);
            return;
        }
        rt_.on = true;
        rt_.res16 = rt_.tgt16;
        rt_.resDisp = rt_.tgtDisp.copy();
        rt_.mask.assign((size_t)w * h, 0);
        view_[0]->setLive(&rt_.resDisp, &rt_.mask);
        view_[0]->setMaskShown(maskAct_->isChecked());
        view_[0]->setBrush(true, brushSize_->value(), brushHard_->value());
        view_[0]->setLabel(entries_[rt_.target].name + "  (retouch)");
        flipAct_->setEnabled(false);
        for (QAction* a : {maskAct_, resetAct_, saveAct_}) a->setEnabled(true);
        undoAct_->setEnabled(false);
        brushSize_->setEnabled(true);
        brushHard_->setEnabled(true);
        retouchStatus();
    } else {
        if (rt_.dirty) {
            auto r = QMessageBox::question(this, "xLamina", "Leave Retouch and drop the unsaved strokes?", QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
            if (r != QMessageBox::Yes) { retouchAct_->blockSignals(true); retouchAct_->setChecked(true); retouchAct_->blockSignals(false); return; }
        }
        view_[0]->setLive(nullptr, nullptr);
        view_[0]->setBrush(false, brushSize_->value(), brushHard_->value());
        rt_ = Retouch{};
        flipAct_->setEnabled(true);
        for (QAction* a : {maskAct_, undoAct_, resetAct_, saveAct_}) a->setEnabled(false);
        brushSize_->setEnabled(false);
        brushHard_->setEnabled(false);
        if (sel_[0] >= 0) refresh(0);
        statusBar()->showMessage("Retouch off.", 3000);
    }
}

// a new source: the strokes so far are committed to the target (they were painted from the old source)
bool CompareWindow::retouchSetSource(int index) {
    if (index < 0 || entries_[index].image.isNull()) return false;
    std::vector<uint16_t> src;
    int w = 0, h = 0;
    if (!load16(entries_[index], src, w, h, nullptr)) return false;
    if (w != rt_.w || h != rt_.h || entries_[index].image.width() != w || entries_[index].image.height() != h) {
        if (rt_.on) statusBar()->showMessage(QString("%1 is not the result's size; the source stays %2.").arg(entries_[index].name, entries_[rt_.source].name), 6000);
        return false;
    }
    if (rt_.on && rt_.source >= 0 && rt_.source != index) retouchBake();
    rt_.source = index;
    rt_.src16 = std::move(src);
    rt_.srcDisp = entries_[index].image;
    if (rt_.on) retouchStatus();
    return true;
}

void CompareWindow::retouchBake() {
    rt_.tgt16 = rt_.res16;
    rt_.tgtDisp = rt_.resDisp.copy();
    std::fill(rt_.mask.begin(), rt_.mask.end(), 0);
    rt_.undo.clear();
    undoAct_->setEnabled(false);
    view_[0]->patch(QRect(0, 0, rt_.w, rt_.h));
}

// the mask before the stroke touches r: extend the stroke's snapshot to cover r
void CompareWindow::growSnapshot(const QRect& r) {
    Stroke& s = rt_.current;
    if (s.r.contains(r)) return;
    QRect u = s.r.isEmpty() ? r : s.r.united(r);
    std::vector<uint8_t> m((size_t)u.width() * u.height());
    for (int y = 0; y < u.height(); ++y) {
        uint8_t* dst = m.data() + (size_t)y * u.width();
        memcpy(dst, rt_.mask.data() + (size_t)(u.y() + y) * rt_.w + u.x(), u.width());   // the current mask (untouched there)
        if (!s.r.isEmpty()) {   // the old snapshot over its own area
            int yy = u.y() + y - s.r.y();
            if (yy >= 0 && yy < s.r.height()) memcpy(dst + (s.r.x() - u.x()), s.mask.data() + (size_t)yy * s.r.width(), s.r.width());
        }
    }
    s.r = u;
    s.mask = std::move(m);
}

// one dab of the brush at p (image coordinates): the mask takes the brush's alpha where it is larger
void CompareWindow::dab(QPointF p) {
    double rad = view_[0]->brushRadius(), hard = brushHard_->value();
    int x0 = std::max(0, (int)std::floor(p.x() - rad)), x1 = std::min(rt_.w - 1, (int)std::ceil(p.x() + rad));
    int y0 = std::max(0, (int)std::floor(p.y() - rad)), y1 = std::min(rt_.h - 1, (int)std::ceil(p.y() + rad));
    if (x0 > x1 || y0 > y1) return;
    QRect r(x0, y0, x1 - x0 + 1, y1 - y0 + 1);
    growSnapshot(r);
    for (int y = y0; y <= y1; ++y) {
        uint8_t* m = rt_.mask.data() + (size_t)y * rt_.w;
        for (int x = x0; x <= x1; ++x) {
            double d = std::hypot(x + 0.5 - p.x(), y + 0.5 - p.y()) / rad;
            if (d >= 1.0) continue;
            double a = d <= hard ? 1.0 : (1.0 - d) / std::max(1e-6, 1.0 - hard);
            a = a * a * (3 - 2 * a);   // smoothstep
            int v = (int)std::lround(a * 255);
            if (v > m[x]) m[x] = (uint8_t)v;
        }
    }
    composite(r);
    view_[0]->patch(r);
}

// result = target + mask * (source - target), 16-bit and the 8-bit display, inside r
void CompareWindow::composite(const QRect& r) {
    for (int y = r.top(); y <= r.bottom(); ++y) {
        const uint8_t* m = rt_.mask.data() + (size_t)y * rt_.w;
        uint16_t* res = rt_.res16.data() + (size_t)y * rt_.w * 3;
        const uint16_t* tg = rt_.tgt16.data() + (size_t)y * rt_.w * 3;
        const uint16_t* sr = rt_.src16.data() + (size_t)y * rt_.w * 3;
        QRgb* rd = reinterpret_cast<QRgb*>(rt_.resDisp.scanLine(y));
        const QRgb* td = reinterpret_cast<const QRgb*>(rt_.tgtDisp.constScanLine(y));
        const QRgb* sd = reinterpret_cast<const QRgb*>(rt_.srcDisp.constScanLine(y));
        for (int x = r.left(); x <= r.right(); ++x) {
            int a = m[x];
            if (a == 0) { for (int k = 0; k < 3; ++k) res[x * 3 + k] = tg[x * 3 + k]; rd[x] = td[x]; continue; }
            if (a == 255) { for (int k = 0; k < 3; ++k) res[x * 3 + k] = sr[x * 3 + k]; rd[x] = sd[x]; continue; }
            for (int k = 0; k < 3; ++k) res[x * 3 + k] = (uint16_t)((tg[x * 3 + k] * (255 - a) + sr[x * 3 + k] * a + 127) / 255);
            int rr = (qRed(td[x]) * (255 - a) + qRed(sd[x]) * a + 127) / 255;
            int gg = (qGreen(td[x]) * (255 - a) + qGreen(sd[x]) * a + 127) / 255;
            int bb = (qBlue(td[x]) * (255 - a) + qBlue(sd[x]) * a + 127) / 255;
            rd[x] = qRgb(rr, gg, bb);
        }
    }
}

void CompareWindow::retouchUndo() {
    if (!rt_.on || rt_.undo.empty()) return;
    Stroke s = std::move(rt_.undo.back());
    rt_.undo.pop_back();
    for (int y = 0; y < s.r.height(); ++y) memcpy(rt_.mask.data() + (size_t)(s.r.y() + y) * rt_.w + s.r.x(), s.mask.data() + (size_t)y * s.r.width(), s.r.width());
    composite(s.r);
    view_[0]->patch(s.r);
    undoAct_->setEnabled(!rt_.undo.empty());
    rt_.dirty = !rt_.undo.empty();
    retouchStatus();
}

void CompareWindow::retouchReset() {
    if (!rt_.on) return;
    std::fill(rt_.mask.begin(), rt_.mask.end(), 0);
    rt_.undo.clear();
    composite(QRect(0, 0, rt_.w, rt_.h));
    view_[0]->patch(QRect(0, 0, rt_.w, rt_.h));
    undoAct_->setEnabled(false);
    rt_.dirty = false;
    retouchStatus();
}

void CompareWindow::retouchSave() {
    if (!rt_.on) return;
    const Entry& t = entries_[rt_.target];
    QString dir = !t.path.isEmpty() ? QFileInfo(t.path).absolutePath() : QSettings().value("compare/lastSweep").toString();
    QString base = !t.path.isEmpty() ? QFileInfo(t.path).completeBaseName() : t.name;
    QString def = QDir(dir).filePath(base + "_retouched.tif");
    QString fn = QFileDialog::getSaveFileName(this, "Save the retouched image", def, "16-bit TIFF (*.tif)");
    if (fn.isEmpty()) return;
    QApplication::setOverrideCursor(Qt::WaitCursor);
    std::vector<float> rgb(rt_.res16.size());
    for (size_t i = 0; i < rgb.size(); ++i) rgb[i] = rt_.res16[i] / 65535.0f;
    bool ok = stackcore::write_tiff_rgb16(fn.toLocal8Bit().toStdString(), rgb, rt_.w, rt_.h, rt_.icc);
    QApplication::restoreOverrideCursor();
    if (!ok) { QMessageBox::warning(this, "xLamina", QString("Could not write %1").arg(QDir::toNativeSeparators(fn))); return; }
    rt_.dirty = false;
    statusBar()->showMessage(QString("Saved %1").arg(QDir::toNativeSeparators(fn)), 6000);
    QString name = QFileInfo(fn).fileName();
    Entry e;
    e.name = name; e.path = fn; e.note = QString("retouch of %1 from %2").arg(entries_[rt_.target].name, entries_[rt_.source].name);
    e.image = rt_.resDisp.copy();
    addEntry(std::move(e));
}

void CompareWindow::retouchStatus() {
    if (!rt_.on) return;
    size_t painted = 0;
    for (uint8_t m : rt_.mask) if (m) ++painted;
    status_->setText(QString("Retouch: %1  <-  %2   |   brush %3 px, hardness %4   |   %5 % painted, %6 stroke%7")
                         .arg(entries_[rt_.target].name, entries_[rt_.source].name).arg((int)view_[0]->brushRadius()).arg(brushHard_->value(), 0, 'f', 1)
                         .arg(100.0 * painted / std::max<size_t>(1, rt_.mask.size()), 0, 'f', 2).arg(rt_.undo.size()).arg(rt_.undo.size() == 1 ? "" : "s"));
}

// ----------------------------------------------------------------------------- drops

void CompareWindow::dragEnterEvent(QDragEnterEvent* e) {
    if (e->mimeData()->hasUrls()) e->acceptProposedAction();
}

void CompareWindow::dropEvent(QDropEvent* e) {
    for (const QUrl& u : e->mimeData()->urls()) {
        QString p = u.toLocalFile();
        QFileInfo fi(p);
        if (fi.isDir()) {
            if (QFileInfo::exists(QDir(p).filePath("sweep.json"))) openSweep(p);
            else if (QFileInfo::exists(QDir(p).filePath("B.tif"))) addFile(fi.fileName(), QDir(p).filePath("B.tif"), QDir::toNativeSeparators(p));
            continue;
        }
        addFile(fi.fileName(), p, QDir::toNativeSeparators(fi.absolutePath()));
        if (sel_[1] < 0) showIn(1, entries_.size() - 1);
        else if (sel_[0] < 0) showIn(0, entries_.size() - 1);
    }
}
