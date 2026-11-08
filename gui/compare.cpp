#include "compare.h"

#include <QAction>
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

#include "stackcore/tiff.h"

using stackcore::TiffInfo;
using stackcore::tiff_probe;

namespace {
const int kKeepDecoded = 8;   // file-backed images held in memory (about 120 MB each at full resolution)
}

CompareWindow::CompareWindow(QWidget* parent) : QMainWindow(parent) {
    setWindowFlag(Qt::Window);
    setWindowTitle("xLamina — two-up");
    setAcceptDrops(true);
    resize(1500, 900);

    QToolBar* tb = addToolBar("Compare");
    tb->setMovable(false);
    QAction* openSweep = tb->addAction("Open sweep…", this, &CompareWindow::openSweepDialog);
    openSweep->setStatusTip("A folder written by stackcli sweep (holds sweep.json): its B variants join the list.");
    QAction* openImage = tb->addAction("Add image…", this, &CompareWindow::openImageDialog);
    openImage->setStatusTip("Any TIFF, JPEG or PNG, for example another stacker's output.");
    tb->addSeparator();
    QAction* flip = tb->addAction("Flip", this, &CompareWindow::flip);
    flip->setShortcuts({QKeySequence(Qt::Key_Space), QKeySequence("F")});
    flip->setStatusTip("Swap the two panes (Space): the image changes under your eye, at the same zoom and position.");
    QAction* fit = tb->addAction("Fit", this, [this] { if (view_[0]->image().isNull()) return; view_[0]->fit(); });
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

    // the list of images: name and the sweep options (or where it came from)
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
        pick_[p]->setToolTip(p == 0 ? "The left pane's image" : "The right pane's image");
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
            if (syncing_ || view_[b]->image().isNull()) return;
            syncing_ = true; view_[b]->syncFrom(view_[a]); syncing_ = false;
        });
    }

    QSplitter* outer = new QSplitter(Qt::Horizontal);
    outer->addWidget(list_);
    outer->addWidget(split_);
    outer->setStretchFactor(0, 0);
    outer->setStretchFactor(1, 1);
    outer->setSizes({340, 1100});
    setCentralWidget(outer);
    status_ = new QLabel;
    statusBar()->addWidget(status_, 1);
    statusBar()->showMessage("Open a sweep folder or add images, then pick one for each pane. Space flips them.");

    connect(&loader_, &ThumbLoader::thumbnail, this, &CompareWindow::onLoaded);
    QSettings s;
    restoreGeometry(s.value("compare/geometry").toByteArray());
    if (s.contains("compare/outer")) outer->restoreState(s.value("compare/outer").toByteArray());
}

CompareWindow::~CompareWindow() { loader_.stop(); }

void CompareWindow::closeEvent(QCloseEvent* e) {
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
    if (i >= 0) {   // the same name again: a new render's output, or a sweep opened twice
        bool shown = sel_[0] == i || sel_[1] == i;
        entries_[i] = std::move(e);
        rebuildList();
        if (shown) { if (sel_[0] == i) refresh(0); if (sel_[1] == i) refresh(1); }
        return;
    }
    entries_.push_back(std::move(e));
    rebuildList();
}

void CompareWindow::addImage(const QString& name, const QImage& img, const QString& note) {
    if (img.isNull()) return;
    Entry e;
    e.name = name; e.note = note; e.image = img;
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

// sweep.json: {"variants": [{"name", "dir", "options", "rc", "seconds"}, ...]}
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
    if (!e.path.isEmpty()) { recent_.removeAll(i); recent_.push_back(i); }
    syncing_ = true;   // a new image must not re-fit the other pane: it takes the other pane's view instead
    view_[pane]->setImage(e.image, true);
    int other = 1 - pane;
    if (!view_[other]->image().isNull()) view_[pane]->syncFrom(view_[other]);
    syncing_ = false;
    view_[pane]->setLabel(e.name);
    status_->setText(QString("%1  |  %2").arg(sel_[0] >= 0 ? entries_[sel_[0]].name : "—", sel_[1] >= 0 ? entries_[sel_[1]].name : "—"));
    trimCache();
    updateOverlay();
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
    if (sel_[0] < 0 || sel_[1] < 0) return;
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
    if (e->key() == Qt::Key_Space) { flip(); return; }
    QMainWindow::keyPressEvent(e);
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

// keep the last few decoded files; the two on screen always stay
void CompareWindow::trimCache() {
    while (recent_.size() > kKeepDecoded) {
        int victim = -1;
        for (int i : recent_) if (i != sel_[0] && i != sel_[1]) { victim = i; break; }
        if (victim < 0) return;
        recent_.removeAll(victim);
        entries_[victim].image = QImage();
    }
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
