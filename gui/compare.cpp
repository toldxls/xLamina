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
    list_ = new QTableWidget(0, 2);
    list_->setHorizontalHeaderLabels({"Image", "Options"});
    list_->horizontalHeader()->setStretchLastSection(true);
    list_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
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

void CompareWindow::addFile(const QString& name, const QString& path, const QString& note) {
    Entry e;
    e.name = name; e.path = path; e.note = note;
    addEntry(std::move(e));
}

void CompareWindow::rebuildList() {
    list_->setRowCount(entries_.size());
    for (int i = 0; i < entries_.size(); ++i) {
        auto* n = new QTableWidgetItem(entries_[i].name);
        n->setToolTip(entries_[i].path.isEmpty() ? entries_[i].note : QDir::toNativeSeparators(entries_[i].path));
        list_->setItem(i, 0, n);
        list_->setItem(i, 1, new QTableWidgetItem(entries_[i].note));
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
    for (const QJsonValue& v : doc.object().value("variants").toArray()) {
        QJsonObject o = v.toObject();
        if (o.value("rc").toInt(1) != 0) { ++failed; continue; }
        QString vdir = o.value("dir").toString();
        if (QDir::isRelativePath(vdir)) vdir = QDir(dir).filePath(vdir);
        QString b = QDir(vdir).filePath("B.tif");
        if (!QFileInfo::exists(b)) { ++failed; continue; }
        QString opts = o.value("options").toString();
        addFile(o.value("name").toString(), b, opts.isEmpty() ? "base settings" : opts);
        ++added;
    }
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
