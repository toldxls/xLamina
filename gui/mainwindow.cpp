#include "mainwindow.h"

#include <QApplication>
#include <QCloseEvent>
#include <QDir>
#include <QMenu>
#include <QDockWidget>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QImageWriter>
#include <QMessageBox>
#include <QMimeData>
#include <QSplitter>
#include <QStatusBar>
#include <QToolBar>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>

#include "stackcore/tiff.h"
#include "version.h"

using namespace stackcore;

static const double kDeg = 180.0 / 3.14159265358979323846;
static const char* kOutputs[] = {"B", "C", "Cn", "A", "B0", "CB", "CBn", "Cd", "Cg", "Cg2", "H"};
static const char* kOutputHelp[] = {
    "B: depth map. Each pixel from the frame where it is sharpest (regularised, refined, soft edges). Cleanest tones.",
    "C: Laplacian pyramid. Per-detail selection across all frames. Most detail; see Smoothing for its noise.",
    "Cn: noise-aware pyramid. C where the detail is above the noise, B elsewhere.",
    "A: focus-weighted average of all frames. Smooth, soft.",
    "B0: raw per-pixel argmax of the focus measure (unregularised B).",
    "CB: pyramid, falling back to B where C's choice disagrees with the depth map.",
    "CBn: CB with the noise-aware rule.",
    "Cd: pyramid locked to B's depth at the fine levels (against glints from other depths).",
    "Cg: pyramid with coarse levels from B (against glare from defocused highlights).",
    "Cg2: Cg with one more coarse level from B.",
    "H: luma from CBn, chroma from B."};

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
    setWindowTitle(QString("xLamina %1").arg(XLAMINA_VERSION));
    resize(1500, 950);
    setAcceptDrops(true);
    buildUi();
    connect(&worker_, &Worker::progress, this, &MainWindow::onProgress);
    connect(&worker_, &Worker::preview, this, &MainWindow::onPreview);
    connect(&worker_, &Worker::frame, this, &MainWindow::onFrame);
    connect(&worker_, &Worker::logLine, this, &MainWindow::onLog);
    connect(&worker_, &Worker::finished, this, &MainWindow::onFinished);
    connect(&thumbs_, &ThumbLoader::thumbnail, this, &MainWindow::onThumbnail);
    connect(&frameLoader_, &ThumbLoader::thumbnail, this, &MainWindow::onFullFrame);
}

QImage MainWindow::forDisplay(QImage img) const {
    if (!colorSpace_.isValid() || img.isNull()) return img;
    img.setColorSpace(colorSpace_);
    img.convertToColorSpace(QColorSpace::SRgb);
    return img;
}

void MainWindow::buildUi() {
    // ---- toolbar
    QToolBar* tb = addToolBar("Main");
    tb->setMovable(false);
    tb->setToolButtonStyle(Qt::ToolButtonTextOnly);
    QAction* open = tb->addAction("Open folder…", this, &MainWindow::chooseFolder);
    open->setShortcut(QKeySequence::Open);
    tb->addSeparator();
    renderBtn_ = new QPushButton("Render");
    renderBtn_->setDefault(true);
    cancelBtn_ = new QPushButton("Cancel");
    cancelBtn_->setEnabled(false);
    saveBtn_ = new QPushButton("Save as…");
    saveBtn_->setEnabled(false);
    tb->addWidget(renderBtn_);
    tb->addWidget(cancelBtn_);
    tb->addWidget(saveBtn_);
    tb->addSeparator();
    tb->addAction("Fit", [this] { view_->fit(); });
    tb->addAction("1:1", [this] { view_->actualSize(); });
    tb->addSeparator();
    dropFlagged_ = tb->addAction("Drop flagged", this, &MainWindow::dropFlagged);
    dropFlagged_->setEnabled(false);
    dropFlagged_->setToolTip("Untick the frames whose wobble (shift from the previous frame) is an outlier; then Render again.");
    connect(renderBtn_, &QPushButton::clicked, this, &MainWindow::render);
    connect(cancelBtn_, &QPushButton::clicked, this, &MainWindow::cancel);
    connect(saveBtn_, &QPushButton::clicked, this, &MainWindow::saveCurrent);

    // ---- centre: viewer with the output tabs under it
    QWidget* centre = new QWidget;
    QVBoxLayout* cl = new QVBoxLayout(centre);
    cl->setContentsMargins(0, 0, 0, 0);
    cl->setSpacing(0);
    view_ = new ImageView;
    liveView_ = new ImageView;            // right pane while rendering: the map being built
    liveView_->setToolTip("What the render is building: the detail map (pass 1), the depth map (pass 2), the composition (pass 3)");
    split_ = new QSplitter(Qt::Horizontal);
    split_->addWidget(view_);
    split_->addWidget(liveView_);
    split_->setChildrenCollapsible(false);
    liveView_->hide();
    tabs_ = new QTabBar;
    tabs_->setExpanding(false);
    tabs_->setDrawBase(false);
    cl->addWidget(split_, 1);
    cl->addWidget(tabs_);
    setCentralWidget(centre);
    connect(tabs_, &QTabBar::currentChanged, this, &MainWindow::showOutput);
    connect(view_, &ImageView::zoomChanged, this, [this](double z) { status_->setText(QString("%1 %").arg(z * 100, 0, 'f', 0)); });

    // ---- left: frames
    QDockWidget* frames = new QDockWidget("Frames", this);
    frames->setFeatures(QDockWidget::DockWidgetMovable);
    frameList_ = new QListWidget;
    frameList_->setAlternatingRowColors(true);
    frameList_->setIconSize(QSize(144, 96));
    // uniform sizes are needed for speed (without them the live view's setCurrentRow re-measures every
    // thumbnail row three times per frame: a 17 s fusion took 111 s), so every item, frame or left-out
    // file, gets the same explicit row height in openFolder; mixed heights made the rows overlap
    frameList_->setUniformItemSizes(true);
    frameList_->setDragDropMode(QAbstractItemView::InternalMove);   // drag to reorder: the list order is the stack order
    frameList_->setDefaultDropAction(Qt::MoveAction);
    frameList_->setContextMenuPolicy(Qt::CustomContextMenu);
    frameList_->setToolTip("The stack's frames, in stack order: drag to reorder, right-click to add a frame again or move it.\n"
                           "Click one to see it at full resolution; greyed files are not part of the stack.");
    frames->setWidget(frameList_);
    addDockWidget(Qt::LeftDockWidgetArea, frames);
    connect(frameList_, &QListWidget::currentRowChanged, this, &MainWindow::showFrame);
    connect(frameList_, &QListWidget::customContextMenuRequested, this, &MainWindow::frameMenu);

    // ---- right: parameters
    QDockWidget* params = new QDockWidget("Render", this);
    params->setFeatures(QDockWidget::DockWidgetMovable);
    QWidget* pw = new QWidget;
    QVBoxLayout* pl = new QVBoxLayout(pw);

    QGroupBox* og = new QGroupBox("Outputs");
    QGridLayout* ol = new QGridLayout(og);
    for (int i = 0; i < 11; ++i) {
        QCheckBox* cb = new QCheckBox(kOutputs[i]);
        cb->setToolTip(kOutputHelp[i]);
        cb->setChecked(i < 3);
        outputs_[kOutputs[i]] = cb;
        ol->addWidget(cb, i / 3, i % 3);
    }
    pl->addWidget(og);

    QGroupBox* sg = new QGroupBox("Settings");
    QFormLayout* fl = new QFormLayout(sg);
    scale_ = new QComboBox;
    scale_->addItems({"Full resolution", "Half resolution (fast preview)"});
    fl->addRow("Output", scale_);
    ref_ = new QComboBox;
    ref_->addItems({"Middle frame", "First frame", "Last frame"});
    ref_->setToolTip("The frame whose pixel grid the output uses. Middle minimises the largest warp.");
    fl->addRow("Reference", ref_);
    lockScale_ = new QCheckBox("Lock scale (telecentric optics)");
    lockScale_->setToolTip("Apply no scale change between frames; use for telecentric objectives.");
    fl->addRow("", lockScale_);
    liveRes_ = new QComboBox;
    liveRes_->addItem("Low (1/12)", 12);
    liveRes_->addItem("Medium (1/6)", 6);
    liveRes_->addItem("High (1/3)", 3);
    liveRes_->addItem("Full", 1);
    liveRes_->setCurrentIndex(1);
    liveRes_->setToolTip("Resolution of the live view while rendering, as a fraction of the output.\n"
                         "Each frame shown is copied from the GPU; Full copies 365 MB per frame and slows the render.");
    fl->addRow("Live view", liveRes_);
    smoothing_ = new QDoubleSpinBox;
    smoothing_->setRange(0, 6);
    smoothing_->setSingleStep(0.5);
    smoothing_->setDecimals(1);
    smoothing_->setValue(2);
    smoothing_->setToolTip("Smoothing of C: detail weaker than this many times the noise is taken from B instead of the sharpest frame.\n"
                           "0 = raw selection (noisiest), 2 is about Helicon's Smoothing 1, 3 equals Cn.");
    fl->addRow("Smoothing (C)", smoothing_);
    cradius_ = new QDoubleSpinBox;
    cradius_->setRange(0, 200);
    cradius_->setSingleStep(4);
    cradius_->setDecimals(0);
    cradius_->setSuffix(" px");
    cradius_->setValue(32);
    cradius_->setToolTip("Radius of C: each frame's pyramid energy is aggregated over this many sensor pixels before the\n"
                         "sharpest frame is chosen, so the frame of a strong edge also wins the band beside it. Stops the\n"
                         "defocus rims of plate edges being taken from frames far away as ghost outlines.\n"
                         "0 = raw selection. Larger values lose small features that sit near a stronger edge at another depth.");
    fl->addRow("Radius (C)", cradius_);
    bradius_ = new QDoubleSpinBox;
    bradius_->setRange(0, 200);
    bradius_->setSingleStep(4);
    bradius_->setDecimals(0);
    bradius_->setSuffix(" px");
    bradius_->setValue(0);
    bradius_->setToolTip("Radius of B: each frame's focus slice is blurred over this many sensor pixels before the depth map\n"
                         "is solved, so the frame of a strong edge also claims the band beside it instead of the defocus rims\n"
                         "of that edge from frames far away. 0 = off (the validated depth pipeline). Larger values lose the\n"
                         "depth of small features that sit near a stronger edge at another depth.");
    fl->addRow("Radius (B)", bradius_);
    fov_ = new QDoubleSpinBox;
    fov_->setRange(0, 1000);
    fov_->setDecimals(4);
    fov_->setSuffix(" mm");
    fov_->setSpecialValueText("unknown");
    fov_->setToolTip("Horizontal field of view of one frame; gives the report its µm/px.");
    fl->addRow("Field of view", fov_);
    outDir_ = new QLineEdit;
    outDir_->setPlaceholderText("<folder>_stacker");
    fl->addRow("Output folder", outDir_);
    pl->addWidget(sg);

    progress_ = new QProgressBar;
    progress_->setRange(0, 1000);
    progress_->setValue(0);
    stage_ = new QLabel("Open a folder of TIFF frames, choose the outputs, press Render.");
    stage_->setWordWrap(true);
    pl->addWidget(stage_);
    pl->addWidget(progress_);
    pl->addStretch(1);
    params->setWidget(pw);
    addDockWidget(Qt::RightDockWidgetArea, params);

    // ---- bottom: geometry + log
    QDockWidget* geo = new QDockWidget("Geometry report", this);
    geo->setFeatures(QDockWidget::DockWidgetMovable);
    geometry_ = new QLabel("—");
    geometry_->setTextInteractionFlags(Qt::TextSelectableByMouse);
    geometry_->setMargin(6);
    geo->setWidget(geometry_);
    addDockWidget(Qt::BottomDockWidgetArea, geo);
    QDockWidget* lg = new QDockWidget("Log", this);
    lg->setFeatures(QDockWidget::DockWidgetMovable | QDockWidget::DockWidgetClosable);
    log_ = new QPlainTextEdit;
    log_->setReadOnly(true);
    log_->setMaximumBlockCount(2000);
    lg->setWidget(log_);
    addDockWidget(Qt::BottomDockWidgetArea, lg);
    tabifyDockWidget(geo, lg);
    geo->raise();
    resizeDocks({geo, lg}, {110, 110}, Qt::Vertical);
    resizeDocks({frames, params}, {200, 300}, Qt::Horizontal);

    status_ = new QLabel("");
    statusBar()->addPermanentWidget(status_);
    statusBar()->showMessage("xLamina: multi-GPU focus stacker. Drop a folder of frames here.");
}

void MainWindow::chooseFolder() {
    QString d = QFileDialog::getExistingDirectory(this, "Folder with the stack's TIFF frames", folder_);
    if (!d.isEmpty()) openFolder(d);
}

void MainWindow::openFolder(const QString& dir) {
    QDir d(dir);
    QStringList files = d.entryList({"*.tif", "*.tiff", "*.TIF", "*.TIFF"}, QDir::Files, QDir::Name);
    if (files.size() < 2) {
        QMessageBox::warning(this, "xLamina", QString("%1\n\nholds %2 TIFF files; a stack needs at least 2.").arg(dir).arg(files.size()));
        return;
    }
    thumbs_.stop();
    folder_ = dir;
    frames_.clear();
    jobItems_.clear();
    flagged_.clear();
    dropFlagged_->setEnabled(false);
    frameList_->clear();
    liveView_->hide();
    colorSpace_ = QColorSpace();
    // the stack is the TIFFs of the most common size; other files in the folder are listed greyed out
    QMap<QPair<unsigned, unsigned>, int> sizes;
    QList<TiffInfo> infos;
    for (const QString& f : files) {
        infos << tiff_probe(d.filePath(f).toStdWString());
        if (infos.last().ok()) ++sizes[{infos.last().width, infos.last().height}];
    }
    QPair<unsigned, unsigned> best{0, 0};
    int bestn = 0;
    for (auto it = sizes.begin(); it != sizes.end(); ++it) if (it.value() > bestn) { best = it.key(); bestn = it.value(); }
    int skipped = 0;
    QPixmap blankPx(frameList_->iconSize());
    blankPx.fill(QColor(40, 40, 40));
    QIcon blank(blankPx);
    for (int i = 0; i < files.size(); ++i) {
        bool ok = infos[i].ok() && infos[i].width == best.first && infos[i].height == best.second;
        QListWidgetItem* it = new QListWidgetItem(files[i]);
        it->setSizeHint(QSize(0, frameList_->iconSize().height() + 6));   // the same for every row (uniform sizes)
        if (ok) {
            frames_ << d.filePath(files[i]);
            it->setIcon(blank);   // holds the row height until the thumbnail arrives
            it->setFlags(it->flags() | Qt::ItemIsUserCheckable);
            it->setCheckState(Qt::Checked);
            it->setToolTip("Untick to leave this frame out of the render. Wobble numbers appear here after a render.");
            if (!colorSpace_.isValid() && !infos[i].icc.empty())
                colorSpace_ = QColorSpace::fromIccProfile(QByteArray((const char*)infos[i].icc.data(), (int)infos[i].icc.size()));
        } else {
            it->setForeground(Qt::gray);
            it->setFlags(it->flags() & ~Qt::ItemIsSelectable);
            it->setToolTip(infos[i].ok() ? QString("%1 x %2: not the stack's size, left out").arg(infos[i].width).arg(infos[i].height)
                                         : QString::fromStdString(infos[i].error));
            ++skipped;
        }
        it->setData(Qt::UserRole, ok ? (int)frames_.size() - 1 : -1);
        frameList_->addItem(it);
    }
    outDir_->setText(QDir::toNativeSeparators(dir + "_stacker"));
    setWindowTitle(QString("xLamina %1 — %2").arg(XLAMINA_VERSION, QDir::toNativeSeparators(dir)));
    QString note = QString("%1 frames of %2 x %3").arg(frames_.size()).arg(best.first).arg(best.second);
    if (skipped) note += QString(" (%1 other TIFF%2 left out)").arg(skipped).arg(skipped > 1 ? "s" : "");
    if (colorSpace_.isValid()) note += QString(", %1").arg(colorSpace_.description().isEmpty() ? "ICC profile" : colorSpace_.description());
    stage_->setText(note + ". Press Render.");
    statusBar()->showMessage(note + " in " + QDir::toNativeSeparators(dir));
    for (int i = 0; i < frameList_->count(); ++i)
        if (frameList_->item(i)->data(Qt::UserRole).toInt() == frames_.size() / 2) { frameList_->setCurrentRow(i); break; }
    thumbs_.start(frames_, 144);
}

void MainWindow::onThumbnail(int index, QImage image) {
    if (index < 0 || index >= frames_.size()) return;
    QIcon icon(QPixmap::fromImage(forDisplay(image)));
    for (int r = 0; r < frameList_->count(); ++r)   // every row showing this frame (a frame can be listed more than once)
        if (frameList_->item(r)->data(Qt::UserRole).toInt() == index) frameList_->item(r)->setIcon(icon);
}

// Right-click on a frame: add it to the stack again (Helicon's "add the same photo twice"), remove a
// copy, or move it; the list order is the stack order, so "move to top" makes it the first frame.
void MainWindow::frameMenu(const QPoint& pos) {
    QListWidgetItem* it = frameList_->itemAt(pos);
    if (!it || worker_.running()) return;
    int idx = it->data(Qt::UserRole).toInt();
    if (idx < 0) return;
    int copies = 0;
    for (int r = 0; r < frameList_->count(); ++r) if (frameList_->item(r)->data(Qt::UserRole).toInt() == idx) ++copies;
    QMenu menu(this);
    QAction* again = menu.addAction("Add this frame again (after this one)");
    QAction* remove = menu.addAction("Remove this copy");
    remove->setEnabled(copies > 1);
    menu.addSeparator();
    QAction* top = menu.addAction("Move to top (first frame)");
    QAction* bottom = menu.addAction("Move to bottom (last frame)");
    QAction* reverse = menu.addAction("Reverse the whole order");
    QAction* chosen = menu.exec(frameList_->viewport()->mapToGlobal(pos));
    if (!chosen) return;
    int row = frameList_->row(it);
    if (chosen == again) {
        QListWidgetItem* c = new QListWidgetItem(QFileInfo(frames_[idx]).fileName() + "  (again)");
        c->setSizeHint(it->sizeHint());
        c->setIcon(it->icon());
        c->setFlags(it->flags());
        c->setCheckState(Qt::Checked);
        c->setData(Qt::UserRole, idx);
        c->setToolTip("A second copy of this frame in the stack. Right-click to remove it.");
        frameList_->insertItem(row + 1, c);
    } else if (chosen == remove) {
        delete frameList_->takeItem(row);
    } else if (chosen == top || chosen == bottom) {
        QListWidgetItem* moved = frameList_->takeItem(row);
        frameList_->insertItem(chosen == top ? 0 : frameList_->count(), moved);
        frameList_->setCurrentItem(moved);
    } else if (chosen == reverse) {
        QList<QListWidgetItem*> items;
        while (frameList_->count()) items.prepend(frameList_->takeItem(0));
        for (QListWidgetItem* i : items) frameList_->addItem(i);
    }
}

// The live view: while frames stream (reading, pass 1 forward, pass 2 backward,
// pass 3 forward) the left pane shows each frame as it is processed (raw while
// reading, so the stack's wobble is visible, then warped onto the output grid)
// and the right pane what is being built from them.
void MainWindow::onFrame(int k, QImage image) {
    if (k < 0 || k >= jobItems_.size()) return;
    QListWidgetItem* item = jobItems_[k];   // the job's frame -> its row (rows may have been reordered or duplicated)
    k = item->data(Qt::UserRole).toInt();   // the folder's frame
    selecting_ = true;
    frameList_->setCurrentItem(item);
    selecting_ = false;
    bool raw = stage_->text().startsWith("reading");
    current_ = "frame";
    view_->setImage(forDisplay(image), true);
    const QString& st = stage_->text();
    QString what = raw ? "Reading frame" : st.startsWith("pass 2") ? "Fine focus, frame" : st.startsWith("pass 3") ? "Composing, frame" : "Aligned frame";
    statusBar()->showMessage(QString("%1 %2 of %3").arg(what).arg(k + 1).arg(frames_.size()));
}

void MainWindow::showFrame(int listRow) {
    if (selecting_ || listRow < 0 || listRow >= frameList_->count() || worker_.running()) return;
    int row = frameList_->item(listRow)->data(Qt::UserRole).toInt();
    if (row < 0 || row >= frames_.size()) return;
    // the source frame at full resolution, read on a thread. The image on screen stays until it
    // arrives (showing the thumbnail meanwhile pixelated the view and moved it, which made
    // frame-to-frame comparison impossible); only an empty view gets the thumbnail
    pendingFrame_ = row;
    QIcon icon = frameList_->item(listRow)->icon();
    if (view_->image().isNull() && !icon.isNull()) { current_ = "frame"; view_->setImage(icon.pixmap(icon.availableSizes().value(0)).toImage(), false); }
    statusBar()->showMessage(QString("%1 — loading at full resolution…").arg(QFileInfo(frames_[row]).fileName()));
    frameLoader_.start({frames_[row]}, 1 << 20);
}

void MainWindow::onFullFrame(int index, QImage image) {
    if (index != 0 || pendingFrame_ < 0 || pendingFrame_ >= frames_.size() || worker_.running()) return;
    current_ = "frame";
    view_->setImage(forDisplay(image), true);
    statusBar()->showMessage(QString("%1 — %2 x %3 (full resolution)").arg(QFileInfo(frames_[pendingFrame_]).fileName()).arg(image.width()).arg(image.height()));
}

void MainWindow::render() {
    if (folder_.isEmpty()) { chooseFolder(); if (folder_.isEmpty()) return; }
    if (worker_.running()) return;
    JobOptions jo;
    jo.align.dir = folder_.toStdWString();
    // the ticked frames, in list order (the list may have been reordered, and a frame may appear more than once)
    jobItems_.clear();
    for (int r = 0; r < frameList_->count(); ++r) {
        QListWidgetItem* it = frameList_->item(r);
        int idx = it->data(Qt::UserRole).toInt();
        if (idx >= 0 && it->checkState() == Qt::Checked) {
            jobItems_ << it;
            jo.align.paths.push_back(frames_[idx].toStdWString());
        }
    }
    if (jobItems_.size() < 2) { QMessageBox::information(this, "xLamina", "A stack needs at least 2 ticked frames."); return; }
    jo.align.ref_rule = ref_->currentIndex() == 0 ? "middle" : ref_->currentIndex() == 1 ? "first" : "last";
    jo.align.lock_scale = lockScale_->isChecked();
    if (fov_->value() > 0) jo.align.um_per_px = -fov_->value();
    jo.stack.po.g.scale = scale_->currentIndex() == 0 ? 1.0 : 0.5;
    jo.stack.po.opt.streams_per_gpu = 2;
    jo.stack.streams = 2;
    jo.stack.c_smoothing = smoothing_->value();
    jo.stack.po.c_radius_px = cradius_->value();
    jo.stack.po.b_radius_px = bradius_->value();
    jo.stack.outputs.clear();
    for (const char* n : kOutputs) if (outputs_[n]->isChecked()) jo.stack.outputs.push_back(n);
    if (jo.stack.outputs.empty()) { QMessageBox::information(this, "xLamina", "Choose at least one output."); return; }
    jo.out = outDir_->text().isEmpty() ? std::string() : outDir_->text().toStdString();
    cache_.clear();
    result_.reset();
    while (tabs_->count()) tabs_->removeTab(0);
    log_->clear();
    liveView_->hide();
    setBusy(true);
    worker_.start(jo, liveRes_->currentData().toInt());
}

void MainWindow::cancel() {
    worker_.cancel();
    stage_->setText("Cancelling after the current frames…");
}

void MainWindow::setBusy(bool busy) {
    renderBtn_->setEnabled(!busy);
    cancelBtn_->setEnabled(busy);
    saveBtn_->setEnabled(!busy && result_ && !result_->result.names.empty());
    // the list stays enabled (a disabled list greys the thumbnails); showFrame ignores clicks while rendering
    progress_->setValue(0);
}

void MainWindow::onProgress(QString stage, double fraction) {
    stage_->setText(stage);
    progress_->setValue((int)std::lround(std::clamp(fraction, 0.0, 1.0) * 1000));
}

void MainWindow::onPreview(QString name, QImage image) {
    bool grey = name.startsWith("depth") || name.startsWith("focus");
    if (!grey) image = forDisplay(image);
    if (name.endsWith("(building)") || name.startsWith("depth map")) {   // the right pane: what the render is building
        if (liveView_->isHidden()) {
            liveView_->show();
            int half = split_->width() / 2;
            split_->setSizes({half, half});
            view_->fit();
        }
        liveView_->setImage(image, true);
        liveView_->setToolTip(name);
        return;
    }
    current_ = name;
    view_->setImage(image, true);
    statusBar()->showMessage(QString("Preview: %1").arg(name));
}

void MainWindow::onLog(QString line) { log_->appendPlainText(line); }

void MainWindow::setStatusGeometry(const JobResultData& d) {
    const auto& A = d.align;
    int cw = A.rect[2] - A.rect[0] + 1, ch = A.rect[3] - A.rect[1] + 1;
    QString g = QString("%1 frames %2 x %3 → %4 x %5 (crop L%6 R%7 T%8 B%9), reference #%10, scale spread %11 %, "
                        "%12/%13 pairs used, %14 outliers, NCC median %15")
                    .arg(A.paths.size()).arg(A.W).arg(A.H).arg(cw).arg(ch).arg(A.rect[0]).arg(A.W - 1 - A.rect[2]).arg(A.rect[1])
                    .arg(A.H - 1 - A.rect[3]).arg(A.ref).arg(100 * (A.smax / A.smin - 1), 0, 'f', 3)
                    .arg(A.G.pairs_used).arg(A.G.pairs_total).arg(A.G.outliers).arg(A.G.ncc_median, 0, 'f', 4);
    if (fov_->value() > 0) {
        double upp = fov_->value() * 1000.0 / A.W;
        g += QString("\n%1 µm/px at the reference depth, %2–%3 µm/px over the stack depth; output FOV %4 x %5 mm")
                 .arg(upp, 0, 'f', 5).arg(upp * A.smin, 0, 'f', 5).arg(upp * A.smax, 0, 'f', 5)
                 .arg(cw * upp / 1000, 0, 'f', 4).arg(ch * upp / 1000, 0, 'f', 4);
    }
    if (!wobble_.isEmpty()) g += "\n" + wobble_;
    g += QString("\nfusion %1 s; report.json and the TIFFs in %2").arg(d.result.seconds, 0, 'f', 1).arg(QDir::toNativeSeparators(d.outDir));
    geometry_->setText(g);
}

// Per-frame wobble: each frame's shift (and scale, rotation) relative to the previous
// frame, from the alignment. Steps far from the median are flagged as outliers.
void MainWindow::labelWobble(const stackcore::AlignOut& A) {
    flagged_.clear();
    int n = (int)A.T.size();
    if (n != jobItems_.size() || n < 2) return;
    QVector<double> step(n, 0.0), sstep(n, 0.0), rstep(n, 0.0);
    for (int k = 1; k < n; ++k) {
        const Similarity &a = A.T[k - 1], &b = A.T[k];
        step[k] = std::hypot(b.tx - a.tx, b.ty - a.ty);
        double sa = std::hypot(a.zr, a.zi), sb = std::hypot(b.zr, b.zi);
        sstep[k] = 100.0 * (sb / sa - 1);
        rstep[k] = (std::atan2(b.zi, b.zr) - std::atan2(a.zi, a.zr)) * kDeg;
    }
    // robust threshold on the shift step: median + 4 x 1.4826 MAD, at least 0.5 px above the median
    QVector<double> v(step.begin() + 1, step.end());
    std::sort(v.begin(), v.end());
    double med = v[v.size() / 2];
    QVector<double> dev;
    for (double x : v) dev << std::abs(x - med);
    std::sort(dev.begin(), dev.end());
    double mad = 1.4826 * dev[dev.size() / 2];
    double thr = med + std::max(4 * mad, 0.5);
    double worst = 0;
    int worstk = -1;
    for (int k = 0; k < n; ++k) {
        QListWidgetItem* it = jobItems_[k];
        QString name = QFileInfo(frames_[it->data(Qt::UserRole).toInt()]).fileName();
        if (k == 0) {
            it->setText(name + "\nfirst frame");
        } else {
            it->setText(QString("%1\nΔ %2 px  %3 %").arg(name).arg(step[k], 0, 'f', 2).arg(sstep[k], 0, 'f', 3));
        }
        bool flag = k > 0 && step[k] > thr;
        it->setForeground(flag ? QBrush(QColor(230, 80, 80)) : QBrush());
        it->setToolTip(QString("To the reference: shift (%1, %2) px, scale %3, rotation %4°\n"
                               "From the previous frame: %5 px, %6 %, %7°%8")
                           .arg(A.T[k].tx, 0, 'f', 2).arg(A.T[k].ty, 0, 'f', 2).arg(std::hypot(A.T[k].zr, A.T[k].zi), 0, 'f', 5)
                           .arg(std::atan2(A.T[k].zi, A.T[k].zr) * kDeg, 0, 'f', 4)
                           .arg(step[k], 0, 'f', 3).arg(sstep[k], 0, 'f', 4).arg(rstep[k], 0, 'f', 4)
                           .arg(flag ? QString("\nFLAGGED: above %1 px (median %2 px)").arg(thr, 0, 'f', 2).arg(med, 0, 'f', 2) : ""));
        if (flag) flagged_ << it;
        if (step[k] > worst) { worst = step[k]; worstk = it->data(Qt::UserRole).toInt(); }
    }
    wobble_ = QString("wobble: median step %1 px, largest %2 px (%3), %4 flagged")
                  .arg(med, 0, 'f', 2).arg(worst, 0, 'f', 2).arg(worstk >= 0 ? QFileInfo(frames_[worstk]).fileName() : "-").arg(flagged_.size());
    dropFlagged_->setEnabled(!flagged_.isEmpty());
}

void MainWindow::dropFlagged() {
    for (QListWidgetItem* it : flagged_) it->setCheckState(Qt::Unchecked);
    statusBar()->showMessage(QString("%1 frame%2 unticked; press Render to stack without them.").arg(flagged_.size()).arg(flagged_.size() == 1 ? "" : "s"));
    flagged_.clear();
    dropFlagged_->setEnabled(false);
}

void MainWindow::onFinished(std::shared_ptr<JobResultData> data) {
    result_ = data;
    setBusy(false);
    liveView_->hide();
    labelWobble(data->align);
    if (data->rc == 3) { stage_->setText("Cancelled."); return; }
    if (data->rc != 0) { stage_->setText("Failed; see the log."); QMessageBox::warning(this, "xLamina", "The render failed. The log has the details."); return; }
    stage_->setText(QString("Done in %1 s.").arg(data->result.seconds, 0, 'f', 1));
    progress_->setValue(1000);
    setStatusGeometry(*data);
    tabs_->blockSignals(true);
    for (const auto& n : data->result.names) tabs_->addTab(QString::fromStdString(n));
    tabs_->addTab("depth");
    tabs_->blockSignals(false);
    tabs_->setCurrentIndex(0);
    showOutput(0);
}

QImage MainWindow::outputImage(const QString& name) {
    if (cache_.contains(name)) return cache_[name];
    if (!result_) return QImage();
    const auto& R = result_->result;
    QImage img;
    if (name == "depth") {
        int K = (int)result_->align.paths.size();
        std::vector<float> rgb((size_t)R.w * R.h * 3);
        for (size_t i = 0; i < (size_t)R.w * R.h; ++i) { float v = R.depth[i] / (float)std::max(K - 1, 1); rgb[i * 3] = rgb[i * 3 + 1] = rgb[i * 3 + 2] = v; }
        img = toQImage(rgb.data(), R.w, R.h);
    } else {
        for (size_t i = 0; i < R.names.size(); ++i)
            if (QString::fromStdString(R.names[i]) == name) { img = forDisplay(toQImage(R.images[i].data(), R.w, R.h)); break; }
    }
    if (!img.isNull()) cache_[name] = img;
    return img;
}

void MainWindow::showOutput(int tab) {
    if (tab < 0 || tab >= tabs_->count()) return;
    QString name = tabs_->tabText(tab);
    QImage img = outputImage(name);
    if (img.isNull()) return;
    current_ = name;
    view_->setImage(img, true);   // the zoomed area carries over from the live view / the other outputs
    statusBar()->showMessage(QString("%1 — %2 x %3").arg(name).arg(img.width()).arg(img.height()));
}

void MainWindow::saveCurrent() {
    if (!result_ || current_.isEmpty() || current_ == "frame") return;
    QString def = QDir(outDir_->text().isEmpty() ? folder_ + "_stacker" : outDir_->text()).filePath(current_ + ".tif");
    QString fn = QFileDialog::getSaveFileName(this, "Save " + current_, def, "16-bit TIFF (*.tif);;JPEG (*.jpg);;PNG (*.png)");
    if (fn.isEmpty()) return;
    const auto& R = result_->result;
    bool ok = false;
    if (fn.endsWith(".tif", Qt::CaseInsensitive) || fn.endsWith(".tiff", Qt::CaseInsensitive)) {
        for (size_t i = 0; i < R.names.size(); ++i)
            if (QString::fromStdString(R.names[i]) == current_) ok = write_tiff_rgb16(fn.toStdString(), R.images[i], R.w, R.h, R.icc);   // the frames' profile
        if (current_ == "depth") ok = outputImage("depth").save(fn);
    } else {
        QImageWriter w(fn);   // JPEG / PNG: the displayed (sRGB) image
        w.setQuality(95);
        ok = w.write(outputImage(current_));
    }
    statusBar()->showMessage(ok ? QString("Saved %1").arg(QDir::toNativeSeparators(fn)) : QString("Could not save %1").arg(fn), 5000);
}

void MainWindow::dragEnterEvent(QDragEnterEvent* e) {
    if (e->mimeData()->hasUrls()) e->acceptProposedAction();
}

void MainWindow::dropEvent(QDropEvent* e) {
    for (const QUrl& u : e->mimeData()->urls()) {
        QString p = u.toLocalFile();
        QFileInfo fi(p);
        openFolder(fi.isDir() ? p : fi.absolutePath());
        break;
    }
}

void MainWindow::closeEvent(QCloseEvent* e) {
    if (worker_.running()) {
        if (QMessageBox::question(this, "xLamina", "A render is running. Cancel it and quit?") != QMessageBox::Yes) { e->ignore(); return; }
        worker_.cancel();
    }
    e->accept();
}
