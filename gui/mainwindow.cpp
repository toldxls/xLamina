#include "mainwindow.h"

#include <QApplication>
#include <QCloseEvent>
#include <QDir>
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
#include <QStatusBar>
#include <QToolBar>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>

#include "stackcore/tiff.h"
#include "version.h"

using namespace stackcore;

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
    connect(renderBtn_, &QPushButton::clicked, this, &MainWindow::render);
    connect(cancelBtn_, &QPushButton::clicked, this, &MainWindow::cancel);
    connect(saveBtn_, &QPushButton::clicked, this, &MainWindow::saveCurrent);

    // ---- centre: viewer with the output tabs under it
    QWidget* centre = new QWidget;
    QVBoxLayout* cl = new QVBoxLayout(centre);
    cl->setContentsMargins(0, 0, 0, 0);
    cl->setSpacing(0);
    view_ = new ImageView;
    tabs_ = new QTabBar;
    tabs_->setExpanding(false);
    tabs_->setDrawBase(false);
    cl->addWidget(view_, 1);
    cl->addWidget(tabs_);
    setCentralWidget(centre);
    connect(tabs_, &QTabBar::currentChanged, this, &MainWindow::showOutput);
    connect(view_, &ImageView::zoomChanged, this, [this](double z) { status_->setText(QString("%1 %").arg(z * 100, 0, 'f', 0)); });

    // ---- left: frames
    QDockWidget* frames = new QDockWidget("Frames", this);
    frames->setFeatures(QDockWidget::DockWidgetMovable);
    QWidget* fw = new QWidget;
    QVBoxLayout* fl0 = new QVBoxLayout(fw);
    fl0->setContentsMargins(0, 0, 0, 0);
    frameList_ = new QListWidget;
    frameList_->setAlternatingRowColors(true);
    frameList_->setIconSize(QSize(144, 96));
    frameList_->setUniformItemSizes(true);
    frameList_->setToolTip("The stack's frames. Click one to see it; greyed files are not part of the stack.");
    frameView_ = new QLabel;
    frameView_->setAlignment(Qt::AlignCenter);
    frameView_->setMinimumHeight(140);
    frameView_->setStyleSheet("background: #202020;");
    frameView_->setToolTip("While the frames are read and aligned: the stack building. Afterwards: the frame being processed.");
    fl0->addWidget(frameList_, 1);
    fl0->addWidget(frameView_);
    frames->setWidget(fw);
    addDockWidget(Qt::LeftDockWidgetArea, frames);
    connect(frameList_, &QListWidget::currentRowChanged, this, &MainWindow::showFrame);

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
    smoothing_ = new QDoubleSpinBox;
    smoothing_->setRange(0, 6);
    smoothing_->setSingleStep(0.5);
    smoothing_->setDecimals(1);
    smoothing_->setValue(2);
    smoothing_->setToolTip("Smoothing of C: detail weaker than this many times the noise is taken from B instead of the sharpest frame.\n"
                           "0 = raw selection (noisiest), 2 is about Helicon's Smoothing 1, 3 equals Cn.");
    fl->addRow("Smoothing (C)", smoothing_);
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
    if (files.size() < 3) {
        QMessageBox::warning(this, "xLamina", QString("%1\n\nholds %2 TIFF files; a stack needs at least 3.").arg(dir).arg(files.size()));
        return;
    }
    thumbs_.stop();
    folder_ = dir;
    frames_.clear();
    frameRows_.clear();
    frameList_->clear();
    frameView_->clear();
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
    for (int i = 0; i < files.size(); ++i) {
        bool ok = infos[i].ok() && infos[i].width == best.first && infos[i].height == best.second;
        QListWidgetItem* it = new QListWidgetItem(files[i]);
        if (ok) {
            frames_ << d.filePath(files[i]);
            frameRows_ << i;
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
    if (index < 0 || index >= frameRows_.size()) return;
    frameList_->item(frameRows_[index])->setIcon(QIcon(QPixmap::fromImage(forDisplay(image))));
}

// While the frames are read, aligned and folded the main view shows each frame as
// it is processed (raw first, so the stack's wobble is visible, then warped onto
// the output grid) and the small pane shows the result building; from the depth
// map on, the main view shows the result and the small pane the frame.
void MainWindow::onFrame(int k, QImage image) {
    if (k < 0 || k >= frameRows_.size()) return;
    selecting_ = true;
    frameList_->setCurrentRow(frameRows_[k]);
    selecting_ = false;
    image = forDisplay(image);
    bool raw = stage_->text().startsWith("reading");
    bool framesPhase = raw || stage_->text().startsWith("aligning") || stage_->text().startsWith("pass 1");
    if (framesPhase) {
        current_ = "frame";
        view_->setImage(image, true);
        statusBar()->showMessage(QString("%1 %2 of %3%4").arg(raw ? "Reading frame" : "Aligned frame").arg(k + 1).arg(frames_.size())
                                     .arg(raw ? "" : " (warped onto the output grid)"));
    } else {
        frameView_->setPixmap(QPixmap::fromImage(image).scaled(frameView_->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }
}

void MainWindow::showFrame(int listRow) {
    if (selecting_ || listRow < 0 || listRow >= frameList_->count() || worker_.running()) return;
    int row = frameList_->item(listRow)->data(Qt::UserRole).toInt();
    if (row < 0 || row >= frames_.size()) return;
    // a quick 1/4 preview of the source frame, read on this thread (the RAM-disk / NVMe read dominates)
    TiffInfo info = tiff_probe(frames_[row].toStdWString());
    if (!info.ok() || info.samples < 3) { statusBar()->showMessage(QString("%1: %2").arg(frames_[row], QString::fromStdString(info.error))); return; }
    std::vector<uint8_t> buf(tiff_buffer_capacity(info.pixel_bytes()) + kReadAlign);
    uint8_t* base = buf.data() + (kReadAlign - ((uintptr_t)buf.data() % kReadAlign)) % kReadAlign;
    TiffRead rd = tiff_read_pixels(frames_[row].toStdWString(), info, base, buf.size() - (base - buf.data()));
    if (!rd.ok()) return;
    const uint8_t* px = base + rd.data_offset_in_buf;
    int down = 4, pw = info.width / down, ph = info.height / down;
    QImage img(pw, ph, QImage::Format_RGB32);
    size_t bpp = info.samples * (info.bits / 8);
    for (int y = 0; y < ph; ++y) {
        QRgb* out = reinterpret_cast<QRgb*>(img.scanLine(y));
        const uint8_t* srow = px + (size_t)(y * down) * info.width * bpp;
        for (int x = 0; x < pw; ++x) {
            const uint8_t* p = srow + (size_t)(x * down) * bpp;
            int c[3];
            for (int k = 0; k < 3; ++k) {
                if (info.bits == 16) { unsigned v = info.big_endian ? (p[2 * k] << 8) | p[2 * k + 1] : (p[2 * k + 1] << 8) | p[2 * k]; c[k] = v >> 8; }
                else c[k] = p[k];
            }
            out[x] = qRgb(c[0], c[1], c[2]);
        }
    }
    current_ = "frame";
    view_->setImage(forDisplay(img), false);
    QFileInfo fi(frames_[row]);
    statusBar()->showMessage(QString("%1 — %2 x %3, %4-bit").arg(fi.fileName()).arg(info.width).arg(info.height).arg(info.bits));
}

void MainWindow::render() {
    if (folder_.isEmpty()) { chooseFolder(); if (folder_.isEmpty()) return; }
    if (worker_.running()) return;
    JobOptions jo;
    jo.align.dir = folder_.toStdWString();
    jo.align.ref_rule = ref_->currentIndex() == 0 ? "middle" : ref_->currentIndex() == 1 ? "first" : "last";
    jo.align.lock_scale = lockScale_->isChecked();
    if (fov_->value() > 0) jo.align.um_per_px = -fov_->value();
    jo.stack.po.g.scale = scale_->currentIndex() == 0 ? 1.0 : 0.5;
    jo.stack.po.opt.streams_per_gpu = 2;
    jo.stack.streams = 2;
    jo.stack.c_smoothing = smoothing_->value();
    jo.stack.outputs.clear();
    for (const char* n : kOutputs) if (outputs_[n]->isChecked()) jo.stack.outputs.push_back(n);
    if (jo.stack.outputs.empty()) { QMessageBox::information(this, "xLamina", "Choose at least one output."); return; }
    jo.out = outDir_->text().isEmpty() ? std::string() : outDir_->text().toStdString();
    cache_.clear();
    result_.reset();
    while (tabs_->count()) tabs_->removeTab(0);
    log_->clear();
    setBusy(true);
    worker_.start(jo);
}

void MainWindow::cancel() {
    worker_.cancel();
    stage_->setText("Cancelling after the current frames…");
}

void MainWindow::setBusy(bool busy) {
    renderBtn_->setEnabled(!busy);
    cancelBtn_->setEnabled(busy);
    saveBtn_->setEnabled(!busy && result_ && !result_->result.names.empty());
    frameList_->setEnabled(!busy);
    progress_->setValue(0);
}

void MainWindow::onProgress(QString stage, double fraction) {
    stage_->setText(stage);
    progress_->setValue((int)std::lround(std::clamp(fraction, 0.0, 1.0) * 1000));
}

void MainWindow::onPreview(QString name, QImage image) {
    if (!name.startsWith("depth")) image = forDisplay(image);
    if (name == "A" && stage_->text().startsWith("pass 1")) {   // the stack building, while the frames take the main view
        frameView_->setPixmap(QPixmap::fromImage(image).scaled(frameView_->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
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
    g += QString("\nfusion %1 s; report.json and the TIFFs in %2").arg(d.result.seconds, 0, 'f', 1).arg(QDir::toNativeSeparators(d.outDir));
    geometry_->setText(g);
}

void MainWindow::onFinished(std::shared_ptr<JobResultData> data) {
    result_ = data;
    setBusy(false);
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
    bool keep = current_ != "frame" && !current_.contains("building");
    current_ = name;
    view_->setImage(img, keep);
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
