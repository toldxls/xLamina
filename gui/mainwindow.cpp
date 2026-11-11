#include "mainwindow.h"

#include <QActionGroup>
#include <QApplication>
#include <QClipboard>
#include <QCloseEvent>
#include <QCollator>
#include <QSignalBlocker>
#include <QDesktopServices>
#include <QDialog>
#include <QDialogButtonBox>
#include <QDir>
#include <QDockWidget>
#include <QEventLoop>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QImageReader>
#include <QImageWriter>
#include <QMenuBar>
#include <QMessageBox>
#include <QMimeData>
#include <QScrollArea>
#include <QSplitter>
#include <QStatusBar>
#include <QTabWidget>
#include <QDateTime>
#include <QTextBrowser>
#include <QToolBar>
#include <QToolButton>
#include <tuple>
#include <QUrl>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>

#include "stackcore/tiff.h"
#include "version.h"

using namespace stackcore;

static const double kDeg = 180.0 / 3.14159265358979323846;
static const char* kOutputs[] = {"B", "C", "Cn", "A", "B0", "CB", "CBn", "Cx", "Cd", "Cg", "Cg2", "H"};
static const int kNumOutputs = 12;
static const char* kOutputHelp[] = {
    "B: depth map. Each pixel from the frame where it is sharpest (regularised, refined, soft edges). Cleanest tones.",
    "C: Laplacian pyramid. Per-detail selection across all frames. Most detail; see Smoothing for its noise.",
    "Cn: noise-aware pyramid. C where the detail is above the noise, B elsewhere.",
    "A: focus-weighted average of all frames. Smooth, soft.",
    "B0: raw per-pixel argmax of the focus measure (unregularised B).",
    "CB: pyramid, falling back to B where C's choice disagrees with the depth map (no noise rule: flat areas keep C's grain).",
    "CBn: CB plus the noise rule: pyramid detail only where it is above the noise, B elsewhere.",
    "Cx: CBn with the finer level's support in place of the dominance test: where C's choice disagrees with the depth map it\n"
    "stays only if the level below chose the same frame. Against the glow of defocused highlights (FINDINGS 50).",
    "Cd: pyramid locked to B's depth at the fine levels (against glints from other depths).",
    "Cg: pyramid with coarse levels from B (against glare from defocused highlights).",
    "Cg2: Cg with one more coarse level from B.",
    "H: luma from CBn, chroma from B."};
// A, B0 and CB are not in the Outputs list: Tools > Developer outputs (developer mode) ticks them
// (CB since 0.6.23: CBn is the better of the two)
static bool isMenuOutput(int i) { return i == 3 || i == 4 || i == 5; }
static const int kMaxRecent = 8;

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
    setWindowTitle(QString("xLamina %1").arg(XLAMINA_VERSION));
    resize(1500, 950);
    setAcceptDrops(true);
    gpus_ = enumerateGpus();
    buildMenus();
    buildUi();
    loadSettings();
    connect(&worker_, &Worker::progress, this, &MainWindow::onProgress);
    connect(worker_.player(), &LivePlayer::previewReady, this, &MainWindow::onPreview);   // converted off the GUI thread
    connect(worker_.player(), &LivePlayer::view, this, &MainWindow::onLiveView);   // paced, in stack order (0.6.32)
    connect(view_, &ImageView::viewChanged, this, &MainWindow::updateViewWindow);
    // the engine asks, from its threads, what part of the frame to render for each live view (0.6.33)
    worker_.setViewWindowProvider([this](int W, int H) {
        QRectF f;
        int vw;
        {
            std::lock_guard<std::mutex> lk(viewWinM_);
            f = viewFrac_;
            vw = viewportW_;
        }
        stackcore::ViewWindow w;
        // a margin of a fifth around the visible part, so a small pan does not show the edge of the crop
        double mx = f.width() * 0.2, my = f.height() * 0.2;
        double x0 = std::max(0.0, f.x() - mx), y0 = std::max(0.0, f.y() - my);
        double x1 = std::min(1.0, f.x() + f.width() + mx), y1 = std::min(1.0, f.y() + f.height() + my);
        w.x0 = (int)std::floor(x0 * W);
        w.y0 = (int)std::floor(y0 * H);
        w.w = std::max(1, (int)std::ceil((x1 - x0) * W));
        w.h = std::max(1, (int)std::ceil((y1 - y0) * H));
        // the screen shows the visible part over the viewport's width: sample every down-th pixel to match it
        w.down = std::max(1, (int)std::floor(f.width() * W / std::max(1, vw)));
        return w;
    });
    connect(&worker_, &Worker::logLine, this, &MainWindow::onLog);
    connect(&worker_, &Worker::finished, this, &MainWindow::onFinished);
    connect(&updater_, &Updater::available, this, &MainWindow::onUpdateAvailable);
    connect(&updater_, &Updater::upToDate, this, [this](const QString& v) {
        if (updateManual_) {
            // a build installed by hand runs ahead of the releases ("0.6.29 is the latest" while running 0.6.33)
            QString msg = Updater::compareVersions(v, XLAMINA_VERSION) < 0
                              ? QString("You are running xLamina %1, newer than the latest release (%2).").arg(XLAMINA_VERSION, v)
                              : QString("xLamina %1 is up to date: %1 is the latest release.").arg(v);
            QMessageBox::information(this, "Check for updates", msg);
        }
        updateManual_ = false;
    });
    connect(&updater_, &Updater::failed, this, [this](const QString& why) {
        if (updateDlg_) return;   // the dialog shows its own download failures
        if (updateManual_) QMessageBox::warning(this, "Check for updates", why);
        else onLog("Update check: " + why);
        updateManual_ = false;
    });
    if (appSettings().value("updates/auto", true).toBool()) QTimer::singleShot(2500, this, [this] { if (!updater_.busy()) updater_.check(); });
    connect(&thumbs_, &ThumbLoader::thumbnail, this, &MainWindow::onThumbnail);
    connect(&frameLoader_, &ThumbLoader::thumbnail, this, &MainWindow::onFullFrame);
    connect(&compareLoader_, &ThumbLoader::thumbnail, this, &MainWindow::onComparison);
    tick_.setInterval(250);
    connect(&tick_, &QTimer::timeout, this, [this] {
        setStage(QString("%1 — %2 s").arg(stageText_).arg(elapsed_.elapsed() / 1000.0, 0, 'f', 1));
    });
    listTick_.setInterval(250);
    connect(&listTick_, &QTimer::timeout, this, [this] {
        if (liveItem_ == markedItem_) return;
        if (markedItem_) markedItem_->setBackground(QBrush());
        markedItem_ = liveItem_;
        if (markedItem_) {
            markedItem_->setBackground(QColor(70, 130, 170, 90));   // a calm blue-grey, unlike the selection
            frameList_->scrollToItem(markedItem_, QAbstractItemView::EnsureVisible);
        }
    });
    QStringList g;
    for (const GpuInfo& gi : gpus_) g << QString("%1 (%2 GB)").arg(QString(gi.name).remove("NVIDIA ").remove("GeForce ")).arg(gi.memory / 1073741824.0, 0, 'f', 0);
    statusBar()->showMessage(gpus_.isEmpty() ? "No CUDA GPU found: rendering is not possible on this machine."
                                             : QString("%1 GPU%2: %3. Drop a folder of frames here.").arg(gpus_.size()).arg(gpus_.size() == 1 ? "" : "s").arg(g.join(", ")));
}

QImage MainWindow::forDisplay(QImage img) const {
    if (!colorSpace_.isValid() || img.isNull()) return img;
    img.setColorSpace(colorSpace_);
    img.convertToColorSpace(QColorSpace::SRgb);
    return img;
}

// ----------------------------------------------------------------------------- menus

void MainWindow::buildMenus() {
    QMenu* file = menuBar()->addMenu("&File");
    QAction* open = file->addAction("&Open folder…", QKeySequence::Open, this, &MainWindow::chooseFolder);
    open->setStatusTip("Open a folder of TIFF or JPEG frames (or drop one onto the window).");
    recentMenu_ = file->addMenu("Open &recent");
    batchAct_ = file->addAction("&Batch render…", QKeySequence("Ctrl+Shift+B"), this, &MainWindow::batch);
    batchAct_->setStatusTip("Queue several folders and render them one after the other with the current settings.");
    file->addSeparator();
    QAction* prefs = file->addAction("&Preferences…", QKeySequence("Ctrl+,"), this, &MainWindow::preferences);
    prefs->setStatusTip("Theme, updates, developer controls, the hardware settings, and this machine's details.");
    file->addSeparator();
    QAction* cmp = file->addAction("Add &comparison image…", QKeySequence("Ctrl+I"), this, &MainWindow::addComparison);
    cmp->setStatusTip("Load another stacker's output (TIFF, JPEG, PNG) as a tab, to flip against xLamina's at the same zoom.");
    saveAct_ = file->addAction("&Save as…", QKeySequence::Save, this, &MainWindow::saveCurrent);
    saveAct_->setEnabled(false);
    openOutAct_ = file->addAction("Open output &folder", QKeySequence("Ctrl+Shift+O"), this, &MainWindow::openOutputFolder);
    openOutAct_->setEnabled(false);
    file->addSeparator();
    file->addAction("&Quit", QKeySequence("Ctrl+Q"), this, &QWidget::close);

    QMenu* render = menuBar()->addMenu("&Render");
    renderAct_ = render->addAction("&Render", QKeySequence("F5"), this, &MainWindow::render);
    renderAct_->setShortcuts({QKeySequence("F5"), QKeySequence("Ctrl+R")});
    cancelAct_ = render->addAction("&Cancel", QKeySequence("Esc"), this, &MainWindow::cancel);
    cancelAct_->setEnabled(false);
    render->addSeparator();
    dropShifted_ = render->addAction("Drop s&hifted frames", this, &MainWindow::dropShifted);
    dropShifted_->setEnabled(false);
    dropShifted_->setToolTip("Untick the frames in red: the shift from the previous frame is an outlier (the stage moved); then Render again.");
    dropSoft_ = render->addAction("Drop s&oft frames", this, &MainWindow::dropSoft);
    dropSoft_->setEnabled(false);
    dropSoft_->setToolTip("Untick the frames in amber: the focus energy fell under the neighbours' (blurred during the exposure); then Render again.");
    dropWarped_ = render->addAction("Drop &warped frames", this, &MainWindow::dropWarped);
    dropWarped_->setEnabled(false);
    dropWarped_->setToolTip("Untick the frames in violet: their rolling-shutter displacement from band to band is far above the stack's\n"
                            "(shaken during the readout, beyond what Correct rolling-shutter warp can put back); then Render again.");
    tickAll_ = render->addAction("&Tick all frames", this, &MainWindow::tickAllFrames);
    tickAll_->setToolTip("Tick every frame in the list again (after dropping frames, or to start over).");
    render->addSeparator();
    // the list is the stack order ("more options for sorting frames. By time taken, alphabetical")
    sortName_ = render->addAction("Sort frames by &name", this, [this] { sortFrames(false); });
    sortName_->setToolTip("Order the list, and so the stack, by file name (numbers in order: IMG_9 before IMG_10).");
    sortTime_ = render->addAction("Sort frames by time ta&ken", this, [this] { sortFrames(true); });
    sortTime_->setToolTip("Order the list by the capture time in each file's EXIF (DateTimeOriginal with its sub-seconds).\n"
                          "A file without one (a JPEG, or a TIFF saved without EXIF) takes its modification time.");
    render->addSeparator();
    render->addAction("Reset settings to &defaults", this, &MainWindow::resetSettings);

    QMenu* view = menuBar()->addMenu("&View");
    view->addAction("&Fit", QKeySequence("Ctrl+0"), this, [this] { view_->fit(); });
    view->addAction("&1:1", QKeySequence("Ctrl+1"), this, [this] { view_->actualSize(); });
    view->addSeparator();
    view->addAction("&Next image", QKeySequence("PgDown"), this, [this] { stepOutput(1); });
    view->addAction("&Previous image", QKeySequence("PgUp"), this, [this] { stepOutput(-1); });
    view->addSeparator();
    QMenu* theme = view->addMenu("&Theme");
    QActionGroup* tg = new QActionGroup(this);
    for (auto [name, key] : std::initializer_list<std::pair<const char*, const char*>>{{"&Dark", "dark"}, {"&Light", "light"}, {"&System (after restart)", "system"}}) {
        QAction* a = theme->addAction(name);
        a->setCheckable(true);
        a->setData(key);
        tg->addAction(a);
        connect(a, &QAction::triggered, this, [this, k = QString(key)] { setTheme(k); });
    }
    theme->setProperty("group", QVariant::fromValue((QObject*)tg));
    view->addSeparator();
    devAct_ = view->addAction("&Developer controls");
    devAct_->setCheckable(true);
    devAct_->setStatusTip("Show the Advanced depth / composition group, the C and B radii, and the outputs beyond B and C.");
    connect(devAct_, &QAction::toggled, this, &MainWindow::setDeveloper);
    // the panels (filled in buildUi, after the docks exist)
    view->addSeparator()->setObjectName("panels");

    QMenu* cmpMenu = menuBar()->addMenu("&Compare");
    QAction* two = cmpMenu->addAction("&Two-up viewer…", QKeySequence("Ctrl+2"), this, &MainWindow::twoUp);
    two->setStatusTip("Two images side by side with one pan and zoom: this render's outputs, comparison images, sweep variants. Space flips them.");
    QAction* run = cmpMenu->addAction("&Run sweep…", QKeySequence("Ctrl+Shift+R"), this, &MainWindow::runSweep);
    run->setStatusTip("Render B with several settings at once: one alignment, one pass 1 per group, a B per combination; then compare them two up.");
    QAction* sweep = cmpMenu->addAction("Open &sweep folder…", QKeySequence("Ctrl+Shift+W"), this, &MainWindow::openSweep);
    sweep->setStatusTip("A folder written by a sweep: its B variants, with their options, into the two-up viewer.");

    QMenu* tools = menuBar()->addMenu("&Tools");
    QAction* bench = tools->addAction("&Benchmark…", QKeySequence("Ctrl+B"), this, &MainWindow::benchmark);
    bench->setStatusTip("Time the engine under different hardware settings and apply the fastest.");
    tools->addSeparator();
    tools->addAction("Copy &log", this, [this] { QApplication::clipboard()->setText(log_->toPlainText()); statusBar()->showMessage("Log copied.", 3000); });
    tools->addAction("Copy &geometry report", this, [this] { QApplication::clipboard()->setText(geometry_->text()); statusBar()->showMessage("Geometry report copied.", 3000); });
    tools->addAction("Copy report.&json", this, [this] {
        if (result_) { QApplication::clipboard()->setText(result_->reportJson); statusBar()->showMessage("report.json copied.", 3000); }
    });
    tools->addSeparator();
    // the outputs that left the Outputs list (A, B0) stay reachable here in developer mode;
    // the actions mirror hidden check boxes, filled in with the Outputs group below
    devOutMenu_ = tools->addMenu("Developer &outputs");
    devOutMenu_->setStatusTip("A (focus-weighted average) and B0 (raw argmax): developer outputs, rendered when ticked.");

    QMenu* help = menuBar()->addMenu("&Help");
    QAction* chk = help->addAction("Check for &updates…", this, &MainWindow::checkForUpdates);
    chk->setStatusTip("Ask GitHub for the latest release; a newer one can be installed from here and xLamina restarts.");
    autoUpdateAct_ = help->addAction("Check for updates at &start");
    autoUpdateAct_->setCheckable(true);
    autoUpdateAct_->setChecked(appSettings().value("updates/auto", true).toBool());
    autoUpdateAct_->setStatusTip("A few seconds after xLamina starts, ask GitHub once; a newer release lights the Update button.");
    connect(autoUpdateAct_, &QAction::toggled, this, [](bool on) { appSettings().setValue("updates/auto", on); });
    help->addSeparator();
    help->addAction("&About xLamina", this, &MainWindow::about);
}

// ----------------------------------------------------------------------------- the window

void MainWindow::buildUi() {
    // ---- toolbar
    QToolBar* tb = addToolBar("Main");
    tb->setObjectName("toolbar");
    tb->setMovable(false);
    tb->setToolButtonStyle(Qt::ToolButtonTextOnly);
    tb->addAction("Open folder…", this, &MainWindow::chooseFolder);
    tb->addSeparator();
    renderBtn_ = new QPushButton("Render");
    renderBtn_->setDefault(true);
    renderBtn_->setToolTip("Align and fuse the ticked frames (F5)");
    cancelBtn_ = new QPushButton("Cancel");
    cancelBtn_->setEnabled(false);
    saveBtn_ = new QPushButton("Save as…");
    saveBtn_->setEnabled(false);
    tb->addWidget(renderBtn_);
    tb->addWidget(cancelBtn_);
    tb->addWidget(saveBtn_);
    tb->addSeparator();
    tb->addAction("Fit", [this] { view_->fit(); })->setToolTip("Fit the image in the view (Ctrl+0, or double-click)");
    tb->addAction("1:1", [this] { view_->actualSize(); })->setToolTip("One image pixel per screen pixel (Ctrl+1)");
    tb->addSeparator();
    tb->addAction(dropShifted_);
    tb->addAction(dropSoft_);
    tb->addAction(dropWarped_);
    tb->addSeparator();
    tb->addAction("Benchmark…", this, &MainWindow::benchmark);
    // the Update button, at the right end, shown only when a newer release exists
    QWidget* spacer = new QWidget;
    spacer->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    tb->addWidget(spacer);
    updateBtn_ = new QToolButton;
    updateBtn_->setToolButtonStyle(Qt::ToolButtonTextOnly);
    updateBtn_->setStyleSheet("QToolButton { background: #7a2fc9; color: white; font-weight: bold; padding: 5px 14px; border-radius: 4px; }"
                              "QToolButton:hover { background: #9a4fe9; }");
    connect(updateBtn_, &QToolButton::clicked, this, &MainWindow::showUpdate);
    updateAct_ = tb->addWidget(updateBtn_);
    updateAct_->setVisible(false);
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
    // the active output must be unmistakable: the selected tab in the accent colour, and the viewer names its image
    // colour only, no bold: a bold selected tab was wider, so stepping B -> C re-laid the whole row
    tabs_->setStyleSheet("QTabBar::tab { padding: 6px 16px; margin-right: 2px; }"
                         "QTabBar::tab:selected { background: #7a2fc9; color: white; }");
    tabs_->setToolTip("The outputs of the last render and any comparison images; PgUp / PgDown step through them at the same zoom.");
    // Compare and Retouch, centred above the view: the same as Compare > Two-up viewer, and the
    // viewer with retouching switched on
    QWidget* topRow = new QWidget;
    QHBoxLayout* tl = new QHBoxLayout(topRow);
    tl->setContentsMargins(0, 4, 0, 4);
    tl->setSpacing(2);
    tl->addStretch(1);
    for (auto [label, tip, slot] : {std::tuple<const char*, const char*, void (MainWindow::*)()>{
                                        "Compare", "Two images side by side with one pan and zoom (Ctrl+2)", &MainWindow::twoUp},
                                    {"Retouch", "Paint one image into another in the two-up viewer (R there)", &MainWindow::retouch}}) {
        QToolButton* b = new QToolButton;
        b->setText(label);
        b->setToolTip(tip);
        b->setToolButtonStyle(Qt::ToolButtonTextOnly);
        b->setStyleSheet("QToolButton { padding: 5px 18px; border: 1px solid #7a2fc9; border-radius: 4px; }"
                         "QToolButton:hover { background: #7a2fc9; color: white; }");
        connect(b, &QToolButton::clicked, this, slot);
        tl->addWidget(b);
    }
    tl->addStretch(1);
    cl->addWidget(topRow);
    cl->addWidget(split_, 1);
    // the stage and its progress bar under the view (they hid at the bottom of the Render column)
    QWidget* progRow = new QWidget;
    QHBoxLayout* prl = new QHBoxLayout(progRow);
    prl->setContentsMargins(6, 4, 6, 2);
    prl->setSpacing(10);
    stage_ = new QLabel;
    stage_->hide();                       // the text lives inside the bar (setStage); the label keeps it for the code that reads it
    progress_ = new QProgressBar;
    progress_->setRange(0, 1000);
    progress_->setValue(0);
    progress_->setMinimumHeight(24);
    progress_->setTextVisible(true);
    progress_->setAlignment(Qt::AlignCenter);
    prl->addWidget(progress_, 1);
    cl->addWidget(progRow);
    setStage("Open a folder of TIFF frames, choose the outputs, press Render.");
    cl->addWidget(tabs_);
    setCentralWidget(centre);
    connect(tabs_, &QTabBar::currentChanged, this, &MainWindow::showOutput);
    connect(view_, &ImageView::zoomChanged, this, [this](double z) { status_->setText(QString("%1 %").arg(z * 100, 0, 'f', 0)); });

    // ---- left: frames
    framesDock_ = new QDockWidget("Frames", this);
    framesDock_->setObjectName("frames");
    framesDock_->setFeatures(QDockWidget::DockWidgetMovable | QDockWidget::DockWidgetClosable);
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
    // the master tick under the dock's title, in line with the rows' ticks: ticked = every frame
    // in the render, clear = none, partly filled = some; a click ticks every frame unless all were ticked, then it clears them
    QWidget* framesBox = new QWidget;
    QVBoxLayout* fbl = new QVBoxLayout(framesBox);
    fbl->setContentsMargins(0, 0, 0, 0);
    fbl->setSpacing(0);
    QWidget* head = new QWidget;
    QHBoxLayout* hl = new QHBoxLayout(head);
    hl->setContentsMargins(4, 3, 4, 3);
    allFrames_ = new QCheckBox("All frames");
    allFrames_->setTristate(true);
    allFrames_->setToolTip("Tick or untick every frame of the stack at once. Partly filled: some frames are unticked.");
    hl->addWidget(allFrames_);
    hl->addStretch(1);
    fbl->addWidget(head);
    fbl->addWidget(frameList_, 1);
    connect(allFrames_, &QCheckBox::clicked, this, [this] {
        if (worker_.running()) { statusBar()->showMessage("A render is running; the list changes when it has finished.", 4000); syncAllFramesBox(); return; }
        // the box has already moved on from the state it showed; decide from the rows themselves
        int ticked = 0, total = 0;
        for (int r = 0; r < frameList_->count(); ++r) {
            QListWidgetItem* o = frameList_->item(r);
            if (o->data(Qt::UserRole).toInt() < 0) continue;
            ++total;
            if (o->checkState() == Qt::Checked) ++ticked;
        }
        const Qt::CheckState to = ticked == total && total > 0 ? Qt::Unchecked : Qt::Checked;
        for (int r = 0; r < frameList_->count(); ++r) {
            QListWidgetItem* o = frameList_->item(r);
            if (o->data(Qt::UserRole).toInt() >= 0) o->setCheckState(to);
        }
        syncAllFramesBox();
        statusBar()->showMessage(to == Qt::Checked ? "Every frame is in the next render." : "Every frame unticked; tick the ones to stack.", 4000);
    });
    connect(frameList_, &QListWidget::itemChanged, this, [this](QListWidgetItem*) { syncAllFramesBox(); });
    framesDock_->setWidget(framesBox);
    addDockWidget(Qt::LeftDockWidgetArea, framesDock_);
    connect(frameList_, &QListWidget::currentRowChanged, this, &MainWindow::showFrame);
    connect(frameList_, &QListWidget::customContextMenuRequested, this, &MainWindow::frameMenu);

    // ---- right: parameters (in a scroll area: the panel is taller than small screens)
    QDockWidget* params = new QDockWidget("Render", this);
    params->setObjectName("render");
    params->setFeatures(QDockWidget::DockWidgetMovable | QDockWidget::DockWidgetClosable);
    QWidget* pw = new QWidget;
    QVBoxLayout* pl = new QVBoxLayout(pw);

    QGroupBox* og = new QGroupBox("Outputs");
    QGridLayout* ol = new QGridLayout(og);
    int pos = 0;
    for (int i = 0; i < kNumOutputs; ++i) {
        QCheckBox* cb = new QCheckBox(kOutputs[i], og);
        cb->setToolTip(kOutputHelp[i]);
        cb->setChecked(i < 2);   // B and C; the rest are developer outputs (View > Developer controls)
        outputs_[kOutputs[i]] = cb;
        if (isMenuOutput(i)) {   // A and B0: off the list, ticked from Tools > Developer outputs instead (the box stays the state)
            cb->hide();
            QAction* act = devOutMenu_->addAction(kOutputs[i]);
            act->setCheckable(true);
            act->setChecked(cb->isChecked());
            act->setStatusTip(kOutputHelp[i]);
            connect(act, &QAction::toggled, cb, &QCheckBox::setChecked);
            connect(cb, &QCheckBox::toggled, act, &QAction::setChecked);
            continue;
        }
        ol->addWidget(cb, pos / 4, pos % 4);
        ++pos;
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
    // the rolling shutter's two corrections (FINDINGS 51). Both are measured by every alignment and shown in the frame
    // list; these put the measurements into the warps. They live in File > Preferences > General; the warp correction
    // is on by default (it removes most doubled edges), banding off.
    fixRs_ = new QCheckBox("Correct rolling-shutter warp");
    fixRs_->setChecked(true);
    fixRs_->setToolTip("Checked: each frame's rows are shifted back by the rolling-shutter displacement the alignment measured\n"
                       "(the \"warp\" value in the frame list), which removes doubled edges. Saved files get \"_rs\" in the name.\n"
                       "Unchecked: no correction.");
    fixBanding_ = new QCheckBox("Correct LED banding");
    fixBanding_->setToolTip("Checked: each frame's rows are divided by the brightness ripple the LED's flicker left across them\n"
                            "(the \"band\" value in the frame list), so regions taken from different frames meet at the same brightness.\n"
                            "Saved files get \"_band\" in the name. Unchecked: no correction.");
    liveRes_ = new QComboBox;
    liveRes_->addItem("Off (fastest)", 0);
    liveRes_->addItem("Low (1/12)", 12);
    liveRes_->addItem("Medium (1/6)", 6);
    liveRes_->addItem("High (1/3)", 3);
    liveRes_->addItem("Full (what the view shows)", 1);
    liveRes_->setCurrentIndex(2);
    liveRes_->setToolTip("Resolution of the live view while rendering, as a fraction of the output. Full renders only the part of the\n"
                         "frame the view shows, at the screen's resolution: a quarter-resolution frame at Fit, a full-resolution crop at\n"
                         "1:1, about 2 Mpx either way, so it sweeps as smoothly as High. Every image copied off a GPU costs its uploads\n"
                         "about 3 ms, so views are limited to 30 a second. Off shows only the finished outputs.");
    fl->addRow("Live view", liveRes_);
    // the warp's resampling filter (Helicon's Preferences offer the same choice)
    interp_ = new QComboBox;
    interp_->addItems({"Bilinear", "Bicubic", "Lanczos 3", "Lanczos 8"});   // the order of stackcore::Interp
    interp_->setCurrentIndex(1);
    interp_->setToolTip("How every frame is resampled onto the output grid when it is aligned.\n"
                        "Bicubic (the default, OpenCV's a = -0.75) is the reference the engine was validated with. Lanczos 3 keeps about\n"
                        "1 % more fine detail for 0.3 s more per render; Lanczos 8 about 3 % more for 3.3 s more (about 150 frames, FINDINGS 39).\n"
                        "Bilinear is softest and fastest (8 % less). Changing it changes every output.");
    // (not in the Render column: File > Preferences > General holds it; the setting is kept as before)
    // the soft-frame cutoff (FINDINGS 49)
    softCutoff_ = new QDoubleSpinBox;
    softCutoff_->setRange(0.50, 0.99);
    softCutoff_->setSingleStep(0.01);
    softCutoff_->setDecimals(2);
    softCutoff_->setValue(0.90);
    softCutoff_->setToolTip("A frame is soft (amber, Render > Drop soft frames) when its focus energy is under this share of its neighbours'\n"
                            "(the median of the five frames around it; the list shows the share as \"sharp 0.97\"). 0.90 by default: a frame\n"
                            "blurred by movement during its exposure loses a tenth or more; the frame-to-frame scatter of a steady stack is\n"
                            "under 0.05. Changing it re-flags the last render's frames at once, no render needed.");
    connect(softCutoff_, &QDoubleSpinBox::valueChanged, this, [this](double) {
        if (result_ && (int)result_->align.T.size() == jobItems_.size()) labelWobble(result_->align, &result_->result);
    });
    mainForm_ = fl;
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
                         "0 = raw selection. Larger values lose small features that sit near a stronger edge at another depth.\n"
                         "Also changes B, whose guide image is C.");
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
    fov_->setDecimals(2);   // two decimals; the report keeps four
    fov_->setSingleStep(0.01);
    fov_->setSuffix(" mm");
    fov_->setSpecialValueText("unknown");
    fov_->setToolTip("Horizontal field of view of one frame; gives the report its µm/px.");
    fl->addRow("Field of view", fov_);
    outDir_ = new QLineEdit;
    outDir_->setPlaceholderText("<folder>_stacker");
    outDir_->setToolTip("Where the TIFFs and report.json are written after a render.");
    fl->addRow("Output folder", outDir_);
    pl->addWidget(sg);

    // ---- advanced: the depth / composition parameters, at the validated defaults unless ticked
    advanced_ = new QGroupBox("Advanced (depth and composition)");
    advanced_->setCheckable(true);
    advanced_->setChecked(false);
    advanced_->setToolTip("The parameters of B's depth pipeline and the composition. Unticked, the validated defaults are used.\n"
                          "They were tuned region by region against Helicon's output; change one at a time and compare at 1:1.");
    QFormLayout* al = new QFormLayout(advanced_);
    auto dspin = [](double lo, double hi, double step, double val, int dec, const QString& suffix, const QString& tip) {
        QDoubleSpinBox* s = new QDoubleSpinBox;
        s->setRange(lo, hi);
        s->setSingleStep(step);
        s->setDecimals(dec);
        s->setValue(val);
        s->setSuffix(suffix);
        s->setToolTip(tip);
        return s;
    };
    focusRadius_ = dspin(1, 30, 1, 6, 0, " px", "Focus measure window (pass 1), in sensor pixels. Default 6.");
    al->addRow("Focus window", focusRadius_);
    focusPower_ = dspin(1, 8, 0.5, 4, 1, "", "Exponent on the focus measure when the frames compete. Default 4.");
    al->addRow("Focus power", focusPower_);
    depthRes_ = new QComboBox;
    depthRes_->addItem("1/4 of the sensor (fast)", 0.25);
    depthRes_->addItem("1/2 of the sensor (default)", 0.5);
    depthRes_->addItem("Full (small stacks only)", 1.0);
    depthRes_->setCurrentIndex(1);
    depthRes_->setToolTip("Resolution of the depth volume. 1/2 (the default since 0.6.1) removes the doubled edges of bright plates that 1/4\n"
                          "left; it needs about 6 GB of GPU memory for 200 full frames and costs about 3 s. Full needs about 25 GB for such a stack.");
    al->addRow("Depth resolution", depthRes_);
    depthFmPx_ = dspin(1, 30, 1, 6, 0, " px", "Window of the normalised Laplacian in the depth volume, in depth-volume pixels. Default 6;\n"
                                               "3 at 1/2 resolution gave the tightest edges on the synthetic stack (FINDINGS 31).");
    al->addRow("Depth window", depthFmPx_);
    volAggPx_ = dspin(0, 128, 4, 32, 0, " px", "Edge-aware aggregation of the depth volume, in sensor pixels; against ghost lines beside edges. Default 32.");
    al->addRow("Volume aggregation", volAggPx_);
    refineR_ = new QSpinBox;
    refineR_->setRange(0, 80);
    refineR_->setValue(12);
    refineR_->setSuffix(" frames");
    refineR_->setToolTip("Fine focus search around the coarse depth, in frames either way. Default 12; 0 skips the refinement.");
    al->addRow("Refine range", refineR_);
    refineFmPx_ = dspin(1, 40, 1, 9, 0, " px", "Window of the fine focus measure in the refinement pass. Default 9.");
    al->addRow("Refine window", refineFmPx_);
    softPx_ = dspin(0, 30, 0.5, 3, 1, " px", "Softness of the frame labels when B is composed; the blur of the seams. Default 3.");
    al->addRow("Soft labels", softPx_);
    haloPrior_ = dspin(0, 10, 0.5, 2, 1, "", "B's preference for darker frames beside bright objects (the spill zone). Default 2; raise it to push the band\n"
                                              "beside a bright plate to its own frame.");
    al->addRow("Halo prior", haloPrior_);
    darkPrior_ = dspin(0, 10, 0.5, 1, 1, "", "B's preference for darker frames in featureless pixels (defocused background). Default 1. It is switched\n"
                                              "off within the dark reach of sharp content, where a featureless pixel is a gap between needles or the inside\n"
                                              "of a rim and must follow its neighbours instead (FINDINGS 34).");
    al->addRow("Dark prior", darkPrior_);
    darkReach_ = dspin(0, 200, 4, 24, 0, " px", "How far from sharp content the dark prior stays off, in sensor pixels. Default 24; 0 applies the prior to\n"
                                                 "every featureless pixel (the 0.5.x behaviour, which put far frames into the gaps of needle fans at 1/2 resolution).");
    al->addRow("Dark reach", darkReach_);
    flatBlur_ = dspin(0, 512, 16, 256, 0, " px", "Defocused background (no sharp detail within the dark reach): a band of about this width beside sharp detail\n"
                                                  "takes that detail's frame, beyond it each pixel takes the frame where its own focus measure peaks, smoothed\n"
                                                  "blob by blob (FINDINGS 40), and the composition blends up to 32 frames there (FINDINGS 41); its focus curves\n"
                                                  "are also smoothed over this width (FINDINGS 34). Default 256 (128 before 0.6.22; FINDINGS 44); 0 = off\n"
                                                  "(the depth stage's own choice, which breaks the background into patches with visible seams).");
    al->addRow("Flat smoothing", flatBlur_);
    QPushButton* resetAdv = new QPushButton("Defaults");
    resetAdv->setToolTip("Put every advanced parameter back to its validated default.");
    connect(resetAdv, &QPushButton::clicked, this, [this] {
        depthRes_->setCurrentIndex(1);
        focusRadius_->setValue(6); focusPower_->setValue(4); depthFmPx_->setValue(6); volAggPx_->setValue(32);
        refineR_->setValue(12); refineFmPx_->setValue(9); softPx_->setValue(3); haloPrior_->setValue(2); darkPrior_->setValue(1);
        darkReach_->setValue(24); flatBlur_->setValue(256);
    });
    al->addRow("", resetAdv);
    pl->addWidget(advanced_);

    // ---- performance: what the benchmark tunes. Machine settings, not render settings: the group lives in
    // File > Preferences, built here because perfToUi / perfFromUi read its widgets
    perfGroup_ = new QGroupBox("Performance");
    QGroupBox* perf = perfGroup_;
    perf->setToolTip("Hardware settings. They change the speed, never the result: B and C are bit-identical across all of them.\n"
                     "Tools > Benchmark tries them and applies the fastest.");
    QFormLayout* pfl = new QFormLayout(perf);
    QWidget* gw = new QWidget;
    QVBoxLayout* gl = new QVBoxLayout(gw);
    gl->setContentsMargins(0, 0, 0, 0);
    gl->setSpacing(2);
    for (const GpuInfo& g : gpus_) {
        QCheckBox* cb = new QCheckBox(g.label());
        cb->setChecked(true);
        cb->setToolTip(QString("%1 streaming multiprocessors. Untick to leave this card out.").arg(g.sms));
        gpuBoxes_ << cb;
        gl->addWidget(cb);
    }
    if (gpus_.isEmpty()) gl->addWidget(new QLabel("no CUDA GPU found"));
    pfl->addRow("GPUs", gw);
    streams_ = new QSpinBox;
    streams_->setRange(1, 4);
    streams_->setValue(2);
    streams_->setToolTip("Upload / compute streams per GPU in the fusion passes. More overlaps transfers with kernels but costs GPU memory per slot.");
    pfl->addRow("Streams per GPU", streams_);
    readers_ = new QSpinBox;
    readers_->setRange(1, 64);
    readers_->setValue(16);
    readers_->setToolTip("TIFF reader threads. Matters when the frames come from disk; resident frames are uploaded without reading.");
    pfl->addRow("Reader threads", readers_);
    pairThreads_ = new QSpinBox;
    pairThreads_->setRange(1, 8);
    pairThreads_->setValue(3);
    pairThreads_->setToolTip("Alignment pairs solved at once per GPU (each on its own stream). The arithmetic per pair is unchanged.");
    pfl->addRow("Pairs in flight", pairThreads_);
    cacheFrames_ = new QCheckBox("Keep the frames resident in host memory");
    cacheFrames_->setChecked(true);
    cacheFrames_->setToolTip("Frames are read once, while the folder's thumbnails load, and kept pinned (up to about half the machine's RAM) for\n"
                             "every pass and the next render of the same folder; a render starts at the alignment and the GPUs pull the frames at\n"
                             "PCIe speed instead of the disk's. A frame changed on disk is read again. Untick to read from disk every pass.");
    pfl->addRow("", cacheFrames_);
    QPushButton* benchBtn = new QPushButton("Benchmark…");
    connect(benchBtn, &QPushButton::clicked, this, &MainWindow::benchmark);
    pfl->addRow("", benchBtn);
    // (not added to the Render column: the Preferences dialog takes the group)

    pl->addStretch(1);
    QScrollArea* scroll = new QScrollArea;
    scroll->setWidget(pw);
    scroll->setWidgetResizable(true);
    scroll->setFrameShape(QFrame::NoFrame);
    scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);   // a narrow dock scrolls instead of clipping the spin boxes
    params->setWidget(scroll);
    params->setMinimumWidth(380);   // the form's fields collapsed to nothing when the window was made small
    addDockWidget(Qt::RightDockWidgetArea, params);

    // ---- bottom: geometry + log
    QDockWidget* geo = new QDockWidget("Geometry report", this);
    geo->setObjectName("geometry");
    geo->setFeatures(QDockWidget::DockWidgetMovable | QDockWidget::DockWidgetClosable);
    geometry_ = new QLabel("—");
    geometry_->setTextInteractionFlags(Qt::TextSelectableByMouse);
    geometry_->setMargin(6);
    geo->setWidget(geometry_);
    addDockWidget(Qt::BottomDockWidgetArea, geo);
    QDockWidget* lg = new QDockWidget("Log", this);
    lg->setObjectName("log");
    lg->setFeatures(QDockWidget::DockWidgetMovable | QDockWidget::DockWidgetClosable);
    log_ = new QPlainTextEdit;
    log_->setReadOnly(true);
    log_->setMaximumBlockCount(2000);
    lg->setWidget(log_);
    addDockWidget(Qt::BottomDockWidgetArea, lg);
    tabifyDockWidget(geo, lg);
    // the live hardware panel (the Preferences line is a start-up snapshot and cannot show a run)
    hwDock_ = new QDockWidget("Hardware", this);
    hwDock_->setObjectName("hardware");
    hwDock_->setFeatures(QDockWidget::DockWidgetMovable | QDockWidget::DockWidgetClosable);
    hw_ = new HwMonitor(gpus_);
    hw_->setPinnedProvider([this] { return worker_.cache()->bytes(); });
    hwDock_->setWidget(hw_);
    addDockWidget(Qt::BottomDockWidgetArea, hwDock_);
    tabifyDockWidget(lg, hwDock_);
    geo->raise();
    resizeDocks({geo, lg}, {110, 110}, Qt::Vertical);
    resizeDocks({framesDock_, params}, {200, 330}, Qt::Horizontal);

    // the panel toggles in the View menu
    for (QMenu* m : menuBar()->findChildren<QMenu*>())
        if (m->title() == "&View")
            for (QDockWidget* d : {framesDock_, params, geo, lg, hwDock_}) m->addAction(d->toggleViewAction());

    status_ = new QLabel("");
    statusBar()->addPermanentWidget(status_);
}

// ----------------------------------------------------------------------------- settings

void MainWindow::loadSettings() {
    QSettings s = appSettings();
    restoreGeometry(s.value("ui/geometry").toByteArray());
    restoreState(s.value("ui/state").toByteArray());
    theme_ = s.value("ui/theme", "dark").toString();
    setDeveloper(s.value("ui/developer", false).toBool());
    for (QMenu* m : menuBar()->findChildren<QMenu*>())
        if (m->title() == "&Theme")
            for (QAction* a : m->actions()) a->setChecked(a->data().toString() == theme_);
    recent_ = s.value("recent").toStringList();
    updateRecent();
    s.beginGroup("render");
    for (const char* n : kOutputs) outputs_[n]->setChecked(s.value(QString("out_%1").arg(n), outputs_[n]->isChecked()).toBool());
    if (!s.contains("cbToMenu")) { outputs_["CB"]->setChecked(false); s.setValue("cbToMenu", true); }   // 0.6.23: CB left the list, once
    scale_->setCurrentIndex(s.value("scale", 0).toInt());
    ref_->setCurrentIndex(s.value("ref", 0).toInt());
    lockScale_->setChecked(s.value("lockScale", false).toBool());
    fixRs_->setChecked(s.value("fixRs", true).toBool());
    fixBanding_->setChecked(s.value("fixBanding", false).toBool());
    liveRes_->setCurrentIndex(s.value("liveRes", 2).toInt());
    interp_->setCurrentIndex(std::clamp(s.value("interp", 1).toInt(), 0, interp_->count() - 1));   // an odd ini value: never a blank combo
    softCutoff_->setValue(s.value("softCutoff", 0.9).toDouble());
    smoothing_->setValue(s.value("smoothing", 2.0).toDouble());
    cradius_->setValue(s.value("cradius", 32.0).toDouble());
    bradius_->setValue(s.value("bradius", 0.0).toDouble());
    fov_->setValue(s.value("fov", 0.0).toDouble());
    s.endGroup();
    s.beginGroup("advanced");
    advanced_->setChecked(s.value("enabled", false).toBool());
    depthRes_->setCurrentIndex(s.value("depthRes", 1).toInt());
    focusRadius_->setValue(s.value("focusRadius", 6.0).toDouble());
    focusPower_->setValue(s.value("focusPower", 4.0).toDouble());
    depthFmPx_->setValue(s.value("depthFmPx", 6.0).toDouble());
    volAggPx_->setValue(s.value("volAggPx", 32.0).toDouble());
    refineR_->setValue(s.value("refineR", 12).toInt());
    refineFmPx_->setValue(s.value("refineFmPx", 9.0).toDouble());
    softPx_->setValue(s.value("softPx", 3.0).toDouble());
    haloPrior_->setValue(s.value("haloPrior", 2.0).toDouble());
    // 0.6.2 saved dark prior 0 (its default then); from 0.6.3 the prior is back on with a reach, so a file from
    // before the reach existed takes the new default rather than the old one
    darkPrior_->setValue(s.contains("darkReach") ? s.value("darkPrior", 1.0).toDouble() : 1.0);
    darkReach_->setValue(s.value("darkReach", 24.0).toDouble());
    // 0.6.4 .. 0.6.21 saved 128, their default; a stored 128 is that default, not a choice, and takes the new 256
    double fb = s.value("flatBlur", 256.0).toDouble();
    flatBlur_->setValue(fb == 128.0 ? 256.0 : fb);
    s.endGroup();
    perfToUi(PerfSettings::load(s));
}

void MainWindow::saveSettings() {
    QSettings s = appSettings();
    s.setValue("ui/geometry", saveGeometry());
    s.setValue("ui/state", saveState());
    s.setValue("ui/theme", theme_);
    s.setValue("ui/developer", developer_);
    s.setValue("recent", recent_);
    s.beginGroup("render");
    for (const char* n : kOutputs) s.setValue(QString("out_%1").arg(n), outputs_[n]->isChecked());
    s.setValue("scale", scale_->currentIndex());
    s.setValue("ref", ref_->currentIndex());
    s.setValue("lockScale", lockScale_->isChecked());
    s.setValue("fixRs", fixRs_->isChecked());
    s.setValue("fixBanding", fixBanding_->isChecked());
    s.setValue("liveRes", liveRes_->currentIndex());
    s.setValue("interp", interp_->currentIndex());
    s.setValue("softCutoff", softCutoff_->value());
    s.setValue("smoothing", smoothing_->value());
    s.setValue("cradius", cradius_->value());
    s.setValue("bradius", bradius_->value());
    s.setValue("fov", fov_->value());
    s.endGroup();
    s.beginGroup("advanced");
    s.setValue("enabled", advanced_->isChecked());
    s.setValue("depthRes", depthRes_->currentIndex());
    s.setValue("focusRadius", focusRadius_->value());
    s.setValue("focusPower", focusPower_->value());
    s.setValue("depthFmPx", depthFmPx_->value());
    s.setValue("volAggPx", volAggPx_->value());
    s.setValue("refineR", refineR_->value());
    s.setValue("refineFmPx", refineFmPx_->value());
    s.setValue("softPx", softPx_->value());
    s.setValue("haloPrior", haloPrior_->value());
    s.setValue("darkPrior", darkPrior_->value());
    s.setValue("darkReach", darkReach_->value());
    s.setValue("flatBlur", flatBlur_->value());
    s.endGroup();
    perfFromUi().save(s);
}

void MainWindow::resetSettings() {
    if (QMessageBox::question(this, "xLamina", "Put every render, advanced and performance setting back to its default?") != QMessageBox::Yes) return;
    for (int i = 0; i < kNumOutputs; ++i) outputs_[kOutputs[i]]->setChecked(i < 2);
    scale_->setCurrentIndex(0);
    ref_->setCurrentIndex(0);
    lockScale_->setChecked(false);
    fixRs_->setChecked(true);
    fixBanding_->setChecked(false);
    liveRes_->setCurrentIndex(2);
    interp_->setCurrentIndex(1);
    softCutoff_->setValue(0.9);
    smoothing_->setValue(2);
    cradius_->setValue(32);
    bradius_->setValue(0);
    advanced_->setChecked(false);
    depthRes_->setCurrentIndex(1);
    focusRadius_->setValue(6); focusPower_->setValue(4); depthFmPx_->setValue(6); volAggPx_->setValue(32);
    refineR_->setValue(12); refineFmPx_->setValue(9); softPx_->setValue(3); haloPrior_->setValue(2); darkPrior_->setValue(1);
    darkReach_->setValue(24); flatBlur_->setValue(256);
    perfToUi(PerfSettings());
    statusBar()->showMessage("Settings reset to the defaults.", 4000);
}

void MainWindow::updateRecent() {
    recentMenu_->clear();
    for (const QString& r : recent_) {
        QAction* a = recentMenu_->addAction(QDir::toNativeSeparators(r));
        connect(a, &QAction::triggered, this, [this, r] { openFolder(r); });
    }
    recentMenu_->setEnabled(!recent_.isEmpty());
    if (!recent_.isEmpty()) {
        recentMenu_->addSeparator();
        recentMenu_->addAction("Clear", this, [this] { recent_.clear(); updateRecent(); });
    }
}

// Developer controls: the Advanced depth / composition group, the C and B radii and the outputs beyond B and C are
// hidden by default (0.6.8). Hidden outputs are not rendered even when ticked; the values of hidden controls still
// apply (an Advanced group left ticked keeps its values), so toggling the menu changes nothing about a render.
void MainWindow::setDeveloper(bool on) {
    developer_ = on;
    if (devAct_ && devAct_->isChecked() != on) { devAct_->blockSignals(true); devAct_->setChecked(on); devAct_->blockSignals(false); }
    if (advanced_) advanced_->setVisible(on);
    if (mainForm_) { mainForm_->setRowVisible(cradius_, on); mainForm_->setRowVisible(bradius_, on); }
    for (int i = 2; i < kNumOutputs; ++i) if (!isMenuOutput(i)) outputs_[kOutputs[i]]->setVisible(on);   // A, B0 stay hidden boxes
    if (devOutMenu_) devOutMenu_->menuAction()->setVisible(on);
}

std::vector<std::string> MainWindow::selectedOutputs() const {
    std::vector<std::string> out;
    for (int i = 0; i < kNumOutputs; ++i)
        if (outputs_[kOutputs[i]]->isChecked() && (developer_ || i < 2)) out.push_back(kOutputs[i]);
    return out;
}

void MainWindow::setTheme(const QString& theme) {
    theme_ = theme;
    applyTheme(theme);
    appSettings().setValue("ui/theme", theme);
    if (theme == "system") statusBar()->showMessage("The system style applies fully after a restart.", 6000);
}

PerfSettings MainWindow::perfFromUi() const {
    PerfSettings p;
    bool all = true;
    for (int i = 0; i < gpuBoxes_.size(); ++i) {
        if (gpuBoxes_[i]->isChecked()) p.gpus << gpus_[i].index;
        else all = false;
    }
    if (all || p.gpus.isEmpty()) p.gpus.clear();   // all cards (none ticked also means all)
    p.streams = streams_->value();
    p.readers = readers_->value();
    p.pairThreads = pairThreads_->value();
    p.cacheFrames = cacheFrames_->isChecked();
    return p;
}

void MainWindow::perfToUi(const PerfSettings& p) {
    for (int i = 0; i < gpuBoxes_.size(); ++i) gpuBoxes_[i]->setChecked(p.gpus.isEmpty() || p.gpus.contains(gpus_[i].index));
    streams_->setValue(p.streams);
    readers_->setValue(p.readers);
    pairThreads_->setValue(p.pairThreads);
    cacheFrames_->setChecked(p.cacheFrames);
}

// ----------------------------------------------------------------------------- folders and frames

void MainWindow::chooseFolder() {
    QString d = QFileDialog::getExistingDirectory(this, "Folder with the stack's frames", folder_.isEmpty() ? recent_.value(0) : folder_);
    if (!d.isEmpty()) openFolder(d);
}

void MainWindow::openFolder(const QString& dir) {
    if (worker_.running()) {   // the running job refers to the list's rows (jobItems_, liveItem_, markedItem_)
        statusBar()->showMessage("A render is running; open the folder when it has finished.", 4000);
        return;
    }
    QDir d(dir);
    QStringList files = frameFilesIn(dir);   // TIFF, JPEG (raw too in a build with LibRaw)
    if (files.size() < 2) {
        if (batchRunning_) { batchSkip(QString("%1 image files, a stack needs at least 2").arg(files.size())); return; }
        QMessageBox::warning(this, "xLamina", QString("%1\n\nholds %2 image files (TIFF or JPEG); a stack needs at least 2.").arg(dir).arg(files.size()));
        return;
    }
    thumbs_.stop();
    thumbsDone_.clear();
    folder_ = dir;
    frames_.clear();
    frameTimes_.clear();
    jobItems_.clear();
    shifted_.clear();
    soft_.clear();
    warped_.clear();
    dropShifted_->setEnabled(false);
    dropSoft_->setEnabled(false);
    dropWarped_->setEnabled(false);
    frameList_->clear();
    liveItem_ = markedItem_ = nullptr;   // the rows they pointed at are gone
    liveView_->hide();
    colorSpace_ = QColorSpace();
    worker_.player()->setColorSpace(colorSpace_);
    // the stack is the files of the most common size; other files in the folder are listed greyed out. Between
    // two sizes with as many files each (raw files beside their exports) the developed files win: see run_align
    QMap<QPair<unsigned, unsigned>, int> sizes;
    QList<TiffInfo> infos;
    for (const QString& f : files) {
        infos << tiff_probe(d.filePath(f).toStdWString());
        if (infos.last().ok()) sizes[{infos.last().width, infos.last().height}] += infos.last().raw ? 2 : 3;
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
            frameTimes_ << QString::fromStdString(infos[i].datetime);
            it->setIcon(blank);   // holds the row height until the thumbnail arrives
            it->setFlags(it->flags() | Qt::ItemIsUserCheckable);
            it->setCheckState(Qt::Checked);
            it->setToolTip("Untick to leave this frame out of the render. Wobble numbers appear here after a render.");
            if (!colorSpace_.isValid() && !infos[i].icc.empty())
                colorSpace_ = QColorSpace::fromIccProfile(QByteArray((const char*)infos[i].icc.data(), (int)infos[i].icc.size()));
                worker_.player()->setColorSpace(colorSpace_);   // the live view converts to sRGB off the GUI thread
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
    framesDock_->setWindowTitle(QString("Frames (%1)").arg(frames_.size()));
    syncAllFramesBox();
    QString note = QString("%1 frames of %2 x %3").arg(frames_.size()).arg(best.first).arg(best.second);
    if (skipped) note += QString(" (%1 other file%2 left out)").arg(skipped).arg(skipped > 1 ? "s" : "");
    if (colorSpace_.isValid()) note += QString(", %1").arg(colorSpace_.description().isEmpty() ? "ICC profile" : colorSpace_.description());
    setStage(note + ". Press Render.");
    statusBar()->showMessage(note + " in " + QDir::toNativeSeparators(dir));
    for (int i = 0; i < frameList_->count(); ++i)
        if (frameList_->item(i)->data(Qt::UserRole).toInt() == frames_.size() / 2) { frameList_->setCurrentRow(i); break; }
    // the middle frame is selected (shown in the viewer) but the list opens at its first frame: setCurrentRow
    // scrolled it to the middle, and so did the list's deferred layout once the rows had their height
    frameList_->scrollToTop();
    QTimer::singleShot(0, frameList_, &QListWidget::scrollToTop);
    // the thumbnails' read fills the frame cache (0.6.24): the stack is resident when Render is pressed
    warmCache();
    thumbs_.start(frames_, 144);
    recent_.removeAll(dir);
    recent_.prepend(dir);
    while (recent_.size() > kMaxRecent) recent_.removeLast();
    updateRecent();
}

// The thumbnail loader reads every frame from disk; it is stopped for the duration of a render (it
// competed with the engine's readers for the disk and the frames' memory) and resumed afterwards.
void MainWindow::resumeThumbnails() {
    QStringList files;
    QList<int> idx;
    for (int i = 0; i < frames_.size(); ++i)
        if (!thumbsDone_.contains(i)) { files << frames_[i]; idx << i; }
    if (files.isEmpty()) return;
    warmCache();
    thumbs_.start(files, 144, idx);
}

// The frame cache takes the folder's frames as its set and a budget from the machine's free memory, and the
// thumbnail loader reads into it (the frames were read once for the thumbnails and again for
// the render). With the cache switched off in Performance the thumbnails read as before.
void MainWindow::warmCache() {
    bool on = cacheFrames_ && cacheFrames_->isChecked() && !worker_.running();
    if (on) worker_.prepareCache(frames_);
    thumbs_.setCache(on ? worker_.cache() : nullptr);
}

void MainWindow::onThumbnail(int index, QImage image) {
    if (index < 0 || index >= frames_.size()) return;
    thumbsDone_.insert(index);
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
    menu.addSeparator();
    QAction* tickAll = menu.addAction("Tick all frames");
    QAction* untickAbove = menu.addAction("Untick the frames above this one");
    QAction* untickBelow = menu.addAction("Untick the frames below this one");
    menu.addSeparator();
    QAction* byName = menu.addAction("Sort by name");
    QAction* byTime = menu.addAction("Sort by time taken");
    QAction* chosen = menu.exec(frameList_->viewport()->mapToGlobal(pos));
    if (!chosen) return;
    int row = frameList_->row(it);
    if (chosen == byName || chosen == byTime) { sortFrames(chosen == byTime); return; }
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
    } else {
        for (int r = 0; r < frameList_->count(); ++r) {
            QListWidgetItem* o = frameList_->item(r);
            if (o->data(Qt::UserRole).toInt() < 0) continue;
            if (chosen == tickAll) o->setCheckState(Qt::Checked);
            else if (chosen == untickAbove && r < row) o->setCheckState(Qt::Unchecked);
            else if (chosen == untickBelow && r > row) o->setCheckState(Qt::Unchecked);
        }
    }
}

// The live view: while frames stream (reading, pass 1 forward, pass 2 backward,
// pass 3 forward) the left pane shows each frame as it is processed (raw while
// reading, so the stack's wobble is visible, then warped onto the output grid)
// and the right pane what is being built from them.
void MainWindow::updateViewWindow() {
    QRectF scene = view_->sceneRect();
    if (scene.isEmpty()) return;
    QRectF vis = view_->fitted() ? scene : view_->visibleSceneRect();
    std::lock_guard<std::mutex> lk(viewWinM_);
    viewFrac_ = QRectF(vis.x() / scene.width(), vis.y() / scene.height(), vis.width() / scene.width(), vis.height() / scene.height());
    viewportW_ = std::max(1, view_->viewport()->width());
}

// the live view: the player's next frame of the sweep, already 8-bit and in sRGB (LivePlayer, its own thread);
// win / full say which part of the frame the image shows
void MainWindow::onLiveView(int k, QImage image, QRect win, QSize full) {
    if (image.isNull() || (bench_ && bench_->running()) || k < 0 || k >= jobItems_.size()) return;
    liveItem_ = jobItems_[k];   // the job's frame -> its row (rows may be reordered or duplicated)
    static const bool logViews = qEnvironmentVariableIsSet("XLAMINA_LOG_VIEWS");   // developer aid: the sweep's order and pace in the log
    if (logViews) onLog(QString("view: frame %1 at %2 ms, %3x%4 of %5x%6 at (%7,%8)").arg(k).arg(elapsed_.elapsed()).arg(image.width()).arg(image.height()).arg(full.width()).arg(full.height()).arg(win.x()).arg(win.y()));
    k = liveItem_->data(Qt::UserRole).toInt();   // the folder's frame
    current_ = "frame";
    if (full.isValid() && win.isValid() && (win.size() != full || image.size() != full)) view_->setLiveWindow(image, win, full);
    else view_->setImage(image, true);
    // the count is the progress signal's (progressStatus_); the view only says which frame is on screen
    QString name = k >= 0 && k < frames_.size() ? QFileInfo(frames_[k]).fileName() : QString();
    statusBar()->showMessage(name.isEmpty() ? progressStatus_ : QString("%1 — on screen: %2").arg(progressStatus_, name));
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
    view_->setLabel(QFileInfo(frames_[pendingFrame_]).fileName());
    statusBar()->showMessage(QString("%1 — %2 x %3 (full resolution)").arg(QFileInfo(frames_[pendingFrame_]).fileName()).arg(image.width()).arg(image.height()));
}

// ----------------------------------------------------------------------------- comparison images

// Another stacker's output (Helicon's, say) as a tab beside xLamina's, so the two can be flipped at the
// same zoom and position. Uncompressed TIFFs go through the engine's reader; anything else through Qt.
void MainWindow::addComparison() {
    QString start = result_ ? result_->outDir : folder_;
    QString fn = QFileDialog::getOpenFileName(this, "Comparison image", start, "Images (*.tif *.tiff *.jpg *.jpeg *.png);;All files (*)");
    if (fn.isEmpty()) return;
    QString name = QFileInfo(fn).fileName();
    TiffInfo info = tiff_probe(fn.toStdWString());
    if (info.ok() && info.samples >= 3) {
        pendingCompare_ = name;
        statusBar()->showMessage(QString("%1 — loading…").arg(name));
        compareLoader_.start({fn}, 1 << 20);
        return;
    }
    QImageReader rd(fn);
    rd.setAutoTransform(true);
    QImage img = rd.read();
    if (img.isNull()) { QMessageBox::warning(this, "xLamina", QString("Could not read %1:\n%2").arg(fn, rd.errorString())); return; }
    if (img.colorSpace().isValid()) img.convertToColorSpace(QColorSpace::SRgb);
    img = img.convertToFormat(QImage::Format_RGB32);
    compare_[name] = img;
    rebuildTabs();
    tabs_->setCurrentIndex(tabs_->count() - 1);
    showOutput(tabs_->currentIndex());
}

// the two-up viewer, fed with whatever this window holds: the last render's outputs and the comparison images
void MainWindow::twoUp() {
    if (!twoUp_) twoUp_ = new CompareWindow(this);
    if (result_ && result_->rc == 0) {
        for (const auto& n : result_->result.names) {   // the written TIFF, when there is one, gives retouching its 16-bit data
            QString name = QString::fromStdString(n), tif = QDir(result_->outDir).filePath(name + ".tif");
            twoUp_->addImage(name, outputImage(name), "this render", QFileInfo::exists(tif) ? tif : QString());
        }
    }
    for (auto it = compare_.begin(); it != compare_.end(); ++it) twoUp_->addImage(it.key(), it.value(), "comparison image");
    twoUp_->show();
    twoUp_->raise();
    twoUp_->activateWindow();
    if (result_ && result_->rc == 0 && !current_.isEmpty() && current_ != "frame") twoUp_->select(0, current_);
}

// the Retouch button above the view: the two-up viewer with retouching on (it needs an image in each pane)
void MainWindow::retouch() {
    twoUp();
    if (twoUp_) twoUp_->setRetouch(true);
}

void MainWindow::openSweep() {
    QString start = result_ ? QFileInfo(result_->outDir).absolutePath() : folder_;
    QString d = QFileDialog::getExistingDirectory(this, "Sweep folder (holds sweep.json)", start);
    if (d.isEmpty()) return;
    twoUp();
    twoUp_->openSweep(d);
}

// a sweep from the panel's settings: the dialog picks what varies, the worker renders, onFinished opens the viewer
void MainWindow::runSweep() {
    if (folder_.isEmpty()) { chooseFolder(); if (folder_.isEmpty()) return; }
    if (worker_.running()) { statusBar()->showMessage("A render is running; start the sweep when it has finished.", 4000); return; }
    if (bench_ && bench_->running()) { statusBar()->showMessage("The benchmark is running; stop it first.", 4000); return; }
    JobOptions jo;
    if (!buildJob(jo, false)) return;
    if (jobItems_.size() < 2) { QMessageBox::information(this, "xLamina", "A stack needs at least 2 ticked frames."); return; }
    jo.stack.outputs = {"B"};
    jo.stack.diagnostics = false;
    QString base = outDir_->text().isEmpty() ? folder_ + "_stacker" : outDir_->text();
    SweepDialog dlg(jo.stack, base + "_sweep", this);
    if (dlg.exec() != QDialog::Accepted) return;
    std::vector<SweepVariant> vars = dlg.variants();
    if (vars.empty()) return;
    if (!jo.stack.cache_frames) worker_.releaseCache();
    thumbs_.stop();   // resumed in onFinished
    log_->clear();
    liveView_->hide();
    saveSettings();
    setBusy(true);
    stageText_ = "starting the sweep";
    elapsed_.start();
    tick_.start();
    if (hwDock_) hwDock_->raise();   // the hardware panel to the front for the run
    liveItem_ = markedItem_ = nullptr;
    lastView_.invalidate();
    listTick_.start();
    int live = liveRes_->currentData().toInt();
    worker_.startSweep(jo, vars, dlg.outDir(), live > 0 ? live : 12);
}

void MainWindow::onComparison(int index, QImage image) {
    if (index != 0 || pendingCompare_.isEmpty()) return;
    compare_[pendingCompare_] = forDisplay(image);   // read like a frame: the stack's profile is the best guess for its colours
    pendingCompare_.clear();
    rebuildTabs();
    for (int i = 0; i < tabs_->count(); ++i) if (compare_.contains(tabs_->tabText(i))) tabs_->setCurrentIndex(i);
    showOutput(tabs_->currentIndex());
}

// the outputs of the last render, then the comparison images
void MainWindow::rebuildTabs() {
    QString cur = tabs_->currentIndex() >= 0 ? tabs_->tabText(tabs_->currentIndex()) : QString();
    tabs_->blockSignals(true);
    while (tabs_->count()) tabs_->removeTab(0);
    if (result_ && result_->rc == 0) {
        for (const auto& n : result_->result.names) tabs_->addTab(QString::fromStdString(n));
        // (the depth map is no longer a tab or a viewer entry; outputImage("depth") still renders it)
    }
    for (auto it = compare_.begin(); it != compare_.end(); ++it) {
        int t = tabs_->addTab(it.key());
        tabs_->setTabToolTip(t, "Comparison image (not an xLamina output). Close it with the middle mouse button.");
    }
    tabs_->blockSignals(false);
    for (int i = 0; i < tabs_->count(); ++i) if (tabs_->tabText(i) == cur) tabs_->setCurrentIndex(i);
}

void MainWindow::stepOutput(int delta) {
    if (tabs_->count() == 0) return;
    int i = tabs_->currentIndex() + delta;
    if (i < 0) i = tabs_->count() - 1;
    if (i >= tabs_->count()) i = 0;
    tabs_->setCurrentIndex(i);
}

// ----------------------------------------------------------------------------- rendering

bool MainWindow::buildJob(JobOptions& jo, bool interactive) {
    jo = JobOptions();
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
    jo.align.ref_rule = ref_->currentIndex() == 0 ? "middle" : ref_->currentIndex() == 1 ? "first" : "last";
    jo.align.lock_scale = lockScale_->isChecked();
    jo.stack.fix_rs = fixRs_->isChecked();
    jo.stack.fix_banding = fixBanding_->isChecked();
    if (fov_->value() > 0) jo.align.um_per_px = -fov_->value();
    jo.stack.po.g.scale = scale_->currentIndex() == 0 ? 1.0 : 0.5;
    jo.stack.po.g.interp = interp_->currentIndex();
    jo.stack.soft_cutoff = softCutoff_->value();
    jo.stack.c_smoothing = smoothing_->value();
    jo.stack.po.c_radius_px = cradius_->value();
    jo.stack.po.b_radius_px = bradius_->value();
    // the settings that name the saved files: interpolation, smoothing, C and B radii
    jobTag_ = QString("%1_s%2_c%3_b%4").arg(interp_->currentText().toLower().remove(' ')).arg(smoothing_->value(), 0, 'g', 2)
                  .arg((int)cradius_->value()).arg((int)bradius_->value());
    if (fixRs_->isChecked()) jobTag_ += "_rs";             // the rolling-shutter corrections (FINDINGS 51) tell renders apart
    if (fixBanding_->isChecked()) jobTag_ += "_band";
    if (advanced_->isChecked()) {
        jo.stack.po.depth_fraction = depthRes_->currentData().toDouble();
        jo.stack.po.radius_px = focusRadius_->value();
        jo.stack.po.power = focusPower_->value();
        jo.stack.po.depth_fm_px = depthFmPx_->value();
        jo.stack.po.vol_agg_px = volAggPx_->value();
        jo.stack.refine_r = refineR_->value();
        jo.stack.refine_fm_px = refineFmPx_->value();
        jo.stack.soft_px = softPx_->value();
        jo.stack.halo_prior = haloPrior_->value();
        jo.stack.dark_prior = darkPrior_->value();
        jo.stack.dark_reach_px = darkReach_->value();
        jo.stack.po.flat_blur_px = flatBlur_->value();
    }
    jo.stack.outputs = selectedOutputs();
    jo.stack.diagnostics = false;   // no depth.f32 and friends in the output folder
    jo.out = outDir_->text().isEmpty() ? std::string() : outDir_->text().toStdString();
    perfFromUi().apply(jo);
    if (interactive) {
        if (jobItems_.size() < 2) { QMessageBox::information(this, "xLamina", "A stack needs at least 2 ticked frames."); return false; }
        if (jo.stack.outputs.empty()) { QMessageBox::information(this, "xLamina", "Choose at least one output."); return false; }
    }
    return true;
}

void MainWindow::render() {
    if (folder_.isEmpty()) { chooseFolder(); if (folder_.isEmpty()) return; }
    if (worker_.running()) return;
    if (bench_ && bench_->running()) { statusBar()->showMessage("The benchmark is running; stop it first.", 4000); return; }
    JobOptions jo;
    if (!buildJob(jo, true)) return;
    if (!jo.stack.cache_frames) worker_.releaseCache();
    thumbs_.stop();   // resumed in onFinished
    cache_.clear();
    result_.reset();
    rebuildTabs();
    log_->clear();
    liveView_->hide();
    saveSettings();
    setBusy(true);
    stageText_ = "starting";
    elapsed_.start();
    tick_.start();
    if (hwDock_) hwDock_->raise();   // the hardware panel to the front for the run
    liveItem_ = markedItem_ = nullptr;
    lastView_.invalidate();
    listTick_.start();
    int live = liveRes_->currentData().toInt();
    worker_.start(jo, live > 0 ? live : 12, live == 0);
}

void MainWindow::cancel() {
    if (!worker_.running() || (bench_ && bench_->running())) return;
    worker_.cancel();
    stageText_ = "Cancelling after the current frames…";
    setStage(stageText_);
}

void MainWindow::setBusy(bool busy) {
    renderBtn_->setEnabled(!busy);
    renderAct_->setEnabled(!busy);
    cancelBtn_->setEnabled(busy);
    cancelAct_->setEnabled(busy);
    bool canSave = !busy && ((result_ && !result_->result.names.empty()) || !compare_.isEmpty());
    saveBtn_->setEnabled(canSave);
    saveAct_->setEnabled(canSave);
    openOutAct_->setEnabled(!busy && result_ && result_->rc == 0);
    // the list stays enabled (a disabled list greys the thumbnails); showFrame ignores clicks while rendering
    progress_->setValue(0);
}

void MainWindow::onProgress(QString stage, double fraction) {
    if (bench_ && bench_->running()) {   // the benchmark's runs: only the bar
        progress_->setValue((int)std::lround(std::clamp(fraction, 0.0, 1.0) * 1000));
        return;
    }
    stageText_ = stage;
    setStage(QString("%1 — %2 s").arg(stage).arg(elapsed_.elapsed() / 1000.0, 0, 'f', 1));
    progress_->setValue((int)std::lround(std::clamp(fraction, 0.0, 1.0) * 1000));
    // the frame count comes from here, not from the live view: the view is throttled and skips frames, so its
    // last frame index stalled short of the total ("194 of 200"). Every progress call is delivered.
    if (stage.startsWith("reading") || stage.startsWith("frames from") || stage.startsWith("pass")) {
        const int total = jobItems_.size();
        const int n = (int)std::lround(std::clamp(fraction, 0.0, 1.0) * total);
        // "frames from memory": the stack was read while the thumbnails loaded; this pass only converts it for the alignment
        const QString what = stage.startsWith("reading") ? "Read" : stage.startsWith("frames from") ? "From memory"
                           : stage.startsWith("pass 1") ? "Aligned" : stage.startsWith("pass 2") ? "Fine-focused" : "Composed";
        progressStatus_ = QString("%1 %2 of %3 frames").arg(what).arg(n).arg(total);
    } else if (!stage.isEmpty()) {   // "aligning pairs" (a share of the pairs), "depth map", "finalize", "done"
        progressStatus_ = stage;
        progressStatus_[0] = progressStatus_[0].toUpper();
        if (fraction > 0.0 && fraction < 1.0) progressStatus_ += QString(" — %1 %").arg((int)std::lround(fraction * 100));
    }
    statusBar()->showMessage(progressStatus_);
}

// a building map or an output preview, already 8-bit and in sRGB (the LivePlayer converts them on its thread)
void MainWindow::onPreview(QString name, QImage image) {
    if (image.isNull() || (bench_ && bench_->running())) return;
    if (name.endsWith("(building)") || name.startsWith("depth map")) {   // the right pane: what the render is building
        if (liveView_->isHidden()) {
            liveView_->show();
            int half = split_->width() / 2;
            split_->setSizes({half, half});
            view_->fit();
        }
        liveView_->setImage(image, true);
        liveView_->setToolTip(name);
        liveView_->setLabel(name);
        return;
    }
    current_ = name;
    view_->setImage(image, true);
    view_->setLabel(name + "  (preview)");
    statusBar()->showMessage(QString("Preview: %1").arg(name));
}

// the stage text is drawn inside the progress bar; the hidden label keeps the plain text
void MainWindow::setStage(const QString& text) {
    stage_->setText(text);
    progress_->setFormat(QString(text).replace('%', "%%"));   // a literal % in the text, not a QProgressBar field
}

void MainWindow::onLog(QString line) {
    log_->appendPlainText(line);
    // developer aid: XLAMINA_LOG=C:\x\gui.log mirrors the Log tab to a file (unattended runs)
    static const QString path = qEnvironmentVariable("XLAMINA_LOG");
    if (!path.isEmpty()) {
        QFile f(path);
        if (f.open(QIODevice::Append | QIODevice::Text)) { f.write(line.toUtf8()); f.write("\n"); }
    }
}

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
    const auto& R = d.result;
    g += QString("\nalignment %1 s, fusion %2 s (pass 1 %3, depth %4, refine %5, compose %6, finalize %7, write %8), %9 frames/s in pass 1; %10")
             .arg(A.seconds, 0, 'f', 1).arg(R.seconds, 0, 'f', 1).arg(R.pass1_seconds, 0, 'f', 1).arg(R.depth_seconds, 0, 'f', 1)
             .arg(R.refine_seconds, 0, 'f', 1).arg(R.compose_seconds, 0, 'f', 1).arg(R.finalize_seconds, 0, 'f', 1).arg(R.write_seconds, 0, 'f', 1)
             .arg(R.frames_per_second, 0, 'f', 0).arg(perfFromUi().summary());
    g += QString("\nreport.json and the TIFFs in %1").arg(QDir::toNativeSeparators(d.outDir));
    geometry_->setText(g);
}

// Per-frame wobble: each frame's shift (and scale, rotation) relative to the previous
// frame, from the alignment. Steps far from the median are flagged as outliers.
// Per-frame sharpness (FINDINGS 49): pass 1's focus energy against the four nearest frames' (frame_sharpness);
// a frame blurred during its exposure is flagged soft. Render > Drop shifted frames / Drop soft frames untick each kind.
void MainWindow::labelWobble(const stackcore::AlignOut& A, const stackcore::StackOptions::Result* R) {
    shifted_.clear();
    soft_.clear();
    warped_.clear();
    int n = (int)A.T.size();
    if (n != jobItems_.size() || n < 2) return;
    stackcore::FrameSharpness sharp;
    if (R && (int)R->frame_energy.size() == n) sharp = stackcore::frame_sharpness(R->frame_energy, softCutoff_->value());
    const bool haveSharp = sharp.threshold > 0;
    const bool haveRs = A.rs_bands > 0 && (int)A.rs_px.size() == n, haveBand = (int)A.band_pct.size() == n;
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
    // the warped outliers (FINDINGS 51): the band-to-band displacement far above the stack's, the same robust rule,
    // at least 2 px above the median (the 8-band correction is too coarse for a frame shaken that hard)
    double warpThr = 0;
    if (haveRs) {
        QVector<double> wv(A.rs_px.begin(), A.rs_px.end());
        std::sort(wv.begin(), wv.end());
        double wmed = wv[wv.size() / 2];
        QVector<double> wdev;
        for (double x : wv) wdev << std::abs(x - wmed);
        std::sort(wdev.begin(), wdev.end());
        warpThr = wmed + std::max(4 * 1.4826 * wdev[wdev.size() / 2], 2.0);
    }
    double worst = 0;
    int worstk = -1;
    int nWobble = 0, softest = -1, nWarped = 0;
    for (int k = 0; k < n; ++k) {
        QListWidgetItem* it = jobItems_[k];
        QString name = QFileInfo(frames_[it->data(Qt::UserRole).toInt()]).fileName();
        // the sharpness: the ratio to the neighbours (1.00 = as expected), shown on every row that has one
        QString sharpTxt;
        const bool hasRatio = haveSharp && sharp.ratio[k] > 0;
        const bool soft = hasRatio && sharp.soft[k];
        if (hasRatio) sharpTxt = QString("  sharp %1").arg(sharp.ratio[k], 0, 'f', 2);
        // the rolling shutter's two marks (FINDINGS 51): the frame's band-to-band displacement and its flicker banding
        QString rsTxt;
        if (haveRs) rsTxt += QString("warp %1 px").arg(A.rs_px[k], 0, 'f', 1);
        if (haveBand) rsTxt += QString("%1band %2 %").arg(rsTxt.isEmpty() ? "" : "  ").arg(A.band_pct[k], 0, 'f', 1);
        if (!rsTxt.isEmpty()) rsTxt = "\n" + rsTxt;
        if (k == 0) {
            it->setText(name + "\nfirst frame" + sharpTxt + rsTxt);
        } else {
            it->setText(QString("%1\nΔ %2 px  %3 %%4%5").arg(name).arg(step[k], 0, 'f', 2).arg(sstep[k], 0, 'f', 3).arg(sharpTxt).arg(rsTxt));
        }
        bool flag = k > 0 && step[k] > thr;
        const bool warped = haveRs && A.rs_px[k] > warpThr;
        // red = the position jumped (wobble), amber = blurred (soft), violet = warped (rolling shutter); the first that applies
        it->setForeground(flag ? QBrush(QColor(230, 80, 80)) : soft ? QBrush(QColor(235, 160, 40)) : warped ? QBrush(QColor(170, 130, 255)) : QBrush());
        QString tip = QString("To the reference: shift (%1, %2) px, scale %3, rotation %4°\n"
                              "From the previous frame: %5 px, %6 %, %7°%8")
                          .arg(A.T[k].tx, 0, 'f', 2).arg(A.T[k].ty, 0, 'f', 2).arg(std::hypot(A.T[k].zr, A.T[k].zi), 0, 'f', 5)
                          .arg(std::atan2(A.T[k].zi, A.T[k].zr) * kDeg, 0, 'f', 4)
                          .arg(step[k], 0, 'f', 3).arg(sstep[k], 0, 'f', 4).arg(rstep[k], 0, 'f', 4)
                          .arg(flag ? QString("\nFLAGGED: above %1 px (median %2 px)").arg(thr, 0, 'f', 2).arg(med, 0, 'f', 2) : "");
        if (hasRatio) {
            tip += QString("\nFocus energy %1 % of the sharpest frame's, %2 of the neighbours' (soft below %3)")
                       .arg(100.0 * sharp.energy[k] / sharp.peak, 0, 'f', 1).arg(sharp.ratio[k], 0, 'f', 3).arg(sharp.threshold, 0, 'f', 3);
            if (soft) tip += "\nSOFT: blurred against its neighbours (movement during the exposure)";
        }
        if (haveRs) {
            QString bands;
            for (int b = 0; b < A.rs_bands; ++b)
                bands += QString("%1(%2, %3)").arg(b ? " " : "").arg(A.rs_dx[(size_t)k * A.rs_bands + b], 0, 'f', 1).arg(A.rs_dy[(size_t)k * A.rs_bands + b], 0, 'f', 1);
            tip += QString("\nRolling shutter: displacement per row band, top to bottom, px: %1\n  band-to-band spread %2 px (stack median %3 px, warped above %4 px)")
                       .arg(bands).arg(A.rs_px[k], 0, 'f', 2).arg(A.rs_median_px, 0, 'f', 2).arg(warpThr, 0, 'f', 2);
            if (warped) tip += "\nWARPED: shaken during the readout, far beyond the stack's rolling shutter";
        }
        if (haveBand)
            tip += QString("\nBrightness banding: LED flicker %1 % peak-to-peak across the rows (stack median %2 %, strongest period %3 rows)")
                       .arg(A.band_pct[k], 0, 'f', 2).arg(A.band_median_pct, 0, 'f', 2).arg(A.band_peak_cycles > 0 ? A.H / A.band_peak_cycles : 0);
        it->setToolTip(tip);
        if (flag) ++nWobble;
        if (flag) shifted_ << it;
        if (soft) soft_ << it;
        if (warped) { warped_ << it; ++nWarped; }
        if (step[k] > worst) { worst = step[k]; worstk = it->data(Qt::UserRole).toInt(); }
        if (soft && (softest < 0 || sharp.ratio[k] < sharp.ratio[softest])) softest = k;
    }
    wobble_ = QString("wobble: median step %1 px, largest %2 px (%3), %4 flagged")
                  .arg(med, 0, 'f', 2).arg(worst, 0, 'f', 2).arg(worstk >= 0 ? QFileInfo(frames_[worstk]).fileName() : "-").arg(nWobble);
    if (haveSharp) {
        wobble_ += QString("; sharpness: %1 soft frame%2 (focus energy under %3 of the neighbours')")
                       .arg(sharp.n_soft).arg(sharp.n_soft == 1 ? "" : "s").arg(sharp.threshold, 0, 'f', 2);
        if (softest >= 0)
            wobble_ += QString(", softest %1 at %2").arg(QFileInfo(frames_[jobItems_[softest]->data(Qt::UserRole).toInt()]).fileName()).arg(sharp.ratio[softest], 0, 'f', 2);
    }
    if (haveRs) wobble_ += QString("; rolling shutter: band-to-band displacement median %1 px, %2 warped frame%3 (above %4 px)")
                               .arg(A.rs_median_px, 0, 'f', 2).arg(nWarped).arg(nWarped == 1 ? "" : "s").arg(warpThr, 0, 'f', 2);
    if (haveBand) wobble_ += QString("; banding: flicker median %1 %, period %2 rows").arg(A.band_median_pct, 0, 'f', 2).arg(A.band_peak_cycles > 0 ? A.H / A.band_peak_cycles : 0);
    dropShifted_->setEnabled(!shifted_.isEmpty());
    dropSoft_->setEnabled(!soft_.isEmpty());
    dropWarped_->setEnabled(!warped_.isEmpty());
}

static void untick(QList<QListWidgetItem*>& rows, QAction* act, QStatusBar* bar, const char* what) {
    for (QListWidgetItem* it : rows) it->setCheckState(Qt::Unchecked);
    bar->showMessage(QString("%1 %2 frame%3 unticked; press Render to stack without them.").arg(rows.size()).arg(what).arg(rows.size() == 1 ? "" : "s"));
    rows.clear();
    act->setEnabled(false);
}

void MainWindow::dropShifted() { untick(shifted_, dropShifted_, statusBar(), "shifted"); }
void MainWindow::dropSoft() { untick(soft_, dropSoft_, statusBar(), "soft"); }
void MainWindow::dropWarped() { untick(warped_, dropWarped_, statusBar(), "warped"); }

void MainWindow::syncAllFramesBox() {
    if (!allFrames_) return;
    int ticked = 0, total = 0;
    for (int r = 0; r < frameList_->count(); ++r) {
        QListWidgetItem* o = frameList_->item(r);
        if (o->data(Qt::UserRole).toInt() < 0) continue;
        ++total;
        if (o->checkState() == Qt::Checked) ++ticked;
    }
    QSignalBlocker block(allFrames_);
    allFrames_->setEnabled(total > 0);
    allFrames_->setCheckState(total == 0 || ticked == 0 ? Qt::Unchecked : ticked == total ? Qt::Checked : Qt::PartiallyChecked);
    allFrames_->setText(total == 0 ? "All frames" : ticked == total ? "All frames" : QString("%1 of %2 frames").arg(ticked).arg(total));
}

void MainWindow::tickAllFrames() {
    if (worker_.running()) { statusBar()->showMessage("A render is running; the list changes when it has finished.", 4000); return; }
    int n = 0;
    for (int r = 0; r < frameList_->count(); ++r) {
        QListWidgetItem* o = frameList_->item(r);
        if (o->data(Qt::UserRole).toInt() < 0) continue;
        if (o->checkState() != Qt::Checked) { o->setCheckState(Qt::Checked); ++n; }
    }
    statusBar()->showMessage(n ? QString("%1 frame%2 ticked again; every frame is in the next render.").arg(n).arg(n == 1 ? "" : "s")
                               : QString("Every frame was ticked already."), 4000);
}

// The list is the stack order: sort its rows by file name (natural order, so IMG_9 precedes IMG_10) or by the capture
// time (the EXIF DateTimeOriginal read at folder open, the file's modification time for a frame without one). Rows of
// files left out of the stack (grey) go to the end; a second copy of a frame sorts with its original; ties keep the
// list's order; the selected row stays selected.
void MainWindow::sortFrames(bool byTime) {
    if (worker_.running()) { statusBar()->showMessage("A render is running; the list changes when it has finished.", 4000); return; }
    if (frameList_->count() < 2) return;
    QCollator coll;
    coll.setNumericMode(true);
    coll.setCaseSensitivity(Qt::CaseInsensitive);
    int noTime = 0;
    QStringList key(frames_.size());
    for (int i = 0; i < frames_.size(); ++i) {
        if (byTime) {
            key[i] = frameTimes_.value(i);
            if (key[i].isEmpty()) { key[i] = QFileInfo(frames_[i]).lastModified().toString("yyyy:MM:dd HH:mm:ss.zzz"); ++noTime; }
        } else key[i] = QFileInfo(frames_[i]).fileName();
    }
    QListWidgetItem* current = frameList_->currentItem();
    QList<QListWidgetItem*> rows;
    while (frameList_->count()) rows << frameList_->takeItem(0);
    std::stable_sort(rows.begin(), rows.end(), [&](QListWidgetItem* a, QListWidgetItem* b) {
        int ia = a->data(Qt::UserRole).toInt(), ib = b->data(Qt::UserRole).toInt();
        if (ia < 0 || ib < 0) return ia >= 0 && ib < 0;   // the greyed rows last, among themselves as they were
        int c = byTime ? QString::compare(key[ia], key[ib]) : coll.compare(key[ia], key[ib]);
        if (c == 0 && byTime) c = coll.compare(QFileInfo(frames_[ia]).fileName(), QFileInfo(frames_[ib]).fileName());
        return c < 0;
    });
    for (QListWidgetItem* r : rows) frameList_->addItem(r);
    if (current) { frameList_->setCurrentItem(current); frameList_->scrollToItem(current); }
    QString msg = byTime ? "Frames sorted by the time they were taken" : "Frames sorted by file name";
    if (byTime && noTime) msg += QString(" (%1 without a capture time in the file: their modification time was used)").arg(noTime);
    statusBar()->showMessage(msg + ".", 6000);
}

void MainWindow::onFinished(std::shared_ptr<JobResultData> data) {
    if (bench_ && bench_->running()) return;   // the benchmark's runs are its own
    tick_.stop();
    listTick_.stop();
    if (markedItem_) markedItem_->setBackground(QBrush());
    liveItem_ = markedItem_ = nullptr;
    if (data->sweep) {   // the last render's outputs stay; the variants go to the two-up viewer
        setBusy(false);
        liveView_->hide();
        resumeThumbnails();
        if (data->rc == 3) { setStage("Sweep cancelled."); return; }
        setStage(data->rc ? "Sweep finished with failures; see the log." : QString("Sweep done: %1 variants in %2 s.").arg(data->variants).arg(data->seconds, 0, 'f', 1));
        progress_->setValue(0);   // the bar empties when the work is done; its text says so
        if (QFileInfo::exists(QDir(data->outDir).filePath("sweep.json"))) { twoUp(); twoUp_->openSweep(data->outDir); }
        return;
    }
    result_ = data;
    // the render's finish time first, as Helicon names its files: B_2026-11-08_14-15-27_bicubic_s2_c32_b0
    result_->settingsTag = QDateTime::currentDateTime().toString("yyyy-MM-dd_HH-mm-ss") + "_" + jobTag_;
    // the output's field of view after the crop, last: the panel's FOV is the sensor width's at
    // the reference depth, so the output's is its share of the width, as the geometry report states it
    if (fov_->value() > 0 && result_->align.W > 0) {
        const auto& A = result_->align;
        result_->settingsTag += QString("_%1mm").arg(fov_->value() * (A.rect[2] - A.rect[0] + 1) / A.W, 0, 'f', 2);
    }
    setBusy(false);
    liveView_->hide();
    resumeThumbnails();
    labelWobble(data->align, &data->result);
    if (data->rc == 3) { setStage("Cancelled."); if (batchRunning_) batchFinish(true); return; }
    if (data->rc != 0) {
        setStage("Failed; see the log.");
        if (batchRunning_) {   // no box in a batch: note it and go on
            batchLog_ << QString("%1: FAILED (%2)").arg(QDir::toNativeSeparators(batchCurrent_), log_->toPlainText().trimmed().section('\n', -1));
            QTimer::singleShot(0, this, &MainWindow::batchNext);
        } else QMessageBox::warning(this, "xLamina", "The render failed. The log has the details.");
        return;
    }
    setStage(QString("Done in %1 s (alignment %2 s, fusion %3 s).").arg(data->seconds, 0, 'f', 1).arg(data->align.seconds, 0, 'f', 1).arg(data->result.seconds, 0, 'f', 1));
    progress_->setValue(0);   // the bar empties when the work is done; its text says so
    setStatusGeometry(*data);
    rebuildTabs();
    tabs_->setCurrentIndex(0);
    showOutput(0);
    if (batchRunning_) {
        ++batchDone_;
        batchLog_ << QString("%1: %2 frames in %3 s -> %4").arg(QDir::toNativeSeparators(batchCurrent_)).arg(data->align.paths.size())
                         .arg(data->seconds, 0, 'f', 1).arg(QDir::toNativeSeparators(data->outDir));
        QTimer::singleShot(0, this, &MainWindow::batchNext);
    }
}

// ----------------------------------------------------------------------------- batch

QStringList MainWindow::frameFilesIn(const QString& dir) {
    QStringList pats;
    for (const std::string& p : frame_file_patterns()) pats << QString::fromStdString(p);
    QStringList files = QDir(dir).entryList(pats, QDir::Files, QDir::Name);
    files.removeDuplicates();
    return files;
}

void MainWindow::batch() { batchDialog({}); }

// The queue: folders to render one after the other with the Render panel's settings as they are at Start.
// Each folder is opened as File > Open folder would open it (every frame ticked) and rendered; its outputs
// and report.json go to <folder>_stacker. A folder that fails or holds no stack is noted and skipped.
void MainWindow::batchDialog(const QStringList& preset) {
    if (batchRunning_ || worker_.running() || (bench_ && bench_->running())) {
        statusBar()->showMessage("A render is running; queue a batch when it has finished.", 4000);
        return;
    }
    QDialog dlg(this);
    dlg.setWindowTitle("Batch render");
    dlg.setWindowFlag(Qt::WindowContextHelpButtonHint, false);
    dlg.resize(680, 440);
    dlg.setAcceptDrops(false);
    QVBoxLayout* vl = new QVBoxLayout(&dlg);
    QLabel* note = new QLabel("The folders are rendered one after the other with the Render panel's current settings (outputs, field of view, "
                              "preferences). Each folder's outputs and report.json go to <folder>_stacker beside it.");
    note->setWordWrap(true);
    vl->addWidget(note);
    QListWidget* list = new QListWidget;
    list->setSelectionMode(QAbstractItemView::ExtendedSelection);
    vl->addWidget(list, 1);
    QLabel* count = new QLabel;
    vl->addWidget(count);
    QPushButton* start = new QPushButton("Start");
    auto refresh = [&] {
        count->setText(list->count() == 0 ? "No folders yet." : QString("%1 folder%2 queued.").arg(list->count()).arg(list->count() == 1 ? "" : "s"));
        start->setEnabled(list->count() > 0);
    };
    // a folder joins the queue when it holds at least two frame files and is not there yet
    auto add = [&](const QString& dir) {
        QString d = QDir::toNativeSeparators(QDir(dir).absolutePath());
        for (int i = 0; i < list->count(); ++i) if (list->item(i)->text().compare(d, Qt::CaseInsensitive) == 0) return false;
        int n = frameFilesIn(dir).size();
        if (n < 2) return false;
        QListWidgetItem* it = new QListWidgetItem(d);
        it->setToolTip(QString("%1 image files").arg(n));
        list->addItem(it);
        return true;
    };
    for (const QString& p : preset) add(p);
    QSettings s = appSettings();
    QHBoxLayout* bl = new QHBoxLayout;
    QPushButton* addOne = new QPushButton("Add folder…");
    QPushButton* addSub = new QPushButton("Add subfolders of…");
    addSub->setToolTip("Adds every folder directly inside the chosen one that holds at least two image files\n(folders named ..._stacker, the outputs of earlier renders, are left out).");
    QPushButton* remove = new QPushButton("Remove");
    QPushButton* clear = new QPushButton("Clear");
    QPushButton* close = new QPushButton("Close");
    start->setDefault(true);
    for (QPushButton* b : {addOne, addSub, remove, clear}) bl->addWidget(b);
    bl->addStretch(1);
    bl->addWidget(start);
    bl->addWidget(close);
    vl->addLayout(bl);
    auto startDir = [&] { return s.value("batch/lastDir", folder_.isEmpty() ? recent_.value(0) : QFileInfo(folder_).absolutePath()).toString(); };
    connect(addOne, &QPushButton::clicked, &dlg, [&] {
        QString d = QFileDialog::getExistingDirectory(&dlg, "A folder with a stack's frames", startDir());
        if (d.isEmpty()) return;
        s.setValue("batch/lastDir", QFileInfo(d).absolutePath());
        if (!add(d)) count->setText(QString("%1 is already queued or holds fewer than two image files.").arg(QDir::toNativeSeparators(d)));
        else refresh();
    });
    connect(addSub, &QPushButton::clicked, &dlg, [&] {
        QString d = QFileDialog::getExistingDirectory(&dlg, "A folder whose subfolders are the stacks", startDir());
        if (d.isEmpty()) return;
        s.setValue("batch/lastDir", d);
        int added = 0;
        for (const QString& sub : QDir(d).entryList(QDir::Dirs | QDir::NoDotAndDotDot, QDir::Name))
            if (!sub.endsWith("_stacker", Qt::CaseInsensitive) && add(QDir(d).filePath(sub))) ++added;
        refresh();
        if (!added) count->setText("No subfolder there holds two or more image files.");
    });
    connect(remove, &QPushButton::clicked, &dlg, [&] { qDeleteAll(list->selectedItems()); refresh(); });
    connect(clear, &QPushButton::clicked, &dlg, [&] { list->clear(); refresh(); });
    connect(close, &QPushButton::clicked, &dlg, &QDialog::reject);
    connect(start, &QPushButton::clicked, &dlg, [&] {
        if (selectedOutputs().empty()) { count->setText("Tick at least one output in the Render panel first."); return; }
        dlg.accept();
    });
    refresh();
    if (dlg.exec() != QDialog::Accepted || list->count() == 0) return;
    QStringList dirs;
    for (int i = 0; i < list->count(); ++i) dirs << QDir::fromNativeSeparators(list->item(i)->text());
    startBatch(dirs);
}

void MainWindow::startBatch(const QStringList& dirs) {
    if (batchRunning_ || worker_.running() || dirs.isEmpty()) return;
    batchQueue_ = dirs;
    batchLog_.clear();
    batchTotal_ = batchQueue_.size();
    batchDone_ = 0;
    batchRunning_ = true;
    batchAct_->setEnabled(false);
    batchClock_.start();
    batchNext();
}

void MainWindow::batchSkip(const QString& why) {
    batchSkipped_ = true;
    batchLog_ << QString("%1: skipped (%2)").arg(QDir::toNativeSeparators(batchCurrent_), why);
}

void MainWindow::batchNext() {
    if (!batchRunning_) return;
    if (worker_.running()) { QTimer::singleShot(200, this, &MainWindow::batchNext); return; }   // the worker's thread is still winding down
    if (batchQueue_.isEmpty()) { batchFinish(false); return; }
    batchCurrent_ = batchQueue_.takeFirst();
    batchSkipped_ = false;
    openFolder(batchCurrent_);
    if (!batchSkipped_ && (QDir(folder_) != QDir(batchCurrent_) || frames_.size() < 2)) batchSkip("no two frames of one size");
    if (!batchSkipped_) {
        setWindowTitle(windowTitle() + QString("  [batch %1 of %2]").arg(batchTotal_ - batchQueue_.size()).arg(batchTotal_));
        render();
        if (!worker_.running()) batchSkip("the render could not start");
    }
    if (batchSkipped_) QTimer::singleShot(0, this, &MainWindow::batchNext);
}

void MainWindow::batchFinish(bool cancelled) {
    batchRunning_ = false;
    batchAct_->setEnabled(true);
    batchQueue_.clear();
    double took = batchClock_.elapsed() / 1000.0;
    QString sum = QString("Batch %1: %2 of %3 folders rendered in %4.").arg(cancelled ? "cancelled" : "done").arg(batchDone_).arg(batchTotal_)
                      .arg(took < 120 ? QString("%1 s").arg(took, 0, 'f', 0) : QString("%1 min").arg(took / 60, 0, 'f', 1));
    onLog("");
    onLog(sum);
    for (const QString& l : batchLog_) onLog("  " + l);
    setStage(sum + (batchDone_ < batchTotal_ ? " See the log." : ""));
    statusBar()->showMessage(sum);
    if (!folder_.isEmpty()) setWindowTitle(QString("xLamina %1 — %2").arg(XLAMINA_VERSION, QDir::toNativeSeparators(folder_)));
}

QImage MainWindow::outputImage(const QString& name) {
    if (compare_.contains(name)) return compare_[name];
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
    view_->setLabel(compare_.contains(name) ? name + "  (comparison)" : name);
    statusBar()->showMessage(QString("%1 — %2 x %3%4").arg(name).arg(img.width()).arg(img.height()).arg(compare_.contains(name) ? " (comparison image)" : ""));
}

void MainWindow::saveCurrent() {
    if (current_.isEmpty() || current_ == "frame") return;
    bool isCompare = compare_.contains(current_);
    if (!result_ && !isCompare) return;
    // a render's file carries its finish time, settings and output field of view:
    // B_2026-11-08_14-15-27_bicubic_s2_c32_b0_1.49mm.tif (interpolation, smoothing, C and B radii, FOV after the
    // crop); the engine itself writes B.tif etc. to the output folder
    QString base = isCompare ? QFileInfo(current_).completeBaseName() + "_copy"
                             : (result_->settingsTag.isEmpty() ? current_ : current_ + "_" + result_->settingsTag);
    // the format: 16-bit TIFF with the frames' profile, or JPEG at 100 % / PNG of the displayed sRGB image; the last
    // choice is remembered
    const QString fTif = "16-bit TIFF (*.tif)", fJpg = "JPEG, 100 % quality (*.jpg)", fPng = "PNG (*.png)";
    QString filter = isCompare ? fPng + ";;" + fJpg : fTif + ";;" + fJpg + ";;" + fPng;
    QString selected = isCompare ? fPng : appSettings().value("save/format", fTif).toString();
    if (!filter.contains(selected)) selected = isCompare ? fPng : fTif;
    QString ext = selected == fJpg ? ".jpg" : selected == fPng ? ".png" : ".tif";
    QString def = QDir(outDir_->text().isEmpty() ? folder_ + "_stacker" : outDir_->text()).filePath(base + ext);
    QString fn = QFileDialog::getSaveFileName(this, "Save " + current_, def, filter, &selected);
    if (fn.isEmpty()) return;
    if (!isCompare) appSettings().setValue("save/format", selected);
    // the chosen format decides the extension when the name has none of the three
    if (!fn.endsWith(".tif", Qt::CaseInsensitive) && !fn.endsWith(".tiff", Qt::CaseInsensitive) && !fn.endsWith(".jpg", Qt::CaseInsensitive)
        && !fn.endsWith(".jpeg", Qt::CaseInsensitive) && !fn.endsWith(".png", Qt::CaseInsensitive))
        fn += selected == fJpg ? ".jpg" : selected == fPng ? ".png" : ".tif";
    bool ok = false;
    if (!isCompare && (fn.endsWith(".tif", Qt::CaseInsensitive) || fn.endsWith(".tiff", Qt::CaseInsensitive))) {
        const auto& R = result_->result;
        for (size_t i = 0; i < R.names.size(); ++i)
            if (QString::fromStdString(R.names[i]) == current_) ok = write_tiff_rgb16(fn.toStdString(), R.images[i], R.w, R.h, R.icc);   // the frames' profile
        if (current_ == "depth") ok = outputImage("depth").save(fn);
    } else {
        QImageWriter w(fn);   // JPEG / PNG: the displayed (sRGB) image, 8 bits per channel
        w.setQuality(100);
        ok = w.write(outputImage(current_));
    }
    statusBar()->showMessage(ok ? QString("Saved %1").arg(QDir::toNativeSeparators(fn)) : QString("Could not save %1").arg(fn), 5000);
}

void MainWindow::openOutputFolder() {
    QString d = result_ && !result_->outDir.isEmpty() ? result_->outDir : outDir_->text();
    if (d.isEmpty() || !QDir(d).exists()) { statusBar()->showMessage("No output folder yet.", 3000); return; }
    QDesktopServices::openUrl(QUrl::fromLocalFile(d));
}

// ----------------------------------------------------------------------------- tools

void MainWindow::benchmark() {
    if (bench_) { bench_->raise(); bench_->activateWindow(); return; }
    if (worker_.running()) { statusBar()->showMessage("A render is running; the benchmark can start when it has finished.", 4000); return; }
    JobOptions jo;
    QStringList frames;
    if (!folder_.isEmpty()) {
        buildJob(jo, false);
        for (const auto& p : jo.align.paths) frames << QString::fromStdWString(p);
    } else {
        jo.stack.outputs = selectedOutputs();
        if (jo.stack.outputs.empty()) jo.stack.outputs = {"B", "C"};
        jo.stack.c_smoothing = smoothing_->value();
        jo.stack.po.c_radius_px = cradius_->value();
        jo.stack.po.g.scale = scale_->currentIndex() == 0 ? 1.0 : 0.5;
        jo.stack.po.g.interp = interp_->currentIndex();
    }
    if (frames.size() < 2) frames.clear();
    bench_ = new BenchmarkDialog(worker_, jo, frames, QDir::toNativeSeparators(folder_), perfFromUi(), this);
    bench_->setAttribute(Qt::WA_DeleteOnClose);
    connect(bench_, &BenchmarkDialog::applySettings, this, [this](PerfSettings p) {
        perfToUi(p);
        saveSettings();
        statusBar()->showMessage("Performance settings applied: " + p.summary(), 8000);
    });
    // Render is held while the benchmark runs; the main window shows no live view of its runs
    connect(bench_, &BenchmarkDialog::runningChanged, this, [this](bool on) {
        renderBtn_->setEnabled(!on);
        renderAct_->setEnabled(!on);
        if (on) setStage("Benchmark running…");
        else { progress_->setValue(0); setStage(result_ ? QString("Done in %1 s.").arg(result_->seconds, 0, 'f', 1) : "Benchmark finished."); }
    });
    bench_->show();
}

void MainWindow::autoBenchmark(bool sample, const QString& reportPath) {
    benchmark();
    if (!bench_) return;
    connect(bench_, &QObject::destroyed, qApp, &QApplication::quit, Qt::QueuedConnection);
    bench_->autoRun(sample, reportPath);
}

// ----------------------------------------------------------------------------- updates

void MainWindow::checkForUpdates() {
    updateManual_ = true;
    if (update_.valid()) { showUpdate(); return; }   // the start-up check already found one
    if (updater_.busy()) return;
    statusBar()->showMessage("Checking GitHub for the latest release…", 4000);
    updater_.check();
}

void MainWindow::onUpdateAvailable(const ReleaseInfo& r) {
    update_ = r;
    bool skipped = appSettings().value("updates/skip").toString() == r.tag;
    updateBtn_->setText(QString("Update to %1").arg(r.version));
    updateBtn_->setToolTip(QString("xLamina %1 is available (this is %2). Click to read what changed and update.").arg(r.version, XLAMINA_VERSION));
    updateAct_->setVisible(!skipped || updateManual_);
    if (updateManual_) { updateManual_ = false; showUpdate(); }
    else if (!skipped) statusBar()->showMessage(QString("xLamina %1 is available: the Update button at the right of the toolbar.").arg(r.version), 12000);
}

void MainWindow::showUpdate() {
    if (!update_.valid()) return;
    if (updateDlg_) { updateDlg_->raise(); updateDlg_->activateWindow(); return; }
    QDialog* d = new QDialog(this);
    updateDlg_ = d;
    d->setAttribute(Qt::WA_DeleteOnClose);
    d->setWindowTitle(QString("xLamina %1 is available").arg(update_.version));
    d->resize(680, 520);
    QVBoxLayout* l = new QVBoxLayout(d);
    l->addWidget(new QLabel(QString("<b>xLamina %1</b> is available; this is %2.").arg(update_.version, XLAMINA_VERSION)));
    QTextBrowser* notes = new QTextBrowser;
    notes->setOpenExternalLinks(true);
    notes->setMarkdown(update_.notes.isEmpty() ? QString("No release notes.") : update_.notes);
    l->addWidget(notes, 1);
    QLabel* page = new QLabel(QString("<a href=\"%1\">Release page</a> · %2 MB").arg(update_.page).arg(update_.size / 1048576.0, 0, 'f', 1));
    page->setOpenExternalLinks(true);
    l->addWidget(page);
    QProgressBar* bar = new QProgressBar;
    bar->setRange(0, 1000);
    bar->hide();
    l->addWidget(bar);
    QLabel* state = new QLabel;
    state->setWordWrap(true);
    l->addWidget(state);
    QDialogButtonBox* bb = new QDialogButtonBox;
    QPushButton* go = bb->addButton("Update and restart", QDialogButtonBox::AcceptRole);
    QPushButton* skip = bb->addButton("Skip this version", QDialogButtonBox::DestructiveRole);
    QPushButton* later = bb->addButton("Later", QDialogButtonBox::RejectRole);
    l->addWidget(bb);
    go->setDefault(true);
    QString here = QDir::toNativeSeparators(QCoreApplication::applicationDirPath());
    bool writable = Updater::canWriteAppDir();
    bool busy = rendering() || (bench_ && bench_->running());
    if (!writable) {
        go->setText("Open download page");
        state->setText(QString("This xLamina cannot write to its own folder (%1): download the zip and unpack it there yourself.").arg(here));
    } else if (busy) {
        go->setEnabled(false);
        state->setText("A render is running: update when it has finished.");
    } else {
        state->setText(QString("The new version replaces this one in %1 and xLamina restarts. Your settings are kept.").arg(here));
    }
    auto enable = [go, skip](bool on) { go->setEnabled(on); skip->setEnabled(on); };   // Later stays: it cancels a download
    connect(later, &QPushButton::clicked, d, &QDialog::reject);
    connect(skip, &QPushButton::clicked, d, [this, d] {
        appSettings().setValue("updates/skip", update_.tag);
        updateAct_->setVisible(false);
        d->reject();
    });
    connect(d, &QObject::destroyed, this, [this] { updater_.abort(); });   // Later / the close box cancel a download
    connect(go, &QPushButton::clicked, d, [this, bar, state, enable, writable] {
        if (!writable) { QDesktopServices::openUrl(QUrl(update_.page)); return; }
        if (rendering() || updater_.busy()) return;
        enable(false);
        bar->show();
        state->setText("Downloading…");
        updater_.download(update_);
    });
    connect(&updater_, &Updater::downloadProgress, d, [bar](qint64 got, qint64 total) {
        if (total > 0) { bar->setRange(0, 1000); bar->setValue((int)(got * 1000 / total)); }
        else bar->setRange(0, 0);
    });
    connect(&updater_, &Updater::failed, d, [bar, state, enable](const QString& why) {
        bar->hide();
        state->setText("<span style=\"color:#e05050\">" + why.toHtmlEscaped() + "</span>");
        enable(true);
    });
    connect(&updater_, &Updater::downloaded, d, [this, d, bar, state, enable](const QString& zip) {
        state->setText("Installing…");
        bar->setRange(0, 0);
        QCoreApplication::processEvents(QEventLoop::ExcludeUserInputEvents);   // show it: the swap takes a second or two
        QString err = Updater::install(zip);
        if (!err.isEmpty()) {
            bar->hide();
            state->setText("<span style=\"color:#e05050\">" + err.toHtmlEscaped() + "</span> Nothing was changed.");
            enable(true);
            return;
        }
        state->setText("Restarting…");
        saveSettings();
        d->disconnect(this);   // the destroyed handler would abort nothing, but keep it quiet
        Updater::restart({"--updated-from", XLAMINA_VERSION});
        QTimer::singleShot(0, qApp, &QCoreApplication::quit);
    });
    d->show();
}

void MainWindow::notifyUpdated(const QString& from) {
    QString line = QString("Updated to xLamina %1 (from %2).").arg(XLAMINA_VERSION, from);
    statusBar()->showMessage(line, 15000);
    onLog(line);
}

void MainWindow::selfUpdate() {
    connect(&updater_, &Updater::available, this, [this](const ReleaseInfo& r) { updater_.download(r); });
    connect(&updater_, &Updater::downloaded, this, [this](const QString& zip) {
        QString err = Updater::install(zip);
        if (!err.isEmpty()) { onLog("Self-update: " + err); return; }
        saveSettings();
        Updater::restart({"--updated-from", XLAMINA_VERSION});
        QTimer::singleShot(0, qApp, &QCoreApplication::quit);
    });
    updater_.check();
}

// File > Preferences: what is not a render setting. General (theme, updates, developer controls), Performance (the
// hardware group, moved out of the Render column) and System (this machine, read-only; it was in About, where the
// long lines wrapped). Built once; the Performance group is reparented into it.
void MainWindow::preferences() {
    if (prefsDlg_) { prefsDlg_->show(); prefsDlg_->raise(); prefsDlg_->activateWindow(); return; }
    QDialog* d = new QDialog(this);
    prefsDlg_ = d;
    d->setWindowTitle("Preferences");
    d->resize(640, 460);
    QVBoxLayout* l = new QVBoxLayout(d);
    QTabWidget* tabs = new QTabWidget;
    l->addWidget(tabs, 1);

    // ---- General
    QWidget* gen = new QWidget;
    QFormLayout* gf = new QFormLayout(gen);
    gf->setFieldGrowthPolicy(QFormLayout::ExpandingFieldsGrow);
    QComboBox* theme = new QComboBox;
    theme->addItem("Dark", "dark");
    theme->addItem("Light", "light");
    theme->addItem("System (after restart)", "system");
    theme->setCurrentIndex(std::max(0, theme->findData(theme_)));
    connect(theme, &QComboBox::currentIndexChanged, this, [this, theme](int i) { setTheme(theme->itemData(i).toString()); });
    gf->addRow("Theme", theme);
    gf->addRow("Interpolation", interp_);   // the warp's resampling filter, built with the Render column (its setting and file-name tag)
    gf->addRow("Soft frame below", softCutoff_);
    gf->addRow("Rolling shutter", fixRs_);   // built with the Render column (its settings and file-name tags)
    gf->addRow("", fixBanding_);
    QCheckBox* upd = new QCheckBox("Check for updates when xLamina starts");
    upd->setChecked(autoUpdateAct_->isChecked());
    connect(upd, &QCheckBox::toggled, autoUpdateAct_, &QAction::setChecked);
    connect(autoUpdateAct_, &QAction::toggled, upd, &QCheckBox::setChecked);
    gf->addRow("Updates", upd);
    QPushButton* chk = new QPushButton("Check now…");
    connect(chk, &QPushButton::clicked, this, &MainWindow::checkForUpdates);
    gf->addRow("", chk);
    QCheckBox* dev = new QCheckBox("Developer controls: the Advanced group, the C and B radii, the outputs beyond B and C");
    dev->setChecked(devAct_->isChecked());
    connect(dev, &QCheckBox::toggled, devAct_, &QAction::setChecked);
    connect(devAct_, &QAction::toggled, dev, &QCheckBox::setChecked);
    gf->addRow("Developer", dev);
    QPushButton* reset = new QPushButton("Reset all settings to the defaults");
    connect(reset, &QPushButton::clicked, this, &MainWindow::resetSettings);
    gf->addRow("", reset);
    tabs->addTab(gen, "General");

    // ---- Performance: the group built with the Render column, moved here
    QWidget* perfTab = new QWidget;
    QVBoxLayout* pv = new QVBoxLayout(perfTab);
    QLabel* pnote = new QLabel("Hardware settings. They change the speed, never the result: B and C are bit-identical across all of them. "
                               "Benchmark… tries them and applies the fastest.");
    pnote->setWordWrap(true);
    pv->addWidget(pnote);
    pv->addWidget(perfGroup_);
    pv->addStretch(1);
    tabs->addTab(perfTab, "Performance");

    // ---- System: read-only
    QWidget* sys = new QWidget;
    QFormLayout* sf = new QFormLayout(sys);
    sf->setFieldGrowthPolicy(QFormLayout::ExpandingFieldsGrow);
    auto ro = [](const QString& t) {
        QLabel* lb = new QLabel(t);
        lb->setTextInteractionFlags(Qt::TextSelectableByMouse);   // no word wrap: a wrapping label in a form row gets one line's height and clips the rest
        return lb;
    };
    sf->addRow("xLamina", ro(XLAMINA_VERSION));
    QStringList g;
    for (const GpuInfo& gi : gpus_) g << gi.label();
    sf->addRow("GPUs", ro(g.isEmpty() ? "no CUDA GPU found" : g.join("\n")));
    sf->addRow("Host", ro(hostSummary()));
    sf->addRow("Qt", ro(qVersion()));
    sf->addRow("Program folder", ro(QDir::toNativeSeparators(QCoreApplication::applicationDirPath())));
    sf->addRow("Settings file", ro(QDir::toNativeSeparators(appSettings().fileName())));
    tabs->addTab(sys, "System");

    QDialogButtonBox* bb = new QDialogButtonBox(QDialogButtonBox::Close);
    connect(bb, &QDialogButtonBox::rejected, d, &QDialog::hide);
    l->addWidget(bb);
    d->show();
}

void MainWindow::about() {
    QMessageBox box(this);
    box.setWindowTitle("About xLamina");
    box.setIconPixmap(QPixmap(":/xlamina-128.png"));
    box.setTextFormat(Qt::RichText);
    box.setText(QString("<h3>xLamina %1</h3>"
                        "<p>Multi-GPU focus stacker for photomicroscopy.<br>"
                        "<a href=\"https://github.com/toldxls/xLamina\">github.com/toldxls/xLamina</a></p>"
                        "<p>Travis A. Olds, Ph.D.<br>"
                        "<a href=\"https://www.mindat.org/user-8894.html\">mindat.org/user-8894.html</a></p>"
                        "<p style=\"color:#888\">This machine's details: File &gt; Preferences &gt; System.</p>")
                    .arg(XLAMINA_VERSION));
    box.exec();
}

// ----------------------------------------------------------------------------- events

void MainWindow::dragEnterEvent(QDragEnterEvent* e) {
    if (e->mimeData()->hasUrls()) e->acceptProposedAction();
}

void MainWindow::dropEvent(QDropEvent* e) {
    // several folders dropped at once: a batch
    QStringList dirs;
    for (const QUrl& u : e->mimeData()->urls()) if (QFileInfo(u.toLocalFile()).isDir()) dirs << u.toLocalFile();
    if (dirs.size() > 1) { QTimer::singleShot(0, this, [this, dirs] { batchDialog(dirs); }); return; }
    for (const QUrl& u : e->mimeData()->urls()) {
        QString p = u.toLocalFile();
        QFileInfo fi(p);
        if (fi.isDir()) { openFolder(p); break; }
        // a dropped image file: a comparison image when a result is on screen, otherwise its folder
        QString ext = fi.suffix().toLower();
        if (result_ && (ext == "tif" || ext == "tiff" || ext == "jpg" || ext == "jpeg" || ext == "png")) {
            TiffInfo info = tiff_probe(p.toStdWString());
            if (info.ok() && info.samples >= 3) { pendingCompare_ = fi.fileName(); compareLoader_.start({p}, 1 << 20); }
            else { QImage img(p); if (!img.isNull()) { compare_[fi.fileName()] = img.convertToFormat(QImage::Format_RGB32); rebuildTabs(); } }
            break;
        }
        openFolder(fi.absolutePath());
        break;
    }
}

void MainWindow::closeEvent(QCloseEvent* e) {
    if (worker_.running()) {
        if (QMessageBox::question(this, "xLamina", "A render is running. Cancel it and quit?") != QMessageBox::Yes) { e->ignore(); return; }
        batchRunning_ = false;
        worker_.cancel();
    }
    saveSettings();
    e->accept();
}
