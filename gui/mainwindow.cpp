#include "mainwindow.h"

#include <QActionGroup>
#include <QApplication>
#include <QClipboard>
#include <QCloseEvent>
#include <QDesktopServices>
#include <QDir>
#include <QDockWidget>
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
// A and B0 are not in the Outputs list: Tools > Developer outputs (developer mode) ticks them (Travis, 2026-11-08)
static bool isMenuOutput(int i) { return i == 3 || i == 4; }
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
    connect(&worker_, &Worker::preview, this, &MainWindow::onPreview);
    connect(&worker_, &Worker::frame, this, &MainWindow::onFrame);
    connect(&worker_, &Worker::logLine, this, &MainWindow::onLog);
    connect(&worker_, &Worker::finished, this, &MainWindow::onFinished);
    connect(&thumbs_, &ThumbLoader::thumbnail, this, &MainWindow::onThumbnail);
    connect(&frameLoader_, &ThumbLoader::thumbnail, this, &MainWindow::onFullFrame);
    connect(&compareLoader_, &ThumbLoader::thumbnail, this, &MainWindow::onComparison);
    tick_.setInterval(250);
    connect(&tick_, &QTimer::timeout, this, [this] {
        stage_->setText(QString("%1 — %2 s").arg(stageText_).arg(elapsed_.elapsed() / 1000.0, 0, 'f', 1));
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
    open->setStatusTip("Open a folder of TIFF frames (or drop one onto the window).");
    recentMenu_ = file->addMenu("Open &recent");
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
    dropFlagged_ = render->addAction("&Drop flagged frames", this, &MainWindow::dropFlagged);
    dropFlagged_->setEnabled(false);
    dropFlagged_->setToolTip("Untick the frames whose wobble (shift from the previous frame) is an outlier; then Render again.");
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
    // the outputs that left the Outputs list (A, B0: Travis, 2026-11-08) stay reachable here in developer mode;
    // the actions mirror hidden check boxes, filled in with the Outputs group below
    devOutMenu_ = tools->addMenu("Developer &outputs");
    devOutMenu_->setStatusTip("A (focus-weighted average) and B0 (raw argmax): developer outputs, rendered when ticked.");

    QMenu* help = menuBar()->addMenu("&Help");
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
    tb->addAction(dropFlagged_);
    tb->addSeparator();
    tb->addAction("Benchmark…", this, &MainWindow::benchmark);
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
    tabs_->setStyleSheet("QTabBar::tab { padding: 6px 16px; margin-right: 2px; }"
                         "QTabBar::tab:selected { background: #7a2fc9; color: white; font-weight: bold; }");
    tabs_->setToolTip("The outputs of the last render and any comparison images; PgUp / PgDown step through them at the same zoom.");
    // Compare and Retouch, centred above the view (Travis, 2026-11-08): the same as Compare > Two-up viewer, and the
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
    framesDock_->setWidget(frameList_);
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
    for (int i = 0; i < 11; ++i) {
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
    liveRes_ = new QComboBox;
    liveRes_->addItem("Off (fastest)", 0);
    liveRes_->addItem("Low (1/12)", 12);
    liveRes_->addItem("Medium (1/6)", 6);
    liveRes_->addItem("High (1/3)", 3);
    liveRes_->addItem("Full", 1);
    liveRes_->setCurrentIndex(2);
    liveRes_->setToolTip("Resolution of the live view while rendering, as a fraction of the output.\n"
                         "Every image copied off a GPU costs its uploads about 3 ms, so the views are rate-limited; the live view\n"
                         "still adds about a second to a 163-frame render. Off shows only the finished outputs.");
    fl->addRow("Live view", liveRes_);
    // the warp's resampling filter (Travis, 2026-11-08: Helicon's Preferences offer the same choice)
    interp_ = new QComboBox;
    interp_->addItems({"Bilinear", "Bicubic", "Lanczos 3", "Lanczos 8"});   // the order of stackcore::Interp
    interp_->setCurrentIndex(1);
    interp_->setToolTip("How every frame is resampled onto the output grid when it is aligned.\n"
                        "Bicubic (the default, OpenCV's a = -0.75) is the reference the engine was validated with. Lanczos 3 keeps a\n"
                        "little more fine detail and costs about twice the warp time; Lanczos 8 is sharper still and roughly 15 times\n"
                        "the warp time (some 10 s more on 163 frames). Bilinear is softest and fastest. Changing it changes every output.");
    fl->addRow("Interpolation", interp_);
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
    fov_->setDecimals(4);
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
                          "left; it needs about 5 GB of GPU memory for 163 full frames and costs about 3 s. Full needs 20 GB for such a stack.");
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
    flatBlur_ = dspin(0, 512, 16, 128, 0, " px", "Defocused background (no sharp detail within the dark reach): a band of about this width beside sharp detail\n"
                                                  "takes that detail's frame, beyond it the background takes the end of the stack where it is quietest, with\n"
                                                  "soft transitions; its focus curves are also smoothed over this width (FINDINGS 34). Default 128; 0 = off\n"
                                                  "(the depth stage's own choice, which breaks the background into patches with visible seams).");
    al->addRow("Flat smoothing", flatBlur_);
    QPushButton* resetAdv = new QPushButton("Defaults");
    resetAdv->setToolTip("Put every advanced parameter back to its validated default.");
    connect(resetAdv, &QPushButton::clicked, this, [this] {
        depthRes_->setCurrentIndex(1);
        focusRadius_->setValue(6); focusPower_->setValue(4); depthFmPx_->setValue(6); volAggPx_->setValue(32);
        refineR_->setValue(12); refineFmPx_->setValue(9); softPx_->setValue(3); haloPrior_->setValue(2); darkPrior_->setValue(1);
        darkReach_->setValue(24); flatBlur_->setValue(128);
    });
    al->addRow("", resetAdv);
    pl->addWidget(advanced_);

    // ---- performance: what the benchmark tunes
    QGroupBox* perf = new QGroupBox("Performance");
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
    cacheFrames_->setToolTip("Frames are read once and kept pinned (up to about half the machine's RAM) for every pass and the next render of the\n"
                             "same folder; the GPUs then pull them at PCIe speed instead of the disk's. Untick to read from disk every pass.");
    pfl->addRow("", cacheFrames_);
    QPushButton* benchBtn = new QPushButton("Benchmark…");
    connect(benchBtn, &QPushButton::clicked, this, &MainWindow::benchmark);
    pfl->addRow("", benchBtn);
    pl->addWidget(perf);

    progress_ = new QProgressBar;
    progress_->setRange(0, 1000);
    progress_->setValue(0);
    stage_ = new QLabel("Open a folder of TIFF frames, choose the outputs, press Render.");
    stage_->setWordWrap(true);
    pl->addWidget(stage_);
    pl->addWidget(progress_);
    pl->addStretch(1);
    QScrollArea* scroll = new QScrollArea;
    scroll->setWidget(pw);
    scroll->setWidgetResizable(true);
    scroll->setFrameShape(QFrame::NoFrame);
    scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    params->setWidget(scroll);
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
    geo->raise();
    resizeDocks({geo, lg}, {110, 110}, Qt::Vertical);
    resizeDocks({framesDock_, params}, {200, 330}, Qt::Horizontal);

    // the panel toggles in the View menu
    for (QMenu* m : menuBar()->findChildren<QMenu*>())
        if (m->title() == "&View")
            for (QDockWidget* d : {framesDock_, params, geo, lg}) m->addAction(d->toggleViewAction());

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
    scale_->setCurrentIndex(s.value("scale", 0).toInt());
    ref_->setCurrentIndex(s.value("ref", 0).toInt());
    lockScale_->setChecked(s.value("lockScale", false).toBool());
    liveRes_->setCurrentIndex(s.value("liveRes", 2).toInt());
    interp_->setCurrentIndex(s.value("interp", 1).toInt());
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
    flatBlur_->setValue(s.value("flatBlur", 128.0).toDouble());
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
    s.setValue("liveRes", liveRes_->currentIndex());
    s.setValue("interp", interp_->currentIndex());
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
    for (int i = 0; i < 11; ++i) outputs_[kOutputs[i]]->setChecked(i < 2);
    scale_->setCurrentIndex(0);
    ref_->setCurrentIndex(0);
    lockScale_->setChecked(false);
    liveRes_->setCurrentIndex(2);
    interp_->setCurrentIndex(1);
    smoothing_->setValue(2);
    cradius_->setValue(32);
    bradius_->setValue(0);
    advanced_->setChecked(false);
    depthRes_->setCurrentIndex(1);
    focusRadius_->setValue(6); focusPower_->setValue(4); depthFmPx_->setValue(6); volAggPx_->setValue(32);
    refineR_->setValue(12); refineFmPx_->setValue(9); softPx_->setValue(3); haloPrior_->setValue(2); darkPrior_->setValue(1);
    darkReach_->setValue(24); flatBlur_->setValue(128);
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
    for (int i = 2; i < 11; ++i) if (!isMenuOutput(i)) outputs_[kOutputs[i]]->setVisible(on);   // A, B0 stay hidden boxes
    if (devOutMenu_) devOutMenu_->menuAction()->setVisible(on);
}

std::vector<std::string> MainWindow::selectedOutputs() const {
    std::vector<std::string> out;
    for (int i = 0; i < 11; ++i)
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
    QString d = QFileDialog::getExistingDirectory(this, "Folder with the stack's TIFF frames", folder_.isEmpty() ? recent_.value(0) : folder_);
    if (!d.isEmpty()) openFolder(d);
}

void MainWindow::openFolder(const QString& dir) {
    if (worker_.running()) {   // the running job refers to the list's rows (jobItems_, liveItem_, markedItem_)
        statusBar()->showMessage("A render is running; open the folder when it has finished.", 4000);
        return;
    }
    QDir d(dir);
    QStringList files = d.entryList({"*.tif", "*.tiff", "*.TIF", "*.TIFF"}, QDir::Files, QDir::Name);
    if (files.size() < 2) {
        QMessageBox::warning(this, "xLamina", QString("%1\n\nholds %2 TIFF files; a stack needs at least 2.").arg(dir).arg(files.size()));
        return;
    }
    thumbs_.stop();
    thumbsDone_.clear();
    folder_ = dir;
    frames_.clear();
    jobItems_.clear();
    flagged_.clear();
    dropFlagged_->setEnabled(false);
    frameList_->clear();
    liveItem_ = markedItem_ = nullptr;   // the rows they pointed at are gone
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
    framesDock_->setWindowTitle(QString("Frames (%1)").arg(frames_.size()));
    QString note = QString("%1 frames of %2 x %3").arg(frames_.size()).arg(best.first).arg(best.second);
    if (skipped) note += QString(" (%1 other TIFF%2 left out)").arg(skipped).arg(skipped > 1 ? "s" : "");
    if (colorSpace_.isValid()) note += QString(", %1").arg(colorSpace_.description().isEmpty() ? "ICC profile" : colorSpace_.description());
    stage_->setText(note + ". Press Render.");
    statusBar()->showMessage(note + " in " + QDir::toNativeSeparators(dir));
    for (int i = 0; i < frameList_->count(); ++i)
        if (frameList_->item(i)->data(Qt::UserRole).toInt() == frames_.size() / 2) { frameList_->setCurrentRow(i); break; }
    // the middle frame is selected (shown in the viewer) but the list opens at its first frame: setCurrentRow
    // scrolled it to the middle, and so did the list's deferred layout once the rows had their height
    frameList_->scrollToTop();
    QTimer::singleShot(0, frameList_, &QListWidget::scrollToTop);
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
    if (!files.isEmpty()) thumbs_.start(files, 144, idx);
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
void MainWindow::onFrame(int k, FloatImage rgb, int w, int h) {
    if (!rgb) return;
    // at most ~30 views a second, fewer when they are large (Full: 30 Mpx to convert, ~300 ms each)
    const qint64 minMs = std::max<qint64>(33, (qint64)w * h / 100000);
    bool show = !(bench_ && bench_->running()) && k >= 0 && k < jobItems_.size() && (!lastView_.isValid() || lastView_.elapsed() >= minMs);
    QImage image = show ? toQImage(rgb->data(), w, h) : QImage();
    worker_.viewShown(std::move(rgb));   // the buffer goes back to the pool
    if (k >= 0 && k < jobItems_.size()) liveItem_ = jobItems_[k];   // the job's frame -> its row (rows may be reordered or duplicated)
    if (image.isNull()) return;
    lastView_.start();
    k = liveItem_->data(Qt::UserRole).toInt();   // the folder's frame
    current_ = "frame";
    view_->setImage(forDisplay(image), true);
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
        twoUp_->addImage("depth", outputImage("depth"), "this render");
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
        tabs_->addTab("depth");
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
    if (fov_->value() > 0) jo.align.um_per_px = -fov_->value();
    jo.stack.po.g.scale = scale_->currentIndex() == 0 ? 1.0 : 0.5;
    jo.stack.po.g.interp = interp_->currentIndex();
    jo.stack.c_smoothing = smoothing_->value();
    jo.stack.po.c_radius_px = cradius_->value();
    jo.stack.po.b_radius_px = bradius_->value();
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
    stage_->setText(stageText_);
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
    stage_->setText(QString("%1 — %2 s").arg(stage).arg(elapsed_.elapsed() / 1000.0, 0, 'f', 1));
    progress_->setValue((int)std::lround(std::clamp(fraction, 0.0, 1.0) * 1000));
    // the frame count comes from here, not from the live view: the view is throttled and skips frames, so its
    // last frame index stalled short of the total ("157 of 163"). Every progress call is delivered.
    if (stage.startsWith("reading") || stage.startsWith("pass")) {
        const int total = jobItems_.size();
        const int n = (int)std::lround(std::clamp(fraction, 0.0, 1.0) * total);
        const QString what = stage.startsWith("reading") ? "Read" : stage.startsWith("pass 1") ? "Aligned"
                           : stage.startsWith("pass 2") ? "Fine-focused" : "Composed";
        progressStatus_ = QString("%1 %2 of %3 frames").arg(what).arg(n).arg(total);
    } else if (!stage.isEmpty()) {   // "aligning pairs" (a share of the pairs), "depth map", "finalize", "done"
        progressStatus_ = stage;
        progressStatus_[0] = progressStatus_[0].toUpper();
        if (fraction > 0.0 && fraction < 1.0) progressStatus_ += QString(" — %1 %").arg((int)std::lround(fraction * 100));
    }
    statusBar()->showMessage(progressStatus_);
}

void MainWindow::onPreview(QString name, FloatImage rgb, int w, int h) {
    if (!rgb) return;
    if (bench_ && bench_->running()) { worker_.previewShown(std::move(rgb)); return; }
    float gain = 1.f;
    if (name.startsWith("focus")) {   // the focus measure has no fixed scale: show it up to its 99th percentile
        std::vector<float> v;
        v.reserve((size_t)w * h / 4 + 1);
        for (size_t i = 0; i < (size_t)w * h; i += 4) v.push_back((*rgb)[i * 3]);
        if (!v.empty()) {
            size_t q = v.size() * 99 / 100;
            std::nth_element(v.begin(), v.begin() + q, v.end());
            gain = v[q] > 1e-9f ? 1.f / v[q] : 1.f;
        }
    }
    QImage image = toQImage(rgb->data(), w, h, gain);
    worker_.previewShown(std::move(rgb));   // the buffer goes back to the pool
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
        liveView_->setLabel(name);
        return;
    }
    current_ = name;
    view_->setImage(image, true);
    view_->setLabel(name + "  (preview)");
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
    if (bench_ && bench_->running()) return;   // the benchmark's runs are its own
    tick_.stop();
    listTick_.stop();
    if (markedItem_) markedItem_->setBackground(QBrush());
    liveItem_ = markedItem_ = nullptr;
    if (data->sweep) {   // the last render's outputs stay; the variants go to the two-up viewer
        setBusy(false);
        liveView_->hide();
        resumeThumbnails();
        if (data->rc == 3) { stage_->setText("Sweep cancelled."); return; }
        stage_->setText(data->rc ? "Sweep finished with failures; see the log." : QString("Sweep done: %1 variants in %2 s.").arg(data->variants).arg(data->seconds, 0, 'f', 1));
        progress_->setValue(1000);
        if (QFileInfo::exists(QDir(data->outDir).filePath("sweep.json"))) { twoUp(); twoUp_->openSweep(data->outDir); }
        return;
    }
    result_ = data;
    setBusy(false);
    liveView_->hide();
    resumeThumbnails();
    labelWobble(data->align);
    if (data->rc == 3) { stage_->setText("Cancelled."); return; }
    if (data->rc != 0) { stage_->setText("Failed; see the log."); QMessageBox::warning(this, "xLamina", "The render failed. The log has the details."); return; }
    stage_->setText(QString("Done in %1 s (alignment %2 s, fusion %3 s).").arg(data->seconds, 0, 'f', 1).arg(data->align.seconds, 0, 'f', 1).arg(data->result.seconds, 0, 'f', 1));
    progress_->setValue(1000);
    setStatusGeometry(*data);
    rebuildTabs();
    tabs_->setCurrentIndex(0);
    showOutput(0);
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
    QString base = isCompare ? QFileInfo(current_).completeBaseName() + "_copy" : current_;
    QString def = QDir(outDir_->text().isEmpty() ? folder_ + "_stacker" : outDir_->text()).filePath(base + (isCompare ? ".png" : ".tif"));
    QString filter = isCompare ? "PNG (*.png);;JPEG (*.jpg)" : "16-bit TIFF (*.tif);;JPEG (*.jpg);;PNG (*.png)";
    QString fn = QFileDialog::getSaveFileName(this, "Save " + current_, def, filter);
    if (fn.isEmpty()) return;
    bool ok = false;
    if (!isCompare && (fn.endsWith(".tif", Qt::CaseInsensitive) || fn.endsWith(".tiff", Qt::CaseInsensitive))) {
        const auto& R = result_->result;
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
        if (on) stage_->setText("Benchmark running…");
        else { progress_->setValue(0); stage_->setText(result_ ? QString("Done in %1 s.").arg(result_->seconds, 0, 'f', 1) : "Benchmark finished."); }
    });
    bench_->show();
}

void MainWindow::autoBenchmark(bool sample, const QString& reportPath) {
    benchmark();
    if (!bench_) return;
    connect(bench_, &QObject::destroyed, qApp, &QApplication::quit, Qt::QueuedConnection);
    bench_->autoRun(sample, reportPath);
}

void MainWindow::about() {
    QStringList g;
    for (const GpuInfo& gi : gpus_) g << gi.label();
    QMessageBox box(this);
    box.setWindowTitle("About xLamina");
    box.setIconPixmap(QPixmap(":/xlamina-128.png"));
    box.setTextFormat(Qt::RichText);
    box.setText(QString("<h3>xLamina %1</h3>"
                        "<p>Multi-GPU focus stacker for photomicroscopy.<br>"
                        "<a href=\"https://github.com/toldxls/xLamina\">github.com/toldxls/xLamina</a></p>"
                        "<p><b>GPUs</b><br>%2</p><p>%3<br>Qt %4</p>"
                        "<p>Settings file: %5</p>")
                    .arg(XLAMINA_VERSION, g.isEmpty() ? "none found" : g.join("<br>"), hostSummary(), qVersion(),
                         QDir::toNativeSeparators(appSettings().fileName())));
    box.exec();
}

// ----------------------------------------------------------------------------- events

void MainWindow::dragEnterEvent(QDragEnterEvent* e) {
    if (e->mimeData()->hasUrls()) e->acceptProposedAction();
}

void MainWindow::dropEvent(QDropEvent* e) {
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
        worker_.cancel();
    }
    saveSettings();
    e->accept();
}
