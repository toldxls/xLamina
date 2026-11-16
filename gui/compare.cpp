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
#include <QPainter>
#include <QRegularExpression>
#include <QScrollBar>
#include <QSettings>
#include <QSignalBlocker>
#include <QStatusBar>
#include <QTimer>
#include <QListView>
#include <QCheckBox>
#include <QMenu>
#include <QToolBar>
#include <QToolButton>
#include <QToolTip>
#include <QWidgetAction>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <thread>

#include "diaglog.h"
#include "prefs.h"
#include "stackcore/compose.h"
#include "stackcore/depth.h"
#include "stackcore/tiff.h"

using stackcore::TiffInfo;
using stackcore::tiff_probe;

namespace {
const int kKeepDecoded = 8;   // file-backed images held in memory (about 120 MB each at full resolution)
const int kUndoDepth = 40;
}

CompareWindow::CompareWindow(QWidget* parent) : QMainWindow(parent) {
    setWindowFlag(Qt::Window);
    setWindowTitle("xLamina — compare");
    setAcceptDrops(true);
    resize(1500, 900);

    QToolBar* tb = mainTb_ = addToolBar("Compare");
    tb->setMovable(false);
    QAction* openSweep = tb->addAction("Open sweep…", this, &CompareWindow::openSweepDialog);
    openSweep->setStatusTip("A folder written by a sweep (holds sweep.json): its B variants join the list.");
    QAction* openImage = tb->addAction("Add image…", this, &CompareWindow::openImageDialog);
    openImage->setStatusTip("Any TIFF, JPEG or PNG, for example another stacker's output.");
    QAction* frames = tb->addAction("Add frames", this, &CompareWindow::addFrames);
    frames->setToolTip("Put every frame of the stack in the list. A frame is shown lined up with the outputs,\n"
                       "so it can be compared with them and painted from. PgUp / PgDn step through the list.");
    tb->addSeparator();
    flipAct_ = tb->addAction("Flip", this, &CompareWindow::flip);
    flipAct_->setShortcuts({QKeySequence(Qt::Key_Space), QKeySequence("F")});
    flipAct_->setStatusTip("Swap the two panes (Space): the image changes under your eye, at the same zoom and position.");
    alignAct_ = tb->addAction("Align");
    alignAct_->setCheckable(true);
    alignAct_->setShortcut(QKeySequence("A"));
    alignAct_->setToolTip("Line the right pane's image up with the left one (A): for an image of another size or crop,\n"
                          "such as another program's stack of the same frames.\n"
                          "Checked = the right image is moved and scaled onto the left one. Unchecked = each as it is.");
    connect(alignAct_, &QAction::toggled, this, [this](bool on) {
        if (!on) align_.left = align_.right = 0;   // on again: the whole-image fit afresh (Align here had moved it)
        applyAlign();
    });
    alignHereAct_ = tb->addAction("Align here", this, &CompareWindow::alignHere);
    alignHereAct_->setShortcut(QKeySequence("Shift+A"));
    alignHereAct_->setToolTip("Line the two images up exactly for the part on screen (Shift+A).");
    QAction* fit = tb->addAction("Fit", this, [this] { view_[0]->fit(); });
    fit->setShortcut(QKeySequence("Ctrl+0"));
    QAction* one = tb->addAction("1:1", this, [this] { view_[0]->actualSize(); });
    one->setShortcut(QKeySequence("Ctrl+1"));
    // measuring, as in the main window: a line drawn in one pane is shown at the same place in the other
    measureAct_ = tb->addAction("Measure");
    measureAct_->setCheckable(true);
    measureAct_->setShortcut(QKeySequence("M"));
    measureAct_->setToolTip("Measure: drag a line on either image and its length is written beside it, in both panes (M).\n"
                            "Shift keeps the line level, upright or at 45°. The middle or right button pans.");
    clearMeasuresAct_ = tb->addAction("Clear lines", this, [this] { for (ImageView* v : view_) v->clearMeasures(); });
    clearMeasuresAct_->setToolTip("Remove the measured lines (Backspace removes the last one)");
    clearMeasuresAct_->setVisible(false);
    measureUnit_ = new QComboBox;
    measureUnit_->addItems({"auto", "px", "µm", "mm", "cm"});
    measureUnit_->setToolTip("The unit of the measured lengths. auto = µm or mm, whichever reads best.\n"
                             "µm, mm and cm need Field of view in the main window; without it lengths are in pixels.\n"
                             "The scale bar has its own unit: click the bar to change it.");
    connect(measureUnit_, QOverload<int>::of(&QComboBox::activated), this, [this](int u) {
        for (ImageView* v : view_) v->setMeasureUnit(u);
        emit unitsChanged(u, view_[0]->barUnit());
    });
    tb->addWidget(measureUnit_);
    connect(measureAct_, &QAction::toggled, this, [this](bool on) {
        // one tool has the left button: the region tools go off, and while retouching the brush rests until Measure is off
        if (on) for (QAction* a : {rectAct_, lassoAct_, frameHereAct_}) a->setChecked(false);
        if (rt_.on && !view_[0]->regionTool()) view_[0]->setBrush(!on, brushSize_->value(), brushHard_->value());
        for (ImageView* v : view_) v->setMeasure(on);
        clearMeasuresAct_->setVisible(on);
        if (on) statusBar()->showMessage(scaleUm_ > 0 ? "Measure: drag a line on either image. Shift keeps it level or upright; the right button pans; Backspace removes the last line."
                                                      : "Measure: lengths are in pixels (Field of view is not set in the main window, or these images are not this render's outputs).", 10000);
    });
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
    paneCount_ = new QComboBox;
    paneCount_->addItems({"2 panes", "3 panes", "4 panes"});
    paneCount_->setToolTip("How many images are on show at once, all at the same place and zoom.\n"
                           "A click in the list fills the right pane; with three or four, the pane you clicked last (its box is outlined).\n"
                           "While retouching, the outlined pane is the one painted from: click another pane to paint from that one.\n"
                           "Flip, Align and Difference work on the first two.");
    connect(paneCount_, QOverload<int>::of(&QComboBox::activated), this, [this](int i) { setPaneCount(i + 2); });
    tb->addWidget(paneCount_);
    verticalAct_ = tb->addAction("Stack vertically");
    verticalAct_->setCheckable(true);
    verticalAct_->setStatusTip("Put the panes under each other (for wide crops). Four panes are always two rows of two.");
    connect(verticalAct_, &QAction::toggled, this, [this](bool) { setPaneCount(nPanes_); });
    {   // Help, at the row's right end
        QWidget* gap = new QWidget;
        gap->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
        tb->addWidget(gap);
        QAction* helpAct = tb->addAction("Help", this, [this] { emit helpRequested("compare"); });
        helpAct->setShortcut(QKeySequence::HelpContents);
        helpAct->setToolTip("The manual's pages on this window, in the browser (F1): comparing, retouching, refining regions, composites.");
    }
    QAction* next = new QAction(this);
    next->setShortcut(QKeySequence("PgDown"));
    connect(next, &QAction::triggered, this, [this] { if (heldFrame_ >= 0) stepLayer(1); else step(active_, 1); });   // a held layer: the next layer
    addAction(next);
    QAction* prev = new QAction(this);
    prev->setShortcut(QKeySequence("PgUp"));
    connect(prev, &QAction::triggered, this, [this] { if (heldFrame_ >= 0) stepLayer(-1); else step(active_, -1); });
    addAction(prev);

    // ---- retouch (a row of its own)
    addToolBarBreak();
    QToolBar* rb = retouchTb_ = addToolBar("Retouch");
    rb->setMovable(false);
    retouchAct_ = rb->addAction("Retouch");
    retouchAct_->setCheckable(true);
    retouchAct_->setShortcut(QKeySequence("R"));
    retouchAct_->setStatusTip("Paint the right pane's image into the left one (R). Left button paints, middle button pans, [ ] or Ctrl+wheel size the brush, Space peeks at the source.");
    connect(retouchAct_, &QAction::toggled, this, &CompareWindow::retouchToggle);
    connect(retouchAct_, &QAction::toggled, this, [this] { markRows(); });   // (after retouchToggle: the row shows the mode)
    rb->addWidget(new QLabel(" Size "));
    brushSize_ = new QDoubleSpinBox;
    brushSize_->setRange(2, 2000);
    brushSize_->setDecimals(0);
    brushSize_->setSuffix(" px");
    brushSize_->setValue(60);
    brushSize_->setToolTip("Brush radius in image pixels ([ and ], or Ctrl + wheel over the image).");
    connect(brushSize_, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double) { if (rt_.on) restBrush(); });
    rb->addWidget(brushSize_);
    rb->addWidget(new QLabel(" Hardness "));
    brushHard_ = new QDoubleSpinBox;
    brushHard_->setRange(0, 1);
    brushHard_->setSingleStep(0.1);
    brushHard_->setDecimals(1);
    brushHard_->setValue(0.5);
    brushHard_->setToolTip("Share of the radius painted at full strength; the rest fades out.");
    connect(brushHard_, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double) { if (rt_.on) restBrush(); });
    rb->addWidget(brushHard_);
    rb->addSeparator();
    cloneAct_ = rb->addAction("Clone");
    cloneAct_->setCheckable(true);
    cloneAct_->setShortcut(QKeySequence("C"));
    cloneAct_->setToolTip("Copy from another place of the left image (C), for a dust speck or a stray mark:\n"
                          "hold Alt and click the place to copy from, then paint over the spot.\n"
                          "Checked = the brush copies from that place; unchecked = it paints from the other pane.");
    connect(cloneAct_, &QAction::toggled, this, [this](bool on) {
        view_[0]->setAltPick(on && rt_.on);
        showCloneMark();
        if (on) statusBar()->showMessage(clone_.have ? "Clone: paint over the spot. Alt + click picks another place to copy from."
                                                     : "Clone: hold Alt and click the place to copy from, then paint over the spot.", 12000);
    });
    matchAct_ = rb->addAction("Match tone");
    matchAct_->setCheckable(true);
    matchAct_->setChecked(appSettings().value("compare/cloneMatch", true).toBool());
    matchAct_->setToolTip("For Clone. Checked = the copy takes the colour and brightness of the place it is painted on,\n"
                          "and only its detail comes from the place copied. Unchecked = an exact copy.");
    connect(matchAct_, &QAction::toggled, this, [](bool on) { QSettings s = appSettings(); s.setValue("compare/cloneMatch", on); });
    for (QAction* a : {cloneAct_, matchAct_}) a->setEnabled(false);
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
    rb->addSeparator();
    frameHereAct_ = rb->addAction("Frame here");
    frameHereAct_->setCheckable(true);
    frameHereAct_->setShortcut(QKeySequence("H"));
    frameHereAct_->setToolTip("Find the frame to paint from (H): click a spot on an output, and the frame of the stack\n"
                              "that is sharpest there comes up in the right pane, lined up with the outputs.\n"
                              "PgUp / PgDn then step to the frames before and after it.");
    connect(frameHereAct_, &QAction::toggled, this, [this](bool on) {
        // one tool has the left button: the others go off, and while retouching the brush rests for the one click
        if (on) for (QAction* a : {rectAct_, lassoAct_, measureAct_}) a->setChecked(false);
        if (rt_.on && !view_[0]->regionTool() && !measureAct_->isChecked()) view_[0]->setBrush(!on, brushSize_->value(), brushHard_->value());
        const bool depth = depthShown();
        for (ImageView* v : view_) v->setPrecise(on || depth);
        if (on) statusBar()->showMessage("Frame here: click the spot. The frame of the stack that is sharpest there comes up on the right.", 12000);
    });

    // ---- regions of the left pane's output rendered again (a row of its own)
    addToolBarBreak();
    regionTb_ = addToolBar("Regions");
    regionTb_->setMovable(false);
    regionLabel_ = new QLabel(" Refine regions: ");
    regionLabel_->setToolTip("Renders marked parts of the left pane's image again, with more detail.\n"
                             "The left pane must hold an output of the last render (B, C, CBn …), not a file, a frame or a composite.");
    regionTb_->addWidget(regionLabel_);
    rectAct_ = regionTb_->addAction("Rectangle");
    rectAct_->setCheckable(true);
    rectAct_->setToolTip("Mark a region to render again: drag a rectangle on either pane.\nMark as many as needed, then press Refine. The middle or right button pans.");
    lassoAct_ = regionTb_->addAction("Lasso");
    lassoAct_->setCheckable(true);
    lassoAct_->setToolTip("Mark a region to render again: draw around it on either pane (the outline closes itself).\nMark as many as needed, then press Refine. The middle or right button pans.");
    connect(rectAct_, &QAction::toggled, this, [this](bool on) {
        if (on && lassoAct_->isChecked()) { QSignalBlocker b(lassoAct_); lassoAct_->setChecked(false); }
        regionTool(on ? 1 : 0);
    });
    connect(lassoAct_, &QAction::toggled, this, [this](bool on) {
        if (on && rectAct_->isChecked()) { QSignalBlocker b(rectAct_); rectAct_->setChecked(false); }
        regionTool(on ? 2 : 0);
    });
    regionTb_->addWidget(new QLabel("  More detail "));
    strength_ = new QComboBox;
    strength_->addItems({"Low", "Medium", "High"});
    strength_->setCurrentIndex(1);
    strength_->setToolTip("How much weak detail the region keeps when it is rendered again (for the next region marked, and for the selected one).\n"
                          "Low = a little more than the render, High = the most, with more risk of blobs and noise.");
    connect(strength_, QOverload<int>::of(&QComboBox::activated), this, [this](int s) {
        const int i = selectedRegion();
        if (i < 0 || refining_ || regions_[i].strength == s) return;
        regions_[i].strength = s;
        if (regions_[i].state != 1) regions_[i].state = regions_[i].rendered == s && regions_[i].hasPatch() ? 2 : 0;
        showRegions();
        if (regions_[i].state == 0) statusBar()->showMessage(QString("Region %1: press Refine to render it at this strength.").arg(i + 1), 6000);
    });
    regionTb_->addWidget(strength_);
    regionTb_->addWidget(new QLabel("  Edge "));
    feather_ = new QDoubleSpinBox;
    feather_->setRange(0, 200);
    feather_->setDecimals(0);
    feather_->setSuffix(" px");
    feather_->setSpecialValueText("Auto");   // shown at 0
    feather_->setValue(0);
    feather_->setToolTip("Width of the soft edge between a region and the image around it, for regions rendered from now on.\n"
                         "Auto = a tenth of the region's shorter side, between 8 and 64 px; the list shows the width each region got.\n"
                         "Measure (M) with its unit on px gives a width in pixels.");
    regionTb_->addWidget(feather_);
    regionTb_->addSeparator();
    refineAct_ = regionTb_->addAction("Refine", this, &CompareWindow::refine);
    refineAct_->setToolTip("Render the marked regions again and lay them into the left pane's image. The rest of the image is not touched.");
    if (QWidget* b = regionTb_->widgetForAction(refineAct_)) b->setObjectName("primary");   // drawn as a button (markRows)
    refineCancelAct_ = regionTb_->addAction("Cancel", this, [this] { if (refining_) emit refineCancelRequested(); });
    refineCancelAct_->setVisible(false);
    removeRegionAct_ = regionTb_->addAction("Remove", this, [this] { removeRegion(selectedRegion()); });
    removeRegionAct_->setToolTip("Remove the selected region; its part of the image goes back to what it was.");
    clearRegionsAct_ = regionTb_->addAction("Remove all", this, &CompareWindow::clearRegions);
    clearRegionsAct_->setToolTip("Remove every region; the image goes back to what it was (brush strokes stay).");
    // (the row is always there: hidden without a render of this session, it looked as if the feature had gone; its
    // tools say what they need when pressed)

    regionList_ = new QTableWidget(0, 4);
    regionList_->setHorizontalHeaderLabels({"Region", "Detail", "Edge", "State"});
    regionList_->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
    regionList_->horizontalHeader()->setStretchLastSection(true);
    regionList_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    regionList_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    regionList_->verticalHeader()->setVisible(false);
    regionList_->setSelectionBehavior(QAbstractItemView::SelectRows);
    regionList_->setSelectionMode(QAbstractItemView::SingleSelection);
    regionList_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    regionList_->setToolTip("The regions marked for Refine. Ticked = the rendered region is in the image; unticked = the image as it was there.");
    regionList_->hide();
    QAction* delRegion = new QAction(regionList_);
    delRegion->setShortcut(QKeySequence::Delete);
    delRegion->setShortcutContext(Qt::WidgetShortcut);
    connect(delRegion, &QAction::triggered, this, [this] { removeRegion(selectedRegion()); });
    regionList_->addAction(delRegion);
    connect(regionList_, &QTableWidget::itemChanged, this, [this](QTableWidgetItem* it) {
        if (fillingRegions_ || !it || it->column() != 0) return;
        setRegionOn(it->row(), it->checkState() == Qt::Checked);
    });
    connect(regionList_, &QTableWidget::currentCellChanged, this, [this](int row, int, int, int) {
        if (fillingRegions_) return;
        if (row >= 0 && row < (int)regions_.size()) strength_->setCurrentIndex(regions_[row].strength);
        showRegions();
    });

    // ---- the depth map's colours (a row of its own, shown when the render's depth map is in the list)
    addToolBarBreak();
    depthTb_ = addToolBar("Depth");
    depthTb_->setMovable(false);
    depthBar_ = new DepthBar(true);
    depthTb_->addWidget(depthBar_);
    depthTb_->hide();
    connect(depthBar_, &DepthBar::changed, this, [this] {
        const DepthStyle st = depthBar_->style();
        const bool samePicture = st.sameImage(depthStyle_);
        depthStyle_ = st;
        if (!samePicture) recolourDepth();
        else updateOverlay();
        emit depthStyleChanged(depthStyle_);
    });
    connect(depthBar_, &DepthBar::fitRequested, this, [this] {
        if (!depth_) return;
        const auto& R = depth_->result;
        depthRange(R.depth.data(), R.depth.size(), (int)depth_->align.paths.size(), depthStyle_.lo, depthStyle_.hi);
        depthBar_->setStyle(depthStyle_);
        recolourDepth();
        emit depthStyleChanged(depthStyle_);
    });

    // the list of images: name, change vs base, and the sweep options (or where it came from)
    list_ = new QTableWidget(0, 4);
    list_->setHorizontalHeaderLabels({"Image", "Change", "Options", "+"});
    list_->horizontalHeaderItem(3)->setToolTip("Tick two or more images of one size, then press Make composite under the list.");
    list_->horizontalHeader()->moveSection(3, 0);   // the tick boxes first
    list_->horizontalHeader()->setSectionResizeMode(3, QHeaderView::ResizeToContents);
    list_->horizontalHeaderItem(1)->setToolTip("Share of B's pixels that differ from the base variant by more than 2 % of full scale: how much there is to see.");
    list_->horizontalHeader()->setStretchLastSection(true);
    list_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    list_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    list_->verticalHeader()->setVisible(false);
    list_->setSelectionBehavior(QAbstractItemView::SelectRows);
    list_->setSelectionMode(QAbstractItemView::SingleSelection);
    list_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    list_->setToolTip("Click: show in the right pane (with three or four panes: the pane clicked last). Double-click: show in the left pane.\n"
                      "PgUp / PgDown step that pane through the list.");
    connect(list_, &QTableWidget::cellClicked, this, [this](int row, int col) { if (col != 3) showIn(active_, row); });   // (a tick is not a pick)
    connect(list_, &QTableWidget::cellDoubleClicked, this, [this](int row, int col) { if (col != 3) pickLeft(row); });
    connect(list_, &QTableWidget::itemChanged, this, [this](QTableWidgetItem* it) {
        if (fillingList_ || !it || it->column() != 3 || it->row() >= entries_.size()) return;
        entries_[it->row()].pick = it->checkState() == Qt::Checked;
        updateCompositeButton();
    });
    framesBtn_ = new QPushButton("Hide frames");
    framesBtn_->setToolTip("Takes the stack's frames out of the list and the panes' pickers, or puts them back.\n"
                           "The outputs and the added images stay; a pane showing a frame keeps it.");
    framesBtn_->hide();   // there with the first frame in the list
    framesHidden_ = appSettings().value("compare/hideFrames", false).toBool();
    connect(framesBtn_, &QPushButton::clicked, this, [this] { setFramesHidden(!framesHidden_); });
    compositeBtn_ = new QPushButton("Make composite");
    compositeBtn_->setEnabled(false);
    compositeBtn_->setToolTip("Stacks the ticked images again, as if they were frames, with the Render panel's settings,\n"
                              "once for each method ticked under Methods. The result is named by what went in: 2B1C = two B and one C.");
    connect(compositeBtn_, &QPushButton::clicked, this, &CompareWindow::makeComposite);
    // the methods a composite is made with: B, C and CBn to begin with, the others a tick away (the menu stays open
    // while its boxes are ticked)
    methodsBtn_ = new QToolButton;
    methodsBtn_->setPopupMode(QToolButton::InstantPopup);
    methodsBtn_->setToolButtonStyle(Qt::ToolButtonTextOnly);
    methodsBtn_->setToolTip("The methods a composite is made with: one image for each ticked method.");
    {
        QMenu* m = new QMenu(methodsBtn_);
        const QStringList on = appSettings().value("compare/compositeMethods", "B,C,CBn").toString().split(',', Qt::SkipEmptyParts);
        for (const char* name : {"B", "C", "CBn", "Cn", "Cx", "Cd", "Cg", "Cg2", "H"}) {
            QCheckBox* box = new QCheckBox(name);
            box->setStyleSheet("QCheckBox { padding: 4px 14px 4px 8px; }");
            box->setChecked(on.contains(name));
            QWidgetAction* a = new QWidgetAction(m);
            a->setDefaultWidget(box);
            if (methodBox_.size() == kFirstMethods) m->addSeparator();
            m->addAction(a);
            methodBox_ << box;
            methodAct_ << a;
            connect(box, &QCheckBox::toggled, this, [this] {
                QStringList ticked;
                for (QCheckBox* b : methodBox_) if (b->isChecked()) ticked << b->text();
                QSettings s = appSettings();
                s.setValue("compare/compositeMethods", ticked.join(','));
                updateCompositeButton();
            });
        }
        methodsBtn_->setMenu(m);
    }

    split_ = new QSplitter(Qt::Horizontal);
    split2_ = new QSplitter(Qt::Horizontal);
    rows_ = new QSplitter(Qt::Vertical);
    rows_->addWidget(split_);
    rows_->addWidget(split2_);
    split2_->hide();
    for (int p = 0; p < kPanes; ++p) {
        QWidget* w = new QWidget;
        paneBox_[p] = w;
        QVBoxLayout* l = new QVBoxLayout(w);
        l->setContentsMargins(0, 0, 0, 0);
        l->setSpacing(2);
        pick_[p] = new QComboBox;
        pick_[p]->setToolTip(p == 0 ? "The left pane's image (the result, when retouching)" : p == 1 ? "The right pane's image (the source, when retouching)"
                                    : "This pane's image. Click the pane and the list fills it.");
        connect(pick_[p], QOverload<int>::of(&QComboBox::activated), this, [this, p](int i) { if (p == 0) pickLeft(i); else showIn(p, i); });
        view_[p] = new ImageView;
        l->addWidget(pick_[p]);
        l->addWidget(view_[p], 1);
        split_->addWidget(w);
        if (p >= 2) w->hide();
        if (p >= 1) pick_[p]->installEventFilter(this);   // its picker, like a click in the pane itself, makes it the pane the list fills
    }
    split_->setSizes({1, 1});
    // one pan and zoom for all: every scroll or zoom of one pane is applied to the others on show
    for (int a = 0; a < kPanes; ++a) {
        connect(view_[a]->horizontalScrollBar(), &QScrollBar::valueChanged, this, [this, a](int v) {
            if (syncing_ || a >= nPanes_) return;
            syncing_ = true;
            for (int b = 0; b < nPanes_; ++b) if (b != a) view_[b]->horizontalScrollBar()->setValue(v);
            syncing_ = false;
        });
        connect(view_[a]->verticalScrollBar(), &QScrollBar::valueChanged, this, [this, a](int v) {
            if (syncing_ || a >= nPanes_) return;
            syncing_ = true;
            for (int b = 0; b < nPanes_; ++b) if (b != a) view_[b]->verticalScrollBar()->setValue(v);
            syncing_ = false;
        });
        connect(view_[a], &ImageView::zoomChanged, this, [this, a](double) {
            if (syncing_ || a >= nPanes_) return;
            syncing_ = true;
            for (int b = 0; b < nPanes_; ++b) if (b != a && (!view_[b]->image().isNull() || rt_.on)) view_[b]->syncFrom(view_[a]);
            syncing_ = false;
        });
        connect(view_[a], &ImageView::hovered, this, [this, a](int x, int y) { depthHover(a, x, y); });
        connect(view_[a], &ImageView::clicked, this, [this, a](int x, int y) {
            if (frameHereAct_->isChecked()) { frameHere(a, x, y); return; }
            if (a >= 1 && a < nPanes_ && a != active_) setActive(a);   // the pane the list fills, and the brush paints from
            if (!depthShown() || rt_.on || sel_[a] < 0) return;
            const auto& R = depth_->result;
            if (x < 0 || y < 0 || x >= R.w || y >= R.h || entries_[sel_[a]].image.size() != QSize(R.w, R.h)) return;
            const int k = std::clamp((int)std::lround(R.depth[(size_t)y * R.w + x]), 0, (int)depth_->align.paths.size() - 1);
            holdLayer(k == heldFrame_ ? -1 : k);
        });
        connect(view_[a], &ImageView::stepKey, this, [this](int d) { stepLayer(d); });
        connect(view_[a], &ImageView::escapePressed, this, [this] { if (heldFrame_ >= 0) holdLayer(-1); else if (frameHereAct_->isChecked()) frameHereAct_->setChecked(false); });
        connect(view_[a], &ImageView::viewChanged, this, [this] { if (heldFrame_ >= 0) markLayer(); });
        connect(view_[a], &ImageView::measuresChanged, this, [this, a] { for (int b = 0; b < kPanes; ++b) if (b != a) view_[b]->setMeasures(view_[a]->measures()); });
        connect(view_[a], &ImageView::measured, this, [this](QString len) { statusBar()->showMessage(QString("Measured: %1").arg(len), 10000); });
        connect(view_[a], &ImageView::regionDrawn, this, [this, a](QPolygonF frac) { regionDrawn(a, frac); });
        connect(view_[a], &ImageView::barUnitChanged, this, [this, a](int u) { for (int b = 0; b < kPanes; ++b) if (b != a) view_[b]->setBarUnit(u); emit unitsChanged(measureUnit_->currentIndex(), u); });
    }
    // the brush paints on the left pane
    connect(view_[0], &ImageView::altClicked, this, [this](QPointF p) { setCloneSource(p); });
    connect(view_[0], &ImageView::strokeBegin, this, [this](QPointF p) {
        if (!rt_.on) return;
        if (cloneAct_->isChecked()) {   // a copy from another place of this image (the other pane is not needed)
            if (!clone_.have) {
                refuse("Clone: hold Alt and click the place to copy from first.");
                return;
            }
            if (!ensureCloneLayer()) return;
            if (clone_.pending) { clone_.off = (clone_.src - p).toPoint(); clone_.pending = false; showCloneMark(); }
            rt_.inStroke = rt_.cloning = true;
            rt_.current = Stroke{};
            rt_.last = p;
            cloneDab(p);
            return;
        }
        if (rt_.blocked) {
            refuse("Nothing is painted: the image in the pane painted from is not the size of the result. Pick an output, or one of this stack's frames.", 9000);
            return;
        }
        rt_.inStroke = true;
        rt_.erasing = rt_.hasClone();
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
        for (int i = 1; i <= n; ++i) { const QPointF q = rt_.last + d * (double(i) / n); if (rt_.cloning) cloneDab(q); else dab(q); }
        rt_.last = p;
    });
    connect(view_[0], &ImageView::strokeEnd, this, [this] {
        if (!rt_.on || !rt_.inStroke) return;
        rt_.inStroke = false;
        if (rt_.cloning) cloneCommit();
        else if (rt_.erasing) eraseCommit();
        if (!rt_.current.r.isEmpty()) {
            rt_.undo.push_back(std::move(rt_.current));
            if ((int)rt_.undo.size() > kUndoDepth) { rt_.undo.erase(rt_.undo.begin()); rt_.dropped = true; }   // that stroke stays, past undoing
            rt_.dirty = true;
        }
        rt_.current = Stroke{};
        undoAct_->setEnabled(!rt_.undo.empty());
        retouchStatus();
    });
    connect(view_[0], &ImageView::brushResized, this, [this](double r) { brushSize_->blockSignals(true); brushSize_->setValue(r); brushSize_->blockSignals(false); retouchStatus(); });

    QSplitter* left = new QSplitter(Qt::Vertical);   // the images, and under them the regions once one is marked
    {
        QWidget* w = new QWidget;
        QVBoxLayout* l = new QVBoxLayout(w);
        l->setContentsMargins(0, 0, 0, 0);
        l->setSpacing(2);
        l->addWidget(framesBtn_);
        l->addWidget(list_, 1);
        l->addWidget(compositeBtn_);
        methodsBtn_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
        l->addWidget(methodsBtn_);
        left->addWidget(w);
    }
    left->addWidget(regionList_);
    left->setStretchFactor(0, 1);
    left->setStretchFactor(1, 0);
    QSplitter* outer = new QSplitter(Qt::Horizontal);
    outer->addWidget(left);
    outer->addWidget(rows_);
    outer->setStretchFactor(0, 0);
    outer->setStretchFactor(1, 1);
    outer->setSizes({340, 1100});
    setCentralWidget(outer);
    status_ = new QLabel;
    statusBar()->addWidget(status_, 1);
    statusBar()->showMessage("Open a sweep folder or add images, then pick one for each pane. Space flips them; R retouches the left one from the right.");

    connect(&loader_, &ThumbLoader::thumbnail, this, &CompareWindow::onLoaded);
    QSettings s = appSettings();   // the same file as every other setting (it was the registry)
    restoreGeometry(s.value("compare/geometry").toByteArray());
    if (s.contains("compare/outer")) outer->restoreState(s.value("compare/outer").toByteArray());
    setPaneCount(s.value("compare/panes", 2).toInt());
    markRows();
}

CompareWindow::~CompareWindow() { loader_.stop(); }

void CompareWindow::closeEvent(QCloseEvent* e) {
    if (rt_.on && rt_.dirty) {
        auto r = QMessageBox::question(this, "xLamina", "The retouch has unsaved strokes. Close anyway?", QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
        if (r != QMessageBox::Yes) { e->ignore(); return; }
    }
    QSettings s = appSettings();   // the same file as every other setting (it was the registry)
    s.setValue("compare/geometry", saveGeometry());
    if (auto* outer = qobject_cast<QSplitter*>(centralWidget())) s.setValue("compare/outer", outer->saveState());
    QMainWindow::closeEvent(e);
}

// ----------------------------------------------------------------------------- the entries

int CompareWindow::indexOf(const QString& name) const {
    for (int i = 0; i < entries_.size(); ++i) if (entries_[i].name == name) return i;
    return -1;
}

// the same file, whatever the spelling (Windows paths compare without case)
static bool sameFile(const QString& a, const QString& b) {
    return !a.isEmpty() && !b.isEmpty() &&
           QFileInfo(a).absoluteFilePath().compare(QFileInfo(b).absoluteFilePath(), Qt::CaseInsensitive) == 0;
}

void CompareWindow::addEntry(Entry e) {
    auto replace = [&](int i) {
        // the image being retouched, or painted from, stays as it is: the strokes belong to it (a new render's output
        // of the same name slid in under them, and Save result wrote the old pixels under the new name)
        if (rt_.on && (i == rt_.target || i == rt_.source)) {
            statusBar()->showMessage(QString("%1 is being retouched: it was not replaced. Leave Retouch, then press Compare again.").arg(entries_[i].name), 8000);
            return;
        }
        overlayFor_[0] = overlayFor_[1] = -1;   // the difference shown was the old image's
        const bool shown = onShow(i);
        entries_[i] = std::move(e);
        rebuildList();
        if (shown) for (int p = 0; p < nPanes_; ++p) if (sel_[p] == i) refresh(p);
    };
    int i = indexOf(e.name);
    if (i >= 0) {
        // the same name again -> replace: a new render's outputs (in-memory images, whichever folder they were
        // written to: the path is only for retouching's 16-bit data), the same file re-opened, a sweep opened twice.
        // A FILE of the same name from another folder (B.tif of two renders) is a different image: keep both, the
        // newcomer named by its folder, and by its full path when that folder name is taken too
        const bool same_source = (!e.fileBacked && !entries_[i].fileBacked) || e.path.isEmpty() || entries_[i].path.isEmpty() ||
                                 sameFile(e.path, entries_[i].path);
        if (same_source) { replace(i); return; }
        QFileInfo fi(e.path);
        QString folder = QFileInfo(fi.absolutePath()).fileName();
        if (folder.isEmpty()) folder = QDir::toNativeSeparators(fi.absolutePath());   // a drive root: Q:\ /B.tif
        QString named = folder + "/" + e.name;
        int j = indexOf(named);
        if (j >= 0 && !sameFile(e.path, entries_[j].path)) {   // another file of that name in a folder of that name
            named = QDir::toNativeSeparators(fi.absoluteFilePath());
            j = indexOf(named);
        }
        e.name = named;
        if (j >= 0) { replace(j); return; }   // that very file again
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
    fillingList_ = true;
    list_->setRowCount(entries_.size());
    for (int i = 0; i < entries_.size(); ++i) {
        auto* n = new QTableWidgetItem(entries_[i].name);
        n->setToolTip(entries_[i].path.isEmpty() ? entries_[i].note : QDir::toNativeSeparators(entries_[i].path));
        list_->setItem(i, 0, n);
        auto* c = new QTableWidgetItem(entries_[i].change < 0 ? QString() : entries_[i].change == 0 ? QString("base") : QString("%1 %").arg(entries_[i].change, 0, 'f', 1));
        c->setTextAlignment(Qt::AlignRight | Qt::AlignVCenter);
        list_->setItem(i, 1, c);
        list_->setItem(i, 2, new QTableWidgetItem(entries_[i].note));
        auto* t = new QTableWidgetItem;
        t->setFlags(Qt::ItemIsEnabled | Qt::ItemIsUserCheckable);
        t->setCheckState(entries_[i].pick ? Qt::Checked : Qt::Unchecked);
        t->setToolTip("Ticked = goes into the composite");
        list_->setItem(i, 3, t);
    }
    fillingList_ = false;
    updateCompositeButton();
    for (int p = 0; p < kPanes; ++p) {
        pick_[p]->blockSignals(true);
        pick_[p]->clear();
        for (auto& e : entries_) pick_[p]->addItem(e.name);
        pick_[p]->setCurrentIndex(sel_[p]);
        pick_[p]->blockSignals(false);
    }
    applyFramesHidden();
}

// The stack's frames out of the list and the pickers, or back: with a few hundred of them the outputs, the added images
// and the composites are hard to find. The rows stay (an entry's index is its row): they are only not shown.
void CompareWindow::setFramesHidden(bool hidden) {
    framesHidden_ = hidden;
    QSettings s = appSettings();   // kept: Frame here lists every frame, and who hid them once wants them hidden then too
    if (s.value("compare/hideFrames", false).toBool() != hidden) s.setValue("compare/hideFrames", hidden);
    applyFramesHidden();
}

void CompareWindow::applyFramesHidden() {
    int n = 0;
    for (int i = 0; i < entries_.size(); ++i) if (entries_[i].owner) ++n;
    for (int i = 0; i < entries_.size(); ++i) {
        const bool hide = framesHidden_ && entries_[i].owner;
        list_->setRowHidden(i, hide);
        for (int p = 0; p < kPanes; ++p)
            if (auto* lv = qobject_cast<QListView*>(pick_[p]->view())) lv->setRowHidden(i, hide);
    }
    framesBtn_->setVisible(n > 0);
    framesBtn_->setText(QString(framesHidden_ ? "Show frames (%1)" : "Hide frames (%1)").arg(n));
}

QString CompareWindow::testHideFrames(bool hidden) {
    setFramesHidden(hidden);
    int rows = 0, items = 0;
    for (int i = 0; i < entries_.size(); ++i) {
        if (!list_->isRowHidden(i)) ++rows;
        if (auto* lv = qobject_cast<QListView*>(pick_[1]->view()); lv && !lv->isRowHidden(i)) ++items;
    }
    return QString("%1 of %2 rows on show, %3 items in the right picker | button \"%4\" %5").arg(rows).arg(entries_.size()).arg(items)
        .arg(framesBtn_->text(), framesBtn_->isHidden() ? "hidden" : "there");
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
    QSettings s = appSettings();   // the same file as every other setting (it was the registry)
    QString d = QFileDialog::getExistingDirectory(this, "Sweep folder (holds sweep.json)", s.value("compare/lastSweep").toString());
    if (d.isEmpty()) return;
    s.setValue("compare/lastSweep", d);
    openSweep(d);
}

void CompareWindow::openImageDialog() {
    QSettings s = appSettings();   // the same file as every other setting (it was the registry)
    QStringList fns = QFileDialog::getOpenFileNames(this, "Images to compare", s.value("compare/lastImage").toString(), "Images (*.tif *.tiff *.jpg *.jpeg *.png);;All files (*)");
    if (fns.isEmpty()) return;
    s.setValue("compare/lastImage", QFileInfo(fns.first()).absolutePath());
    for (const QString& fn : fns) addFile(QFileInfo(fn).fileName(), fn, QDir::toNativeSeparators(QFileInfo(fn).absolutePath()));
    if (sel_[1] < 0) showIn(1, entries_.size() - 1);
}

// ----------------------------------------------------------------------------- showing

void CompareWindow::setScale(double um, QSize size) {
    scaleUm_ = um;
    scaleSize_ = size;
    for (int p = 0; p < kPanes; ++p) applyScale(p);
}

void CompareWindow::setScaleBar(bool on) {
    for (ImageView* v : view_) v->setScaleBar(on);
}

void CompareWindow::setUnits(int measure, int bar) {
    for (ImageView* v : view_) { v->setMeasureUnit(measure); v->setBarUnit(bar); }
    QSignalBlocker b(measureUnit_);
    measureUnit_->setCurrentIndex(view_[0]->measureUnit());
}

// a pane's image has the scale when it is the size of the render's outputs (its B, C, depth map, a sweep's variants)
void CompareWindow::applyScale(int pane) {
    const int i = pane == 1 && view_[1]->placed() ? sel_[0] : sel_[pane];   // lined up with the left image: its scale
    const bool same = i >= 0 && i < entries_.size() && !entries_[i].image.isNull() && entries_[i].image.size() == scaleSize_;
    view_[pane]->setSceneWidthUm(same ? scaleUm_ : 0);
}

// ----------------------------------------------------------------------------- lining the right image up with the left

namespace {
struct Plane { int w = 0, h = 0; std::vector<float> v; };

Plane halved(const Plane& p) {
    Plane q;
    q.w = std::max(1, p.w / 2); q.h = std::max(1, p.h / 2);
    q.v.resize((size_t)q.w * q.h);
    for (int y = 0; y < q.h; ++y)
        for (int x = 0; x < q.w; ++x) {
            const int x1 = std::min(p.w - 1, 2 * x + 1), y1 = std::min(p.h - 1, 2 * y + 1);
            q.v[(size_t)y * q.w + x] = 0.25f * (p.v[(size_t)2 * y * p.w + 2 * x] + p.v[(size_t)2 * y * p.w + x1] + p.v[(size_t)y1 * p.w + 2 * x] + p.v[(size_t)y1 * p.w + x1]);
        }
    return q;
}

// the plane with its local mean (a box of 9 x 9) removed: the picture's structure, whatever its tone curve
Plane structure(const Plane& p) {
    Plane m = p;
    std::vector<float> t(p.v.size());
    const int r = 4, w = p.w, h = p.h;
    for (int y = 0; y < h; ++y) {
        double acc = 0;
        for (int k = -r; k <= r; ++k) acc += p.v[(size_t)y * w + std::clamp(k, 0, w - 1)];
        for (int x = 0; x < w; ++x) {
            t[(size_t)y * w + x] = (float)(acc / (2 * r + 1));
            acc += p.v[(size_t)y * w + std::min(w - 1, x + r + 1)] - p.v[(size_t)y * w + std::max(0, x - r)];
        }
    }
    for (int x = 0; x < w; ++x) {
        double acc = 0;
        for (int k = -r; k <= r; ++k) acc += t[(size_t)std::clamp(k, 0, h - 1) * w + x];
        for (int y = 0; y < h; ++y) {
            m.v[(size_t)y * w + x] = p.v[(size_t)y * w + x] - (float)(acc / (2 * r + 1));
            acc += t[(size_t)std::min(h - 1, y + r + 1) * w + x] - t[(size_t)std::max(0, y - r) * w + x];
        }
    }
    return m;
}

// the correlation of L with R placed at L = s * R + t, over the part where both exist (every `step`-th pixel of L);
// -1 when they share under a quarter of L
double placedMatch(const Plane& L, const Plane& R, double s, double tx, double ty, int step) {
    double sab = 0, saa = 0, sbb = 0;
    long n = 0, all = 0;
    for (int y = 0; y < L.h; y += step) {
        const double yr = (y - ty) / s;
        const int y0 = (int)std::floor(yr);
        const bool rowIn = y0 >= 0 && y0 + 1 < R.h;
        const float fy = (float)(yr - y0);
        for (int x = 0; x < L.w; x += step) {
            ++all;
            if (!rowIn) continue;
            const double xr = (x - tx) / s;
            const int x0 = (int)std::floor(xr);
            if (x0 < 0 || x0 + 1 >= R.w) continue;
            const float fx = (float)(xr - x0);
            const float* r0 = R.v.data() + (size_t)y0 * R.w + x0;
            const float* r1 = r0 + R.w;
            const float b = (r0[0] * (1 - fx) + r0[1] * fx) * (1 - fy) + (r1[0] * (1 - fx) + r1[1] * fx) * fy;
            const float a = L.v[(size_t)y * L.w + x];
            sab += a * b; saa += a * a; sbb += b * b;
            ++n;
        }
    }
    if (n * 4 < all || saa <= 0 || sbb <= 0) return -1;
    return sab / std::sqrt(saa * sbb);
}
}  // namespace

bool estimatePlacement(const QImage& left, const QImage& right, double& scale, QPointF& pos, double* match, bool fixedScale) {
    if (left.isNull() || right.isNull() || left.width() < 64 || left.height() < 64 || right.width() < 64 || right.height() < 64) return false;
    // both at the same reduction (so the scale between them is the images' own): the finest level about 3000 px of
    // the left image's width, halved down to about 100
    double f = std::max(1.0, left.width() / 3072.0);
    auto planeOf = [f](const QImage& img) {
        Plane p;
        p.w = std::max(8, (int)std::lround(img.width() / f)); p.h = std::max(8, (int)std::lround(img.height() / f));
        const QImage g = img.scaled(p.w, p.h, Qt::IgnoreAspectRatio, Qt::SmoothTransformation).convertToFormat(QImage::Format_Grayscale8);
        p.v.resize((size_t)p.w * p.h);
        for (int y = 0; y < p.h; ++y) {
            const uchar* s = g.constScanLine(y);
            for (int x = 0; x < p.w; ++x) p.v[(size_t)y * p.w + x] = s[x] / 255.f;
        }
        return p;
    };
    std::vector<Plane> L{planeOf(left)}, R{planeOf(right)};
    while (L.back().w > 140 && R.back().w > 32 && R.back().h > 32) { L.push_back(halved(L.back())); R.push_back(halved(R.back())); }
    for (auto& p : L) p = structure(p);
    for (auto& p : R) p = structure(p);

    // coarsest level: every scale within a fifth of the starting one, every shift of the centres within 30 % of the image
    const int top = (int)L.size() - 1;
    const double s0 = fixedScale || std::fabs((double)left.width() / right.width() - 1.0) < 0.25 ? 1.0 : (double)left.width() / right.width();
    struct Fit { double s = 1, tx = 0, ty = 0, m = -2; };
    const int ns = fixedScale ? 1 : 49;   // s0 * 1.0075^(-24 .. 24): 0.84 .. 1.20
    std::vector<Fit> best(ns);
    {
        const Plane& l = L[top];
        const Plane& r = R[top];
        std::vector<std::thread> th;
        const int nt = std::max(1u, std::min(8u, std::thread::hardware_concurrency()));
        for (int q = 0; q < nt; ++q)
            th.emplace_back([&, q] {
                for (int k = q; k < ns; k += nt) {
                    const double s = s0 * std::pow(1.0075, k - ns / 2);
                    const double cx = (l.w - s * r.w) / 2, cy = (l.h - s * r.h) / 2;   // the centres on each other
                    const int rx = (int)std::ceil(0.3 * l.w), ry = (int)std::ceil(0.3 * l.h);
                    Fit b;
                    for (int dy = -ry; dy <= ry; ++dy)
                        for (int dx = -rx; dx <= rx; ++dx) {
                            const double m = placedMatch(l, r, s, cx + dx, cy + dy, 1);
                            if (m > b.m) b = {s, cx + dx, cy + dy, m};
                        }
                    best[k] = b;
                }
            });
        for (auto& t : th) t.join();
    }
    Fit fit;
    for (const Fit& b : best) if (b.m > fit.m) fit = b;
    if (fit.m < 0.05) return false;
    // finer levels: the place doubles, the search narrows
    double ds = 0.00375;
    for (int lv = top - 1; lv >= 0; --lv, ds /= 2) {
        const Plane& l = L[lv];
        const Plane& r = R[lv];
        const int step = std::max(1, (int)std::sqrt((double)l.w * l.h / 250000.0));
        Fit b;
        const double tx0 = fit.tx * 2, ty0 = fit.ty * 2;
        for (int j = fixedScale ? 0 : -2; j <= (fixedScale ? 0 : 2); ++j) {
            const double s = fit.s * (1 + j * ds);
            // (a change of scale moves the far corner: the shift is searched about the image's centre)
            const double ax = tx0 + (fit.s - s) * r.w / 2, ay = ty0 + (fit.s - s) * r.h / 2;
            for (int dy = -2; dy <= 2; ++dy)
                for (int dx = -2; dx <= 2; ++dx) {
                    const double m = placedMatch(l, r, s, ax + dx, ay + dy, step);
                    if (m > b.m) b = {s, ax + dx, ay + dy, m};
                }
        }
        if (b.m < -1) return false;
        fit = b;
    }
    // between the pixels of the finest level: the peak of a parabola through the match at -1, 0, +1
    {
        const Plane& l = L[0];
        const Plane& r = R[0];
        const int step = std::max(1, (int)std::sqrt((double)l.w * l.h / 250000.0));
        auto peak = [](double a, double b, double c) { const double d = a - 2 * b + c; return d < 0 ? std::clamp(0.5 * (a - c) / d, -0.5, 0.5) : 0.0; };
        const double mx0 = placedMatch(l, r, fit.s, fit.tx - 1, fit.ty, step), mx1 = placedMatch(l, r, fit.s, fit.tx + 1, fit.ty, step);
        const double my0 = placedMatch(l, r, fit.s, fit.tx, fit.ty - 1, step), my1 = placedMatch(l, r, fit.s, fit.tx, fit.ty + 1, step);
        if (mx0 > -1 && mx1 > -1) fit.tx += peak(mx0, fit.m, mx1);
        if (my0 > -1 && my1 > -1) fit.ty += peak(my0, fit.m, my1);
    }
    if (fit.m < 0.15) return false;   // nothing in common worth the name
    scale = fit.s;
    pos = QPointF(fit.tx * f, fit.ty * f);
    if (match) *match = fit.m;
    return true;
}

void CompareWindow::applyAlign() {
    auto plain = [this] { if (view_[1]->placed()) { view_[1]->clearPlacement(); if (!view_[0]->image().isNull()) { syncing_ = true; view_[1]->syncFrom(view_[0]); syncing_ = false; } } };
    if (!alignAct_->isChecked()) { plain(); return; }
    if (rt_.on) {   // retouching paints pixel on pixel: both images as they are
        plain();
        QSignalBlocker b(alignAct_);
        alignAct_->setChecked(false);
        refuse("Align is off while retouching (the brush paints pixel on pixel). A frame of this stack is lined up when picked on the right.", 9000);
        return;
    }
    if (sel_[0] < 0 || sel_[1] < 0 || entries_[sel_[0]].image.isNull() || entries_[sel_[1]].image.isNull()) { plain(); return; }
    const QImage& l = entries_[sel_[0]].image;
    const QImage& r = entries_[sel_[1]].image;
    if (align_.left != l.cacheKey() || align_.right != r.cacheKey()) {
        QApplication::setOverrideCursor(Qt::WaitCursor);
        align_.left = l.cacheKey(); align_.right = r.cacheKey();
        align_.ok = sel_[0] == sel_[1] ? false : estimatePlacement(l, r, align_.scale, align_.pos, &align_.match);
        QApplication::restoreOverrideCursor();
    }
    if (sel_[0] == sel_[1]) { plain(); return; }
    if (!align_.ok) {
        plain();
        refuse(QString("Align: %1 and %2 could not be lined up (they do not look like the same subject).").arg(entries_[sel_[0]].name, entries_[sel_[1]].name), 9000);
        return;
    }
    const bool same = l.size() == r.size() && std::fabs(align_.scale - 1) < 2e-4 && std::hypot(align_.pos.x(), align_.pos.y()) < 0.3;
    if (same) plain();
    else {
        view_[1]->setPlacement(align_.pos, align_.scale, l.size());
        syncing_ = true; view_[1]->syncFrom(view_[0]); syncing_ = false;
    }
    statusBar()->showMessage(same ? QString("Align: %1 already lies on %2.").arg(entries_[sel_[1]].name, entries_[sel_[0]].name)
                                  : QString("Align: %1 scaled by %2 and moved %3, %4 px onto %5 (match %6).").arg(entries_[sel_[1]].name).arg(align_.scale, 0, 'f', 4)
                                        .arg(align_.pos.x(), 0, 'f', 1).arg(align_.pos.y(), 0, 'f', 1).arg(entries_[sel_[0]].name).arg(align_.match, 0, 'f', 2), 12000);
}

// Two programs' stacks are not one picture moved and scaled: a fit over the whole image can be several pixels off in a
// corner. Here the shift is found again on what the panes show, at the scale already found, and the right image moved
// by the difference: exact where one is looking.
void CompareWindow::alignHere() {
    if (!alignAct_->isChecked()) alignAct_->setChecked(true);   // (-> applyAlign: the whole-image fit first)
    if (!alignAct_->isChecked() || !align_.ok || sel_[0] < 0 || sel_[1] < 0 || sel_[0] == sel_[1]) return;
    const QImage& l = entries_[sel_[0]].image;
    const QImage& r = entries_[sel_[1]].image;
    if (l.isNull() || r.isNull()) return;
    const QRect vis = view_[0]->visibleSceneRect().toAlignedRect().intersected(l.rect());
    // the same part of the right image, in its own pixels, and that at the left image's scale
    const QRectF inRight((vis.x() - align_.pos.x()) / align_.scale, (vis.y() - align_.pos.y()) / align_.scale, vis.width() / align_.scale, vis.height() / align_.scale);
    const QRect rr = inRight.toAlignedRect().intersected(r.rect());
    if (vis.width() < 96 || vis.height() < 96 || rr.width() < 96 || rr.height() < 96) { refuse("Align here: too little of both images is on screen.", 6000); return; }
    QApplication::setOverrideCursor(Qt::WaitCursor);
    const QImage a = l.copy(vis);
    const QImage b = r.copy(rr).scaled(std::max(1, (int)std::lround(rr.width() * align_.scale)), std::max(1, (int)std::lround(rr.height() * align_.scale)), Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
    double s = 1, m = 0;
    QPointF found;
    const bool ok = estimatePlacement(a, b, s, found, &m, true);
    QApplication::restoreOverrideCursor();
    if (!ok) { refuse("Align here: the two do not line up on this part (too little structure on screen?).", 8000); return; }
    // where b's corner was expected in a, from the placement in use
    const QPointF expected(rr.x() * align_.scale + align_.pos.x() - vis.x(), rr.y() * align_.scale + align_.pos.y() - vis.y());
    const QPointF d = found - expected;
    align_.pos += d;
    view_[1]->setPlacement(align_.pos, align_.scale, l.size());
    syncing_ = true; view_[1]->syncFrom(view_[0]); syncing_ = false;
    statusBar()->showMessage(QString("Align here: the right image moved by %1, %2 px for this part (match %3). Align off and on again goes back to the whole-image fit.")
                                 .arg(d.x(), 0, 'f', 1).arg(d.y(), 0, 'f', 1).arg(m, 0, 'f', 2), 12000);
}

void CompareWindow::select(int pane, const QString& name) {
    int i = indexOf(name);
    if (i >= 0) showIn(pane, i);
}

void CompareWindow::refuse(const QString& why, int ms) {
    statusBar()->showMessage(why, ms);
    if (isVisible() && isActiveWindow() && !testAttribute(Qt::WA_DontShowOnScreen))
        QToolTip::showText(QCursor::pos(), QString("<qt>%1</qt>").arg(why.toHtmlEscaped()), this, QRect(), std::min(ms, 9000));   // (rich text wraps)
}

// The image the user picked for the left pane (its picker, a double-click in the list). While retouching, the left
// pane is the result: the retouch moves to the image picked. Unsaved strokes and regions are asked about first.
void CompareWindow::pickLeft(int index) {
    if (index < 0 || index >= entries_.size()) return;
    if (!rt_.on) { showIn(0, index); return; }
    auto back = [this] { QSignalBlocker b(pick_[0]); pick_[0]->setCurrentIndex(sel_[0]); };
    if (index == rt_.target) { back(); return; }
    if (refining_) { back(); refuse("Refine is running: pick another image for the left pane when it has finished."); return; }
    if (rt_.dirty || !regions_.empty()) {
        const auto r = QMessageBox::question(this, "xLamina", QString("Retouch %1 instead?\n\nThe unsaved strokes and regions of %2 are dropped. Save result… keeps them: press No, save, and pick again.")
                                                                  .arg(entries_[index].name, entries_[rt_.target].name), QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
        if (r != QMessageBox::Yes) { back(); return; }
    }
    const bool clone = cloneAct_->isChecked();
    rt_.dirty = false;                 // (asked above)
    retouchAct_->setChecked(false);    // -> retouchToggle
    if (rt_.on) { back(); return; }
    showIn(0, index);
    if (entries_[index].image.isNull()) { statusBar()->showMessage(QString("%1 is being read: press Retouch (R) when it is on show.").arg(entries_[index].name), 9000); return; }
    retouchAct_->setChecked(true);
    if (rt_.on && clone) cloneAct_->setChecked(true);
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
    if (pane == 0) nameRegionRow();   // (before the image is there: a file still being read is not an output either)
    refresh(pane);
}

void CompareWindow::refresh(int pane) {
    int i = sel_[pane];
    if (i < 0) return;
    if (!(rt_.on && pane == 0)) {   // one of the render's frames: shown, and painted from, lined up with the render
        const int j = adoptFrame(i);
        if (j >= 0 && j != i) { showIn(pane, j); return; }
    }
    if (entries_[i].owner && entries_[i].image.isNull() && !loadFrame(i)) {   // (it said why)
        view_[pane]->setLabel(entries_[i].name + " — not lined up yet");
        return;
    }
    Entry& e = entries_[i];
    if (e.image.isNull()) {
        view_[pane]->setLabel(e.name + " — loading…");
        requestLoad(i);
        return;
    }
    if (e.fileBacked) { recent_.removeAll(i); recent_.push_back(i); }
    syncing_ = true;   // a new image must not re-fit the other pane: it takes the other pane's view instead
    view_[pane]->setImage(e.image, true);
    for (int other = 0; other < nPanes_; ++other)
        if (other != pane && !view_[other]->image().isNull()) { view_[pane]->syncFrom(view_[other]); break; }
    syncing_ = false;
    view_[pane]->setLabel(rt_.on && pane == active_ && pane > 0 ? e.name + "  (source)" : e.name);
    if (pane == 0) nameRegionRow();
    applyAlign();   // (a new image in either pane: the right one is lined up with the left one again, when Align is on)
    for (int p = 0; p < kPanes; ++p) applyScale(p);
    updateDepthTools();
    if (frameHereAct_->isChecked()) for (ImageView* v : view_) v->setPrecise(true);
    showNames();
    trimCache();
    updateOverlay();
    if (heldFrame_ >= 0 && depthShown()) holdLayer(heldFrame_);   // the pane's label names the held layer again
    // the image to paint from; one that does not fit the result stops the brush (it used to go on painting from the
    // image before it, whatever the pane showed)
    if (rt_.on && pane == active_) rt_.blocked = i != rt_.source && !retouchSetSource(i);
    showBrushBlocked();
}

// The brush circle is red while the brush would paint nothing: the pane painted from shows an image that is not the
// result's size. (Clone copies within the result itself and needs no other image.)
void CompareWindow::testCursor(double fx, double fy) {
    QWidget* vp = view_[0]->viewport();
    const QPointF p(vp->width() * fx, vp->height() * fy);
    QMouseEvent e(QEvent::MouseMove, p, vp->mapToGlobal(p), Qt::NoButton, Qt::NoButton, Qt::NoModifier);
    QCoreApplication::sendEvent(vp, &e);
}

void CompareWindow::showBrushBlocked() {
    if (view_[0]) view_[0]->setBrushBlocked(rt_.on && rt_.blocked && !(cloneAct_ && cloneAct_->isChecked()));
}

// |luma(left) - luma(right)| as an orange tint: transparent up to the chosen level, opaque at four times it.
// Both panes get the same overlay, so the highlight sits on the pixels in question whichever image is under it.
void CompareWindow::updateOverlay() {
    if (markLayer()) return;   // a held layer is what the panes show over their images
    if (diffAct_->isChecked() && view_[1]->placed()) {   // lined up by Align: the two are not pixel on pixel
        view_[0]->setOverlay(QImage());
        view_[1]->setOverlay(QImage());
        overlayFor_[0] = overlayFor_[1] = -1;
        refuse("Difference compares pixel with pixel: switch Align off for it.", 6000);
        return;
    }
    if (!diffAct_->isChecked() && depthOverlay()) { overlayFor_[0] = overlayFor_[1] = -1; return; }
    for (int p = 0; p < kPanes; ++p) { depthOv_[p] = false; if (p >= 2) view_[p]->setOverlay(QImage()); }   // from here on the overlays are the difference's (the first two panes), or none
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
        refuse("Difference: the two images are not the same size.", 5000);
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

// ----------------------------------------------------------------------------- the depth map

void CompareWindow::setDepth(std::shared_ptr<JobResultData> r, const DepthStyle& st) {
    if (r && (r->rc != 0 || r->result.depth.size() != (size_t)r->result.w * r->result.h || r->result.depth.empty())) r.reset();
    const bool same = r && r == depth_ && st.sameImage(depthStyle_) && indexOf("depth") >= 0;
    depth_ = r;
    const int keep = depthStyle_.overlay;   // the overlay belongs to this window
    depthStyle_ = st;
    depthStyle_.overlay = keep;
    depthIdx_.clear();
    depthHas_.clear();
    heldFrame_ = -1;
    updateDepthTools();
    if (!depth_) { updateOverlay(); return; }
    depthBar_->setFrames((int)depth_->align.paths.size());
    depthBar_->setStyle(depthStyle_);
    if (!same) recolourDepth();
}

void CompareWindow::setDepthStyle(const DepthStyle& st) {
    if (!depth_) return;
    const int keep = depthStyle_.overlay;
    DepthStyle s = st;
    s.overlay = keep;
    if (s == depthStyle_) return;
    const bool samePicture = s.sameImage(depthStyle_);
    depthStyle_ = s;
    depthBar_->setStyle(depthStyle_);
    if (!samePicture) recolourDepth();
}

// the entry of that name is replaced (addEntry), and the panes showing it are refreshed
void CompareWindow::recolourDepth() {
    if (!depth_) return;
    const auto& R = depth_->result;
    for (bool& b : depthOv_) b = false;
    addImage("depth", depthImage(R.depth.data(), R.w, R.h, (int)depth_->align.paths.size(), depthStyle_), "this render's depth map");
    updateOverlay();
}

bool CompareWindow::depthShown() const {
    const int di = depth_ ? indexOf("depth") : -1;
    return di >= 0 && onShow(di);
}

// The Depth toolbar, the crosshair, the note beside the cursor and the arrow keys' stepping: with the depth map in
// a pane. With two other images on show they were controls for something that is not there.
void CompareWindow::updateDepthTools() {
    const bool on = depthShown();
    depthTb_->setVisible(on);
    for (ImageView* v : view_) {
        v->setPrecise(on);
        v->setStepKeys(on);
        if (!on) v->setCursorNote({});
    }
    if (!on && heldFrame_ >= 0) holdLayer(-1);
}

void CompareWindow::holdLayer(int k) {
    if (k >= 0 && !depthShown()) return;
    heldFrame_ = k;
    const int di = indexOf("depth");
    if (k < 0) {
        layerMarked_ = false;
        for (int p = 0; p < kPanes; ++p) { view_[p]->setOverlay(QImage()); depthOv_[p] = false; if (sel_[p] >= 0) view_[p]->setLabel(entries_[sel_[p]].name); }
        overlayFor_[0] = overlayFor_[1] = -1;
        updateOverlay();   // the difference or the depth overlay, when one is on
        statusBar()->showMessage("Layer let go.", 4000);
        return;
    }
    const auto& R = depth_->result;
    const int K = (int)depth_->align.paths.size();
    if (depthIdx_.size() != R.depth.size()) {
        depthIdx_.resize(R.depth.size());
        depthHas_.assign(K, 0);
        for (size_t i = 0; i < R.depth.size(); ++i) { depthIdx_[i] = (uint16_t)std::clamp((int)std::lround(R.depth[i]), 0, K - 1); depthHas_[depthIdx_[i]] = 1; }
    }
    markLayer();
    const QString name = QString::fromStdWString(depth_->align.paths[k].filename().wstring());
    size_t px = 0;
    for (uint16_t v : depthIdx_) px += v == (uint16_t)k;
    for (int p = 0; p < kPanes; ++p) if (sel_[p] == di) view_[p]->setLabel(QString("depth  ·  frame %1 of %2  ·  %3").arg(k + 1).arg(K).arg(name));
    statusBar()->showMessage(QString("Layer held: frame %1 of %2 (%3), %4 % of the image. Arrow keys or PgUp / PgDn step through the layers; Esc or a click on it lets go.")
                                 .arg(k + 1).arg(K).arg(name).arg(100.0 * px / std::max<size_t>(1, depthIdx_.size()), 0, 'f', 2));
}

void CompareWindow::stepLayer(int delta) {
    if (!depthShown() || rt_.on || delta == 0) return;
    const int K = (int)depth_->align.paths.size();
    if (heldFrame_ < 0) { holdLayer(delta > 0 ? 0 : K - 1); if (heldFrame_ >= 0 && depthHas_[heldFrame_]) return; }
    int n = heldFrame_;
    for (int i = heldFrame_ + (delta > 0 ? 1 : -1); i >= 0 && i < K; i += delta > 0 ? 1 : -1) if (depthHas_[i]) { n = i; break; }
    if (n == heldFrame_) { statusBar()->showMessage(delta > 0 ? "This is the last layer the depth map uses." : "This is the first layer the depth map uses.", 4000); return; }
    holdLayer(n);
}

// The held layer over every pane that shows an image on the render's grid, for what each pane shows, at the
// screen's resolution (as the main window draws it).
bool CompareWindow::markLayer() {
    if (heldFrame_ < 0 || !depthShown() || depthIdx_.empty()) return false;
    const auto& R = depth_->result;
    for (int p = 0; p < kPanes; ++p) {
        depthOv_[p] = false;
        if (sel_[p] < 0 || entries_[sel_[p]].image.size() != QSize(R.w, R.h)) { view_[p]->setOverlay(QImage()); continue; }
        const QRect vis = view_[p]->visibleSceneRect().toAlignedRect().intersected(QRect(0, 0, R.w, R.h));
        if (vis.isEmpty()) continue;
        const int s = std::max(1, (int)std::floor(1.0 / std::max(view_[p]->zoom(), 1e-3)));
        view_[p]->setOverlayAt(depthLayerMark(depthIdx_.data(), R.w, heldFrame_, vis, s, depthStyle_.palette), vis.topLeft(), s);
    }
    overlayFor_[0] = overlayFor_[1] = -1;
    layerMarked_ = true;
    return true;
}

bool CompareWindow::depthOverlay() {
    const int di = depth_ && depthStyle_.overlay > 0 && !rt_.on && depthShown() ? indexOf("depth") : -1;
    if (di < 0 || entries_[di].image.isNull()) return false;
    const QImage& map = entries_[di].image;
    const double op = depthStyle_.overlay / 100.0;
    for (int p = 0; p < kPanes; ++p) {
        const bool want = sel_[p] >= 0 && sel_[p] != di && entries_[sel_[p]].image.size() == map.size();
        if (!want) { view_[p]->setOverlay(QImage()); depthOv_[p] = false; }
        else if (depthOv_[p]) view_[p]->setOverlayOpacity(op);
        else { view_[p]->setOverlay(map, op); depthOv_[p] = true; }
    }
    return true;
}

// the frame the depth map chose under the cursor, for any image on the render's pixel grid
void CompareWindow::depthHover(int pane, int x, int y) {
    if (!depthShown() || sel_[pane] < 0) { view_[pane]->setCursorNote({}); return; }
    const auto& R = depth_->result;
    if (x < 0 || y < 0 || x >= R.w || y >= R.h || entries_[sel_[pane]].image.size() != QSize(R.w, R.h)) { view_[pane]->setCursorNote({}); return; }
    // "within" a patch of about 8 screen pixels around the cursor
    const int radius = depthStyle_.radiusAt(view_[pane]->zoom());
    const QString text = depthFrameText(R.depth.data(), R.w, R.h, depth_->align.paths, x, y, radius);
    statusBar()->showMessage(QString("x %1, y %2: %3").arg(x).arg(y).arg(QString(text).replace('\n', "; ")));
    view_[pane]->setCursorNote(depthStyle_.popup ? text : QString());   // drawn by the view beside the cursor
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
    if (sel_[pane] >= 0 && render_ && render_->stack && entries_[sel_[pane]].owner == render_.get()) {
        // a frame: the one before or after it in the stack, wherever its row is in the list
        const int K = (int)render_->stack->po.paths.size(), k = entries_[sel_[pane]].frame + (delta > 0 ? 1 : -1);
        if (k < 0 || k >= K) { statusBar()->showMessage(k < 0 ? "This is the first frame of the stack." : "This is the last frame of the stack.", 4000); return; }
        const int before = entries_.size();
        const int j = ensureFrameEntry(k);
        if (entries_.size() != before) rebuildList();
        if (j < 0) return;
        showIn(pane, j);
        list_->scrollToItem(list_->item(j, 0));
        list_->selectRow(j);
        return;
    }
    const int n = entries_.size(), d = delta > 0 ? 1 : -1;
    int i = sel_[pane] < 0 ? 0 : ((sel_[pane] + d) % n + n) % n;
    for (int tries = 0; tries < n && list_->isRowHidden(i); ++tries) i = ((i + d) % n + n) % n;   // (the frames, when they are hidden)
    showIn(pane, i);
}

void CompareWindow::keyPressEvent(QKeyEvent* e) {
    if (e->key() == Qt::Key_Space) {
        if (rt_.on) {   // peek at the source in the result pane while Space is held
            if (!e->isAutoRepeat() && !rt_.peek) {
                rt_.peek = true;
                if (!regions_.empty()) {   // with regions marked, the image as it was before them (and before the strokes)
                    view_[0]->setLive(rt_.baseDisp.isNull() ? &rt_.tgtDisp : &rt_.baseDisp, nullptr);
                    view_[0]->setLabel(entries_[rt_.target].name + "  (before)");
                } else {
                    view_[0]->setLive(&rt_.srcDisp, nullptr);
                    view_[0]->setLabel(entries_[rt_.source].name + "  (source)");
                }
            }
        } else flip();
        return;
    }
    QMainWindow::keyPressEvent(e);
}

void CompareWindow::keyReleaseEvent(QKeyEvent* e) {
    if (e->key() == Qt::Key_Space && rt_.on && rt_.peek && !e->isAutoRepeat()) {
        rt_.peek = false;
        view_[0]->setLive(&rt_.resDisp, &rt_.mask);
        if (rt_.hasClone()) view_[0]->setSecondMask(&rt_.ca);
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
    // not a frame the engine reads as RGB (a PNG, a grey image): Windows decodes it, not a decoder shipped with the program
    QString why;
    QImage img = readImageViaWindows(e.path, &why);
    if (img.isNull()) statusBar()->showMessage(QString("Could not read %1: %2").arg(e.name, why), 8000);
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
            if (!e.owner) e.image = image;   // (a file that became a frame's row meanwhile shows the lined-up frame)
        } else {
            for (int p = 0; p < nPanes_; ++p) if (sel_[p] == index) view_[p]->setLabel(e.name + " — could not be read");
        }
        for (int p = 0; p < nPanes_; ++p) if (sel_[p] == index && !e.image.isNull()) refresh(p);
    }
    if (loading_ == index) loading_ = -1;
    startNextLoad();
}

// keep the last few decoded files; the two on screen and the retouch's images always stay
void CompareWindow::trimCache() {
    while (recent_.size() > kKeepDecoded) {
        int victim = -1;
        for (int i : recent_) if (!onShow(i) && i != rt_.target && i != rt_.source) { victim = i; break; }
        if (victim < 0) return;
        recent_.removeAll(victim);
        entries_[victim].image = QImage();
    }
}

// ----------------------------------------------------------------------------- retouch

// the 16-bit RGB of an entry: its TIFF when it has one (and the profile for the output), else its 8-bit image x 257
bool CompareWindow::load16(const Entry& e, std::vector<uint16_t>& out, int& w, int& h, std::vector<uint8_t>* icc) {
    if (e.px16 && !e.image.isNull() && e.px16->size() == (size_t)e.image.width() * e.image.height() * 3) {   // a frame lined up with the render
        w = e.image.width(); h = e.image.height();
        out = *e.px16;
        if (icc && render_) *icc = render_->result.icc;
        return true;
    }
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

void CompareWindow::setRetouch(bool on) {
    if (retouchAct_ && retouchAct_->isChecked() != on) retouchAct_->setChecked(on);   // toggled -> retouchToggle
}

void CompareWindow::retouchToggle(bool on) {
    if (on) {
        // the source is the pane the list fills (the outlined one); when that one cannot serve, the right pane
        if (active_ != 1 && sel_[0] >= 0 && (sel_[active_] < 0 || entries_[sel_[active_]].image.isNull() || entries_[sel_[active_]].image.size() != entries_[sel_[0]].image.size())) active_ = 1;
        if (sel_[0] < 0 || sel_[active_] < 0 || entries_[sel_[0]].image.isNull() || entries_[sel_[active_]].image.isNull()) {
            refuse("Retouch needs an image in each pane: the left is the result, the right the source.", 5000);
            retouchAct_->blockSignals(true); retouchAct_->setChecked(false); retouchAct_->blockSignals(false);
            return;
        }
        if (measureAct_->isChecked()) measureAct_->setChecked(false);   // the left button paints from here on
        if (heldFrame_ >= 0) holdLayer(-1);                             // (a held depth layer's marks lay over the pane painted on)
        if (alignAct_->isChecked()) alignAct_->setChecked(false);       // painting is pixel on pixel: both images as they are
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
        if (ok) ok = retouchSetSource(sel_[active_]);
        QApplication::restoreOverrideCursor();
        if (!ok) {
            refuse("Retouch: the two images must be the same size, and readable at 16 bits.", 6000);
            rt_ = Retouch{};
            retouchAct_->blockSignals(true); retouchAct_->setChecked(false); retouchAct_->blockSignals(false);
            return;
        }
        rt_.on = true;
        markActive();
        rt_.res16 = rt_.tgt16;
        rt_.resDisp = rt_.tgtDisp.copy();
        rt_.mask.assign((size_t)w * h, 0);
        view_[0]->setLive(&rt_.resDisp, &rt_.mask);
        view_[0]->setMaskShown(maskAct_->isChecked());
        view_[0]->setBrush(true, brushSize_->value(), brushHard_->value());
        view_[0]->setLabel(entries_[rt_.target].name + "  (retouch)");
        flipAct_->setEnabled(false);
        for (QAction* a : {maskAct_, resetAct_, saveAct_, cloneAct_, matchAct_}) a->setEnabled(true);
        clone_ = Clone{};
        view_[0]->setAltPick(cloneAct_->isChecked());
        showCloneMark();
        undoAct_->setEnabled(false);
        brushSize_->setEnabled(true);
        brushHard_->setEnabled(true);
        retouchStatus();
    } else {
        if (rt_.dirty) {
            auto r = QMessageBox::question(this, "xLamina", "Leave Retouch and drop the unsaved strokes?", QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
            if (r != QMessageBox::Yes) { retouchAct_->blockSignals(true); retouchAct_->setChecked(true); retouchAct_->blockSignals(false); return; }
        }
        // the regions were laid into this result: they go with it
        if (refining_) emit refineCancelRequested();
        for (QAction* a : {rectAct_, lassoAct_}) { QSignalBlocker b(a); a->setChecked(false); }
        for (ImageView* v : view_) v->setRegionTool(0);
        regions_.clear();
        jobMap_.clear();
        showRegions();
        view_[0]->setLive(nullptr, nullptr);
        view_[0]->setBrush(false, brushSize_->value(), brushHard_->value());
        rt_ = Retouch{};
        markActive();
        flipAct_->setEnabled(true);
        for (QAction* a : {maskAct_, undoAct_, resetAct_, saveAct_, cloneAct_, matchAct_}) a->setEnabled(false);
        { QSignalBlocker b(cloneAct_); cloneAct_->setChecked(false); }
        clone_ = Clone{};
        view_[0]->setAltPick(false);
        view_[0]->setCloneMark(0, QPointF());
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
        if (rt_.on) refuse(QString("%1 is not the result's size: the brush paints nothing from it. Pick an output, or one of this stack's frames.").arg(entries_[index].name), 9000);
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
    if (!rt_.undo.empty() || rt_.dropped) rt_.baked = true;   // unsaved work that Undo and Reset no longer reach
    rt_.tgt16 = rt_.res16;
    rt_.tgtDisp = rt_.resDisp.copy();
    // the regions in the result are part of the target from here on, as the strokes are: they leave the list (one
    // that is switched off, or not rendered yet, stays, without its patch)
    if (!rt_.base16.empty() || !regions_.empty()) {
        bool kept = false;
        for (auto& g : regions_) kept = kept || (g.hasPatch() && g.on);
        if (kept) rt_.baked = true;
        regions_.erase(std::remove_if(regions_.begin(), regions_.end(), [](const Region& g) { return g.hasPatch() && g.on; }), regions_.end());
        for (auto& g : regions_) { g.alpha.clear(); g.px16.clear(); g.disp = QImage(); g.rendered = -1; if (g.state != 1) g.state = 0; }
        rt_.base16.clear();
        rt_.baseDisp = QImage();
        showRegions();
    }
    std::fill(rt_.mask.begin(), rt_.mask.end(), 0);
    std::fill(rt_.ca.begin(), rt_.ca.end(), 0);   // (the copies are in the result, which is the target now)
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
            if (rt_.erasing) { uint8_t& s = rt_.sa[(size_t)y * rt_.w + x]; if (v > s) s = (uint8_t)v; }
        }
    }
    composite(r);
    view_[0]->patch(r);
}

// result = target + mask * (source - target), then the clone layer over it by its share; 16-bit and the 8-bit
// display, inside r
void CompareWindow::composite(const QRect& r) {
    const bool lay = rt_.hasClone();
    for (int y = r.top(); y <= r.bottom(); ++y) {
        const uint8_t* ca = lay ? rt_.ca.data() + (size_t)y * rt_.w : nullptr;
        const uint8_t* sa = lay ? rt_.sa.data() + (size_t)y * rt_.w : nullptr;
        const uint16_t* cl = lay ? rt_.cl16.data() + (size_t)y * rt_.w * 3 : nullptr;
        const QRgb* cd = lay ? reinterpret_cast<const QRgb*>(rt_.clDisp.constScanLine(y)) : nullptr;
        const uint8_t* m = rt_.mask.data() + (size_t)y * rt_.w;
        uint16_t* res = rt_.res16.data() + (size_t)y * rt_.w * 3;
        const uint16_t* tg = rt_.tgt16.data() + (size_t)y * rt_.w * 3;
        const uint16_t* sr = rt_.src16.data() + (size_t)y * rt_.w * 3;
        QRgb* rd = reinterpret_cast<QRgb*>(rt_.resDisp.scanLine(y));
        const QRgb* td = reinterpret_cast<const QRgb*>(rt_.tgtDisp.constScanLine(y));
        const QRgb* sd = reinterpret_cast<const QRgb*>(rt_.srcDisp.constScanLine(y));
        for (int x = r.left(); x <= r.right(); ++x) {
            int a = m[x];
            if (a == 0) { for (int k = 0; k < 3; ++k) res[x * 3 + k] = tg[x * 3 + k]; rd[x] = td[x]; }
            else if (a == 255) { for (int k = 0; k < 3; ++k) res[x * 3 + k] = sr[x * 3 + k]; rd[x] = sd[x]; }
            else {
                for (int k = 0; k < 3; ++k) res[x * 3 + k] = (uint16_t)((tg[x * 3 + k] * (255 - a) + sr[x * 3 + k] * a + 127) / 255);
                int rr = (qRed(td[x]) * (255 - a) + qRed(sd[x]) * a + 127) / 255;
                int gg = (qGreen(td[x]) * (255 - a) + qGreen(sd[x]) * a + 127) / 255;
                int bb = (qBlue(td[x]) * (255 - a) + qBlue(sd[x]) * a + 127) / 255;
                rd[x] = qRgb(rr, gg, bb);
            }
            if (!lay || !ca[x]) continue;
            int c = ca[x];
            if (rt_.erasing) c = (c * (255 - sa[x]) + 127) / 255;   // a paint stroke under way: the copy gives way under it
            if (c == 0) continue;
            for (int k = 0; k < 3; ++k) res[x * 3 + k] = (uint16_t)((res[x * 3 + k] * (255 - c) + cl[x * 3 + k] * c + 127) / 255);
            rd[x] = qRgb((qRed(rd[x]) * (255 - c) + qRed(cd[x]) * c + 127) / 255, (qGreen(rd[x]) * (255 - c) + qGreen(cd[x]) * c + 127) / 255,
                         (qBlue(rd[x]) * (255 - c) + qBlue(cd[x]) * c + 127) / 255);
        }
    }
}

// one pixel of the result as composite() shows it, from the layers (the display itself holds a clone stroke's
// preview while one is drawn)
QRgb CompareWindow::dispAt(int x, int y) const {
    const size_t i = (size_t)y * rt_.w + x;
    const int a = rt_.mask[i];
    const QRgb t = reinterpret_cast<const QRgb*>(rt_.tgtDisp.constScanLine(y))[x];
    QRgb v = t;
    if (a) {
        const QRgb s = reinterpret_cast<const QRgb*>(rt_.srcDisp.constScanLine(y))[x];
        v = a == 255 ? s : qRgb((qRed(t) * (255 - a) + qRed(s) * a + 127) / 255, (qGreen(t) * (255 - a) + qGreen(s) * a + 127) / 255, (qBlue(t) * (255 - a) + qBlue(s) * a + 127) / 255);
    }
    if (rt_.hasClone() && rt_.ca[i]) {
        const int c = rt_.ca[i];
        const QRgb d = reinterpret_cast<const QRgb*>(rt_.clDisp.constScanLine(y))[x];
        v = qRgb((qRed(v) * (255 - c) + qRed(d) * c + 127) / 255, (qGreen(v) * (255 - c) + qGreen(d) * c + 127) / 255, (qBlue(v) * (255 - c) + qBlue(d) * c + 127) / 255);
    }
    return v;
}

void CompareWindow::retouchUndo() {
    if (!rt_.on || rt_.undo.empty()) return;
    Stroke s = std::move(rt_.undo.back());
    rt_.undo.pop_back();
    for (int y = 0; y < s.r.height(); ++y) memcpy(rt_.mask.data() + (size_t)(s.r.y() + y) * rt_.w + s.r.x(), s.mask.data() + (size_t)y * s.r.width(), s.r.width());
    const size_t n = (size_t)s.r.width() * s.r.height();
    for (int y = 0; y < s.r.height() && rt_.hasClone(); ++y) {   // the clone layer as it was before the stroke
        const size_t at = (size_t)(s.r.y() + y) * rt_.w + s.r.x(), row = (size_t)y * s.r.width();
        if (s.ca.size() == n) memcpy(rt_.ca.data() + at, s.ca.data() + row, s.r.width());
        if (s.cl.size() == n * 3) memcpy(rt_.cl16.data() + at * 3, s.cl.data() + row * 3, (size_t)s.r.width() * 6);
        if (s.cd.size() == n) memcpy(reinterpret_cast<QRgb*>(rt_.clDisp.scanLine(s.r.y() + y)) + s.r.x(), s.cd.data() + row, (size_t)s.r.width() * sizeof(QRgb));
    }
    composite(s.r);
    view_[0]->patch(s.r);
    undoAct_->setEnabled(!rt_.undo.empty());
    rt_.dirty = !rt_.undo.empty() || rt_.baked || rt_.dropped;   // strokes that can no longer be undone are still unsaved work
    retouchStatus();
}

void CompareWindow::retouchReset() {
    if (!rt_.on) return;
    std::fill(rt_.mask.begin(), rt_.mask.end(), 0);
    std::fill(rt_.ca.begin(), rt_.ca.end(), 0);
    rt_.undo.clear();
    rt_.dropped = false;   // (no stroke is left that Undo could not take back)
    composite(QRect(0, 0, rt_.w, rt_.h));
    view_[0]->patch(QRect(0, 0, rt_.w, rt_.h));
    undoAct_->setEnabled(false);
    rt_.dirty = rt_.baked;   // strokes painted from an earlier source are part of the target now and stay
    retouchStatus();
}

// the field of view written into a file name, as the main window's Save as does: "_1.46mm" before the end
static double fovInName(const QString& name) {
    static const QRegularExpression tag("_(\\d+(?:\\.\\d+)?)mm(?![A-Za-z0-9])");
    double v = 0;
    for (auto it = tag.globalMatch(name); it.hasNext();) v = it.next().captured(1).toDouble();   // (the last one)
    return v;
}

double CompareWindow::fovOf(const Entry& e) const {
    if (const double v = fovInName(e.path.isEmpty() ? e.name : QFileInfo(e.path).completeBaseName()); v > 0) return v;
    if (e.fovMm > 0) return e.fovMm;
    if (scaleUm_ > 0 && !e.image.isNull() && e.image.size() == scaleSize_) return scaleUm_ / 1000.0;
    return 0;
}

// "<the image's name>_retouched", with the field of view after it when it is known and the name does not say it
// already: the main window's Save as ends its names so, and a retouched composite lost it
QString CompareWindow::retouchedName() const {
    if (!rt_.on) return {};
    const Entry& t = entries_[rt_.target];
    QString base = (!t.path.isEmpty() ? QFileInfo(t.path).completeBaseName() : t.name) + "_retouched";
    if (const double fov = fovOf(t); fov > 0 && fovInName(base) <= 0) base += QString("_%1mm").arg(fov, 0, 'f', 2);
    return base + ".tif";
}

void CompareWindow::retouchSave() {
    if (!rt_.on) return;
    const Entry& t = entries_[rt_.target];
    QString dir = !t.path.isEmpty() ? QFileInfo(t.path).absolutePath() : appSettings().value("compare/lastSweep").toString();
    QString def = QDir(dir).filePath(retouchedName());
    QString fn = QFileDialog::getSaveFileName(this, "Save the retouched image", def, "16-bit TIFF (*.tif)");
    if (fn.isEmpty()) return;
    saveResultTo(fn);
}

bool CompareWindow::saveResultTo(const QString& fn) {
    if (!rt_.on) return false;
    QApplication::setOverrideCursor(Qt::WaitCursor);
    std::vector<float> rgb(rt_.res16.size());
    for (size_t i = 0; i < rgb.size(); ++i) rgb[i] = rt_.res16[i] / 65535.0f;
    std::string why;
    bool ok = stackcore::write_tiff_rgb16(fn.toStdString(), rgb, rt_.w, rt_.h, rt_.icc, &why);
    QApplication::restoreOverrideCursor();
    if (!ok) { QMessageBox::warning(this, "xLamina", QString("Could not write %1:\n%2").arg(QDir::toNativeSeparators(fn), QString::fromStdString(why))); return false; }
    rt_.dirty = rt_.baked = rt_.dropped = false;
    statusBar()->showMessage(QString("Saved %1").arg(QDir::toNativeSeparators(fn)), 6000);
    QString name = QFileInfo(fn).fileName();
    Entry e;
    int laid = 0;
    for (const Region& g : regions_) if (g.hasPatch() && g.on) ++laid;
    e.name = name; e.path = fn;
    e.note = rt_.source == rt_.target ? QString("%1 with %2 region%3 rendered again").arg(entries_[rt_.target].name).arg(laid).arg(laid == 1 ? "" : "s")
                                      : QString("retouch of %1 from %2").arg(entries_[rt_.target].name, entries_[rt_.source].name) + (laid ? QString(", %1 region%2 rendered again").arg(laid).arg(laid == 1 ? "" : "s") : QString());
    e.image = rt_.resDisp.copy();
    addEntry(std::move(e));
    return true;
}

// ----------------------------------------------------------------------------- regions rendered again (Refine)

namespace {
// "More detail": the depth map's confidence ramp lowered inside the region, as a share of the render's own. A pixel
// whose focus curve is weaker than the ramp is filled from its surroundings, which is what smooths faint detail away;
// lowered for the whole image the same setting lets defocus blobs through elsewhere.
//
// That holds for a depth volume at 1/2 of the sensor or finer. At 1/4 the ramp changes nothing (the confidence is
// there already): what smooths faint detail away, and blurs the background beside a crystal, is the fine focus's
// short range around a coarse depth that is the neighbour's. There the strengths are 2, 3 and 5 times the render's
// range. (Over a whole image the wide ranges draw arcs around crystals; at 1/2 they draw dark blobs.)
struct DetailStrength { const char* name; double factor; };
const DetailStrength kDetail[3] = {{"low", 0.9}, {"medium", 0.7}, {"high", 0.35}};
const double kRegionMargin = 512;   // sensor px rendered around a region (run_region's margin)

std::vector<std::pair<std::string, std::string>> detailOptions(int strength, const stackcore::StackOptions& so) {
    auto num = [](double v) { return QString::number(v, 'g', 4).toStdString(); };
    strength = std::clamp(strength, 0, 2);
    if (so.po.depth_fraction < 0.4) {
        static const int kTimes[3] = {2, 3, 5};
        const int r = so.refine_r < 0 ? 24 : std::max(12, so.refine_r);   // (-1: from the stack, 12 to 60)
        return {{"--refine-r", num(r * kTimes[strength])}};
    }
    const stackcore::DepthParams dflt;   // the render's own ramp, lowered
    const double lo = so.conf_lo > 0 ? so.conf_lo : dflt.c_lo, hi = so.conf_hi > 0 ? so.conf_hi : dflt.c_hi;
    return {{"--conf-lo", num(lo * kDetail[strength].factor)}, {"--conf-hi", num(hi * kDetail[strength].factor)}};
}

// a box blur of radius r, twice, over a w x h plane: the soft edge of a region's share
void softenPlane(std::vector<float>& m, int w, int h, int r) {
    if (r < 1) return;
    std::vector<float> t(m.size());
    for (int pass = 0; pass < 2; ++pass) {
        for (int y = 0; y < h; ++y) {   // along the rows, the ends repeated
            const float* s = m.data() + (size_t)y * w;
            float* d = t.data() + (size_t)y * w;
            double acc = 0;
            for (int k = -r; k <= r; ++k) acc += s[std::clamp(k, 0, w - 1)];
            for (int x = 0; x < w; ++x) {
                d[x] = (float)(acc / (2 * r + 1));
                acc += s[std::min(w - 1, x + r + 1)] - s[std::max(0, x - r)];
            }
        }
        for (int x = 0; x < w; ++x) {   // down the columns
            double acc = 0;
            for (int k = -r; k <= r; ++k) acc += t[(size_t)std::clamp(k, 0, h - 1) * w + x];
            for (int y = 0; y < h; ++y) {
                m[(size_t)y * w + x] = (float)(acc / (2 * r + 1));
                acc += t[(size_t)std::min(h - 1, y + r + 1) * w + x] - t[(size_t)std::max(0, y - r) * w + x];
            }
        }
    }
}
}  // namespace

bool CompareWindow::onShow(int i) const {
    for (int p = 0; p < nPanes_; ++p) if (sel_[p] == i) return true;
    return false;
}

int CompareWindow::frameEntry(int k) const {
    if (!render_) return -1;
    for (int i = 0; i < entries_.size(); ++i) if (entries_[i].owner == render_.get() && entries_[i].frame == k) return i;
    return -1;
}

int CompareWindow::ensureFrameEntry(int k) {
    if (!render_ || !render_->stack) return -1;
    const auto& paths = render_->stack->po.paths;
    if (k < 0 || k >= (int)paths.size()) return -1;
    if (const int j = frameEntry(k); j >= 0) return j;
    const QString file = QString::fromStdWString(paths[k].wstring());
    Entry e;
    e.name = QFileInfo(file).completeBaseName();
    if (indexOf(e.name) >= 0) e.name = QString("%1 (frame %2)").arg(e.name).arg(k + 1);   // a frame listed twice, or an image of that name
    e.note = QString("frame %1 of %2").arg(k + 1).arg(paths.size());
    e.fileBacked = false;
    e.owner = render_.get();
    e.frame = k;
    e.frameFile = file;
    entries_.push_back(std::move(e));
    return entries_.size() - 1;
}

void CompareWindow::addFrames() {
    if (!render_ || !render_->stack) {
        refuse("Add frames needs this session's render: render the stack, then press Compare in the main window.", 9000);
        return;
    }
    const int K = (int)render_->stack->po.paths.size(), before = entries_.size();
    for (int k = 0; k < K; ++k) ensureFrameEntry(k);
    const int added = entries_.size() - before;
    if (added) rebuildList();
    setFramesHidden(false);   // asked for: on show
    if (added && list_->item(before, 0)) list_->scrollToItem(list_->item(before, 0), QAbstractItemView::PositionAtTop);
    statusBar()->showMessage(added ? QString("%1 frames of the stack are in the list. Click one to see it lined up with the outputs; PgUp / PgDn step through them. Frame here finds the sharpest one at a spot.").arg(added)
                                   : QString("The stack's %1 frames are in the list already.").arg(K), 12000);
}

void CompareWindow::showFrame(const QString& path) {
    int k = -1;
    if (render_ && render_->stack) {
        const QString want = QFileInfo(path).absoluteFilePath();
        const auto& paths = render_->stack->po.paths;
        for (size_t q = 0; q < paths.size() && k < 0; ++q)
            if (QFileInfo(QString::fromStdWString(paths[q].wstring())).absoluteFilePath().compare(want, Qt::CaseInsensitive) == 0) k = (int)q;
    }
    if (k >= 0) {
        const int before = entries_.size();
        const int j = ensureFrameEntry(k);
        if (entries_.size() != before) rebuildList();
        if (j >= 0) { showIn(active_, j); list_->scrollToItem(list_->item(j, 0)); }
        return;
    }
    // not a frame of the render on show (no render yet, or an unticked frame): as it is, like any image
    const QString name = QFileInfo(path).fileName();
    addFile(name, path, QDir::toNativeSeparators(QFileInfo(path).absolutePath()));
    if (const int j = indexOf(name); j >= 0) showIn(active_, j);
}

int CompareWindow::adoptFrame(int i) {
    if (!render_ || !render_->stack || i < 0 || i >= entries_.size()) return -1;
    Entry& e = entries_[i];
    if (e.owner) return e.owner == render_.get() ? i : -1;
    const auto& R = render_->result;
    const auto& paths = render_->stack->po.paths;
    int k = -1;
    if (e.fileBacked && !e.path.isEmpty()) {   // a file opened here
        const QString want = QFileInfo(e.path).absoluteFilePath();
        for (size_t q = 0; q < paths.size() && k < 0; ++q)
            if (QFileInfo(QString::fromStdWString(paths[q].wstring())).absoluteFilePath().compare(want, Qt::CaseInsensitive) == 0) k = (int)q;
    } else if (!e.fileBacked && e.path.isEmpty() && !e.image.isNull() && (e.image.width() != R.w || e.image.height() != R.h)) {
        // a comparison image from the main window: known by its file name only
        for (size_t q = 0; q < paths.size() && k < 0; ++q)
            if (QString::fromStdWString(paths[q].filename().wstring()).compare(e.name, Qt::CaseInsensitive) == 0) k = (int)q;
    }
    if (k < 0) return -1;
    if (const int j = frameEntry(k); j >= 0) return j;
    if (rt_.on && (i == rt_.target || i == rt_.source)) return -1;   // in use as it is
    // the file's row becomes the frame's: a frame as it is on disk is another size than the outputs and lies a little
    // beside them, so there is nothing to compare it with or to paint from
    const QString file = QString::fromStdWString(paths[k].wstring());
    const QString base = QFileInfo(file).completeBaseName();
    if (indexOf(base) < 0) e.name = base;
    e.note = QString("frame %1 of %2").arg(k + 1).arg(paths.size());
    e.path.clear();
    e.fileBacked = false;
    e.loading = false;
    e.image = QImage();
    e.px16.reset();
    e.owner = render_.get();
    e.frame = k;
    e.frameFile = file;
    recent_.removeAll(i);
    loadQueue_.removeAll(i);
    rebuildList();
    return i;
}

bool CompareWindow::loadFrame(int i) {
    if (i < 0 || i >= entries_.size()) return false;
    Entry& e = entries_[i];
    if (!render_ || !render_->stack || e.owner != render_.get()) {
        refuse(QString("%1 is a frame of an earlier render. Press Add frames for this render's.").arg(e.name), 9000);
        return false;
    }
    if (gpuIdle_ && !gpuIdle_()) {
        refuse(QString("%1 is lined up with the outputs when picked again after the render.").arg(e.name), 9000);
        return false;
    }
    const auto& R = render_->result;
    const int k = e.frame, K = (int)render_->stack->po.paths.size();
    QApplication::setOverrideCursor(Qt::WaitCursor);
    std::vector<float> rgb;
    int w = 0, h = 0;
    int rc = 1;
    try { rc = stackcore::warp_frame(*render_->stack, k, rgb, w, h); } catch (const std::exception&) { rc = 1; }
    QApplication::restoreOverrideCursor();
    if (rc != 0 || w < R.w || h < R.h || R.w <= 0 || R.h <= 0) {
        refuse(QString("%1 could not be lined up with the render (%2).").arg(e.name, rc == 4 ? "the graphics card is out of memory" : "the frame could not be read"), 9000);
        return false;
    }
    // the render's size (a render made in strips is up to 3 px smaller than the rectangle), 16 bits as a render writes
    // them, and the screen image as the outputs are shown
    if (w != R.w || h != R.h) {
        std::vector<float> crop((size_t)R.w * R.h * 3);
        for (int y = 0; y < R.h; ++y) memcpy(crop.data() + (size_t)y * R.w * 3, rgb.data() + (size_t)y * w * 3, (size_t)R.w * 3 * sizeof(float));
        rgb.swap(crop);
    }
    auto px = std::make_shared<std::vector<uint16_t>>(rgb.size());
    for (size_t q = 0; q < rgb.size(); ++q) (*px)[q] = (uint16_t)(std::min(std::max(rgb[q], 0.f), 1.f) * 65535.0f + 0.5f);
    QImage d = toQImage(rgb.data(), R.w, R.h);
    if (space_.isValid()) { d.setColorSpace(space_); d.convertToColorSpace(QColorSpace::SRgb); }
    e.image = d.convertToFormat(QImage::Format_RGB32);
    e.px16 = px;
    alignedRecent_.removeAll(i);
    alignedRecent_.push_back(i);
    // each holds about 300 MB: the six used last keep their pixels
    while (alignedRecent_.size() > 6) {
        int victim = -1;
        for (int v : alignedRecent_) if (v != i && !onShow(v) && v != rt_.target && v != rt_.source) { victim = v; break; }
        if (victim < 0) break;
        alignedRecent_.removeAll(victim);
        if (victim < entries_.size() && entries_[victim].owner) { entries_[victim].image = QImage(); entries_[victim].px16.reset(); }
    }
    statusBar()->showMessage(QString("%1: frame %2 of %3, lined up with the render's outputs.").arg(e.name).arg(k + 1).arg(K), 8000);
    return true;
}

// The frame that is sharpest at an output pixel: measured on the frames in memory (focus_at; `second` = another frame
// that is nearly as sharp there, as behind a translucent crystal, or -1), failing that the frame the render's depth
// map chose. -1 = neither is to be had.
int CompareWindow::sharpestAt(int x, int y, int* second, int* how) const {
    if (second) *second = -1;
    if (how) *how = 0;
    if (!render_ || !render_->stack) return -1;
    const auto& R = render_->result;
    if (x < 0 || y < 0 || x >= R.w || y >= R.h) return -1;
    std::vector<double> en;
    bool ok = false;
    try { ok = stackcore::focus_at(*render_->stack, x, y, 32, en); } catch (const std::exception&) { ok = false; }
    const int K = (int)en.size();
    if (ok && K >= 2) {
        std::vector<double> s(K);   // a frame with its neighbours: one noisy frame does not win
        for (int k = 0; k < K; ++k) s[k] = (en[std::max(0, k - 1)] + 2 * en[k] + en[std::min(K - 1, k + 1)]) / 4;
        const int best = (int)(std::max_element(s.begin(), s.end()) - s.begin());
        std::vector<double> sorted = s;
        std::nth_element(sorted.begin(), sorted.begin() + K / 2, sorted.end());
        const double floor = sorted[K / 2];
        // a spot without detail (smooth background) has no peak: every frame measures the same there
        if (s[best] <= 3 * floor || s[best] <= 0) { if (how) *how = 2; }
        else {
            if (how) *how = 1;
            if (second) {   // another peak, well away from the first and at least half as high above the floor
                const int away = std::max(6, K / 16);
                int b2 = -1;
                for (int k = 1; k + 1 < K; ++k)
                    if (std::abs(k - best) >= away && s[k] >= s[k - 1] && s[k] >= s[k + 1] && s[k] - floor > 0.5 * (s[best] - floor) && (b2 < 0 || s[k] > s[b2])) b2 = k;
                *second = b2;
            }
            return best;
        }
    }
    if (R.depth.size() == (size_t)R.w * R.h) return std::clamp((int)std::lround(R.depth[(size_t)y * R.w + x]), 0, (int)render_->stack->po.paths.size() - 1);
    return -1;
}

void CompareWindow::frameHere(int pane, int x, int y) {
    frameHereAct_->setChecked(false);   // one pick: the left button paints, or pans, again
    if (!render_ || !render_->stack) { refuse("Frame here needs this session's render: render the stack, then press Compare in the main window.", 9000); return; }
    const auto& R = render_->result;
    const int shown = rt_.on && pane == 0 ? rt_.target : sel_[pane];
    if (shown < 0 || shown >= entries_.size() || entries_[shown].image.size() != QSize(R.w, R.h) || (pane == 1 && view_[1]->placed())) {
        refuse("Frame here: click on an output of this render (B, C, a composite, a sweep's variant), not on an image of another size.", 9000);
        return;
    }
    if (x < 0 || y < 0 || x >= R.w || y >= R.h) return;
    int second = -1, how = 0;
    QApplication::setOverrideCursor(Qt::WaitCursor);
    const int k = sharpestAt(x, y, &second, &how);
    QApplication::restoreOverrideCursor();
    if (k < 0) { refuse(how == 2 ? "Frame here: no frame stands out at that spot (nothing with detail there?)." : "Frame here: the frames are not in memory and this render kept no depth map.", 9000); return; }
    const int before = entries_.size();
    const int K = (int)render_->stack->po.paths.size();
    for (int q = 0; q < K; ++q) ensureFrameEntry(q);   // every frame, so that PgUp / PgDn step to the ones beside it
    if (entries_.size() != before) rebuildList();
    const int j = frameEntry(k);
    if (j < 0) return;
    showIn(active_, j);
    list_->scrollToItem(list_->item(j, 0));
    list_->selectRow(j);
    QString msg = how == 2 ? QString("No frame stands out at x %1, y %2 (nothing with detail there?): the frame the render chose is shown, frame %3 of %4 (%5)").arg(x).arg(y).arg(k + 1).arg(K).arg(entries_[j].name)
                           : QString("Sharpest at x %1, y %2: frame %3 of %4 (%5)").arg(x).arg(y).arg(k + 1).arg(K).arg(entries_[j].name);
    if (how != 2) msg += how == 1 ? QString(", measured on the frames") : QString(", as the depth map chose (the frames are not in memory)");
    if (second >= 0) msg += QString("; also sharp there: frame %1").arg(second + 1);
    statusBar()->showMessage(msg + ". PgUp / PgDn step to the frames before and after.", 20000);
}

QString CompareWindow::testFrameHere(int x, int y) {
    frameHereAct_->setChecked(true);
    frameHere(0, x, y);
    QString curve;   // the measure itself, every 8th frame and the ends
    std::vector<double> en;
    if (render_ && render_->stack && stackcore::focus_at(*render_->stack, x, y, 32, en))
        for (size_t k = 0; k < en.size(); ++k) if (k % 8 == 0 || k + 3 >= en.size()) curve += QString(" %1:%2").arg(k + 1).arg(en[k], 0, 'f', 1);
    const int d = render_ && render_->result.depth.size() == (size_t)render_->result.w * render_->result.h ? (int)std::lround(render_->result.depth[(size_t)y * render_->result.w + x]) + 1 : -1;
    return QString("pane %1: %2 | %3 | depth map: frame %4 | energy%5").arg(active_ + 1).arg(sel_[active_] >= 0 ? entries_[sel_[active_]].name : "-", statusBar()->currentMessage()).arg(d).arg(curve);
}

// ----------------------------------------------------------------------------- panes

void CompareWindow::setActive(int pane) {
    if (pane < 1 || pane >= nPanes_) pane = 1;
    const bool changed = pane != active_;
    active_ = pane;
    if (rt_.on && changed) {   // the brush paints from this pane's image from here on
        const int i = sel_[active_];
        if (i < 0 || i >= entries_.size() || entries_[i].image.isNull()) {
            rt_.blocked = true;
            refuse("That pane has no image to paint from: pick one for it.", 8000);
        } else rt_.blocked = i != rt_.source && !retouchSetSource(i);   // (it says why when the image does not fit)
    }
    markActive();
}

// With more than two panes the one the list fills has its picker outlined (with two it is always the right one).
// While retouching that pane is the one painted from: a faint frame round its image, and "(source)" after its name.
void CompareWindow::markActive() {
    for (int p = 0; p < kPanes; ++p) {
        const bool mark = nPanes_ > 2 && p == active_;
        pick_[p]->setStyleSheet(mark ? "QComboBox { border: 1px solid #8a4fe0; }" : QString());
        view_[p]->setFrameTint(mark && rt_.on ? QColor(138, 79, 224, 170) : QColor());
        if (p >= 1 && p < nPanes_ && heldFrame_ < 0 && sel_[p] >= 0 && sel_[p] < entries_.size() && !entries_[sel_[p]].image.isNull())
            view_[p]->setLabel(rt_.on && p == active_ ? entries_[sel_[p]].name + "  (source)" : entries_[sel_[p]].name);
    }
    showBrushBlocked();
}

bool CompareWindow::eventFilter(QObject* o, QEvent* e) {
    if (e->type() == QEvent::MouseButtonPress)
        for (int p = 1; p < nPanes_; ++p)
            if (o == pick_[p] && p != active_) setActive(p);
    return QMainWindow::eventFilter(o, e);
}

void CompareWindow::showNames() {
    if (rt_.on) return;   // (the retouch has its own line)
    QStringList names;
    for (int p = 0; p < nPanes_; ++p) names << (sel_[p] >= 0 && sel_[p] < entries_.size() ? entries_[sel_[p]].name : QString("—"));
    status_->setText(names.join("  |  "));
}

void CompareWindow::setPaneCount(int n) {
    n = std::clamp(n, 2, (int)kPanes);
    nPanes_ = n;
    { QSignalBlocker b(paneCount_); paneCount_->setCurrentIndex(n - 2); }
    const bool grid = n == 4;   // four: two rows of two; three: side by side (or under each other)
    (grid ? split2_ : split_)->addWidget(paneBox_[2]);
    (grid ? split2_ : split_)->addWidget(paneBox_[3]);
    for (int p = 0; p < kPanes; ++p) paneBox_[p]->setVisible(p < n);
    split2_->setVisible(grid);
    split_->setOrientation(verticalAct_->isChecked() && !grid ? Qt::Vertical : Qt::Horizontal);
    QList<int> even;
    for (int i = 0; i < split_->count(); ++i) even << 1000;
    split_->setSizes(even);
    split2_->setSizes({1000, 1000});
    rows_->setSizes({1000, 1000});
    for (int p = n; p < kPanes; ++p) sel_[p] = -1;
    setActive(active_);
    // a pane that has just come up shows the next image of the list that no pane shows yet (not a frame: those are
    // made when asked for)
    for (int p = 2; p < n; ++p) {
        if (sel_[p] >= 0) continue;
        for (int i = 0; i < entries_.size(); ++i)
            if (!onShow(i) && !entries_[i].owner && entries_[i].name != "depth") { showIn(p, i); break; }
    }
    updateDepthTools();
    showNames();
    QSettings s = appSettings();
    if (s.value("compare/panes", 2).toInt() != n) s.setValue("compare/panes", n);
}

void CompareWindow::setRender(std::shared_ptr<JobResultData> render, const QColorSpace& space) {
    // while a result is being retouched its regions belong to the render it came from: that one stays (the list's B
    // is not replaced under a retouch either); the main window says so when Refine is pressed after another render
    if (rt_.on && render_ && !regions_.empty()) return;
    const void* was = render_.get();
    render_ = render && render->rc == 0 && render->stack ? render : nullptr;
    space_ = space;
    if (render_.get() != was) {
        // the frames' rows go over to the new render: another alignment, so their pixels are made again when shown
        bool any = false;
        for (int i = 0; i < entries_.size(); ++i) {
            Entry& e = entries_[i];
            if (!e.owner || (rt_.on && (i == rt_.target || i == rt_.source))) continue;
            int k = -1;
            if (render_) {
                const auto& paths = render_->stack->po.paths;
                const QString want = QFileInfo(e.frameFile).absoluteFilePath();
                for (size_t q = 0; q < paths.size() && k < 0; ++q)
                    if (QFileInfo(QString::fromStdWString(paths[q].wstring())).absoluteFilePath().compare(want, Qt::CaseInsensitive) == 0) k = (int)q;
                if (k >= 0) { e.owner = render_.get(); e.frame = k; e.note = QString("frame %1 of %2").arg(k + 1).arg(paths.size()); }
            }
            if (k < 0) e.note = "a frame of an earlier render";
            e.image = QImage();
            e.px16.reset();
            any = true;
        }
        alignedRecent_.clear();
        if (any) {
            rebuildList();
            for (int p = 0; p < nPanes_; ++p) if (sel_[p] >= 0 && entries_[sel_[p]].owner && !(rt_.on && p == 0)) refresh(p);
        }
    }
    nameRegionRow();
}

// An output of the render on show, as the render made it: what Refine can render regions of. Not A (a tile does not
// keep its sums), and not a file, a frame, a composite or the depth map.
bool CompareWindow::refinable(int i) const {
    if (!render_ || i < 0 || i >= entries_.size()) return false;
    const Entry& e = entries_[i];
    const auto& R = render_->result;
    if (e.fileBacked || e.owner || e.name == "A" || e.image.size() != QSize(R.w, R.h)) return false;
    return std::find(R.names.begin(), R.names.end(), e.name.toStdString()) != R.names.end();
}

// the region row names the image its tools work on: the left pane's, when that is one Refine can render
void CompareWindow::nameRegionRow() {
    const int t = rt_.on ? rt_.target : sel_[0];
    regionLabel_->setText(!render_ ? QString(" Refine regions (render first): ") : refinable(t) ? QString(" Refine regions of %1: ").arg(entries_[t].name) : QString(" Refine regions: "));
}

// The brush has the left button while retouching, unless a tool has it: a region tool, Measure or Frame here.
void CompareWindow::restBrush() {
    const bool tool = view_[0]->regionTool() || measureAct_->isChecked() || frameHereAct_->isChecked();
    view_[0]->setBrush(rt_.on && !tool, brushSize_->value(), brushHard_->value());
}

QString CompareWindow::testTool(int mode) {
    QAction* tool = mode == 2 ? lassoAct_ : rectAct_;
    tool->setChecked(true);
    const QString said = QString("%1 %2 | row \"%3\" | retouch %4 on %5 | %6").arg(tool->text(), tool->isChecked() && view_[0]->regionTool() ? "ON" : "refused", regionLabel_->text().trimmed(),
                                                                        rt_.on ? "on" : "off", rt_.on ? entries_[rt_.target].name : QString("-"), statusBar()->currentMessage());
    tool->setChecked(false);
    return said;
}

// the soft edge of a region: the Edge box's width, or (Auto) a tenth of the region's shorter side, at least 8 px so
// the join is never a hard line and at most 64 px so a large region is not mostly edge
int CompareWindow::edgeFor(const Region& g) const {
    const int v = (int)feather_->value();
    if (v > 0) return v;
    const QRectF b = g.poly.boundingRect();
    return std::clamp((int)std::lround(0.1 * std::min(b.width(), b.height())), 8, 64);
}

int CompareWindow::selectedRegion() const {
    const int r = regionList_->currentRow();
    return r >= 0 && r < (int)regions_.size() ? r : -1;
}

void CompareWindow::setRefining(bool on) {
    refining_ = on;
    refineAct_->setEnabled(!on);
    refineCancelAct_->setVisible(on);
    for (QAction* a : {rectAct_, lassoAct_, removeRegionAct_, clearRegionsAct_}) a->setEnabled(!on);
    strength_->setEnabled(!on);
    regionList_->setEnabled(!on);
    if (on) for (QAction* a : {rectAct_, lassoAct_}) a->setChecked(false);   // (-> regionTool(0))
    markRows();
}

// The left pane must hold one of the render's outputs, with retouching on: the regions are laid into the retouch
// result. Retouching is switched on here when it is off, painting from the right pane's image when that fits the
// output and from the output itself when not. A refusal is said in the status bar and beside the cursor: a tool
// that does not switch on, with its reason only in a line at the bottom of the window, looks broken.
bool CompareWindow::regionsReady() {
    auto no = [this](const QString& why) { refuse(why, 12000); return false; };
    if (refining_) return no("Refine is running.");
    if (!render_) return no("Refine renders from the frames and the alignment of a render made since the program was started: there is none yet. "
                            "Press Render in the main window, then Compare. An image opened from disk cannot be refined.");
    const int t = rt_.on ? rt_.target : sel_[0];
    if (!refinable(t)) {
        const QString what = t < 0 ? QString("The left pane is empty.")
                           : entries_[t].name == "A" ? QString("A cannot be rendered in regions.")
                           : QString("%1 is not an output of the last render (it is a file, a frame, a composite or an earlier render's image).").arg(entries_[t].name);
        return no(what + " Refine renders regions of an output of the last render again: double-click B, C or another output in the list.");
    }
    if (!rt_.on) {
        if (sel_[active_] < 0 || entries_[sel_[active_]].image.isNull() || entries_[sel_[active_]].image.size() != entries_[t].image.size()) showIn(active_, t);
        retouchAct_->setChecked(true);   // -> retouchToggle, which says why when it cannot
        if (!rt_.on) return false;
    }
    return true;
}

void CompareWindow::regionTool(int mode) {
    if (mode && !regionsReady()) {
        for (QAction* a : {rectAct_, lassoAct_}) { QSignalBlocker b(a); a->setChecked(false); }
        mode = 0;
    }
    if (mode && measureAct_->isChecked()) measureAct_->setChecked(false);
    if (mode && frameHereAct_->isChecked()) frameHereAct_->setChecked(false);
    // the left button marks regions while a tool is on, and paints again when it is off
    if (rt_.on) view_[0]->setBrush(mode == 0, brushSize_->value(), brushHard_->value());
    for (ImageView* v : view_) v->setRegionTool(mode);
    markRows();
    if (mode) statusBar()->showMessage(QString(mode == 1 ? "Drag a rectangle over a part of %1 to render again. Mark as many as needed, then press Refine."
                                                         : "Draw around a part of %1 to render again. Mark as many as needed, then press Refine.").arg(entries_[rt_.target].name), 10000);
}

void CompareWindow::regionDrawn(int pane, const QPolygonF& frac) {
    if (!regionsReady()) return;
    // an outline is in fractions of the pane's picture: on an image of another size it would mark another place
    if (pane > 0 && (sel_[pane] < 0 || entries_[sel_[pane]].image.size() != QSize(rt_.w, rt_.h))) {
        refuse(QString("Mark regions on the left pane: this pane's image is not the size of %1.").arg(entries_[rt_.target].name));
        return;
    }
    QPolygonF px;
    for (const QPointF& f : frac) px << QPointF(f.x() * rt_.w, f.y() * rt_.h);
    addRegionPx(px, strength_->currentIndex());
}

int CompareWindow::addRegionPx(const QPolygonF& outline, int strength) {
    if (outline.size() < 3 || !regionsReady()) return -1;
    Region g;
    g.poly = outline;
    g.strength = std::clamp(strength, 0, 2);
    regions_.push_back(std::move(g));
    showRegions();
    regionList_->setCurrentCell((int)regions_.size() - 1, 0);
    statusBar()->showMessage(QString("Region %1 marked. Mark more, or press Refine.").arg(regions_.size()), 8000);
    return (int)regions_.size() - 1;
}

QString CompareWindow::testSource(const QString& path, const QString& savePng) {
    const QString name = QFileInfo(path).fileName();
    addFile(name, path, "test");
    select(1, name);
    if (!savePng.isEmpty() && sel_[1] >= 0) entries_[sel_[1]].image.save(savePng);
    setRetouch(true);
    return QString("right pane: %1 | retouch %2 | status: %3").arg(sel_[1] >= 0 ? entries_[sel_[1]].name : "-", rt_.on ? "on" : "off", statusBar()->currentMessage());
}

QString CompareWindow::testStroke(QPointF a, QPointF b) {
    emit view_[0]->strokeBegin(a);
    emit view_[0]->strokeTo(b);
    emit view_[0]->strokeEnd();
    return QString("right pane: %1 | source: %2 | blocked %3 | status: %4").arg(sel_[1] >= 0 ? entries_[sel_[1]].name : "-", rt_.on && rt_.source >= 0 ? entries_[rt_.source].name : "-")
        .arg(rt_.blocked ? 1 : 0).arg(statusBar()->currentMessage());
}

void CompareWindow::testDraw(int mode, QPointF a, QPointF b) {
    QAction* tool = mode == 1 ? rectAct_ : lassoAct_;
    tool->setChecked(true);
    if (!view_[0]->regionTool()) return;   // (it said why)
    QWidget* vp = view_[0]->viewport();
    auto at = [vp](QPointF f) { return QPointF(f.x() * vp->width(), f.y() * vp->height()); };
    auto send = [vp](QEvent::Type type, QPointF p, Qt::MouseButton button, Qt::MouseButtons held) {
        QMouseEvent e(type, p, vp->mapToGlobal(p), button, held, Qt::NoModifier);
        QCoreApplication::sendEvent(vp, &e);
    };
    std::vector<QPointF> path;
    if (mode == 1) path = {at(a), at((a + b) / 2), at(b)};
    else   // once round the ellipse between the two points
        for (int i = 0; i <= 36; ++i) {
            const double t = 6.283185307179586 * i / 37;
            path.push_back(at(QPointF((a.x() + b.x()) / 2 + std::cos(t) * (b.x() - a.x()) / 2, (a.y() + b.y()) / 2 + std::sin(t) * (b.y() - a.y()) / 2)));
        }
    send(QEvent::MouseButtonPress, path.front(), Qt::LeftButton, Qt::LeftButton);
    for (size_t i = 1; i < path.size(); ++i) send(QEvent::MouseMove, path[i], Qt::NoButton, Qt::LeftButton);
    send(QEvent::MouseButtonRelease, path.back(), Qt::LeftButton, Qt::NoButton);
    tool->setChecked(false);
}

void CompareWindow::removeRegion(int i) {
    if (i < 0 || i >= (int)regions_.size() || refining_) return;
    const QRect r = regions_[i].hasPatch() && regions_[i].on ? regions_[i].r : QRect();
    regions_.erase(regions_.begin() + i);
    if (!r.isEmpty()) rebuildTarget(r);
    showRegions();
}

void CompareWindow::clearRegions() {
    if (refining_) return;
    QRect r;
    for (const Region& g : regions_) if (g.hasPatch() && g.on) r = r.united(g.r);
    regions_.clear();
    if (!r.isEmpty()) rebuildTarget(r);
    showRegions();
}

void CompareWindow::setRegionOn(int i, bool on) {
    if (i < 0 || i >= (int)regions_.size() || !regions_[i].hasPatch() || regions_[i].on == on) return;
    regions_[i].on = on;
    rt_.dirty = true;
    rebuildTarget(regions_[i].r);
    showRegions();
}

// the outlines in both panes, and the list's rows
void CompareWindow::showRegions() {
    std::vector<ImageView::RegionMark> marks;
    if (rt_.on && rt_.w > 0 && rt_.h > 0)
        for (const Region& g : regions_) {
            ImageView::RegionMark m;
            for (const QPointF& p : g.poly) m.frac << QPointF(p.x() / rt_.w, p.y() / rt_.h);
            m.state = g.state == 2 ? (g.on ? 2 : 3) : g.state;
            marks.push_back(std::move(m));
        }
    const int sel = selectedRegion();
    // only on images of the result's size: an outline over another image would mark another place
    for (int p = 0; p < kPanes; ++p) {
        const bool fits = p == 0 || (sel_[p] >= 0 && entries_[sel_[p]].image.size() == QSize(rt_.w, rt_.h));
        view_[p]->setRegions(fits ? marks : std::vector<ImageView::RegionMark>(), sel);
    }
    fillingRegions_ = true;
    regionList_->setRowCount((int)regions_.size());
    for (int i = 0; i < (int)regions_.size(); ++i) {
        const Region& g = regions_[i];
        auto* n = new QTableWidgetItem(QString::number(i + 1));
        if (g.hasPatch()) { n->setFlags(n->flags() | Qt::ItemIsUserCheckable); n->setCheckState(g.on ? Qt::Checked : Qt::Unchecked); }
        regionList_->setItem(i, 0, n);
        regionList_->setItem(i, 1, new QTableWidgetItem(strength_->itemText(g.strength)));
        const QString was = g.hasPatch() && g.rendered != g.strength && g.rendered >= 0 ? QString(" (shown: %1)").arg(strength_->itemText(g.rendered).toLower()) : QString();
        regionList_->setItem(i, 2, new QTableWidgetItem(QString("%1 px").arg(g.hasPatch() || g.state == 1 ? g.edge : edgeFor(g))));
        regionList_->setItem(i, 3, new QTableWidgetItem(g.state == 1 ? "rendering…" : g.state == 4 ? "failed" : g.state == 2 ? (g.on ? "in the image" : "switched off") : "marked" + was));
    }
    if (sel >= 0 && sel < (int)regions_.size()) regionList_->setCurrentCell(sel, 0);
    regionList_->setVisible(!regions_.empty());
    fillingRegions_ = false;
}

void CompareWindow::refine() {
    if (!regionsReady()) return;
    std::vector<int> todo;
    for (int i = 0; i < (int)regions_.size(); ++i)
        if (regions_[i].state != 2 || regions_[i].rendered != regions_[i].strength) todo.push_back(i);
    if (todo.empty()) {
        refuse(regions_.empty() ? "Nothing to render: mark a region first (Rectangle or Lasso)." : "Every region is rendered already. Mark another, or change one's strength.", 8000);
        return;
    }
    const stackcore::StackOptions& so = *render_->stack;
    const stackcore::Grid g = so.po.g;
    const auto& R = render_->result;
    const int levels = R.levels > 0 ? R.levels : stackcore::n_levels(R.w, R.h);
    // each region's rectangle in sensor px: its outline's bounds and the soft edge; regions whose rendered
    // rectangles (with the margin around them) would overlap are rendered as one tile
    struct Group { int x0, y0, x1, y1; std::vector<int> members; };
    std::vector<Group> groups;
    for (int i : todo) {
        regions_[i].edge = edgeFor(regions_[i]);
        const double edge = regions_[i].edge + 2;
        const QRectF b = regions_[i].poly.boundingRect().adjusted(-edge, -edge, edge, edge).intersected(QRectF(0, 0, R.w, R.h));
        if (b.isEmpty()) { regions_[i].state = 4; continue; }
        Group q;
        q.x0 = std::clamp(g.x0 + (int)std::floor(b.left() / g.scale), g.x0, g.x1);
        q.y0 = std::clamp(g.y0 + (int)std::floor(b.top() / g.scale), g.y0, g.y1);
        q.x1 = std::clamp(g.x0 + (int)std::ceil(b.right() / g.scale), g.x0, g.x1);
        q.y1 = std::clamp(g.y0 + (int)std::ceil(b.bottom() / g.scale), g.y0, g.y1);
        q.members = {i};
        groups.push_back(std::move(q));
    }
    auto rendered = [&](const Group& q) {
        int a = q.x0, b = q.y0, c = q.x1, d = q.y1;
        stackcore::region_rect(g, levels, kRegionMargin, a, b, c, d);
        return QRect(QPoint(a, b), QPoint(c, d));
    };
    for (bool merged = true; merged;) {
        merged = false;
        for (size_t a = 0; a < groups.size() && !merged; ++a)
            for (size_t b = a + 1; b < groups.size() && !merged; ++b)
                if (rendered(groups[a]).intersects(rendered(groups[b]))) {
                    Group& A = groups[a];
                    const Group& B = groups[b];
                    A.x0 = std::min(A.x0, B.x0); A.y0 = std::min(A.y0, B.y0); A.x1 = std::max(A.x1, B.x1); A.y1 = std::max(A.y1, B.y1);
                    A.members.insert(A.members.end(), B.members.begin(), B.members.end());
                    groups.erase(groups.begin() + b);
                    merged = true;
                }
    }
    if (groups.empty()) { showRegions(); return; }
    auto job = std::make_shared<RegionJobData>();
    job->render = render_;
    const std::string output = entries_[rt_.target].name.toStdString();   // (regionsReady: one of the render's)
    jobMap_.assign(regions_.size(), {-1, -1});
    for (size_t gi = 0; gi < groups.size(); ++gi) {
        stackcore::RegionTile t;
        t.output = output;
        t.x0 = groups[gi].x0; t.y0 = groups[gi].y0; t.x1 = groups[gi].x1; t.y1 = groups[gi].y1;
        for (int i : groups[gi].members) {
            const DetailStrength& ds = kDetail[std::clamp(regions_[i].strength, 0, 2)];
            int vi = 0;
            while (vi < (int)t.variants.size() && t.variants[vi].name != ds.name) ++vi;
            if (vi == (int)t.variants.size()) t.variants.push_back({ds.name, detailOptions(regions_[i].strength, so)});
            jobMap_[i] = {(int)gi, vi};
            regions_[i].state = 1;
        }
        job->tiles.push_back(std::move(t));
    }
    setRefining(true);
    showRegions();
    statusBar()->showMessage(QString("Refine %5: %1 region%2 in %3 tile%4…").arg(todo.size()).arg(todo.size() == 1 ? "" : "s").arg(groups.size()).arg(groups.size() == 1 ? "" : "s").arg(entries_[rt_.target].name));
    emit refineRequested(job);
}

// ----------------------------------------------------------------------------- composite

// what an image is, for the composite's name: an output's method letter (the render's "B", "C", "Cn", "A"; another
// program's file named "... (B,Radius 8,...)"; a sweep's variants are all B), F for a frame lined up with the render,
// X for anything else
static QString kindOf(const QString& name, double change, bool frame) {
    if (frame) return "F";
    static const QRegularExpression own("^(CBn|Cg2|Cn|Cx|Cd|Cg|CB|B0|A|B|C|H)(?![A-Za-z0-9])"), other("\\((A|B|C),");
    if (auto m = own.match(name); m.hasMatch()) return m.captured(1);
    if (auto m = other.match(name); m.hasMatch()) return m.captured(1);
    if (change >= 0) return "B";
    return "X";
}

QString CompareWindow::compositeLabel(const QList<int>& picks) const {
    QString s;
    for (const char* k : {"B", "B0", "C", "CB", "Cn", "CBn", "Cx", "Cd", "Cg", "Cg2", "H", "A", "F", "X"}) {
        int n = 0;
        for (int i : picks) if (kindOf(entries_[i].name, entries_[i].change, entries_[i].owner != nullptr) == k) ++n;
        if (n) s += QString::number(n) + k;
    }
    return s;
}

// the ticked methods that are on offer (the ones past the first three only with the developer controls)
QStringList CompareWindow::compositeMethods() const {
    QStringList out;
    for (int i = 0; i < methodBox_.size(); ++i)
        if (methodBox_[i]->isChecked() && (moreMethods_ || i < kFirstMethods)) out << methodBox_[i]->text();
    return out;
}

void CompareWindow::setMoreMethods(bool on) {
    moreMethods_ = on;
    for (int i = kFirstMethods; i < methodAct_.size(); ++i) methodAct_[i]->setVisible(on);
    updateCompositeButton();
}

void CompareWindow::setMethodHelp(const QString& name, const QString& help) {
    for (QCheckBox* b : methodBox_) if (b->text() == name) b->setToolTip(help);
}

void CompareWindow::updateCompositeButton() {
    if (!methodsBtn_) return;   // (the list is filled before the buttons are made)
    QList<int> picks;
    for (int i = 0; i < entries_.size(); ++i) if (entries_[i].pick) picks << i;
    const QStringList methods = compositeMethods();
    methodsBtn_->setText(methods.isEmpty() ? QString("Methods: none") : methods.size() <= 4 ? "Methods: " + methods.join(", ")
                         : QString("Methods: %1 + %2 more").arg(methods.mid(0, 3).join(", ")).arg(methods.size() - 3));
    methodsBtn_->setEnabled(!compositing_);
    compositeBtn_->setEnabled(picks.size() >= 2 && !methods.isEmpty() && !compositing_ && !refining_);
    compositeBtn_->setText(compositing_ ? "Making the composite…" : picks.size() >= 2 && methods.isEmpty() ? "Make composite: tick a method"
                           : picks.size() >= 2 ? QString("Make composite (%1)").arg(compositeLabel(picks))
                           : picks.size() == 1 ? "Make composite: tick one more" : "Make composite");
}

namespace {
// A field of shifts on a grid of nodes: node (i, j) stands at ((i + 0.5) * cell, (j + 0.5) * cell)
struct Field {
    int nx = 0, ny = 0;
    double cell = 1;
    std::vector<QPointF> d;
    std::vector<char> ok;
    QPointF at(double x, double y) const {   // between the nodes; past the outermost ones, theirs
        const double gx = std::clamp(x / cell - 0.5, 0.0, nx - 1.0), gy = std::clamp(y / cell - 0.5, 0.0, ny - 1.0);
        const int i = std::min((int)gx, std::max(0, nx - 2)), j = std::min((int)gy, std::max(0, ny - 2));
        const double fx = nx > 1 ? gx - i : 0.0, fy = ny > 1 ? gy - j : 0.0;
        const int i1 = std::min(i + 1, nx - 1), j1 = std::min(j + 1, ny - 1);
        return (d[(size_t)j * nx + i] * (1 - fx) + d[(size_t)j * nx + i1] * fx) * (1 - fy) + (d[(size_t)j1 * nx + i] * (1 - fx) + d[(size_t)j1 * nx + i1] * fx) * fy;
    }
};

// The nodes without a measurement take the mean of their measured neighbours, from those outwards; then every node
// takes the median of itself and its neighbours, which drops a node that found another place than the ones around it.
void settle(Field& f) {
    const int nx = f.nx, ny = f.ny;
    for (int pass = 0; pass < nx + ny; ++pass) {
        std::vector<QPointF> d = f.d;
        std::vector<char> ok = f.ok;
        bool any = false, left = false;
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                if (f.ok[(size_t)j * nx + i]) continue;
                QPointF sum;
                int n = 0;
                for (int dj = -1; dj <= 1; ++dj)
                    for (int di = -1; di <= 1; ++di) {
                        const int a = i + di, b = j + dj;
                        if (a >= 0 && b >= 0 && a < nx && b < ny && f.ok[(size_t)b * nx + a]) { sum += f.d[(size_t)b * nx + a]; ++n; }
                    }
                if (n) { d[(size_t)j * nx + i] = sum / n; ok[(size_t)j * nx + i] = 1; any = true; } else left = true;
            }
        f.d = d; f.ok = ok;
        if (!any || !left) break;
    }
    std::vector<QPointF> out = f.d;
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            double xs[9], ys[9];
            int n = 0;
            for (int dj = -1; dj <= 1; ++dj)
                for (int di = -1; di <= 1; ++di) {
                    const int a = i + di, b = j + dj;
                    if (a >= 0 && b >= 0 && a < nx && b < ny) { xs[n] = f.d[(size_t)b * nx + a].x(); ys[n] = f.d[(size_t)b * nx + a].y(); ++n; }
                }
            std::nth_element(xs, xs + n / 2, xs + n);
            std::nth_element(ys, ys + n / 2, ys + n);
            out[(size_t)j * nx + i] = QPointF(xs[n / 2], ys[n / 2]);
        }
    f.d = out;
}

// the correlation of A's window about (cx, cy) with B's the shift (dx, dy) away (every `step`-th pixel); -2 when
// too little of the two is there, or the window has no structure to go by
double windowMatch(const Plane& A, const Plane& B, int cx, int cy, int hw, int dx, int dy, int step) {
    const int x0 = std::max({0, cx - hw, -dx}), x1 = std::min({A.w, cx + hw, B.w - dx});
    const int y0 = std::max({0, cy - hw, -dy}), y1 = std::min({A.h, cy + hw, B.h - dy});
    if (x1 - x0 < hw || y1 - y0 < hw) return -2;
    double sab = 0, saa = 0, sbb = 0;
    long n = 0;
    for (int y = y0; y < y1; y += step) {
        const float* a = A.v.data() + (size_t)y * A.w;
        const float* b = B.v.data() + (size_t)(y + dy) * B.w + dx;
        for (int x = x0; x < x1; x += step) { sab += a[x] * b[x]; saa += a[x] * a[x]; sbb += b[x] * b[x]; ++n; }
    }
    if (n < 64 || saa < 1e-7 * n || sbb < 1e-7 * n) return -2;
    return sab / std::sqrt(saa * sbb);
}

// One size of the two pictures: for each node of a grid of `cell` px, where A's window about it lies in B
// (A(x) = B(x + d)), searched within r px of what the pass before found there (`prev`, at 1 / up of this size)
Field matchLevel(const Plane& A, const Plane& B, int cell, int r, int step, const Field* prev, double up = 2.0) {
    Field f;
    f.cell = cell;
    f.nx = std::max(1, A.w / cell); f.ny = std::max(1, A.h / cell);
    f.d.assign((size_t)f.nx * f.ny, QPointF());
    f.ok.assign((size_t)f.nx * f.ny, 0);
    const int side = 2 * r + 1;
    std::vector<std::thread> th;
    const int nt = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    for (int q = 0; q < nt; ++q)
        th.emplace_back([&, q] {
            std::vector<double> m((size_t)side * side);
            for (int k = q; k < f.nx * f.ny; k += nt) {
                const int cx = (int)((k % f.nx + 0.5) * cell), cy = (int)((k / f.nx + 0.5) * cell);
                const QPointF start = prev ? prev->at(cx / up, cy / up) * up : QPointF();
                const int bx = (int)std::lround(start.x()), by = (int)std::lround(start.y());
                f.d[k] = start;
                int best = -1;
                for (int j = 0; j < side; ++j)
                    for (int i = 0; i < side; ++i) {
                        m[(size_t)j * side + i] = windowMatch(A, B, cx, cy, cell, bx + i - r, by + j - r, step);
                        if (best < 0 || m[(size_t)j * side + i] > m[best]) best = j * side + i;
                    }
                if (best < 0 || m[best] < 0.3) continue;
                const int bi = best % side, bj = best / side;
                auto peak = [](double a, double b, double c) { const double d = a - 2 * b + c; return a > -1.5 && c > -1.5 && d < 0 ? std::clamp(0.5 * (a - c) / d, -0.5, 0.5) : 0.0; };
                const double px = bi > 0 && bi + 1 < side ? peak(m[best - 1], m[best], m[best + 1]) : 0.0;
                const double py = bj > 0 && bj + 1 < side ? peak(m[best - side], m[best], m[best + side]) : 0.0;
                f.d[k] = QPointF(bx + bi - r + px, by + bj - r + py);
                f.ok[k] = 1;
            }
        });
    for (auto& t : th) t.join();
    return f;
}
}  // namespace

// An image of another size or crop (another program's stack of the same frames) on the grid of a reference image,
// for Make composite. Two programs' stacks are not one picture moved and scaled: each gave the frames its own scale
// and place, so what lies at one depth sits elsewhere than what lies at another, by tens of pixels on a deep stack,
// and no single move, scale, rotation or perspective brings the two together (measured on a 276-frame stack: 5 px
// median and 40 px at most left by the best of them). So: the whole-image fit first (estimatePlacement: a scale and a
// shift), then the shift that is left, found part by part from coarse to fine (parts of about 1000 px down to about
// 100 px of the image, each searched about what the size before found). The image is then resampled once, with the
// bicubic kernel, at 16 bits. Where it does not reach, the reference's own pixels stand in, so the stack finds
// nothing new there.
static bool placeOnto(const QImage& refImg, const std::vector<uint16_t>& refPx, int W, int H, const QImage& img, const std::vector<uint16_t>& px, int w, int h,
                      std::vector<uint16_t>& out, QString* why, QString* how) {
    auto no = [why](const QString& s) { if (why) *why = s; return false; };
    if (refImg.width() != W || refImg.height() != H || img.width() != w || img.height() != h || refPx.size() != (size_t)W * H * 3 || px.size() != (size_t)w * h * 3)
        return no("the images are not read in full.");
    if (W < 512 || H < 512) return no("the images are too small to line up.");
    double s = 1, match = 0;
    QPointF pos;
    if (!estimatePlacement(refImg, img, s, pos, &match)) return no("they do not look like the same subject.");
    // both at half size on the reference's grid, as planes of structure, and those halved three times
    const int hwid = W / 2, hhei = H / 2;
    auto planeOf = [](const QImage& g8) {
        Plane p;
        p.w = g8.width(); p.h = g8.height();
        p.v.resize((size_t)p.w * p.h);
        for (int y = 0; y < p.h; ++y) {
            const uchar* row = g8.constScanLine(y);
            for (int x = 0; x < p.w; ++x) p.v[(size_t)y * p.w + x] = row[x] / 255.f;
        }
        return p;
    };
    std::vector<Plane> A, B;
    {
        A.push_back(planeOf(refImg.scaled(hwid, hhei, Qt::IgnoreAspectRatio, Qt::SmoothTransformation).convertToFormat(QImage::Format_Grayscale8)));
        QImage on(hwid, hhei, QImage::Format_RGB32);
        on.fill(Qt::black);
        {
            QPainter p(&on);
            p.setRenderHint(QPainter::SmoothPixmapTransform);
            p.translate(pos.x() / 2, pos.y() / 2);
            p.scale(s / 2, s / 2);
            p.drawImage(0, 0, img);
        }
        B.push_back(planeOf(on.convertToFormat(QImage::Format_Grayscale8)));
    }
    for (int k = 0; k < 3; ++k) { A.push_back(halved(A.back())); B.push_back(halved(B.back())); }
    for (auto& p : A) p = structure(p);
    for (auto& p : B) p = structure(p);
    // coarse to fine: the nodes 32 px apart at each size (48 at the last), the search 4 px wide at first and 3 after
    Field f;
    int measured = 0;
    for (int lv = 3; lv >= 0; --lv) {
        Field g = matchLevel(A[lv], B[lv], lv == 0 ? 48 : 32, lv == 3 ? 4 : 3, lv == 0 ? 2 : 1, lv == 3 ? nullptr : &f);
        if (lv == 3) { int n = 0; for (char k : g.ok) n += k; if (n * 4 < (int)g.ok.size()) return no("too little of the two pictures agrees (are they the same stack?)."); }
        settle(g);
        f = std::move(g);
    }
    {   // once more at the last size with the nodes half as far apart: what changes within a part (two depths side by side)
        Field g = matchLevel(A[0], B[0], 24, 2, 1, &f, 1.0);
        for (char k : g.ok) measured += k;   // (before the nodes without detail take their neighbours')
        settle(g);
        f = std::move(g);
    }
    std::vector<double> len;
    for (const QPointF& d : f.d) len.push_back(2 * std::hypot(d.x(), d.y()));
    std::sort(len.begin(), len.end());
    try { out.resize((size_t)W * H * 3); }
    catch (const std::bad_alloc&) { return no("this PC ran out of memory (RAM)."); }
    // each pixel of the reference's grid: where it lies in the image, and the image there (bicubic, Catmull-Rom)
    auto rows = [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y)
            for (int x = 0; x < W; ++x) {
                const QPointF d = f.at(x / 2.0, y / 2.0) * 2.0;
                const double u = (x + d.x() - pos.x()) / s, v = (y + d.y() - pos.y()) / s;
                const int u0 = (int)std::floor(u), v0 = (int)std::floor(v);
                uint16_t* o = out.data() + ((size_t)y * W + x) * 3;
                if (u0 < 1 || v0 < 1 || u0 + 2 >= w || v0 + 2 >= h) {   // past the image: the reference's own pixel
                    const uint16_t* r = refPx.data() + ((size_t)y * W + x) * 3;
                    o[0] = r[0]; o[1] = r[1]; o[2] = r[2];
                    continue;
                }
                const double tx = u - u0, ty = v - v0;
                const double wx[4] = {((-tx + 2) * tx - 1) * tx / 2, ((3 * tx - 5) * tx * tx + 2) / 2, ((-3 * tx + 4) * tx + 1) * tx / 2, (tx - 1) * tx * tx / 2};
                const double wy[4] = {((-ty + 2) * ty - 1) * ty / 2, ((3 * ty - 5) * ty * ty + 2) / 2, ((-3 * ty + 4) * ty + 1) * ty / 2, (ty - 1) * ty * ty / 2};
                double acc[3] = {0, 0, 0};
                for (int j = 0; j < 4; ++j) {
                    const uint16_t* p = px.data() + ((size_t)(v0 - 1 + j) * w + (u0 - 1)) * 3;
                    for (int i = 0; i < 4; ++i) {
                        const double k = wx[i] * wy[j];
                        acc[0] += k * p[i * 3]; acc[1] += k * p[i * 3 + 1]; acc[2] += k * p[i * 3 + 2];
                    }
                }
                for (int c = 0; c < 3; ++c) o[c] = (uint16_t)std::lround(std::clamp(acc[c], 0.0, 65535.0));
            }
    };
    {
        std::vector<std::thread> th;
        const int nt = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
        for (int q = 0; q < nt; ++q) th.emplace_back(rows, (int)((qint64)H * q / nt), (int)((qint64)H * (q + 1) / nt));
        for (auto& t : th) t.join();
    }
    if (how) *how = QString("scaled by %1, moved %2, %3 px, then part by part: %4 px in the middle of the range, up to %5 px; %6 % of the picture had detail to go by")
                        .arg(s, 0, 'f', 4).arg(pos.x(), 0, 'f', 1).arg(pos.y(), 0, 'f', 1).arg(len[len.size() / 2], 0, 'f', 1).arg(len.back(), 0, 'f', 1)
                        .arg((int)std::lround(100.0 * measured / std::max<size_t>(1, f.ok.size())));
    return true;
}

void CompareWindow::testComposite(const QStringList& names) {
    // (a file added a moment ago is still being read: the button is pressed once every named image is there)
    bool waiting = false;
    for (int i = 0; i < entries_.size(); ++i)
        if (names.contains(entries_[i].name) && entries_[i].image.isNull() && entries_[i].fileBacked) { requestLoad(i); waiting = true; }
    if (waiting) { QTimer::singleShot(400, this, [this, names] { testComposite(names); }); return; }
    for (Entry& e : entries_) e.pick = names.contains(e.name);
    rebuildList();
    makeComposite();
}

void CompareWindow::makeComposite() {
    if (compositing_ || refining_) return;
    QList<int> picks;
    for (int i = 0; i < entries_.size(); ++i) if (entries_[i].pick) picks << i;
    if (picks.size() < 2) { refuse("Make composite: tick two or more images in the list first.", 6000); return; }
    const QStringList methods = compositeMethods();
    if (methods.isEmpty()) { refuse("Make composite: tick a method under Methods first.", 6000); return; }
    auto job = std::make_shared<CompositeJobData>();
    job->outputs = methods;
    auto fail = [this, job](const QString& why) {
        if (!job->temp.isEmpty()) QDir(job->temp).removeRecursively();
        refuse(why, 12000);
        emit compositeFinished(false);
    };
    QApplication::setOverrideCursor(Qt::WaitCursor);
    struct Restore { ~Restore() { QApplication::restoreOverrideCursor(); } } restore;
    // a 16-bit RGB TIFF of the composite's size is read as it is; anything else (an 8-bit file, a frame lined up here,
    // an image held in memory only, an image of another size) goes in as a 16-bit copy, so the stack is of one kind
    struct Src { int i = -1, w = 0, h = 0; QString file; };
    std::vector<Src> srcs;
    for (int i : picks) {
        const Entry& e = entries_[i];
        Src s;
        s.i = i;
        if (!e.path.isEmpty() && !e.px16) {
            const TiffInfo t = tiff_probe(e.path.toStdWString());
            if (t.ok() && t.samples == 3 && t.bits == 16 && !t.jpeg && !t.raw) { s.file = e.path; s.w = (int)t.width; s.h = (int)t.height; }
        }
        if (s.file.isEmpty()) {
            if (e.image.isNull()) { fail(QString("Make composite: %1 is not read yet. Show it in a pane once, then try again.").arg(e.name)); return; }
            s.w = e.image.width(); s.h = e.image.height();
        }
        srcs.push_back(s);
    }
    // the composite's size: the one most of the ticked images have; of sizes with as many images each, the largest.
    // An image of another size (another program's stack of the same frames) is lined up with the first image of
    // that size and resampled onto its grid
    size_t ref = 0;
    {
        auto share = [&](const Src& a) { int n = 0; for (const Src& b : srcs) if (b.w == a.w && b.h == a.h) ++n; return n; };
        for (size_t k = 1; k < srcs.size(); ++k) {
            // (of sizes with as many images each: the render's own, whose scale and frames the composite then keeps;
            // failing that the larger)
            auto own = [&](const Src& q) { return render_ && q.w == render_->result.w && q.h == render_->result.h; };
            const int a = share(srcs[k]), b = share(srcs[ref]);
            const bool larger = (qint64)srcs[k].w * srcs[k].h > (qint64)srcs[ref].w * srcs[ref].h;
            if (a > b || (a == b && (own(srcs[k]) != own(srcs[ref]) ? own(srcs[k]) : larger))) ref = k;
        }
    }
    const int W = srcs[ref].w, H = srcs[ref].h;
    const Entry& re = entries_[srcs[ref].i];
    std::vector<uint16_t> refPx;   // the reference's pixels, read when an image has to be lined up with it
    QStringList names, placed;
    for (const Src& s : srcs) {
        const Entry& e = entries_[s.i];
        QString p = s.file;
        const bool other = s.w != W || s.h != H;
        if (other || p.isEmpty()) {
            std::vector<uint16_t> px;
            int w = 0, h = 0;
            if (!load16(e, px, w, h, nullptr)) { fail(QString("Make composite: %1 is not read yet. Show it in a pane once, then try again.").arg(e.name)); return; }
            if (other) {
                if (re.image.isNull() || e.image.isNull()) {
                    fail(QString("Make composite: %1 is another size than %2 and has to be lined up with it. Show both in a pane once, then try again.").arg(e.name, re.name));
                    return;
                }
                int rw = 0, rh = 0;
                if (refPx.empty() && (!load16(re, refPx, rw, rh, nullptr) || rw != W || rh != H)) { fail(QString("Make composite: %1 could not be read.").arg(re.name)); return; }
                statusBar()->showMessage(QString("Make composite: lining %1 up with %2…").arg(e.name, re.name));
                QApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
                std::vector<uint16_t> on;
                QString why, how;
                if (!placeOnto(re.image, refPx, W, H, e.image, px, w, h, on, &why, &how)) {
                    fail(QString("Make composite: %1 could not be lined up with %2: %3").arg(e.name, re.name, why));
                    return;
                }
                DiagLog::write(QString("composite: %1 (%2 x %3) lined up with %4 (%5 x %6): %7").arg(e.name).arg(w).arg(h).arg(re.name).arg(W).arg(H).arg(how));
                placed << QString("%1 %2").arg(e.name, how);
                px = std::move(on);
                w = W; h = H;
            }
            if (job->temp.isEmpty()) {
                job->temp = QDir(QDir::tempPath()).filePath(QString("xlamina_composite_%1").arg(QCoreApplication::applicationPid()));
                QDir(job->temp).removeRecursively();
                QDir().mkpath(job->temp);
            }
            std::vector<float> rgb(px.size());
            for (size_t q = 0; q < px.size(); ++q) rgb[q] = px[q] / 65535.0f;
            p = QDir(job->temp).filePath(QString("source_%1.tif").arg(names.size() + 1));
            std::string why;
            if (!stackcore::write_tiff_rgb16(p.toStdString(), rgb, w, h, render_ ? render_->result.icc : std::vector<uint8_t>(), &why)) {
                fail(QString("Make composite: a copy of %1 could not be written (%2).").arg(e.name, QString::fromStdString(why)));
                return;
            }
        }
        job->paths.emplace_back(p.toStdWString());
        names << e.name;
    }
    job->label = compositeLabel(picks);
    job->fovMm = fovOf(re);   // the composite is on the reference's grid
    job->sources = names.join(", ") + (placed.isEmpty() ? QString() : QString(" (lined up: %1)").arg(placed.join("; ")));
    // beside the render's outputs; without a render, beside the first image that is a file
    if (render_ && !render_->outDir.isEmpty()) job->dir = render_->outDir;
    else for (int i : picks) if (!entries_[i].path.isEmpty()) { job->dir = QFileInfo(entries_[i].path).absolutePath(); break; }
    compositeTemp_ = job->temp;
    compositing_ = true;
    updateCompositeButton();
    statusBar()->showMessage(QString("Composite %1 (%3): stacking %2…").arg(job->label, job->sources, methods.join(", ")));
    emit compositeRequested(job);
}

void CompareWindow::compositeFailedToStart(const QString& why) {
    if (!compositing_) return;
    compositing_ = false;
    if (!compositeTemp_.isEmpty()) QDir(compositeTemp_).removeRecursively();
    compositeTemp_.clear();
    updateCompositeButton();
    refuse(why, 12000);
    emit compositeFinished(false);
}

void CompareWindow::compositeDone(std::shared_ptr<JobResultData> data) {
    if (!compositing_) return;
    compositing_ = false;
    // (developer aid: XLAMINA_KEEP_TEMP leaves the copies, to look at an image as it was lined up)
    if (!compositeTemp_.isEmpty() && !qEnvironmentVariableIsSet("XLAMINA_KEEP_TEMP")) QDir(compositeTemp_).removeRecursively();
    compositeTemp_.clear();
    updateCompositeButton();
    const auto job = data ? data->composite : nullptr;
    if (!job || data->rc != 0 || data->result.names.empty()) {
        statusBar()->showMessage(data && data->rc == 3 ? QString("Composite cancelled.")
                                 : QString("The composite could not be made%1").arg(data && !data->error.isEmpty() ? ": " + data->error : QString(".")), 15000);
        emit compositeFinished(false);
        return;
    }
    const auto& R = data->result;
    // "2B1C composite (B)"; a second one of the same kind is "2B1C composite 2 (B)": no file and no entry is replaced
    auto nameFor = [&](int n, const QString& out) { return QString("%1 composite%2 (%3)").arg(job->label, n > 1 ? QString(" %1").arg(n) : QString(), out); };
    int n = 1;
    for (bool taken = true; taken && n < 1000; n += taken ? 1 : 0) {
        taken = false;
        for (const std::string& o : R.names) {
            const QString nm = nameFor(n, QString::fromStdString(o));
            if (indexOf(nm) >= 0 || (!job->dir.isEmpty() && QFileInfo::exists(QDir(job->dir).filePath(nm + ".tif")))) taken = true;
        }
    }
    QColorSpace cs = space_;
    if (!cs.isValid() && !R.icc.empty()) cs = QColorSpace::fromIccProfile(QByteArray((const char*)R.icc.data(), (int)R.icc.size()));
    QStringList made, unwritten;
    for (size_t k = 0; k < R.names.size() && k < R.images.size(); ++k) {
        if (R.images[k].size() != (size_t)R.w * R.h * 3) continue;
        const QString nm = nameFor(n, QString::fromStdString(R.names[k]));
        QString file;
        if (!job->dir.isEmpty()) {
            file = QDir(job->dir).filePath(nm + ".tif");
            std::string why;
            if (!stackcore::write_tiff_rgb16(file.toStdString(), R.images[k], R.w, R.h, R.icc, &why)) { unwritten << nm; file.clear(); }
        }
        QImage d = toQImage(R.images[k].data(), R.w, R.h);
        if (cs.isValid()) { d.setColorSpace(cs); d.convertToColorSpace(QColorSpace::SRgb); }   // as the outputs are shown
        addImage(nm, d.convertToFormat(QImage::Format_RGB32), "composite of " + job->sources, file);
        if (const int at = indexOf(nm); at >= 0) entries_[at].fovMm = job->fovMm;
        made << nm;
    }
    if (made.isEmpty()) { statusBar()->showMessage("The composite came back empty.", 12000); emit compositeFinished(false); return; }
    if (!rt_.on) showIn(1, indexOf(made.first()));
    statusBar()->showMessage(QString("%1: stacked from %2 in %3 s%4").arg(made.join(", "), job->sources).arg(data->seconds, 0, 'f', 1)
                                 .arg(!unwritten.isEmpty() ? QString("; not written to disk: %1").arg(unwritten.join(", "))
                                      : job->dir.isEmpty() ? QString("; in memory only") : QString("; saved in %1").arg(QDir::toNativeSeparators(job->dir))), 20000);
    emit compositeFinished(true);
}

void CompareWindow::regionsProgress(const QString& stage, double fraction) {
    if (compositing_) { statusBar()->showMessage(QString("Composite %1 %: %2").arg((int)std::lround(100 * std::clamp(fraction, 0.0, 1.0))).arg(stage)); return; }
    if (!refining_) return;
    statusBar()->showMessage(QString("Refine %1 %: %2").arg((int)std::lround(100 * std::clamp(fraction, 0.0, 1.0))).arg(stage));
}

void CompareWindow::refineFailedToStart(const QString& why) {
    if (!refining_) return;
    for (Region& g : regions_) if (g.state == 1) g.state = 0;
    jobMap_.clear();
    setRefining(false);
    showRegions();
    refuse(why, 12000);
    emit refineFinished(false);
}

void CompareWindow::regionsDone(std::shared_ptr<JobResultData> data) {
    if (!refining_) return;
    setRefining(false);
    const auto job = data ? data->regions : nullptr;
    const bool cancelled = data && data->rc == 3;
    int ok = 0, bad = 0;
    for (int i = 0; i < (int)regions_.size() && i < (int)jobMap_.size(); ++i) {
        const auto [ti, vi] = jobMap_[i];
        if (ti < 0) continue;
        Region& g = regions_[i];
        const stackcore::RegionTile* t = job && ti < (int)job->tiles.size() ? &job->tiles[ti] : nullptr;
        if (rt_.on && !cancelled && t && vi < (int)t->out.size() && t->out[vi].rc == 0 && !t->out[vi].b.empty() && t->w > 0 && t->h > 0) {
            const QRect before = g.hasPatch() && g.on ? g.r : QRect();
            buildPatch(g, *t, t->out[vi].b);
            if (g.hasPatch()) {
                g.state = 2; g.rendered = g.strength; g.on = true;
                rebuildTarget(before.united(g.r));
                ++ok;
                continue;
            }
        }
        g.state = cancelled ? 0 : 4;
        if (!cancelled) ++bad;
    }
    jobMap_.clear();
    if (ok) rt_.dirty = true;
    showRegions();
    if (cancelled) statusBar()->showMessage("Refine cancelled.", 6000);
    else if (bad) statusBar()->showMessage(QString("Refine: %1 region%2 rendered, %3 failed%4").arg(ok).arg(ok == 1 ? "" : "s").arg(bad)
                                               .arg(data && data->rc == 4 ? ": not enough GPU memory for a region that large; mark a smaller one." : data ? ": " + data->error : QString()), 15000);
    else statusBar()->showMessage(QString("Refine: %1 region%2 rendered in %3 s. Hold Space to see the image as it was; untick a region to take it out.")
                                      .arg(ok).arg(ok == 1 ? "" : "s").arg(data ? data->seconds : 0.0, 0, 'f', 1), 15000);
    retouchStatus();
    emit refineFinished(!cancelled && bad == 0 && ok > 0);
}

// A region's patch from its tile's image: the rectangle of its outline and soft edge (inside the image and the tile), the
// pixels at 16 bits as a render writes them and on screen as the outputs are shown, and its share per pixel: 1 inside
// the outline, falling to 0 across the soft edge
void CompareWindow::buildPatch(Region& g, const stackcore::RegionTile& t, const std::vector<float>& b) {
    const int F = g.edge;
    const QRect r = g.poly.boundingRect().toAlignedRect().adjusted(-F - 1, -F - 1, F + 1, F + 1).intersected(QRect(0, 0, rt_.w, rt_.h)).intersected(QRect(t.ox, t.oy, t.w, t.h));
    g.alpha.clear(); g.px16.clear(); g.disp = QImage();
    if (r.isEmpty() || b.size() < (size_t)t.w * t.h * 3) return;
    const int w = r.width(), h = r.height();
    QImage a(w, h, QImage::Format_Grayscale8);
    a.fill(0);
    {
        QPainter p(&a);
        p.setRenderHint(QPainter::Antialiasing);
        p.setPen(Qt::NoPen);
        p.setBrush(Qt::white);
        p.translate(-r.topLeft());
        p.drawPolygon(g.poly);
    }
    std::vector<float> m((size_t)w * h);
    for (int y = 0; y < h; ++y) {
        const uchar* s = a.constScanLine(y);
        for (int x = 0; x < w; ++x) m[(size_t)y * w + x] = s[x] / 255.f;
    }
    softenPlane(m, w, h, F / 2);
    g.alpha.resize((size_t)w * h);
    for (size_t i = 0; i < m.size(); ++i) g.alpha[i] = (uint8_t)std::lround(std::clamp(m[i], 0.f, 1.f) * 255.f);
    g.px16.resize((size_t)w * h * 3);
    std::vector<float> crop((size_t)w * h * 3);
    for (int y = 0; y < h; ++y) {
        const float* s = b.data() + ((size_t)(r.y() + y - t.oy) * t.w + (r.x() - t.ox)) * 3;
        memcpy(crop.data() + (size_t)y * w * 3, s, (size_t)w * 3 * sizeof(float));
        uint16_t* d = g.px16.data() + (size_t)y * w * 3;
        for (int i = 0; i < w * 3; ++i) d[i] = (uint16_t)(std::min(std::max(s[i], 0.f), 1.f) * 65535.0f + 0.5f);   // as write_tiff_rgb16
    }
    QImage d = toQImage(crop.data(), w, h);
    if (space_.isValid()) { d.setColorSpace(space_); d.convertToColorSpace(QColorSpace::SRgb); }   // as the outputs are shown
    g.disp = d.convertToFormat(QImage::Format_RGB32);
    g.r = r;
}

// The retouch target inside r: the image as it was before any region (kept from the first patch on), then the
// patches that are switched on, in the list's order; the result (target + strokes) follows.
void CompareWindow::rebuildTarget(const QRect& rect) {
    if (!rt_.on) return;
    const QRect r = rect.intersected(QRect(0, 0, rt_.w, rt_.h));
    if (r.isEmpty()) return;
    if (rt_.base16.empty()) { rt_.base16 = rt_.tgt16; rt_.baseDisp = rt_.tgtDisp; }
    for (int y = r.top(); y <= r.bottom(); ++y) {
        memcpy(rt_.tgt16.data() + ((size_t)y * rt_.w + r.left()) * 3, rt_.base16.data() + ((size_t)y * rt_.w + r.left()) * 3, (size_t)r.width() * 6);
        memcpy(reinterpret_cast<QRgb*>(rt_.tgtDisp.scanLine(y)) + r.left(), reinterpret_cast<const QRgb*>(rt_.baseDisp.constScanLine(y)) + r.left(), (size_t)r.width() * sizeof(QRgb));
    }
    for (const Region& g : regions_) {
        if (!g.hasPatch() || !g.on) continue;
        const QRect q = g.r.intersected(r);
        for (int y = q.top(); y <= q.bottom() && !q.isEmpty(); ++y) {
            const int py = y - g.r.y();
            const uint8_t* al = g.alpha.data() + (size_t)py * g.r.width();
            const uint16_t* p16 = g.px16.data() + (size_t)py * g.r.width() * 3;
            const QRgb* pd = reinterpret_cast<const QRgb*>(g.disp.constScanLine(py));
            uint16_t* tg = rt_.tgt16.data() + (size_t)y * rt_.w * 3;
            QRgb* td = reinterpret_cast<QRgb*>(rt_.tgtDisp.scanLine(y));
            for (int x = q.left(); x <= q.right(); ++x) {
                const int px = x - g.r.x(), a = al[px];
                if (a == 0) continue;
                if (a == 255) { for (int k = 0; k < 3; ++k) tg[x * 3 + k] = p16[px * 3 + k]; td[x] = pd[px]; continue; }
                for (int k = 0; k < 3; ++k) tg[x * 3 + k] = (uint16_t)((tg[x * 3 + k] * (255 - a) + p16[px * 3 + k] * a + 127) / 255);
                td[x] = qRgb((qRed(td[x]) * (255 - a) + qRed(pd[px]) * a + 127) / 255, (qGreen(td[x]) * (255 - a) + qGreen(pd[px]) * a + 127) / 255,
                             (qBlue(td[x]) * (255 - a) + qBlue(pd[px]) * a + 127) / 255);
            }
        }
    }
    composite(r);
    view_[0]->patch(r);
}

void CompareWindow::retouchStatus() {
    if (!rt_.on) return;
    size_t painted = 0;
    for (size_t i = 0; i < rt_.mask.size(); ++i) if (rt_.mask[i] || (rt_.hasClone() && rt_.ca[i])) ++painted;   // painted or copied
    status_->setText(QString("Retouch: %1  <-  %2   |   brush %3 px, hardness %4   |   %5 % painted, %6 stroke%7")
                         .arg(entries_[rt_.target].name, entries_[rt_.source].name).arg((int)view_[0]->brushRadius()).arg(brushHard_->value(), 0, 'f', 1)
                         .arg(100.0 * painted / std::max<size_t>(1, rt_.mask.size()), 0, 'f', 2).arg(rt_.undo.size()).arg(rt_.undo.size() == 1 ? "" : "s"));
}

// ----------------------------------------------------------------------------- rows

// Each toolbar row in its section's colour: faint as a rule, strong (with a line above and below) while its mode is
// on, so that it is plain at a glance whether the left button paints, marks regions or pans. A checked button is
// filled with the colour, and the row's main button ("primary": Refine) is drawn as a button.
static QString rowStyle(const QColor& c, bool active) {
    const QString rgb = QString("%1, %2, %3").arg(c.red()).arg(c.green()).arg(c.blue());
    QString s = QStringLiteral(
        "QToolBar { background: rgba(RGB, BG); border: 0; border-left: 4px solid rgba(RGB, EDGE); LINES }"
        " QToolButton:checked { background: rgb(RGB); color: white; border: 1px solid rgba(255, 255, 255, 110); border-radius: 3px; }"
        " QToolButton:checked:disabled { background: rgba(RGB, 45); color: rgba(128, 128, 128, 220); border: 1px solid rgba(128, 128, 128, 90); }"
        " QToolButton#primary { background: rgba(RGB, 215); color: white; font-weight: bold; border: 1px solid rgba(255, 255, 255, 90);"
        " border-radius: 4px; padding: 2px 16px; margin: 0 6px; }"
        " QToolButton#primary:hover { background: rgb(RGB); border-color: white; }"
        " QToolButton#primary:pressed { background: rgba(RGB, 140); }"
        " QToolButton#primary:disabled { background: rgba(128, 128, 128, 50); color: rgba(128, 128, 128, 210); border-color: transparent; }");
    s.replace("LINES", active ? "border-top: 1px solid rgb(RGB); border-bottom: 1px solid rgb(RGB);" : "");
    s.replace("BG", active ? "95" : "24");
    s.replace("EDGE", active ? "255" : "150");
    s.replace("RGB", rgb);
    return s;
}

void CompareWindow::markRows() {
    auto set = [](QToolBar* tb, const QString& s) { if (tb && tb->styleSheet() != s) tb->setStyleSheet(s); };
    set(mainTb_, rowStyle(QColor(64, 128, 224), false));                                              // comparing: blue
    set(retouchTb_, rowStyle(QColor(138, 79, 224), rt_.on));                                          // retouch: purple (the source pane's frame)
    set(regionTb_, rowStyle(QColor(40, 160, 104), rectAct_->isChecked() || lassoAct_->isChecked() || refining_));   // refine: green (the regions' outlines)
}

// ----------------------------------------------------------------------------- clone (retouch: a copy from another place)

namespace {
// The sum of each cell's (2r+1)^2 neighbourhood, for a w x h plane of ch values per cell
void boxSum(std::vector<float>& v, int w, int h, int ch, int r) {
    std::vector<float> t(v.size());
    for (int y = 0; y < h; ++y)
        for (int c = 0; c < ch; ++c) {
            const float* in = v.data() + (size_t)y * w * ch + c;
            float* out = t.data() + (size_t)y * w * ch + c;
            double s = 0;
            for (int x = 0; x <= std::min(r, w - 1); ++x) s += in[(size_t)x * ch];
            for (int x = 0; x < w; ++x) {
                out[(size_t)x * ch] = (float)s;
                if (x + r + 1 < w) s += in[(size_t)(x + r + 1) * ch];
                if (x - r >= 0) s -= in[(size_t)(x - r) * ch];
            }
        }
    const size_t row = (size_t)w * ch;
    for (int x = 0; x < w; ++x)
        for (int c = 0; c < ch; ++c) {
            const float* in = t.data() + (size_t)x * ch + c;
            float* out = v.data() + (size_t)x * ch + c;
            double s = 0;
            for (int y = 0; y <= std::min(r, h - 1); ++y) s += in[y * row];
            for (int y = 0; y < h; ++y) {
                out[y * row] = (float)s;
                if (y + r + 1 < h) s += in[(y + r + 1) * row];
                if (y - r >= 0) s -= in[(y - r) * row];
            }
        }
}

// the known cells' values averaged over the known cells within r of each: what is left is the tone, not the texture
void blurKnown(std::vector<float>& v, const std::vector<uint8_t>& k, int w, int h, int ch, int r) {
    std::vector<float> n(k.size());
    for (size_t i = 0; i < k.size(); ++i) {
        n[i] = k[i] ? 1.f : 0.f;
        if (!k[i]) for (int c = 0; c < ch; ++c) v[i * ch + c] = 0.f;
    }
    boxSum(v, w, h, ch, r);
    boxSum(n, w, h, 1, r);
    for (size_t i = 0; i < k.size(); ++i)
        for (int c = 0; c < ch; ++c) v[i * ch + c] = k[i] && n[i] > 0.5f ? v[i * ch + c] / n[i] : 0.f;
}

// The unknown cells (k == 0) of a w x h plane take the values of a membrane stretched over the known ones: no
// detail of their own, and no step where they meet the known cells. Coarse to fine: the half-size plane is filled
// first and gives the start; a few sweeps of "the mean of the four neighbours" settle each size.
void fillSmooth(std::vector<float>& v, const std::vector<uint8_t>& k, int w, int h, int ch) {
    size_t unknown = 0;
    for (uint8_t f : k) if (!f) ++unknown;
    if (unknown == 0) return;
    if (unknown == k.size()) { std::fill(v.begin(), v.end(), 0.f); return; }   // nothing to go by
    const int w2 = (w + 1) / 2, h2 = (h + 1) / 2;
    std::vector<float> v2((size_t)w2 * h2 * ch, 0.f);
    std::vector<uint8_t> k2((size_t)w2 * h2, 0);
    {
        std::vector<int> n2((size_t)w2 * h2, 0);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                const size_t i = (size_t)y * w + x, j = (size_t)(y / 2) * w2 + x / 2;
                if (!k[i]) continue;
                for (int c = 0; c < ch; ++c) v2[j * ch + c] += v[i * ch + c];
                ++n2[j];
            }
        for (size_t j = 0; j < n2.size(); ++j)
            if (n2[j]) { for (int c = 0; c < ch; ++c) v2[j * ch + c] /= (float)n2[j]; k2[j] = 1; }
    }
    fillSmooth(v2, k2, w2, h2, ch);
    std::vector<int> u;
    u.reserve(unknown);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const size_t i = (size_t)y * w + x;
            if (k[i]) continue;
            u.push_back((int)i);
            const size_t j = (size_t)(y / 2) * w2 + x / 2;
            for (int c = 0; c < ch; ++c) v[i * ch + c] = v2[j * ch + c];
        }
    auto relax = [&](int i) {
        const int x = i % w, y = i / w;
        for (int c = 0; c < ch; ++c) {
            float s = 0;
            int n = 0;
            if (x > 0) { s += v[(size_t)(i - 1) * ch + c]; ++n; }
            if (x + 1 < w) { s += v[(size_t)(i + 1) * ch + c]; ++n; }
            if (y > 0) { s += v[(size_t)(i - w) * ch + c]; ++n; }
            if (y + 1 < h) { s += v[(size_t)(i + w) * ch + c]; ++n; }
            if (n) v[(size_t)i * ch + c] = s / n;
        }
    };
    for (int it = 0; it < 12; ++it) {
        if (it % 2 == 0) for (size_t q = 0; q < u.size(); ++q) relax(u[q]);
        else for (size_t q = u.size(); q-- > 0;) relax(u[q]);
    }
}
}  // namespace

void CompareWindow::showCloneMark() {
    const bool on = rt_.on && cloneAct_->isChecked() && clone_.have;
    view_[0]->setCloneMark(!on ? 0 : clone_.pending ? 1 : 2, clone_.pending ? clone_.src : QPointF(clone_.off));
    showBrushBlocked();
}

void CompareWindow::setCloneSource(QPointF p) {
    if (!rt_.on || !cloneAct_->isChecked()) return;
    clone_.have = clone_.pending = true;
    clone_.src = p;
    showCloneMark();
    statusBar()->showMessage(QString("Clone: copying from %1, %2. Paint over the spot; the place copied from moves with the brush. Alt + click picks another.")
                                 .arg((int)p.x()).arg((int)p.y()), 12000);
}

bool CompareWindow::ensureCloneLayer() {
    if (rt_.hasClone()) return true;
    try {
        const size_t n = (size_t)rt_.w * rt_.h;
        rt_.cl16.assign(n * 3, 0);
        rt_.sa.assign(n, 0);
        rt_.clDisp = QImage(rt_.w, rt_.h, QImage::Format_RGB32);
        if (rt_.clDisp.isNull()) throw std::bad_alloc();
        rt_.clDisp.fill(0);
        rt_.ca.assign(n, 0);
        if (!rt_.peek) view_[0]->setSecondMask(&rt_.ca);   // Show mask tints the copied pixels too
    } catch (const std::bad_alloc&) {
        rt_.cl16 = {}; rt_.sa = {}; rt_.ca = {}; rt_.clDisp = QImage();
        statusBar()->showMessage("Clone: this PC has not enough free memory for it on an image this large.", 12000);
        return false;
    }
    return true;
}

// One dab of the clone brush at p, on the screen only: the display takes the pixels from `off` away by the stroke's
// share so far. The layers are not touched until the stroke ends, so every dab copies the image as it was.
void CompareWindow::cloneDab(QPointF p) {
    const double rad = view_[0]->brushRadius(), hard = brushHard_->value();
    const int x0 = std::max(0, (int)std::floor(p.x() - rad)), x1 = std::min(rt_.w - 1, (int)std::ceil(p.x() + rad));
    const int y0 = std::max(0, (int)std::floor(p.y() - rad)), y1 = std::min(rt_.h - 1, (int)std::ceil(p.y() + rad));
    if (x0 > x1 || y0 > y1) return;
    const QRect r(x0, y0, x1 - x0 + 1, y1 - y0 + 1);
    growSnapshot(r);
    const QPoint off = clone_.off;
    for (int y = y0; y <= y1; ++y) {
        uint8_t* sa = rt_.sa.data() + (size_t)y * rt_.w;
        QRgb* rd = reinterpret_cast<QRgb*>(rt_.resDisp.scanLine(y));
        for (int x = x0; x <= x1; ++x) {
            const double d = std::hypot(x + 0.5 - p.x(), y + 0.5 - p.y()) / rad;
            if (d >= 1.0) continue;
            double a = d <= hard ? 1.0 : (1.0 - d) / std::max(1e-6, 1.0 - hard);
            a = a * a * (3 - 2 * a);   // smoothstep
            const int v = (int)std::lround(a * 255);
            if (v <= sa[x]) continue;
            sa[x] = (uint8_t)v;
            const int qx = x + off.x(), qy = y + off.y();
            if (qx < 0 || qy < 0 || qx >= rt_.w || qy >= rt_.h) continue;   // nothing to copy from there
            const QRgb b = dispAt(x, y), c = dispAt(qx, qy);
            rd[x] = qRgb((qRed(b) * (255 - v) + qRed(c) * v + 127) / 255, (qGreen(b) * (255 - v) + qGreen(c) * v + 127) / 255, (qBlue(b) * (255 - v) + qBlue(c) * v + 127) / 255);
        }
    }
    view_[0]->patch(r);
}

// The end of a clone stroke. The copy is the image `off` away plus, with Match tone, a smooth correction: the
// difference between the two places is taken in the ring around the stroke (averaged, so that it carries tone and no
// texture) and stretched across the stroke as a membrane. The copy then meets its surroundings in colour and
// brightness, and its detail is the copied place's. It goes into the clone layer by the stroke's share.
void CompareWindow::cloneCommit() {
    Stroke& s = rt_.current;
    rt_.cloning = false;
    if (s.r.isEmpty()) return;
    const QRect R = s.r, img(0, 0, rt_.w, rt_.h);
    composite(R);   // the preview out of the display: both images as they were before the stroke
    const QPoint off = clone_.off;
    const int W = rt_.w;
    const QRect Q = R.adjusted(-8, -8, 8, 8).intersected(img);
    const int w = Q.width(), h = Q.height();
    const bool match = matchAct_->isChecked();
    auto dispRow = [this](int y) { return reinterpret_cast<const QRgb*>(rt_.resDisp.constScanLine(y)); };
    std::vector<float> d16, dD;
    if (match) {
        d16.assign((size_t)w * h * 3, 0.f);
        dD.assign((size_t)w * h * 3, 0.f);
        std::vector<uint8_t> known((size_t)w * h, 0);
        for (int y = Q.top(); y <= Q.bottom(); ++y) {
            const int qy = y + off.y();
            if (qy < 0 || qy >= rt_.h) continue;
            const QRgb* bd = dispRow(y);
            const QRgb* sd = dispRow(qy);
            for (int x = Q.left(); x <= Q.right(); ++x) {
                const int qx = x + off.x();
                if (qx < 0 || qx >= W || rt_.sa[(size_t)y * W + x]) continue;   // (inside the stroke: to be found)
                const size_t i = (size_t)(y - Q.top()) * w + (x - Q.left());
                known[i] = 1;
                const uint16_t* b = rt_.res16.data() + ((size_t)y * W + x) * 3;
                const uint16_t* c = rt_.res16.data() + ((size_t)qy * W + qx) * 3;
                for (int k = 0; k < 3; ++k) d16[i * 3 + k] = (float)b[k] - (float)c[k];
                dD[i * 3] = (float)(qRed(bd[x]) - qRed(sd[qx]));
                dD[i * 3 + 1] = (float)(qGreen(bd[x]) - qGreen(sd[qx]));
                dD[i * 3 + 2] = (float)(qBlue(bd[x]) - qBlue(sd[qx]));
            }
        }
        blurKnown(d16, known, w, h, 3, 3);
        blurKnown(dD, known, w, h, 3, 3);
        fillSmooth(d16, known, w, h, 3);
        fillSmooth(dD, known, w, h, 3);
    }
    // the clone layer inside R as it was, for Undo
    const size_t n = (size_t)R.width() * R.height();
    s.ca.resize(n); s.cl.resize(n * 3); s.cd.resize(n);
    for (int y = 0; y < R.height(); ++y) {
        const size_t at = (size_t)(R.y() + y) * W + R.x(), row = (size_t)y * R.width();
        memcpy(s.ca.data() + row, rt_.ca.data() + at, R.width());
        memcpy(s.cl.data() + row * 3, rt_.cl16.data() + at * 3, (size_t)R.width() * 6);
        memcpy(s.cd.data() + row, reinterpret_cast<const QRgb*>(rt_.clDisp.constScanLine(R.y() + y)) + R.x(), (size_t)R.width() * sizeof(QRgb));
    }
    // (the result is read below and not written: a place copied from inside the stroke is still the image as it was)
    for (int y = R.top(); y <= R.bottom(); ++y) {
        const int qy = y + off.y();
        QRgb* cd = reinterpret_cast<QRgb*>(rt_.clDisp.scanLine(y));
        for (int x = R.left(); x <= R.right(); ++x) {
            const size_t p = (size_t)y * W + x;
            const int a = rt_.sa[p];
            if (!a) continue;
            rt_.sa[p] = 0;
            const int qx = x + off.x();
            if (qx < 0 || qy < 0 || qx >= W || qy >= rt_.h) continue;
            const size_t i = (size_t)(y - Q.top()) * w + (x - Q.left());
            const uint16_t* c16 = rt_.res16.data() + ((size_t)qy * W + qx) * 3;
            const QRgb cq = dispRow(qy)[qx];
            const int cdv[3] = {qRed(cq), qGreen(cq), qBlue(cq)};
            // the layer's pixel and share after this stroke: over the old ones by the stroke's share
            const int c = rt_.ca[p];
            const uint64_t wOld = (uint64_t)(255 - a) * c, wNew = (uint64_t)a * 255, den = wOld + wNew;
            uint16_t* cl = rt_.cl16.data() + p * 3;
            const int old[3] = {qRed(cd[x]), qGreen(cd[x]), qBlue(cd[x])};
            int nd[3];
            for (int k = 0; k < 3; ++k) {
                const float v16 = std::clamp((float)c16[k] + (match ? d16[i * 3 + k] : 0.f), 0.f, 65535.f);
                const float vd = std::clamp((float)cdv[k] + (match ? dD[i * 3 + k] : 0.f), 0.f, 255.f);
                cl[k] = (uint16_t)((wOld * cl[k] + wNew * (uint64_t)std::lround(v16) + den / 2) / den);
                nd[k] = (int)((wOld * old[k] + wNew * (uint64_t)std::lround(vd) + den / 2) / den);
            }
            cd[x] = qRgb(nd[0], nd[1], nd[2]);
            rt_.ca[p] = (uint8_t)((den + 127) / 255);
        }
    }
    composite(R);
    view_[0]->patch(R);
}

// The end of a paint stroke that ran over cloned pixels: their share goes down by the stroke's (composite() showed
// it so while the stroke ran)
void CompareWindow::eraseCommit() {
    Stroke& s = rt_.current;
    rt_.erasing = false;
    if (s.r.isEmpty() || !rt_.hasClone()) return;
    const QRect R = s.r;
    s.ca.resize((size_t)R.width() * R.height());
    for (int y = 0; y < R.height(); ++y) {
        uint8_t* ca = rt_.ca.data() + (size_t)(R.y() + y) * rt_.w + R.x();
        uint8_t* sa = rt_.sa.data() + (size_t)(R.y() + y) * rt_.w + R.x();
        memcpy(s.ca.data() + (size_t)y * R.width(), ca, R.width());
        for (int x = 0; x < R.width(); ++x) {
            if (!sa[x]) continue;
            ca[x] = (uint8_t)((ca[x] * (255 - sa[x]) + 127) / 255);
            sa[x] = 0;
        }
    }
}

QString CompareWindow::testPickLeft(const QString& name) {
    pickLeft(indexOf(name));
    return QString("retouch %1 on %2 | clone %3 | %4").arg(rt_.on ? "on" : "off", rt_.on ? entries_[rt_.target].name : QString("-"), cloneAct_->isChecked() ? "on" : "off", statusBar()->currentMessage());
}

// a stroke begun with the mouse, and a tool switched on (as its shortcut would) before the button is let go
QString CompareWindow::testBrokenStroke() {
    QWidget* vp = view_[0]->viewport();
    auto send = [vp](QEvent::Type t, QPointF p, Qt::MouseButton b, Qt::MouseButtons held) {
        QMouseEvent e(t, p, vp->mapToGlobal(p), b, held, Qt::NoModifier);
        QCoreApplication::sendEvent(vp, &e);
    };
    const QPointF a(vp->width() * 0.4, vp->height() * 0.5), b(vp->width() * 0.5, vp->height() * 0.5);
    const size_t before = rt_.undo.size();
    send(QEvent::MouseButtonPress, a, Qt::LeftButton, Qt::LeftButton);
    send(QEvent::MouseMove, b, Qt::NoButton, Qt::LeftButton);
    measureAct_->setChecked(true);
    const QString said = QString("strokes %1 -> %2 | in a stroke: %3").arg(before).arg(rt_.undo.size()).arg(rt_.inStroke ? "STILL" : "no");
    send(QEvent::MouseButtonRelease, b, Qt::LeftButton, Qt::NoButton);
    measureAct_->setChecked(false);
    return said;
}

QString CompareWindow::testClone(QPointF from, QPointF a, QPointF b, bool matchTone) {
    cloneAct_->setChecked(true);
    matchAct_->setChecked(matchTone);
    setCloneSource(from);
    emit view_[0]->strokeBegin(a);
    emit view_[0]->strokeTo(b);
    emit view_[0]->strokeEnd();
    return QString("clone %1 | offset %2, %3 | strokes %4 | layer %5 | %6").arg(cloneAct_->isChecked() ? "on" : "off").arg(clone_.off.x()).arg(clone_.off.y())
        .arg(rt_.undo.size()).arg(rt_.hasClone() ? "there" : "none").arg(statusBar()->currentMessage());
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
