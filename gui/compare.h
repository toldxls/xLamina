// The compare window (the "two-up viewer" of the comments): two to four panes with one pan and zoom, for judging B variants and other stackers' outputs at 1:1.
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
#include <QPushButton>
#include <QSplitter>
#include <QStatusBar>
#include <QTableWidget>
#include <QToolBar>
#include <QToolButton>

class QCheckBox;

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "depthview.h"
#include "imageview.h"
#include "worker.h"

// Where `right` lies in `left` when both show the same subject at another crop or scale (two programs' stacks of the
// same frames): left px = scale * right px + pos. Found on the images' structure (luma with its local mean removed),
// coarse to fine, so a tone curve or another stacking method does not matter; no rotation. False when nothing lines up.
// match: the correlation of the two at the place found (1 = the same picture).
// fixedScale: the two are at the same scale already; only the shift is looked for.
bool estimatePlacement(const QImage& left, const QImage& right, double& scale, QPointF& pos, double* match = nullptr, bool fixedScale = false);

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
    void setRetouch(bool on);   // retouching on or off, as the R toolbar button (the main window's Retouch button)
    // the render's depth map (developer controls): a "depth" entry drawn in `st`, the Depth toolbar, the frame under
    // the cursor in the status bar and the overlay. null = no depth map (the toolbar hides)
    void setDepth(std::shared_ptr<JobResultData> r, const DepthStyle& st);
    void setDepthStyle(const DepthStyle& st);   // from the main window's controls: no signal back
    // the scale for measuring and the scale bar: an image of `size` pixels is `um` micrometres wide (the render's
    // outputs, from Field of view). Images of another size have no scale: their lengths are in pixels. um 0 = none
    void setScale(double um, QSize size);
    void setScaleBar(bool on);
    void setUnits(int measure, int bar);        // the units of the measured lines and of the scale bar (ImageView::Unit): no signal back
    // Refine: regions marked on one of the render's outputs (a rectangle or a lasso outline each; the output in the
    // left pane: B, C, CBn …) are rendered again with a recipe
    // and laid into the retouch result; everything outside them stays as it is. `render` is the render whose outputs
    // the list holds (null = none: Refine is not offered), `space` the frames' colour space, for the patches on screen
    void setRender(std::shared_ptr<JobResultData> render, const QColorSpace& space);
    // says whether the graphics cards are free (no render running): lining a frame up with the render uses them
    void setGpuIdle(std::function<bool()> idle) { gpuIdle_ = std::move(idle); }
    void regionsProgress(const QString& stage, double fraction);    // the running Refine's stage
    void regionsDone(std::shared_ptr<JobResultData> data);          // its end: data->regions holds the tiles
    // Make composite: the images ticked in the list (two or more, of one size) are stacked again as if they were
    // frames, with the Render panel's settings and outputs; the results join the list named by what went in ("2B1C")
    void compositeDone(std::shared_ptr<JobResultData> data);        // its end: data->result holds the outputs
    void compositeFailedToStart(const QString& why);                // the main window could not start it
    void testComposite(const QStringList& names);                   // developer aid: these entries ticked, the button pressed
    // the methods offered for a composite: B, C and CBn always, the others with the developer controls; and a
    // method's description, for its tick box
    void setMoreMethods(bool on);
    void setMethodHelp(const QString& name, const QString& help);
    void refineFailedToStart(const QString& why);                   // the main window could not start it
    bool refining() const { return refining_; }
    // developer aids (unattended tests): a region by its outline in image pixels, the Refine button, a region's
    // tick, and Save result without the dialog
    int addRegionPx(const QPolygonF& outline, int strength);
    // ... and one drawn as the mouse would: the tool (1 rectangle, 2 lasso) switched on and the left button dragged
    // from a to b on the left pane (an ellipse between them for the lasso), as fractions of the pane
    void testDraw(int mode, QPointF a, QPointF b);
    void refine();
    void setRegionOn(int index, bool on);
    bool saveResultTo(const QString& fn);
    int regionCount() const { return (int)regions_.size(); }
    // ... a file opened and shown on the right with Retouch on, and a brush stroke from a to b (image pixels); each
    // returns what the window then shows and says
    QString testSource(const QString& path, const QString& savePng = {});
    QString testStroke(QPointF a, QPointF b);
    // ... and a clone stroke: the place copied from picked as Alt + click would, then the brush dragged from a to b
    QString testClone(QPointF from, QPointF a, QPointF b, bool matchTone);
    void testUndo() { retouchUndo(); }
    ImageView* testView(int pane) { return view_[std::clamp(pane, 0, kPanes - 1)]; }   // a pane's view (0 = the left one), for the manual's grabs
    QString testPickLeft(const QString& name);   // ... the image picked for the left pane, as a double-click in the list picks it
    QString testBrokenStroke();                  // ... a stroke cut short by a tool switched on during the drag
    QString testSaveName() const { return retouchedName(); }
    // ... the newest image of the list (a composite just made) retouched from itself: what Save result would call it
    QString testRetouchLast() { setRetouch(false); showIn(0, entries_.size() - 1); showIn(1, entries_.size() - 1); setRetouch(true); return entries_.isEmpty() ? QString() : entries_.last().name + " -> " + retouchedName(); }
    void testCloneOff() { cloneAct_->setChecked(false); }
    // ... the number of panes, every frame added to the list, and Frame here at an image position
    void testPanes(int n) { setPaneCount(n); }
    QString testActive(int pane) { setActive(pane); return QString("active pane %1 | source %2 | blocked %3 | %4").arg(active_ + 1).arg(rt_.on && rt_.source >= 0 ? entries_[rt_.source].name : QString("-")).arg(rt_.blocked ? 1 : 0).arg(statusBar()->currentMessage()); }
    int testAddFrames() { addFrames(); return (int)entries_.size(); }
    QString testHideFrames(bool hidden);   // ... the frames hidden or shown: what the list and the button then say
    QString testTool(int mode);            // ... a region tool pressed (1 rectangle, 2 lasso): whether it came on, and what the window says
    QString testFrameHere(int x, int y);
    QString testStep(int delta) { step(active_, delta); return QString("pane %1: %2 | %3").arg(active_ + 1).arg(sel_[active_] >= 0 ? entries_[sel_[active_]].name : QString("-"), statusBar()->currentMessage()); }
    // one of the stack's frames (by its file) shown in the pane the list fills, lined up with the render when it is
    // one of the render's (the main window's frame list)
    void showFrame(const QString& path);

signals:
    void refineRequested(std::shared_ptr<RegionJobData> job);   // to the main window, which owns the worker
    void refineCancelRequested();
    void refineFinished(bool ok);               // a Refine ended (developer aids)
    void compositeRequested(std::shared_ptr<CompositeJobData> job);   // to the main window, which owns the worker
    void compositeFinished(bool ok);            // a composite ended (developer aids)
    void helpRequested(QString section);        // Help: the manual at that section
    void unitsChanged(int measure, int bar);    // chosen here (the Units box, a click on the scale bar)
    void depthStyleChanged(DepthStyle st);      // the Depth toolbar changed it

protected:
    void keyPressEvent(QKeyEvent* e) override;
    void keyReleaseEvent(QKeyEvent* e) override;
    void dragEnterEvent(QDragEnterEvent* e) override;
    void dropEvent(QDropEvent* e) override;
    void closeEvent(QCloseEvent* e) override;
    bool eventFilter(QObject* o, QEvent* e) override;

private slots:
    void onLoaded(int index, QImage image);

private:
    static constexpr int kPanes = 4;   // the most panes; nPanes_ of them are on show
    struct Entry {
        QString name, path, note;
        QImage image;      // empty until loaded (files) or always set (addImage)
        bool loading = false;
        bool fileBacked = true;   // false: the image came in memory (path, if any, is only for the 16-bit data)
        // a frame of the render, read and warped onto the render's grid (alignedFrame): its 16-bit pixels, the render
        // it was made for and the frame's place in that render's list. The pixels of the older ones are let go and
        // made again when the entry is shown (image empty, px16 empty)
        std::shared_ptr<std::vector<uint16_t>> px16;
        const void* owner = nullptr;
        int frame = -1;
        QString frameFile;    // the frame's file (the entry itself has no path: its pixels are the lined-up ones)
        bool pick = false;    // ticked for Make composite
        double fovMm = 0;     // the width the image stands for, mm, when neither its name nor its size says it (a composite: its reference's)
        double change = -1;   // a sweep variant: % of B's pixels differing > 2 % from the base (sweep.json change_pct); -1 = unknown
    };
    int indexOf(const QString& name) const;
    // the width an image stands for, in mm; 0 = not known. From its file name ("..._1.46mm", as Save as writes it),
    // what it was made from (a composite), or its size being the render's outputs' (the Field of view of the main window)
    double fovOf(const Entry& e) const;
    QString retouchedName() const;   // the name Save result offers
    void addEntry(Entry e);
    void rebuildList();
    void showIn(int pane, int index);
    void pickLeft(int index);    // the user picked an image for the left pane: while retouching, the retouch moves to it
    // a control could not do what was asked: why, in the status bar and beside the cursor (the line at the bottom of
    // the window alone goes unseen, and the control looks broken)
    void refuse(const QString& why, int ms = 9000);
    void refresh(int pane);
    void flip();
    void step(int pane, int delta);
    void requestLoad(int index);
    void startNextLoad();
    void trimCache();
    void openImageDialog();
    void openSweepDialog();
    void updateOverlay();        // the difference highlight over both panes, when it is on and both images are loaded
    // ---- the depth map
    void recolourDepth();        // the "depth" entry redrawn in depthStyle_
    bool depthOverlay();         // the map over the panes that show another image of its size; false when it is off
    void depthHover(int pane, int x, int y);
    std::shared_ptr<JobResultData> depth_;
    DepthStyle depthStyle_;
    DepthBar* depthBar_ = nullptr;
    QToolBar* depthTb_ = nullptr;
    bool depthOv_[kPanes] = {};   // the pane's overlay pixmap is the current depth image
    // the depth map's controls, crosshair, note and overlay are there while a pane shows the depth map, and only then
    bool depthShown() const;
    void updateDepthTools();
    // a layer held on show, as in the main window: a click on either pane holds the frame the map chose there, its
    // pixels marked over every pane on the render's grid; the arrow keys and PgUp / PgDn step, Esc lets go
    int heldFrame_ = -1;
    bool layerMarked_ = false;
    std::vector<uint16_t> depthIdx_;     // the map rounded to frames
    std::vector<char> depthHas_;         // frames the map uses at all
    void holdLayer(int k);               // -1 = let go
    void stepLayer(int delta);
    bool markLayer();                    // the held layer over the panes; false when none is held

    // ---- retouch
    // the mask before the stroke, inside r; and, when the stroke changed the clone layer there, that layer before it
    struct Stroke { QRect r; std::vector<uint8_t> mask, ca; std::vector<uint16_t> cl; std::vector<QRgb> cd; };
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
        bool blocked = false;   // the right pane shows an image that cannot be painted from (another size): the brush paints nothing
        bool baked = false, dropped = false;   // unsaved strokes that Undo no longer reaches: baked into the target, or past the undo depth
        QPointF last;
        std::vector<uint8_t> icc;
        // the target before any region's patch (kept from the first patch on): the target is rebuilt from it and the
        // patches that are switched on
        std::vector<uint16_t> base16;
        QImage baseDisp;
        // the clone layer: parts of the result copied to another place of it, laid over the painted result with
        // their share ca (0..255). Empty until the first clone stroke. sa is the running stroke's own share
        std::vector<uint16_t> cl16;
        QImage clDisp;
        std::vector<uint8_t> ca, sa;
        bool cloning = false;   // the running stroke is a clone stroke
        bool erasing = false;   // the running stroke paints from the source over cloned pixels: they give way under it
        bool hasClone() const { return !ca.empty(); }
    } rt_;
    // the clone brush: copies from the place `off` away from the brush (set by the first stroke after a place was picked)
    struct Clone { bool have = false, pending = false; QPointF src; QPoint off; } clone_;
    QAction* cloneAct_ = nullptr;
    QAction* matchAct_ = nullptr;
    void setCloneSource(QPointF p);     // Alt + click
    bool ensureCloneLayer();
    void cloneDab(QPointF p);           // the running clone stroke on the screen
    void cloneCommit();                 // ... fitted to its surroundings and laid into the clone layer
    void eraseCommit();                 // a paint stroke over cloned pixels: their share taken down
    QRgb dispAt(int x, int y) const;    // the result's pixel on screen, from its layers (as composite makes it)
    void showCloneMark();
    QToolBar* mainTb_ = nullptr;
    QToolBar* retouchTb_ = nullptr;
    void markRows();                    // each toolbar row in its section's colour, stronger while its mode is on

    // ---- regions (Refine)
    struct Region {
        QPolygonF poly;                 // the outline, image pixels
        int strength = 1;               // index into the recipe's strengths
        int state = 0;                  // 0 marked, 1 being rendered, 2 rendered, 4 failed
        bool on = true;                 // rendered and laid into the result
        int rendered = -1;              // the strength the patch was rendered with
        int edge = 0;                   // the soft edge's width, px: set when the region is sent to be rendered (edgeFor)
        QRect r;                        // the patch: its rectangle, 16-bit and display pixels, and its share per pixel
        std::vector<uint16_t> px16;
        QImage disp;
        std::vector<uint8_t> alpha;
        bool hasPatch() const { return !alpha.empty(); }
    };
    std::vector<Region> regions_;
    std::function<bool()> gpuIdle_;
    QList<int> alignedRecent_;   // the lined-up frames that hold their pixels, oldest first
    // The stack's frames in the list: a row per frame, its pixels made when it is first shown (the frame read and
    // warped as the render placed it) and let go again when enough newer ones are held.
    int frameEntry(int k) const;      // the row of the render's frame k; -1 = none
    int ensureFrameEntry(int k);      // ... made when there is none (the list is not rebuilt: the caller does that)
    int adoptFrame(int i);            // entry i is a file of one of the render's frames: it becomes that frame's row (or the row there is, returned)
    bool loadFrame(int i);            // the pixels of a frame's row; false (and why, in the status bar) when they cannot be made now
    void addFrames();                 // a row for every frame of the render
    QPushButton* framesBtn_ = nullptr;   // above the list: the frames' rows hidden or shown
    bool framesHidden_ = false;
    void setFramesHidden(bool hidden);
    void applyFramesHidden();         // the rows and the pickers' items as framesHidden_ says, and the button's text
    void frameHere(int pane, int x, int y);   // the frame sharpest at that spot, into the active pane
    int sharpestAt(int x, int y, int* second, int* how) const;   // how: 1 measured on the frames, 0 the depth map's (frames not in memory), 2 the depth map's (nothing stands out there)
    QAction* frameHereAct_ = nullptr;
    std::shared_ptr<JobResultData> render_;
    QColorSpace space_;
    bool refining_ = false;
    bool compositing_ = false, fillingList_ = false;
    QPushButton* compositeBtn_ = nullptr;
    static constexpr int kFirstMethods = 3;     // B, C, CBn: on offer to everyone, and ticked to begin with
    QToolButton* methodsBtn_ = nullptr;         // "Methods: B, C, CBn": its menu holds a tick box per method
    QList<QCheckBox*> methodBox_;
    QList<QAction*> methodAct_;
    bool moreMethods_ = false;
    QStringList compositeMethods() const;       // the ticked methods on offer
    QString compositeTemp_;                     // the running composite's folder of copies
    void makeComposite();
    void updateCompositeButton();
    QString compositeLabel(const QList<int>& picks) const;   // "2B1C": how many of each kind
    std::vector<std::pair<int, int>> jobMap_;   // per region of the running job: its tile and the variant's index there (-1, -1: not in it)
    QToolBar* regionTb_ = nullptr;
    QAction* rectAct_ = nullptr;
    QAction* lassoAct_ = nullptr;
    QAction* refineAct_ = nullptr;
    QAction* refineCancelAct_ = nullptr;
    QAction* removeRegionAct_ = nullptr;
    QAction* clearRegionsAct_ = nullptr;
    QComboBox* strength_ = nullptr;
    QDoubleSpinBox* feather_ = nullptr;
    QTableWidget* regionList_ = nullptr;
    bool fillingRegions_ = false;
    void regionTool(int mode);          // 0 off, 1 rectangle, 2 lasso: the toolbar's two buttons
    bool regionsReady();                // the left pane is one of the render's outputs, retouching on (switched on when it can be)
    bool refinable(int i) const;        // entry i is an output of the render that a region can be rendered of
    QLabel* regionLabel_ = nullptr;     // "Refine regions of CBn:"
    void nameRegionRow();
    void restBrush();                   // the brush on while retouching, unless a tool has the left button
    void regionDrawn(int pane, const QPolygonF& frac);
    void removeRegion(int i);
    void clearRegions();
    void buildPatch(Region& g, const stackcore::RegionTile& t, const std::vector<float>& b);
    void rebuildTarget(const QRect& r); // the target inside r from the base and the patches switched on, then the result
    int edgeFor(const Region& g) const; // the Edge box's width, or the automatic one for this region's size
    void showRegions();                 // the outlines in both panes and the list's rows
    int selectedRegion() const;
    void setRefining(bool on);
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
    // Pane 0 is the left one (the result, when retouching), pane 1 the right one (the source); panes 2 and 3 show
    // further images at the same place and zoom. Three stand side by side (or under each other), four as two rows.
    ImageView* view_[kPanes] = {};
    QComboBox* pick_[kPanes] = {};
    QWidget* paneBox_[kPanes] = {};
    int sel_[kPanes] = {-1, -1, -1, -1};
    int nPanes_ = 2;
    // the pane a click in the list fills, and the one painted from while retouching: the right one, or pane 3 / 4
    // once it was clicked (never the left one, which is the result)
    int active_ = 1;
    QSplitter* split2_ = nullptr;   // the second row of a four-pane view
    QSplitter* rows_ = nullptr;
    QAction* verticalAct_ = nullptr;
    QComboBox* paneCount_ = nullptr;
    void setPaneCount(int n);
    void setActive(int pane);
    void markActive();           // the active pane's outline; while retouching it is the pane painted from: a faint frame and "(source)"
    bool onShow(int i) const;    // some pane shows entry i
    void showNames();            // the status bar's line of what the panes show
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
    // Align: the right pane's image moved and scaled onto the left one's (estimatePlacement), found once per pair
    QAction* alignAct_ = nullptr;
    struct { qint64 left = 0, right = 0; bool ok = false; double scale = 1; QPointF pos; double match = 0; } align_;
    void applyAlign();
    void alignHere();   // the shift found again on the part of the images on screen: exact there, where the whole-image fit is a few pixels off
    QAction* alignHereAct_ = nullptr;
    QAction* measureAct_ = nullptr;
    QAction* clearMeasuresAct_ = nullptr;
    QComboBox* measureUnit_ = nullptr;   // the measured lines' unit
    double scaleUm_ = 0;
    QSize scaleSize_;
    void applyScale(int pane);
    QAction* retouchAct_ = nullptr;
    QAction* maskAct_ = nullptr;
    QAction* undoAct_ = nullptr;
    QAction* resetAct_ = nullptr;
    QAction* saveAct_ = nullptr;
    QDoubleSpinBox* brushSize_ = nullptr;
    QDoubleSpinBox* brushHard_ = nullptr;
};
