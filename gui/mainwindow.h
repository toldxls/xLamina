#pragma once

#include <QCheckBox>
#include <QColorSpace>
#include <QComboBox>
#include <QDialog>
#include <QDoubleSpinBox>
#include <QElapsedTimer>
#include <QFormLayout>
#include <QGroupBox>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMainWindow>
#include <QMap>
#include <QMenu>
#include <QPlainTextEdit>
#include <QPointer>
#include <QProgressBar>
#include <QPushButton>
#include <QSet>
#include <QSpinBox>
#include <QSplitter>
#include <QTabBar>
#include <QTimer>
#include <QToolButton>

#include <memory>
#include <mutex>

#include "benchmark.h"
#include "compare.h"
#include "depthview.h"
#include "imageview.h"
#include "prefs.h"
#include "sweepdialog.h"
#include "hwmon.h"
#include "updater.h"
#include "worker.h"

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow() override;
    // only, when given, holds file names of the folder: the stack is those files and nothing else of the folder
    void openFolder(const QString& dir, const QStringList& only = {});
    void openFiles(const QStringList& paths);   // File > Open files, or several image files dropped: a stack of just these
    bool rendering() const { return worker_.running(); }
    void startBatch(const QStringList& dirs);   // render these folders one after the other (File > Batch render's Start)
    bool batchRunning() const { return batchRunning_; }
    void autoBenchmark(bool sample, const QString& reportPath);   // developer aid: run the benchmark unattended, quit when done
    void notifyUpdated(const QString& from);                      // the first start after an update: say so
    void selfUpdate();                                            // developer aid: update to the latest release without asking
    void probeHover(double fx, double fy);                        // developer aid: the last tab, and a mouse move at that fraction of the view

protected:
    void dragEnterEvent(QDragEnterEvent* e) override;
    void dropEvent(QDropEvent* e) override;
    void closeEvent(QCloseEvent* e) override;
    bool eventFilter(QObject* o, QEvent* e) override;   // the middle button on a comparison tab closes it

public slots:
    void render();

private slots:
    void chooseFolder();
    void cancel();
    void saveCurrent();
    void addComparison();
    void twoUp();          // the two-up viewer with this render's outputs and the comparison images
    void retouch();        // the two-up viewer with retouching switched on (the centre buttons above the view)
    void openSweep();      // a sweep folder into the two-up viewer
    void runSweep();       // the sweep dialog, then the worker renders the variants and the viewer opens on them
    void openOutputFolder();
    void benchmark();
    void batch();          // File > Batch render: a queue of folders rendered one after the other
    void batchNext();      // open and render the queue's next folder, or finish
    void preferences();                        // File > Preferences: General, Performance, System
    void about();
    // the dust map (dust.h): a white-background frame whose dust is taken out of every frame before stacking
    void chooseDustMap();                      // File > Set dust map
    bool setDustMap(const QString& path, bool quiet);   // read it and find its dust; an empty path drops it
    void reportProblem();                     // Help > Report a problem: the logs, the PC's details and the settings as one file
    void logRunStart(const QString& what, const stackcore::JobOptions& jo);   // the run's block in the session's log
    QString failureMessage(const JobResultData& d) const;                     // a failed job in words, with the next step
    void openManual();                         // Help > Manual (F1): manual\index.html beside the program, in the browser
    void checkForUpdates();                    // Help > Check for updates: the result in a box
    void showUpdate();                         // the dialog: what changed, Update and restart / Skip / Later
    void onUpdateAvailable(const ReleaseInfo& r);
    void resetSettings();
    void showFrame(int row);
    void showOutput(int tab);
    void stepOutput(int delta);
    void onProgress(QString stage, double fraction);
    void onPreview(QString name, QImage image);
    void onLiveView(int k, QImage image, QRect win, QSize full);
    void updateViewWindow();                   // the viewer's visible part, as fractions, for the engine's frame views
    void onThumbnail(int index, QImage image);
    void onFullFrame(int index, QImage image);
    void onComparison(int index, QImage image);
    void onLog(QString line);
    void onFinished(std::shared_ptr<JobResultData> data);
    void dropShifted();   // untick the frames whose shift from the previous frame is an outlier
    void dropSoft();      // untick the frames whose focus energy fell under their neighbours'
    void dropWarped();    // untick the frames whose rolling-shutter displacement is an outlier
    void frameMenu(const QPoint& pos);
    void tickAllFrames();
    void syncAllFramesBox();   // the master tick above the list mirrors the rows: all, none or some ticked
    void sortFrames(bool byTime);   // the list (= the stack order) by file name, or by the capture time (EXIF; file time when absent)

private:
    friend class ManualShots;   // developer aid: the manual's screen grabs (manualshots.h)
    std::shared_ptr<const stackcore::DustFix> dust_;   // the dust map in use (null: none, or not read yet)
    QString dustPath_;
    QLabel* dustLabel_ = nullptr;              // the Render panel's Dust map row
    QToolButton* dustClear_ = nullptr;
    void buildMenus();
    void buildUi();
    void loadSettings();
    void saveSettings();
    void updateRecent();
    static QStringList frameFilesIn(const QString& dir);   // the folder's TIFF, JPEG and raw files, by name
    void batchDialog(const QStringList& preset);           // the queue's dialog, with these folders already in it
    void batchSkip(const QString& why);                     // the current batch folder cannot be rendered: note it
    void batchFinish(bool cancelled);
    void resumeThumbnails();
    void warmCache();      // point the thumbnail loader at the frame cache (or away from it) for the open folder
    void setStage(const QString& text);   // the stage text, drawn inside the progress bar
    // the frame list's per-frame lines: the wobble (shift from the previous frame, alignment) and, when the render ran
    // pass 1, the sharpness (focus energy against the neighbours', StackOptions::Result::frame_energy); outliers flagged
    void labelWobble(const stackcore::AlignOut& A, const stackcore::StackOptions::Result* R = nullptr);
    void setBusy(bool busy);
    void setStatusGeometry(const JobResultData& d);
    void setTheme(const QString& theme);
    // developer controls (View menu, off by default): the Advanced group, the two radii and the outputs beyond B and C
    void setDeveloper(bool on);
    std::vector<std::string> selectedOutputs() const;   // the ticked outputs that are on show
    bool buildJob(stackcore::JobOptions& jo, bool interactive);   // the job as the panel describes it; false when it cannot run
    PerfSettings perfFromUi() const;
    void perfToUi(const PerfSettings& p);
    void rebuildTabs();
    QImage outputImage(const QString& name);
    QImage forDisplay(QImage img) const;   // the frames' colour space -> sRGB
    // the depth map as a tab (developer controls): its colours (the bar under the tabs, shared with the two-up
    // viewer) and the frame under the cursor
    bool hasDepth() const;
    void setDepthStyle(const DepthStyle& st, bool fromTwoUp);
    void depthHover(int x, int y);
    void updateDepthHighlight();           // the frame under the cursor marked across what the view shows
    int hoverFrame_ = -1;                  // that frame, -1 = none
    bool highlightOn_ = false;
    std::vector<uint16_t> depthIdx_;       // the depth map rounded to frames, built when first needed
    DepthStyle depthStyle_;
    DepthBar* depthBar_ = nullptr;

    QString folder_;
    QStringList frames_;
    QStringList frameTimes_;   // per frame: the capture time from the file's EXIF ("YYYY:MM:DD HH:MM:SS[.sss]"), empty when none
    // rows carry the frame index (into frames_) in Qt::UserRole, -1 for files left out; the list order is
    // the stack order and a frame may be listed more than once, so jobs refer to rows, not frames
    QList<QListWidgetItem*> jobItems_;     // the rows of the current / last render, in stack order
    QList<QListWidgetItem*> shifted_;      // rows whose wobble (shift from the previous frame) is an outlier
    QList<QListWidgetItem*> soft_;         // rows whose focus energy fell under their neighbours' (blurred frames)
    QList<QListWidgetItem*> warped_;       // rows whose band-to-band displacement is an outlier (shaken during the readout)
    QString wobble_;                       // the wobble + sharpness summary line
    QAction* tickAll_ = nullptr;
    QAction* sortName_ = nullptr;
    QAction* sortTime_ = nullptr;
    QAction* dropShifted_ = nullptr;
    QAction* dropSoft_ = nullptr;
    QAction* dropWarped_ = nullptr;   // the rolling-shutter outliers (FINDINGS 51)
    QAction* saveAct_ = nullptr;
    QAction* openOutAct_ = nullptr;
    QAction* renderAct_ = nullptr;
    QAction* cancelAct_ = nullptr;
    QMenu* recentMenu_ = nullptr;
    QStringList recent_;
    // the batch queue (File > Batch render): folders still to render, the one in work, a line per folder done
    QStringList batchQueue_, batchLog_;
    QString batchCurrent_;
    int batchTotal_ = 0, batchDone_ = 0;
    bool batchRunning_ = false, batchSkipped_ = false;
    QElapsedTimer batchClock_;
    QAction* batchAct_ = nullptr;
    QString theme_ = "dark";
    bool developer_ = false;
    QAction* devAct_ = nullptr;
    QFormLayout* mainForm_ = nullptr;      // the Render panel's form (Smoothing and the radii rows)
    QColorSpace colorSpace_;               // the frames' ICC profile (invalid = assume sRGB)
    bool selecting_ = false;
    ImageView* liveView_ = nullptr;        // right pane while rendering
    QSplitter* split_ = nullptr;
    ThumbLoader thumbs_;
    QSet<int> thumbsDone_;                 // frames whose thumbnail has arrived (the loader pauses during a render)
    ThumbLoader frameLoader_;              // one frame at full resolution
    ThumbLoader compareLoader_;            // a comparison image (another stacker's output) at full resolution
    QString pendingCompare_;
    int pendingFrame_ = -1;
    QComboBox* liveRes_ = nullptr;
    QComboBox* interp_ = nullptr;      // the warp's resampling filter (Grid::interp)
    std::shared_ptr<JobResultData> result_;
    QString jobTag_;                   // the running render's settings for file names (interpolation, smoothing, radii)
    QMap<QString, QImage> cache_;          // 8-bit versions of outputs and previews
    QMap<QString, QImage> compare_;        // comparison images by tab name; they survive renders
    QString current_;                      // name of the image on screen
    QElapsedTimer elapsed_;
    QTimer tick_;
    QString stageText_;
    QString progressStatus_;               // status-bar count for the per-frame stages ("Read 194 of 200 frames"), from the progress signal
    // the live view: the frame being processed is marked in the list (a tint, moved 4 times a second, never
    // the selection, which jumped about and ended on a random frame) and shown at most ~30 times a second
    QListWidgetItem* liveItem_ = nullptr;
    QListWidgetItem* markedItem_ = nullptr;
    QTimer listTick_;
    QElapsedTimer lastView_;
    QPointer<BenchmarkDialog> bench_;
    QPointer<CompareWindow> twoUp_;
    QMenu* devOutMenu_ = nullptr;      // Tools > Developer outputs: A and B0, mirrored to hidden check boxes
    QList<GpuInfo> gpus_;
    // updates: checked against GitHub's latest release after the start (Help > Check for updates at start); a
    // newer one lights the toolbar button, whose dialog downloads and swaps the files in and restarts
    Updater updater_;
    ReleaseInfo update_;               // the newer release, once found
    QToolButton* updateBtn_ = nullptr;
    QAction* updateAct_ = nullptr;     // the button's place in the toolbar: a toolbar widget shows and hides through its action
    QAction* autoUpdateAct_ = nullptr;
    QPointer<QDialog> updateDlg_;
    QPointer<QDialog> prefsDlg_;       // File > Preferences, built once
    HwMonitor* hw_ = nullptr;          // the live hardware panel (bottom dock), raised when a render starts
    // the live view's window: what the viewer shows as fractions of the image, and the viewport's width in pixels,
    // read by the engine's threads (under the mutex) to render each frame view at the screen's resolution
    std::mutex viewWinM_;
    QRectF viewFrac_{0, 0, 1, 1};
    int viewportW_ = 1600;
    QDockWidget* hwDock_ = nullptr;
    QGroupBox* perfGroup_ = nullptr;   // the hardware settings, built with the Render column, shown in Preferences
    bool updateManual_ = false;        // the running check came from the menu: report its result either way

    ImageView* view_ = nullptr;
    QListWidget* frameList_ = nullptr;
    QDockWidget* framesDock_ = nullptr;
    QCheckBox* allFrames_ = nullptr;   // the master tick in line with the rows' ticks
    QTabBar* tabs_ = nullptr;
    QPlainTextEdit* log_ = nullptr;
    QProgressBar* progress_ = nullptr;
    QLabel* stage_ = nullptr;
    QLabel* status_ = nullptr;
    QLabel* geometry_ = nullptr;
    QPushButton* renderBtn_ = nullptr;
    QPushButton* cancelBtn_ = nullptr;
    QPushButton* saveBtn_ = nullptr;
    QComboBox* scale_ = nullptr;
    QComboBox* ref_ = nullptr;
    QCheckBox* lockScale_ = nullptr;
    QCheckBox* fixRs_ = nullptr;        // correct the rolling-shutter warp in every warp (FINDINGS 51)
    QCheckBox* fixBanding_ = nullptr;   // correct the LED flicker's row banding
    QDoubleSpinBox* fov_ = nullptr;
    QDoubleSpinBox* smoothing_ = nullptr;
    QDoubleSpinBox* softCutoff_ = nullptr;   // Preferences > General: the soft-frame flag's cutoff (FINDINGS 49)
    QDoubleSpinBox* cradius_ = nullptr;
    QDoubleSpinBox* bradius_ = nullptr;
    QLineEdit* outDir_ = nullptr;
    QMap<QString, QCheckBox*> outputs_;
    // performance
    QList<QCheckBox*> gpuBoxes_;
    QSpinBox* streams_ = nullptr;
    QSpinBox* readers_ = nullptr;
    QSpinBox* pairThreads_ = nullptr;
    QCheckBox* cacheFrames_ = nullptr;
    // advanced depth / composition tuning (the validated defaults unless the group is ticked)
    QGroupBox* advanced_ = nullptr;
    QComboBox* depthRes_ = nullptr;
    QDoubleSpinBox* focusRadius_ = nullptr;
    QDoubleSpinBox* focusPower_ = nullptr;
    QDoubleSpinBox* depthFmPx_ = nullptr;
    QDoubleSpinBox* volAggPx_ = nullptr;
    QSpinBox* refineR_ = nullptr;
    QDoubleSpinBox* refineFmPx_ = nullptr;
    QDoubleSpinBox* softPx_ = nullptr;
    QDoubleSpinBox* haloPrior_ = nullptr;
    QDoubleSpinBox* darkPrior_ = nullptr;
    QDoubleSpinBox* darkReach_ = nullptr;
    QDoubleSpinBox* flatBlur_ = nullptr;
    Worker worker_;
};
