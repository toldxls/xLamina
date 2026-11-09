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

#include <memory>

#include "benchmark.h"
#include "compare.h"
#include "imageview.h"
#include "prefs.h"
#include "sweepdialog.h"
#include "updater.h"
#include "worker.h"

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent = nullptr);
    void openFolder(const QString& dir);
    bool rendering() const { return worker_.running(); }
    void autoBenchmark(bool sample, const QString& reportPath);   // developer aid: run the benchmark unattended, quit when done
    void notifyUpdated(const QString& from);                      // the first start after an update: say so
    void selfUpdate();                                            // developer aid: update to the latest release without asking

protected:
    void dragEnterEvent(QDragEnterEvent* e) override;
    void dropEvent(QDropEvent* e) override;
    void closeEvent(QCloseEvent* e) override;

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
    void about();
    void checkForUpdates();                    // Help > Check for updates: the result in a box
    void showUpdate();                         // the dialog: what changed, Update and restart / Skip / Later
    void onUpdateAvailable(const ReleaseInfo& r);
    void resetSettings();
    void showFrame(int row);
    void showOutput(int tab);
    void stepOutput(int delta);
    void onProgress(QString stage, double fraction);
    void onPreview(QString name, FloatImage rgb, int w, int h);
    void onFrame(int k, FloatImage rgb, int w, int h);
    void onThumbnail(int index, QImage image);
    void onFullFrame(int index, QImage image);
    void onComparison(int index, QImage image);
    void onLog(QString line);
    void onFinished(std::shared_ptr<JobResultData> data);
    void dropFlagged();
    void frameMenu(const QPoint& pos);

private:
    void buildMenus();
    void buildUi();
    void loadSettings();
    void saveSettings();
    void updateRecent();
    void resumeThumbnails();
    void warmCache();      // point the thumbnail loader at the frame cache (or away from it) for the open folder
    void labelWobble(const stackcore::AlignOut& A);
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

    QString folder_;
    QStringList frames_;
    // rows carry the frame index (into frames_) in Qt::UserRole, -1 for files left out; the list order is
    // the stack order and a frame may be listed more than once, so jobs refer to rows, not frames
    QList<QListWidgetItem*> jobItems_;     // the rows of the current / last render, in stack order
    QList<QListWidgetItem*> flagged_;      // rows whose wobble is an outlier
    QString wobble_;                       // the wobble summary line
    QAction* dropFlagged_ = nullptr;
    QAction* saveAct_ = nullptr;
    QAction* openOutAct_ = nullptr;
    QAction* renderAct_ = nullptr;
    QAction* cancelAct_ = nullptr;
    QMenu* recentMenu_ = nullptr;
    QStringList recent_;
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
    QString progressStatus_;               // status-bar count for the per-frame stages ("Read 157 of 163 frames"), from the progress signal
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
    bool updateManual_ = false;        // the running check came from the menu: report its result either way

    ImageView* view_ = nullptr;
    QListWidget* frameList_ = nullptr;
    QDockWidget* framesDock_ = nullptr;
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
    QDoubleSpinBox* fov_ = nullptr;
    QDoubleSpinBox* smoothing_ = nullptr;
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
