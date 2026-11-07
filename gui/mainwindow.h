#pragma once

#include <QCheckBox>
#include <QColorSpace>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QElapsedTimer>
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
#include <QSpinBox>
#include <QSplitter>
#include <QTabBar>
#include <QTimer>

#include <memory>

#include "benchmark.h"
#include "imageview.h"
#include "prefs.h"
#include "worker.h"

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent = nullptr);
    void openFolder(const QString& dir);
    bool rendering() const { return worker_.running(); }
    void autoBenchmark(bool sample, const QString& reportPath);   // developer aid: run the benchmark unattended, quit when done

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
    void openOutputFolder();
    void benchmark();
    void about();
    void resetSettings();
    void showFrame(int row);
    void showOutput(int tab);
    void stepOutput(int delta);
    void onProgress(QString stage, double fraction);
    void onPreview(QString name, QImage image);
    void onFrame(int k, QImage image);
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
    void labelWobble(const stackcore::AlignOut& A);
    void setBusy(bool busy);
    void setStatusGeometry(const JobResultData& d);
    void setTheme(const QString& theme);
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
    QColorSpace colorSpace_;               // the frames' ICC profile (invalid = assume sRGB)
    bool selecting_ = false;
    ImageView* liveView_ = nullptr;        // right pane while rendering
    QSplitter* split_ = nullptr;
    ThumbLoader thumbs_;
    ThumbLoader frameLoader_;              // one frame at full resolution
    ThumbLoader compareLoader_;            // a comparison image (another stacker's output) at full resolution
    QString pendingCompare_;
    int pendingFrame_ = -1;
    QComboBox* liveRes_ = nullptr;
    std::shared_ptr<JobResultData> result_;
    QMap<QString, QImage> cache_;          // 8-bit versions of outputs and previews
    QMap<QString, QImage> compare_;        // comparison images by tab name; they survive renders
    QString current_;                      // name of the image on screen
    QElapsedTimer elapsed_;
    QTimer tick_;
    QString stageText_;
    QPointer<BenchmarkDialog> bench_;
    QList<GpuInfo> gpus_;

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
    QDoubleSpinBox* focusRadius_ = nullptr;
    QDoubleSpinBox* focusPower_ = nullptr;
    QDoubleSpinBox* depthFmPx_ = nullptr;
    QDoubleSpinBox* volAggPx_ = nullptr;
    QSpinBox* refineR_ = nullptr;
    QDoubleSpinBox* refineFmPx_ = nullptr;
    QDoubleSpinBox* softPx_ = nullptr;
    QDoubleSpinBox* haloPrior_ = nullptr;
    QDoubleSpinBox* darkPrior_ = nullptr;
    Worker worker_;
};
