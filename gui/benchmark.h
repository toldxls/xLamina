// Tools > Benchmark: runs a stack through the engine under a sweep of hardware
// settings (each GPU alone and all together, streams per GPU, reader threads,
// alignment pairs in flight, frames resident or streamed), one setting varied
// at a time from the current ones, then the best value of each combined; times
// every run and offers to apply the fastest. Like Helicon's benchmark it can
// also run a built-in synthetic stack, so machines can be compared. None of
// these settings changes the result (B and C are bit-identical across them).
#pragma once

#include <QDialog>
#include <QElapsedTimer>
#include <QList>
#include <QStringList>

#include <memory>
#include <thread>

#include "prefs.h"
#include "worker.h"

class QCheckBox;
class QComboBox;
class QLabel;
class QProgressBar;
class QPushButton;
class QTableWidget;

class BenchmarkDialog : public QDialog {
    Q_OBJECT
public:
    // base: the job as the main window would run it now (frames, outputs, parameters); frames may be empty
    BenchmarkDialog(Worker& worker, const stackcore::JobOptions& base, const QStringList& frames, const QString& stackLabel,
                    const PerfSettings& perf, QWidget* parent);
    ~BenchmarkDialog() override;
    bool running() const { return running_; }
    // developer aid (xLamina --bench): run at once, on the built-in sample or the open stack, save the report
    // (and a grab of the dialog beside it) when done, then close
    void autoRun(bool sample, const QString& reportPath);

signals:
    void applySettings(PerfSettings p);
    void runningChanged(bool running);

protected:
    void closeEvent(QCloseEvent* e) override;

private slots:
    void startStop();
    void onFinished(std::shared_ptr<JobResultData> data);
    void onProgress(QString stage, double fraction);
    void applyBest();
    void saveReport();
    void copyReport();

private:
    struct Config {
        QString label, dimension;   // dimension: "", "gpus", "streams", "readers", "pairs", "cache", "combined"
        PerfSettings perf;
        bool cold = false;          // the first run reads the frames from disk; not compared
    };
    struct Row {
        Config cfg;
        int rc = -1;
        double align = 0, fusion = 0, total = 0, fps = 0, pass1 = 0, depth = 0, refine = 0, compose = 0, finalize = 0;
    };
    void updatePlan();
    void buildConfigs();
    void beginSweep();
    void runNext();
    void finishSweep();
    void addRow(const Row& r);
    int bestRow() const;
    QString reportText() const;
    void setRunning(bool on);
    bool useSample() const;
    void makeSample();

    Worker& worker_;
    stackcore::JobOptions base_;
    QStringList frames_;
    QString stackLabel_;
    PerfSettings perf_;
    QList<GpuInfo> gpus_;
    QList<Config> configs_;
    QList<Row> rows_;
    int next_ = 0;
    bool running_ = false, stopping_ = false, combinedDone_ = false;
    QElapsedTimer clock_;
    std::thread sampleThread_;
    QStringList sample_;        // the synthetic frames, once generated
    QString sampleDir_;
    QString autoReport_;        // set by autoRun

    QComboBox* stack_ = nullptr;
    QComboBox* res_ = nullptr;
    QCheckBox* varyGpus_ = nullptr;
    QCheckBox* varyStreams_ = nullptr;
    QCheckBox* varyReaders_ = nullptr;
    QCheckBox* varyPairs_ = nullptr;
    QCheckBox* varyCache_ = nullptr;
    QLabel* info_ = nullptr;
    QLabel* plan_ = nullptr;
    QLabel* stage_ = nullptr;
    QProgressBar* progress_ = nullptr;
    QTableWidget* table_ = nullptr;
    QPushButton* startBtn_ = nullptr;
    QPushButton* applyBtn_ = nullptr;
    QPushButton* saveBtn_ = nullptr;
    QPushButton* copyBtn_ = nullptr;
};
