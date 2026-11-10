// Tools > Benchmark. Quick (the default, about 20 s): up to 40 frames of the
// open stack at half resolution under the current settings, then each GPU
// alone, then with the frames streamed from disk instead of resident; the
// fastest can be applied. Thorough (minutes): the whole stack at the Render
// panel's resolution, every hardware setting varied in turn, then the best of
// each combined. Single run: the whole stack once on the current settings, to
// compare the machine against itself (an overclock); each Start adds a row to
// the table. A built-in synthetic stack stands in when no folder is open,
// or to compare machines. None of these settings changes the result (B and C
// are bit-identical across them).
#pragma once

#include <QDialog>
#include <QElapsedTimer>
#include <QList>
#include <QStringList>

#include <memory>
#include <thread>

#include "prefs.h"
#include "worker.h"

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
    bool quick() const;
    bool thorough() const;
    bool single() const;             // one render on the current settings; its rows accumulate over Starts
    bool useSample() const;
    QStringList runFrames() const;   // the frames a run uses: the quick subset or the whole stack
    void updatePlan();
    void buildConfigs();
    void beginSweep();
    void runNext();
    void finishSweep();
    void addRow(const Row& r);
    int bestRow() const;
    QString reportText() const;
    void setRunning(bool on);
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
    int keepKey_ = 0;           // the last Start: 0 = a sweep, 1 = a single run of the open stack, 2 = of the sample
    bool running_ = false, stopping_ = false, combinedDone_ = false;
    QElapsedTimer clock_, sweepClock_;
    std::thread sampleThread_;
    QStringList sample_;        // the synthetic frames, once generated
    QString sampleDir_;
    QString autoReport_;        // set by autoRun
    int autoStarts_ = 1;        // autoRun: presses of Start still to make

    QComboBox* stack_ = nullptr;
    QComboBox* mode_ = nullptr;
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
