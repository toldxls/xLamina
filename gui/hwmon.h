// Live hardware panel: per GPU the load, memory, temperature and PCIe link with its traffic, and the host's RAM with
// what xLamina has pinned. Sampled once a second on its own thread through NVML (nvml.dll, loaded at run time) and
// Windows' memory status; the Preferences line is a snapshot at start-up and cannot show a run.
#pragma once

#include <QList>
#include <QWidget>

#include <atomic>
#include <functional>
#include <thread>
#include <vector>

#include "prefs.h"

class QLabel;

// A bar with its text inside, coloured by how full it is (calm below 60 %, warm to 85 %, red above)
class HwBar : public QWidget {
    Q_OBJECT
public:
    explicit HwBar(QWidget* parent = nullptr);
    void set(double fraction, const QString& text);
    QSize sizeHint() const override;
    QSize minimumSizeHint() const override { return sizeHint(); }

protected:
    void paintEvent(QPaintEvent*) override;

private:
    double f_ = 0;
    QString text_;
};

class HwMonitor : public QWidget {
    Q_OBJECT
public:
    explicit HwMonitor(const QList<GpuInfo>& gpus, QWidget* parent = nullptr);
    ~HwMonitor() override;
    void setPinnedProvider(std::function<size_t()> f) { pinned_ = std::move(f); }   // bytes xLamina holds pinned (the frame cache)

private:
    struct GpuSample { bool ok = false; unsigned util = 0, temp = 0, gen = 0, width = 0; unsigned long long memUsed = 0, memTotal = 0; double gbps = 0; };
    struct Sample { std::vector<GpuSample> gpu; unsigned long long ramUsed = 0, ramTotal = 0; };
    struct Row { QLabel* name; HwBar *util, *mem, *temp, *pcie; };
    void apply(const Sample& s);

    QList<GpuInfo> gpus_;
    std::vector<Row> rows_;
    HwBar* ram_ = nullptr;
    std::function<size_t()> pinned_;
    std::thread thread_;
    std::atomic<bool> stop_{false};
    struct Nvml;
    Nvml* nvml_ = nullptr;
};
