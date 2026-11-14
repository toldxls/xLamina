// Preferences shared by the main window and the benchmark: the hardware
// settings a job runs with, the GPUs the machine has, the theme, and the
// QSettings store they all persist in (%APPDATA%\xLamina\xLamina.ini).
#pragma once

#include <algorithm>
#include <thread>

#include <QByteArray>
#include <QList>
#include <QSettings>
#include <QString>

#include "stackcore/pipeline.h"

struct GpuInfo {
    int index = 0;
    QString name;
    quint64 memory = 0;                                      // bytes
    int sms = 0;
    int pcieGen = 0, pcieWidth = 0, pcieGenMax = 0, pcieWidthMax = 0;   // as negotiated / the card's maximum; 0 = unknown
    QByteArray busId;                                        // PCI bus id ("0000:01:00.0"), for NVML's live readings
    QString label() const;                                   // "0: NVIDIA GeForce RTX 3080 Ti, 12 GB, PCIe 4 x8"
};
QList<GpuInfo> enumerateGpus();
QString hostSummary();                                       // "RAM 64 GB (41 GB free), 24 logical CPUs"

// The settings that change speed, not the result (B and C are bit-identical across all of them).
struct PerfSettings {
    QList<int> gpus;         // CUDA device indices to use; empty = all
    int streams = 2;         // upload / compute streams per GPU in the fusion passes
    int readers = std::clamp((int)std::thread::hardware_concurrency(), 2, 16);   // TIFF reader threads: 16, fewer on a PC with fewer cores
    int pairThreads = 3;     // alignment pairs in flight per GPU
    bool cacheFrames = true; // keep the frames pinned in host memory between passes and renders

    void apply(stackcore::JobOptions& jo) const;
    QString summary() const; // one line, for reports and the status bar
    bool operator==(const PerfSettings&) const = default;
    static PerfSettings load(QSettings& s);
    void save(QSettings& s) const;
};

QSettings appSettings();                                     // the application's INI store

// "dark" and "light" are Fusion palettes and switch live; "system" is the platform style (restart to return to it).
void applyTheme(const QString& theme);
