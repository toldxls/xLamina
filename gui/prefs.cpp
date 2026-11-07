#include "prefs.h"

#include <cuda_runtime.h>
#include <windows.h>

#include <QApplication>
#include <QPalette>
#include <QStyle>
#include <QStyleFactory>

#include <thread>

// ----------------------------------------------------------------------------- GPUs

// NVML (ships with the driver as nvml.dll) gives the PCIe link; loaded at run time so a machine
// without it still starts.
namespace {
typedef struct nvmlDevice_st* nvmlDevice_t;
typedef int (*PFN_nvmlNoArg)();
typedef int (*PFN_nvmlByBus)(const char*, nvmlDevice_t*);
typedef int (*PFN_nvmlUint)(nvmlDevice_t, unsigned int*);

struct Nvml {
    HMODULE lib = nullptr;
    PFN_nvmlNoArg init = nullptr, shutdown = nullptr;
    PFN_nvmlByBus byBus = nullptr;
    PFN_nvmlUint curGen = nullptr, curWidth = nullptr, maxGen = nullptr, maxWidth = nullptr;
    Nvml() {
        lib = LoadLibraryA("nvml.dll");
        if (!lib) return;
        init = (PFN_nvmlNoArg)GetProcAddress(lib, "nvmlInit_v2");
        shutdown = (PFN_nvmlNoArg)GetProcAddress(lib, "nvmlShutdown");
        byBus = (PFN_nvmlByBus)GetProcAddress(lib, "nvmlDeviceGetHandleByPciBusId_v2");
        curGen = (PFN_nvmlUint)GetProcAddress(lib, "nvmlDeviceGetCurrPcieLinkGeneration");
        curWidth = (PFN_nvmlUint)GetProcAddress(lib, "nvmlDeviceGetCurrPcieLinkWidth");
        maxGen = (PFN_nvmlUint)GetProcAddress(lib, "nvmlDeviceGetMaxPcieLinkGeneration");
        maxWidth = (PFN_nvmlUint)GetProcAddress(lib, "nvmlDeviceGetMaxPcieLinkWidth");
        if (!(init && shutdown && byBus && curGen && curWidth && maxGen && maxWidth) || init() != 0) { FreeLibrary(lib); lib = nullptr; }
    }
    ~Nvml() { if (lib) { shutdown(); FreeLibrary(lib); } }
    void fill(GpuInfo& g, const char* bus) const {
        if (!lib) return;
        nvmlDevice_t h;
        if (byBus(bus, &h) != 0) return;
        unsigned v;
        if (curGen(h, &v) == 0) g.pcieGen = (int)v;
        if (curWidth(h, &v) == 0) g.pcieWidth = (int)v;
        if (maxGen(h, &v) == 0) g.pcieGenMax = (int)v;
        if (maxWidth(h, &v) == 0) g.pcieWidthMax = (int)v;
    }
};
}  // namespace

QList<GpuInfo> enumerateGpus() {
    QList<GpuInfo> out;
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) return out;
    Nvml nvml;
    for (int i = 0; i < n; ++i) {
        cudaDeviceProp p{};
        if (cudaGetDeviceProperties(&p, i) != cudaSuccess) continue;
        GpuInfo g;
        g.index = i;
        g.name = QString::fromLatin1(p.name);
        g.memory = p.totalGlobalMem;
        g.sms = p.multiProcessorCount;
        char bus[64] = {0};
        if (cudaDeviceGetPCIBusId(bus, sizeof bus, i) == cudaSuccess) nvml.fill(g, bus);
        out << g;
    }
    return out;
}

QString GpuInfo::label() const {
    QString s = QString("%1: %2, %3 GB").arg(index).arg(name).arg(memory / 1073741824.0, 0, 'f', 0);
    if (pcieWidth) {   // the width is the slot's; the generation drops to 1 while the card idles, so it is shown as "now"
        s += QString(", PCIe x%1").arg(pcieWidth);
        if (pcieWidthMax && pcieWidthMax != pcieWidth) s += QString(" (card max x%1)").arg(pcieWidthMax);
        if (pcieGen) s += QString(", gen %1 now").arg(pcieGen);
        if (pcieGenMax) s += QString(" of %1").arg(pcieGenMax);
    }
    return s;
}

QString hostSummary() {
    MEMORYSTATUSEX m{};
    m.dwLength = sizeof m;
    GlobalMemoryStatusEx(&m);
    return QString("RAM %1 GB (%2 GB free), %3 logical CPUs")
        .arg(m.ullTotalPhys / 1073741824.0, 0, 'f', 0).arg(m.ullAvailPhys / 1073741824.0, 0, 'f', 0).arg(std::thread::hardware_concurrency());
}

// ----------------------------------------------------------------------------- performance settings

void PerfSettings::apply(stackcore::JobOptions& jo) const {
    std::vector<int> devs(gpus.begin(), gpus.end());
    jo.align.devs = devs;
    jo.stack.po.opt.devices = devs;
    jo.stack.streams = streams;
    jo.stack.po.opt.streams_per_gpu = streams;
    jo.align.readers = readers;
    jo.stack.po.opt.readers = readers;
    jo.align.pair_threads = pairThreads;
    jo.stack.cache_frames = cacheFrames;
}

QString PerfSettings::summary() const {
    QStringList g;
    for (int i : gpus) g << QString::number(i);
    return QString("GPUs %1, %2 stream%3/GPU, %4 readers, %5 pair%6 in flight, frames %7")
        .arg(g.isEmpty() ? "all" : g.join("+")).arg(streams).arg(streams == 1 ? "" : "s").arg(readers)
        .arg(pairThreads).arg(pairThreads == 1 ? "" : "s").arg(cacheFrames ? "resident" : "streamed");
}

PerfSettings PerfSettings::load(QSettings& s) {
    PerfSettings p;
    s.beginGroup("performance");
    for (const QString& v : s.value("gpus").toString().split(',', Qt::SkipEmptyParts)) p.gpus << v.toInt();
    p.streams = s.value("streams", p.streams).toInt();
    p.readers = s.value("readers", p.readers).toInt();
    p.pairThreads = s.value("pairThreads", p.pairThreads).toInt();
    p.cacheFrames = s.value("cacheFrames", p.cacheFrames).toBool();
    s.endGroup();
    return p;
}

void PerfSettings::save(QSettings& s) const {
    QStringList g;
    for (int i : gpus) g << QString::number(i);
    s.beginGroup("performance");
    s.setValue("gpus", g.join(','));
    s.setValue("streams", streams);
    s.setValue("readers", readers);
    s.setValue("pairThreads", pairThreads);
    s.setValue("cacheFrames", cacheFrames);
    s.endGroup();
}

QSettings appSettings() { return QSettings(QSettings::IniFormat, QSettings::UserScope, "xLamina", "xLamina"); }

// ----------------------------------------------------------------------------- theme

void applyTheme(const QString& theme) {
    static QString systemStyle = QApplication::style()->objectName();   // captured before the first switch
    static QPalette systemPalette = QApplication::palette();
    if (theme == "system") {
        QApplication::setStyle(QStyleFactory::create(systemStyle));
        QApplication::setPalette(systemPalette);
        return;
    }
    QApplication::setStyle(QStyleFactory::create("Fusion"));
    if (theme != "dark") { QApplication::setPalette(QApplication::style()->standardPalette()); return; }
    QPalette p;
    const QColor window(45, 45, 45), base(30, 30, 30), alt(38, 38, 38), text(224, 224, 224), button(56, 56, 56);
    const QColor disabled(122, 122, 122), accent(122, 47, 201);   // the icon's purple
    p.setColor(QPalette::Window, window);
    p.setColor(QPalette::WindowText, text);
    p.setColor(QPalette::Base, base);
    p.setColor(QPalette::AlternateBase, alt);
    p.setColor(QPalette::ToolTipBase, QColor(60, 60, 60));
    p.setColor(QPalette::ToolTipText, text);
    p.setColor(QPalette::Text, text);
    p.setColor(QPalette::PlaceholderText, disabled);
    p.setColor(QPalette::Button, button);
    p.setColor(QPalette::ButtonText, text);
    p.setColor(QPalette::BrightText, QColor(255, 96, 96));
    p.setColor(QPalette::Link, QColor(120, 170, 255));
    p.setColor(QPalette::Highlight, accent);
    p.setColor(QPalette::HighlightedText, Qt::white);
    p.setColor(QPalette::Light, QColor(80, 80, 80));
    p.setColor(QPalette::Midlight, QColor(64, 64, 64));
    p.setColor(QPalette::Mid, QColor(50, 50, 50));
    p.setColor(QPalette::Dark, QColor(24, 24, 24));
    p.setColor(QPalette::Shadow, Qt::black);
    for (QPalette::ColorRole r : {QPalette::WindowText, QPalette::Text, QPalette::ButtonText, QPalette::HighlightedText})
        p.setColor(QPalette::Disabled, r, disabled);
    p.setColor(QPalette::Disabled, QPalette::Highlight, QColor(70, 70, 70));
    QApplication::setPalette(p);
}
