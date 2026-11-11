#include "hwmon.h"

#include <windows.h>

#include <QFormLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QMetaObject>
#include <QPainter>
#include <QVBoxLayout>

#include <chrono>

// ----------------------------------------------------------------------------- the bar

HwBar::HwBar(QWidget* parent) : QWidget(parent) { setMinimumWidth(120); }

QSize HwBar::sizeHint() const { return QSize(180, 20); }

void HwBar::set(double fraction, const QString& text) {
    f_ = std::clamp(fraction, 0.0, 1.0);
    text_ = text;
    update();
}

void HwBar::paintEvent(QPaintEvent*) {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);
    QRectF r = rect().adjusted(0.5, 0.5, -0.5, -0.5);
    p.setPen(QColor(90, 90, 90));
    p.setBrush(QColor(38, 38, 38));
    p.drawRoundedRect(r, 3, 3);
    QColor fill = f_ < 0.6 ? QColor(86, 140, 200) : f_ < 0.85 ? QColor(220, 150, 50) : QColor(210, 60, 60);
    if (f_ > 0) {
        QRectF fr = r;
        fr.setWidth(std::max(4.0, r.width() * f_));
        p.setPen(Qt::NoPen);
        p.setBrush(fill);
        p.drawRoundedRect(fr, 3, 3);
    }
    p.setPen(QColor(235, 235, 235));
    p.drawText(rect(), Qt::AlignCenter, text_);
}

// ----------------------------------------------------------------------------- NVML, loaded at run time

namespace {
typedef struct nvmlDevice_st* nvmlDevice_t;
struct nvmlUtilization_t { unsigned gpu, memory; };
struct nvmlMemory_t { unsigned long long total, free, used; };
typedef int (*PFN_noArg)();
typedef int (*PFN_byBus)(const char*, nvmlDevice_t*);
typedef int (*PFN_uint)(nvmlDevice_t, unsigned*);
typedef int (*PFN_util)(nvmlDevice_t, nvmlUtilization_t*);
typedef int (*PFN_temp)(nvmlDevice_t, int, unsigned*);
typedef int (*PFN_mem)(nvmlDevice_t, nvmlMemory_t*);
typedef int (*PFN_pcie)(nvmlDevice_t, int, unsigned*);
}  // namespace

struct HwMonitor::Nvml {
    HMODULE lib = nullptr;
    PFN_noArg init = nullptr, shutdown = nullptr;
    PFN_byBus byBus = nullptr;
    PFN_uint curGen = nullptr, curWidth = nullptr;
    PFN_util util = nullptr;
    PFN_temp temp = nullptr;
    PFN_mem mem = nullptr;
    PFN_pcie pcie = nullptr;   // KB/s over a 20 ms window per call: sampled off the GUI thread
    std::vector<nvmlDevice_t> dev;
    Nvml(const QList<GpuInfo>& gpus) {
        lib = LoadLibraryExA("nvml.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);   // the driver's copy only, never one beside a stack's frames
        if (!lib) return;
        init = (PFN_noArg)GetProcAddress(lib, "nvmlInit_v2");
        shutdown = (PFN_noArg)GetProcAddress(lib, "nvmlShutdown");
        byBus = (PFN_byBus)GetProcAddress(lib, "nvmlDeviceGetHandleByPciBusId_v2");
        curGen = (PFN_uint)GetProcAddress(lib, "nvmlDeviceGetCurrPcieLinkGeneration");
        curWidth = (PFN_uint)GetProcAddress(lib, "nvmlDeviceGetCurrPcieLinkWidth");
        util = (PFN_util)GetProcAddress(lib, "nvmlDeviceGetUtilizationRates");
        temp = (PFN_temp)GetProcAddress(lib, "nvmlDeviceGetTemperature");
        mem = (PFN_mem)GetProcAddress(lib, "nvmlDeviceGetMemoryInfo");
        pcie = (PFN_pcie)GetProcAddress(lib, "nvmlDeviceGetPcieThroughput");
        if (!(init && shutdown && byBus && curGen && curWidth && util && temp && mem && pcie) || init() != 0) { FreeLibrary(lib); lib = nullptr; return; }
        for (const GpuInfo& g : gpus) {
            nvmlDevice_t h = nullptr;
            if (g.busId.isEmpty() || byBus(g.busId.constData(), &h) != 0) h = nullptr;
            dev.push_back(h);
        }
    }
    ~Nvml() { if (lib) { shutdown(); FreeLibrary(lib); } }
};

// ----------------------------------------------------------------------------- the panel

HwMonitor::HwMonitor(const QList<GpuInfo>& gpus, QWidget* parent) : QWidget(parent), gpus_(gpus) {
    QVBoxLayout* l = new QVBoxLayout(this);
    l->setContentsMargins(6, 4, 6, 4);
    l->setSpacing(4);
    for (const GpuInfo& g : gpus_) {
        Row r;
        QHBoxLayout* h = new QHBoxLayout;
        h->setSpacing(6);
        r.name = new QLabel(QString("%1: %2").arg(g.index).arg(QString(g.name).remove("NVIDIA ").remove("GeForce ")));
        r.name->setMinimumWidth(110);
        h->addWidget(r.name);
        r.util = new HwBar; r.util->setToolTip("GPU load");
        r.mem = new HwBar; r.mem->setToolTip("GPU memory in use, by every program");
        r.temp = new HwBar; r.temp->setToolTip("GPU temperature");
        r.pcie = new HwBar; r.pcie->setToolTip("The PCIe link as negotiated now (it drops to gen 1 while the card idles) and its traffic, both directions, against the link's capacity");
        for (HwBar* b : {r.util, r.mem, r.temp, r.pcie}) h->addWidget(b, 1);
        h->setStretch(4, 2);   // the PCIe bar carries the longest text
        l->addLayout(h);
        rows_.push_back(r);
    }
    QHBoxLayout* h = new QHBoxLayout;
    h->setSpacing(6);
    QLabel* rl = new QLabel("Host RAM");
    rl->setMinimumWidth(110);
    h->addWidget(rl);
    ram_ = new HwBar;
    ram_->setToolTip("Physical memory in use, by every program, and the part xLamina holds pinned for the frames (the frame cache)");
    h->addWidget(ram_, 1);
    l->addLayout(h);
    l->addStretch(1);

    nvml_ = new Nvml(gpus_);
    thread_ = std::thread([this] {
        while (!stop_.load()) {
            Sample s;
            for (size_t i = 0; i < (size_t)gpus_.size(); ++i) {
                GpuSample gs;
                nvmlDevice_t h = nvml_->lib && i < nvml_->dev.size() ? nvml_->dev[i] : nullptr;
                if (h) {
                    nvmlUtilization_t u{};
                    nvmlMemory_t m{};
                    unsigned v = 0, tx = 0, rx = 0;
                    gs.ok = true;
                    if (nvml_->util(h, &u) == 0) gs.util = u.gpu;
                    if (nvml_->temp(h, 0, &v) == 0) gs.temp = v;
                    if (nvml_->mem(h, &m) == 0) { gs.memUsed = m.used; gs.memTotal = m.total; }
                    if (nvml_->curGen(h, &v) == 0) gs.gen = v;
                    if (nvml_->curWidth(h, &v) == 0) gs.width = v;
                    if (nvml_->pcie(h, 0, &tx) == 0 && nvml_->pcie(h, 1, &rx) == 0) gs.gbps = (tx + rx) * 1024.0 / 1e9;
                }
                s.gpu.push_back(gs);
            }
            MEMORYSTATUSEX ms{};
            ms.dwLength = sizeof ms;
            if (GlobalMemoryStatusEx(&ms)) { s.ramTotal = ms.ullTotalPhys; s.ramUsed = ms.ullTotalPhys - ms.ullAvailPhys; }
            QMetaObject::invokeMethod(this, [this, s] { apply(s); }, Qt::QueuedConnection);
            for (int i = 0; i < 10 && !stop_.load(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });
}

HwMonitor::~HwMonitor() {
    stop_.store(true);
    if (thread_.joinable()) thread_.join();
    delete nvml_;
}

void HwMonitor::apply(const Sample& s) {
    for (size_t i = 0; i < rows_.size() && i < s.gpu.size(); ++i) {
        const GpuSample& g = s.gpu[i];
        Row& r = rows_[i];
        if (!g.ok) {
            r.util->set(0, "no NVML");
            r.mem->set(0, "—"); r.temp->set(0, "—"); r.pcie->set(0, "—");
            continue;
        }
        r.util->set(g.util / 100.0, QString("GPU %1 %").arg(g.util));
        r.mem->set(g.memTotal ? (double)g.memUsed / g.memTotal : 0, QString("%1 / %2 GB").arg(g.memUsed / 1073741824.0, 0, 'f', 1).arg(g.memTotal / 1073741824.0, 0, 'f', 0));
        r.temp->set(g.temp / 100.0, QString("%1 °C").arg(g.temp));
        // the link's capacity: per lane and direction 0.25 / 0.5 / 0.985 / 1.97 / 3.94 GB/s for gen 1..5; traffic is both directions
        static const double lane[] = {0, 0.25, 0.5, 0.985, 1.97, 3.94, 7.88};
        double cap = g.gen >= 1 && g.gen < 7 ? 2 * lane[g.gen] * g.width : 0;
        r.pcie->set(cap > 0 ? g.gbps / cap : 0, QString("PCIe gen %1 x%2, %3 GB/s").arg(g.gen).arg(g.width).arg(g.gbps, 0, 'f', 1));
    }
    size_t pinned = pinned_ ? pinned_() : 0;
    QString t = QString("%1 / %2 GB").arg(s.ramUsed / 1073741824.0, 0, 'f', 0).arg(s.ramTotal / 1073741824.0, 0, 'f', 0);
    if (pinned) t += QString(", frames pinned %1 GB").arg(pinned / 1073741824.0, 0, 'f', 1);
    ram_->set(s.ramTotal ? (double)s.ramUsed / s.ramTotal : 0, t);
}
