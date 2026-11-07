#include "benchmark.h"

#include <QApplication>
#include <QCheckBox>
#include <QClipboard>
#include <QCloseEvent>
#include <QComboBox>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QProgressBar>
#include <QPushButton>
#include <QStandardPaths>
#include <QTableWidget>
#include <QTextStream>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

#include "stackcore/compose.h"
#include "version.h"

using namespace stackcore;

// ----------------------------------------------------------------------------- the built-in sample

// A synthetic stack with the features a real one has: a textured matrix whose height varies smoothly,
// a raised plate with a bright thin rim (the edge-on plate edges of Travis's specimens), per-frame
// defocus that grows with the distance to the frame's depth, sensor noise, and a little wobble.
// Deterministic, so two machines running it are comparable.
namespace {
const int kSampleW = 4000, kSampleH = 2667, kSampleN = 32;   // 2 GB of frames: runs of a few seconds, not sub-second noise

inline float hash01(int x, int y, int seed) {
    uint32_t h = (uint32_t)x * 374761393u + (uint32_t)y * 668265263u + (uint32_t)seed * 2246822519u;
    h = (h ^ (h >> 13)) * 1274126177u;
    h ^= h >> 16;
    return (h & 0xffffff) / 16777216.f;
}

// value noise, one octave of cell size `cell`
inline float vnoise(float x, float y, float cell, int seed) {
    float fx = x / cell, fy = y / cell;
    int ix = (int)std::floor(fx), iy = (int)std::floor(fy);
    float tx = fx - ix, ty = fy - iy;
    tx = tx * tx * (3 - 2 * tx);
    ty = ty * ty * (3 - 2 * ty);
    float a = hash01(ix, iy, seed), b = hash01(ix + 1, iy, seed), c = hash01(ix, iy + 1, seed), d = hash01(ix + 1, iy + 1, seed);
    return (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty;
}

void gauss_blur(const std::vector<float>& in, std::vector<float>& out, std::vector<float>& tmp, int W, int H, float sigma) {
    if (sigma < 0.05f) { out = in; return; }
    int r = (int)std::ceil(3 * sigma);
    std::vector<float> k(2 * r + 1);
    float sum = 0;
    for (int i = -r; i <= r; ++i) { k[i + r] = std::exp(-0.5f * i * i / (sigma * sigma)); sum += k[i + r]; }
    for (float& v : k) v /= sum;
    tmp.resize(in.size());
    out.resize(in.size());
    for (int y = 0; y < H; ++y) {
        const float* row = in.data() + (size_t)y * W;
        float* o = tmp.data() + (size_t)y * W;
        for (int x = 0; x < W; ++x) {
            float s = 0;
            for (int i = -r; i <= r; ++i) s += k[i + r] * row[std::clamp(x + i, 0, W - 1)];
            o[x] = s;
        }
    }
    for (int y = 0; y < H; ++y) {
        float* o = out.data() + (size_t)y * W;
        for (int x = 0; x < W; ++x) o[x] = 0;
        for (int i = -r; i <= r; ++i) {
            const float* row = tmp.data() + (size_t)std::clamp(y + i, 0, H - 1) * W;
            float w = k[i + r];
            for (int x = 0; x < W; ++x) o[x] += w * row[x];
        }
    }
}

// Writes the frames into dir (skipping those already there); returns their paths in stack order.
QStringList generate_sample(const QString& dir, const std::function<void(const QString&)>& progress) {
    QDir().mkpath(dir);
    QStringList files;
    bool complete = true;
    for (int k = 0; k < kSampleN; ++k) {
        QString f = QDir(dir).filePath(QString("sample_%1.tif").arg(k, 3, 10, QChar('0')));
        files << f;
        if (!QFileInfo::exists(f)) complete = false;
    }
    if (complete) return files;
    const int W = kSampleW, H = kSampleH, N = kSampleN;
    const size_t n = (size_t)W * H;
    progress("generating the sample stack: scene");
    std::vector<float> Z(n), T(n), tintR(n), tintG(n), tintB(n);
    const float cx = 0.6f * W, cy = 0.5f * H, rx = 0.25f * W, ry = 0.30f * H;
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            size_t i = (size_t)y * W + x;
            float ex = (x - cx) / rx, ey = (y - cy) / ry, e = ex * ex + ey * ey;
            float base = (N - 1) * 0.45f * (0.5f + 0.5f * std::sin(6.2832f * 1.3f * x / W + 0.7f) * std::cos(6.2832f * 0.9f * y / H + 1.9f));
            bool plate = e < 1.f;
            Z[i] = plate ? (N - 1) * (0.82f + 0.08f * ex) : base;
            float tex = 0.45f * vnoise((float)x, (float)y, 96, 1) + 0.30f * vnoise((float)x, (float)y, 32, 2) + 0.17f * vnoise((float)x, (float)y, 10, 3) +
                        0.08f * vnoise((float)x, (float)y, 3, 4);
            float v = 0.25f + 0.75f * tex;
            if (plate) v = 0.35f + 0.65f * tex;
            if (std::abs(e - 1.f) < 0.012f) v = 1.0f;   // the plate's bright rim
            T[i] = v;
            float m = 0.8f + 0.4f * vnoise((float)x, (float)y, 200, 5);
            if (plate) { tintR[i] = 0.95f * m; tintG[i] = 0.80f * m; tintB[i] = 0.30f * m; }
            else { tintR[i] = 0.50f * m; tintG[i] = 0.38f * m; tintB[i] = 0.26f * m; }
        }
    const float sigmas[] = {0.f, 0.6f, 1.2f, 2.4f, 4.f, 6.f, 8.5f, 11.f};
    const int L = 8;
    std::vector<std::vector<float>> levels(L);
    std::vector<float> tmp;
    for (int l = 0; l < L; ++l) {
        progress(QString("generating the sample stack: defocus level %1 of %2").arg(l + 1).arg(L));
        gauss_blur(T, levels[l], tmp, W, H, sigmas[l]);
    }
    std::vector<float>().swap(tmp);
    std::vector<float> rgb(n * 3);
    for (int k = 0; k < N; ++k) {
        progress(QString("generating the sample stack: frame %1 of %2").arg(k + 1).arg(N));
        int dx = (int)std::lround(1.5 * std::sin(0.9 * k)), dy = (int)std::lround(1.2 * std::cos(0.7 * k));
        uint32_t rng = 12345u + 7919u * k;
        for (int y = 0; y < H; ++y) {
            int sy = std::clamp(y + dy, 0, H - 1);
            for (int x = 0; x < W; ++x) {
                int sx = std::clamp(x + dx, 0, W - 1);
                size_t s = (size_t)sy * W + sx, o = ((size_t)y * W + x) * 3;
                float d = 0.45f * std::abs(Z[s] - k);
                int l = 0;
                while (l < L - 2 && sigmas[l + 1] < d) ++l;
                float t = std::clamp((d - sigmas[l]) / (sigmas[l + 1] - sigmas[l]), 0.f, 1.f);
                float v = levels[l][s] * (1 - t) + levels[l + 1][s] * t;
                rng = rng * 1664525u + 1013904223u;
                float noise = ((rng >> 8) / 16777216.f - 0.5f) * 0.008f;
                rgb[o] = tintR[s] * v + noise;
                rgb[o + 1] = tintG[s] * v + noise;
                rgb[o + 2] = tintB[s] * v + noise;
            }
        }
        write_tiff_rgb16(files[k].toStdString(), rgb, W, H, {});
    }
    return files;
}
}  // namespace

// ----------------------------------------------------------------------------- the dialog

BenchmarkDialog::BenchmarkDialog(Worker& worker, const JobOptions& base, const QStringList& frames, const QString& stackLabel,
                                 const PerfSettings& perf, QWidget* parent)
    : QDialog(parent), worker_(worker), base_(base), frames_(frames), stackLabel_(stackLabel), perf_(perf) {
    setWindowTitle("Benchmark");
    setWindowFlag(Qt::WindowContextHelpButtonHint, false);
    resize(1000, 640);
    gpus_ = enumerateGpus();
    sampleDir_ = QDir(QStandardPaths::writableLocation(QStandardPaths::TempLocation)).filePath("xLamina-sample");

    QVBoxLayout* vl = new QVBoxLayout(this);
    QStringList sys;
    for (const GpuInfo& g : gpus_) sys << g.label();
    sys << hostSummary();
    info_ = new QLabel(sys.join("\n"));
    info_->setTextInteractionFlags(Qt::TextSelectableByMouse);
    vl->addWidget(info_);

    QGroupBox* what = new QGroupBox("What to run");
    QFormLayout* fl = new QFormLayout(what);
    stack_ = new QComboBox;
    if (!frames_.isEmpty()) stack_->addItem(QString("The open stack: %1 (%2 frames)").arg(stackLabel_).arg(frames_.size()), 0);
    stack_->addItem(QString("Built-in sample: %1 synthetic frames of %2 x %3 (comparable between machines)").arg(kSampleN).arg(kSampleW).arg(kSampleH), 1);
    fl->addRow("Stack", stack_);
    res_ = new QComboBox;
    res_->addItems({"Full resolution", "Half resolution"});
    res_->setCurrentIndex(base_.stack.po.g.scale == 0.5 ? 1 : 0);
    fl->addRow("Output", res_);
    QStringList outs;
    for (const auto& o : base_.stack.outputs) outs << QString::fromStdString(o);
    fl->addRow("Outputs", new QLabel(outs.isEmpty() ? "all" : outs.join(", ") + "  (as ticked in the Render panel; nothing is written to disk)"));
    QWidget* vary = new QWidget;
    QHBoxLayout* hl = new QHBoxLayout(vary);
    hl->setContentsMargins(0, 0, 0, 0);
    varyGpus_ = new QCheckBox("GPUs");
    varyGpus_->setToolTip("Each GPU alone, and all of them together.");
    varyStreams_ = new QCheckBox("Streams per GPU");
    varyStreams_->setToolTip("1, 2 and 3 upload/compute streams per GPU in the fusion passes.");
    varyReaders_ = new QCheckBox("Reader threads");
    varyReaders_->setToolTip("8, 16 and 32 TIFF reader threads (matters when the frames are read from disk).");
    varyPairs_ = new QCheckBox("Pairs in flight");
    varyPairs_->setToolTip("2, 3 and 4 alignment pairs in flight per GPU.");
    varyCache_ = new QCheckBox("Frames resident / streamed");
    varyCache_->setToolTip("Frames kept pinned in host memory between passes, or read from disk every pass.");
    for (QCheckBox* c : {varyGpus_, varyStreams_, varyReaders_, varyPairs_, varyCache_}) {
        c->setChecked(true);
        hl->addWidget(c);
        connect(c, &QCheckBox::toggled, this, &BenchmarkDialog::updatePlan);
    }
    varyGpus_->setEnabled(gpus_.size() > 1);
    hl->addStretch(1);
    fl->addRow("Vary", vary);
    plan_ = new QLabel;
    plan_->setWordWrap(true);
    fl->addRow("", plan_);
    vl->addWidget(what);
    connect(stack_, &QComboBox::currentIndexChanged, this, &BenchmarkDialog::updatePlan);

    table_ = new QTableWidget(0, 11);
    table_->setHorizontalHeaderLabels({"#", "Configuration", "Align s", "Fusion s", "Total s", "Frames/s", "Pass 1 s", "Depth s", "Refine s", "Compose s", "Finalize s"});
    table_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch);
    table_->verticalHeader()->hide();
    table_->setSelectionBehavior(QAbstractItemView::SelectRows);
    table_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    table_->setAlternatingRowColors(true);
    table_->setToolTip("Total is the wall time of the whole job (alignment, fusion, report). The first run of a stack reads the frames from\n"
                       "disk and is marked cold; the comparison is between the warm runs. Bold: the fastest warm run.");
    vl->addWidget(table_, 1);

    stage_ = new QLabel("Press Start.");
    progress_ = new QProgressBar;
    progress_->setRange(0, 1000);
    QHBoxLayout* pl = new QHBoxLayout;
    pl->addWidget(stage_, 1);
    pl->addWidget(progress_, 1);
    vl->addLayout(pl);

    QHBoxLayout* bl = new QHBoxLayout;
    startBtn_ = new QPushButton("Start");
    startBtn_->setDefault(true);
    applyBtn_ = new QPushButton("Apply fastest settings");
    applyBtn_->setEnabled(false);
    applyBtn_->setToolTip("Puts the fastest warm run's GPU, stream, reader, pair and cache settings into the Render panel.");
    saveBtn_ = new QPushButton("Save report…");
    saveBtn_->setEnabled(false);
    copyBtn_ = new QPushButton("Copy");
    copyBtn_->setEnabled(false);
    QPushButton* close = new QPushButton("Close");
    bl->addWidget(startBtn_);
    bl->addWidget(applyBtn_);
    bl->addStretch(1);
    bl->addWidget(copyBtn_);
    bl->addWidget(saveBtn_);
    bl->addWidget(close);
    vl->addLayout(bl);
    connect(startBtn_, &QPushButton::clicked, this, &BenchmarkDialog::startStop);
    connect(applyBtn_, &QPushButton::clicked, this, &BenchmarkDialog::applyBest);
    connect(saveBtn_, &QPushButton::clicked, this, &BenchmarkDialog::saveReport);
    connect(copyBtn_, &QPushButton::clicked, this, &BenchmarkDialog::copyReport);
    connect(close, &QPushButton::clicked, this, &QDialog::close);
    connect(&worker_, &Worker::finished, this, &BenchmarkDialog::onFinished);
    connect(&worker_, &Worker::progress, this, &BenchmarkDialog::onProgress);
    updatePlan();
}

BenchmarkDialog::~BenchmarkDialog() {
    if (sampleThread_.joinable()) sampleThread_.join();
}

bool BenchmarkDialog::useSample() const { return stack_->currentData().toInt() == 1; }

// The sweep: the current settings first (twice when the frames are not resident yet: the cold run reads
// them), then one setting at a time; the combined best is added at the end of the run.
void BenchmarkDialog::buildConfigs() {
    configs_.clear();
    auto add = [&](const QString& label, const QString& dim, PerfSettings p, bool cold = false) {
        for (const Config& c : configs_) if (!cold && !c.cold && c.perf == p) return;   // the same settings once (the cold run aside)
        configs_ << Config{label, dim, p, cold};
    };
    int n = useSample() ? kSampleN : frames_.size();
    bool resident = !useSample() && perf_.cacheFrames && worker_.cache().count() >= n;
    if (perf_.cacheFrames && !resident) add("current settings, frames read from disk (cold)", "", perf_, true);
    add("current settings", "", perf_);
    if (varyGpus_->isChecked() && gpus_.size() > 1) {
        for (const GpuInfo& g : gpus_) {
            PerfSettings p = perf_;
            p.gpus = {g.index};
            add(QString("GPU %1 only (%2)").arg(g.index).arg(g.name), "gpus", p);
        }
        PerfSettings p = perf_;
        p.gpus.clear();
        add("all GPUs", "gpus", p);
    }
    if (varyStreams_->isChecked())
        for (int s : {1, 2, 3}) { PerfSettings p = perf_; p.streams = s; add(QString("%1 stream%2 per GPU").arg(s).arg(s == 1 ? "" : "s"), "streams", p); }
    if (varyReaders_->isChecked())
        for (int r : {8, 16, 32}) { PerfSettings p = perf_; p.readers = r; add(QString("%1 reader threads").arg(r), "readers", p); }
    if (varyPairs_->isChecked())
        for (int t : {2, 3, 4}) { PerfSettings p = perf_; p.pairThreads = t; add(QString("%1 alignment pairs in flight per GPU").arg(t), "pairs", p); }
    if (varyCache_->isChecked()) {
        PerfSettings p = perf_;
        p.cacheFrames = !perf_.cacheFrames;
        add(p.cacheFrames ? "frames resident in host memory" : "frames streamed from disk every pass (no cache)", "cache", p);
    }
}

void BenchmarkDialog::updatePlan() {
    if (running_) return;
    buildConfigs();
    int n = useSample() ? kSampleN : frames_.size();
    plan_->setText(QString("%1 runs of %2 frames, plus the combined best if it differs. Starting from: %3.")
                       .arg(configs_.size()).arg(n).arg(perf_.summary()));
}

void BenchmarkDialog::setRunning(bool on) {
    bool changed = running_ != on;
    running_ = on;
    if (changed) emit runningChanged(on);
    startBtn_->setText(on ? "Stop" : "Start");
    for (QWidget* w : std::initializer_list<QWidget*>{stack_, res_, varyGpus_, varyStreams_, varyReaders_, varyPairs_, varyCache_}) w->setEnabled(!on);
    varyGpus_->setEnabled(!on && gpus_.size() > 1);
    applyBtn_->setEnabled(!on && bestRow() >= 0);
    saveBtn_->setEnabled(!on && !rows_.isEmpty());
    copyBtn_->setEnabled(!on && !rows_.isEmpty());
}

void BenchmarkDialog::startStop() {
    if (running_) {
        stopping_ = true;
        worker_.cancel();
        stage_->setText("Stopping after the current run…");
        startBtn_->setEnabled(false);
        return;
    }
    if (worker_.running()) { stage_->setText("A render is running; wait for it to finish."); return; }
    rows_.clear();
    table_->setRowCount(0);
    stopping_ = false;
    combinedDone_ = false;
    buildConfigs();
    next_ = 0;
    setRunning(true);
    if (useSample() && sample_.isEmpty()) makeSample(); else beginSweep();
}

void BenchmarkDialog::makeSample() {
    stage_->setText("Generating the sample stack…");
    progress_->setRange(0, 0);
    if (sampleThread_.joinable()) sampleThread_.join();
    sampleThread_ = std::thread([this] {
        QStringList files = generate_sample(sampleDir_, [this](const QString& s) {
            QMetaObject::invokeMethod(this, [this, s] { stage_->setText(s); }, Qt::QueuedConnection);
        });
        QMetaObject::invokeMethod(this, [this, files] {
            sample_ = files;
            progress_->setRange(0, 1000);
            beginSweep();
        }, Qt::QueuedConnection);
    });
}

void BenchmarkDialog::beginSweep() {
    if (!running_) return;
    if (useSample() && base_.stack.cache_frames) worker_.releaseCache();   // the open stack's frames would only take the budget
    runNext();
}

void BenchmarkDialog::runNext() {
    if (stopping_ || next_ >= configs_.size()) { finishSweep(); return; }
    const Config& c = configs_[next_];
    JobOptions jo = base_;
    c.perf.apply(jo);
    jo.stack.po.g.scale = res_->currentIndex() == 1 ? 0.5 : 1.0;
    jo.stack.write_outputs = false;
    jo.out = QDir(QStandardPaths::writableLocation(QStandardPaths::TempLocation)).filePath("xLamina-benchmark").toStdString();
    if (useSample()) {
        jo.align.dir = sampleDir_.toStdWString();
        jo.align.paths.clear();
        for (const QString& f : sample_) jo.align.paths.push_back(f.toStdWString());
        jo.align.um_per_px = 0;
    }
    stage_->setText(QString("Run %1 of %2: %3").arg(next_ + 1).arg(configs_.size()).arg(c.label));
    progress_->setValue(0);
    clock_.start();
    worker_.start(jo, 12, true);
}

void BenchmarkDialog::onProgress(QString stage, double f) {
    if (!running_) return;
    stage_->setText(QString("Run %1 of %2: %3 — %4").arg(std::min(next_ + 1, (int)configs_.size())).arg(configs_.size()).arg(configs_.value(next_).label, stage));
    progress_->setValue((int)std::lround(std::clamp(f, 0.0, 1.0) * 1000));
}

void BenchmarkDialog::onFinished(std::shared_ptr<JobResultData> d) {
    if (!running_ || next_ >= configs_.size()) return;
    Row r;
    r.cfg = configs_[next_];
    r.rc = d->rc;
    r.align = d->align.seconds;
    r.fusion = d->result.seconds;
    r.total = clock_.elapsed() / 1000.0;
    r.fps = d->result.frames_per_second;
    r.pass1 = d->result.pass1_seconds;
    r.depth = d->result.depth_seconds;
    r.refine = d->result.refine_seconds;
    r.compose = d->result.compose_seconds;
    r.finalize = d->result.finalize_seconds;
    if (!(stopping_ && d->rc == 3)) { rows_ << r; addRow(r); }
    ++next_;
    runNext();
}

void BenchmarkDialog::addRow(const Row& r) {
    int row = table_->rowCount();
    table_->insertRow(row);
    auto cell = [&](int col, const QString& text, bool numeric = true) {
        QTableWidgetItem* it = new QTableWidgetItem(text);
        if (numeric) it->setTextAlignment(Qt::AlignRight | Qt::AlignVCenter);
        it->setToolTip(r.cfg.perf.summary());
        table_->setItem(row, col, it);
    };
    QString label = r.cfg.label;
    if (r.rc == 3) label += "  (cancelled)";
    else if (r.rc != 0) label += "  (failed, see the log)";
    cell(0, QString::number(row + 1));
    cell(1, label, false);
    if (r.rc == 0) {
        cell(2, QString::number(r.align, 'f', 2));
        cell(3, QString::number(r.fusion, 'f', 2));
        cell(4, QString::number(r.total, 'f', 2));
        cell(5, QString::number(r.fps, 'f', 1));
        cell(6, QString::number(r.pass1, 'f', 2));
        cell(7, QString::number(r.depth, 'f', 2));
        cell(8, QString::number(r.refine, 'f', 2));
        cell(9, QString::number(r.compose, 'f', 2));
        cell(10, QString::number(r.finalize, 'f', 2));
    }
    if (r.cfg.cold) for (int c = 0; c < table_->columnCount(); ++c) if (table_->item(row, c)) table_->item(row, c)->setForeground(QBrush(QColor(140, 140, 140)));
    table_->scrollToBottom();
}

int BenchmarkDialog::bestRow() const {
    int best = -1;
    for (int i = 0; i < rows_.size(); ++i)
        if (rows_[i].rc == 0 && !rows_[i].cfg.cold && (best < 0 || rows_[i].total < rows_[best].total)) best = i;
    return best;
}

// After the one-at-a-time sweep: take the best value of every varied setting and run them together,
// unless that combination has been run already.
void BenchmarkDialog::finishSweep() {
    if (!stopping_ && !combinedDone_ && rows_.size() > 1) {
        combinedDone_ = true;
        PerfSettings comb = perf_;
        auto bestOf = [&](const QString& dim, auto get, auto set) {
            double best = -1;
            decltype(get(perf_)) val = get(perf_);
            for (const Row& r : rows_) {
                if (r.rc != 0 || r.cfg.cold || (r.cfg.dimension != dim && !r.cfg.dimension.isEmpty())) continue;
                if (best < 0 || r.total < best) { best = r.total; val = get(r.cfg.perf); }
            }
            set(comb, val);
        };
        bestOf("gpus", [](const PerfSettings& p) { return p.gpus; }, [](PerfSettings& p, QList<int> v) { p.gpus = v; });
        bestOf("streams", [](const PerfSettings& p) { return p.streams; }, [](PerfSettings& p, int v) { p.streams = v; });
        bestOf("readers", [](const PerfSettings& p) { return p.readers; }, [](PerfSettings& p, int v) { p.readers = v; });
        bestOf("pairs", [](const PerfSettings& p) { return p.pairThreads; }, [](PerfSettings& p, int v) { p.pairThreads = v; });
        bestOf("cache", [](const PerfSettings& p) { return p.cacheFrames; }, [](PerfSettings& p, bool v) { p.cacheFrames = v; });
        bool seen = false;
        for (const Row& r : rows_) if (!r.cfg.cold && r.cfg.perf == comb) seen = true;
        if (!seen) {
            configs_ << Config{"combined best of each setting", "combined", comb, false};
            runNext();
            return;
        }
    }
    int best = bestRow();
    for (int i = 0; i < rows_.size(); ++i)
        for (int c = 0; c < table_->columnCount(); ++c)
            if (QTableWidgetItem* it = table_->item(i, c)) { QFont f = it->font(); f.setBold(i == best); it->setFont(f); }
    if (best >= 0) {
        const Row& b = rows_[best];
        double cur = -1;
        for (const Row& r : rows_) if (r.rc == 0 && !r.cfg.cold && r.cfg.dimension.isEmpty()) cur = r.total;
        QString s = QString("Fastest: %1, %2 s").arg(b.cfg.label).arg(b.total, 0, 'f', 2);
        if (cur > 0 && b.cfg.perf != perf_) s += QString(" (current settings %1 s, %2 %% slower)").arg(cur, 0, 'f', 2).arg(100 * (cur / b.total - 1), 0, 'f', 0);
        else if (cur > 0) s += "; the current settings are already the fastest";
        stage_->setText(s + (stopping_ ? ". Stopped." : "."));
    } else {
        stage_->setText(stopping_ ? "Stopped." : "No run finished; see the log.");
    }
    progress_->setValue(0);
    startBtn_->setEnabled(true);
    setRunning(false);
    if (!autoReport_.isEmpty()) {
        QFile f(autoReport_);
        if (f.open(QIODevice::WriteOnly | QIODevice::Text)) f.write(reportText().toUtf8());
        grab().save(autoReport_ + ".png");
        QString p = autoReport_;
        autoReport_.clear();
        QMetaObject::invokeMethod(this, [this] { close(); }, Qt::QueuedConnection);
    }
}

void BenchmarkDialog::autoRun(bool sample, const QString& reportPath) {
    autoReport_ = reportPath;
    for (int i = 0; i < stack_->count(); ++i) if (stack_->itemData(i).toInt() == (sample ? 1 : 0)) stack_->setCurrentIndex(i);
    QMetaObject::invokeMethod(this, &BenchmarkDialog::startStop, Qt::QueuedConnection);
}

void BenchmarkDialog::applyBest() {
    int b = bestRow();
    if (b < 0) return;
    perf_ = rows_[b].cfg.perf;
    emit applySettings(perf_);
    stage_->setText(QString("Applied: %1.").arg(perf_.summary()));
    updatePlan();
}

QString BenchmarkDialog::reportText() const {
    QString out;
    QTextStream ts(&out);
    ts << "xLamina " << XLAMINA_VERSION << " benchmark, " << QDateTime::currentDateTime().toString("yyyy-MM-dd HH:mm") << "\n";
    if (useSample()) ts << "Stack: built-in sample, " << kSampleN << " frames of " << kSampleW << " x " << kSampleH << "\n";
    else ts << "Stack: " << stackLabel_ << ", " << frames_.size() << " frames\n";
    ts << "Output: " << (res_->currentIndex() == 1 ? "half" : "full") << " resolution; outputs";
    for (const auto& o : base_.stack.outputs) ts << " " << QString::fromStdString(o);
    ts << "\n" << info_->text() << "\n\n";
    ts.setFieldAlignment(QTextStream::AlignLeft);
    auto line = [&](const QString& a, const QString& b, const QString& c, const QString& d, const QString& e, const QString& f, const QString& g,
                    const QString& h, const QString& i, const QString& j, const QString& k) {
        ts << qSetFieldWidth(4) << a << qSetFieldWidth(58) << b.left(57);
        ts.setFieldAlignment(QTextStream::AlignRight);
        ts << qSetFieldWidth(9) << c << d << e << f << g << h << i << j << k << qSetFieldWidth(0) << "\n";
        ts.setFieldAlignment(QTextStream::AlignLeft);
    };
    line("#", "configuration", "align s", "fusion s", "total s", "frames/s", "pass1 s", "depth s", "refine s", "compose", "finalize");
    int best = bestRow();
    for (int i = 0; i < rows_.size(); ++i) {
        const Row& r = rows_[i];
        QString label = r.cfg.label + (i == best ? "  <- fastest" : "") + (r.rc == 3 ? "  (cancelled)" : r.rc ? "  (failed)" : "");
        if (r.rc != 0) { line(QString::number(i + 1), label, "", "", "", "", "", "", "", "", ""); continue; }
        line(QString::number(i + 1), label, QString::number(r.align, 'f', 2), QString::number(r.fusion, 'f', 2), QString::number(r.total, 'f', 2),
             QString::number(r.fps, 'f', 1), QString::number(r.pass1, 'f', 2), QString::number(r.depth, 'f', 2), QString::number(r.refine, 'f', 2),
             QString::number(r.compose, 'f', 2), QString::number(r.finalize, 'f', 2));
    }
    ts << "\nSettings per run:\n";
    for (int i = 0; i < rows_.size(); ++i) ts << (i + 1) << ": " << rows_[i].cfg.perf.summary() << "\n";
    return out;
}

void BenchmarkDialog::saveReport() {
    QString def = QDir(QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation))
                      .filePath(QString("xLamina-benchmark-%1.txt").arg(QDateTime::currentDateTime().toString("yyyyMMdd-HHmm")));
    QString fn = QFileDialog::getSaveFileName(this, "Save the benchmark report", def, "Text (*.txt)");
    if (fn.isEmpty()) return;
    QFile f(fn);
    if (f.open(QIODevice::WriteOnly | QIODevice::Text)) { f.write(reportText().toUtf8()); stage_->setText("Saved " + QDir::toNativeSeparators(fn)); }
    else stage_->setText("Could not write " + fn);
}

void BenchmarkDialog::copyReport() {
    QApplication::clipboard()->setText(reportText());
    stage_->setText("Report copied to the clipboard.");
}

void BenchmarkDialog::closeEvent(QCloseEvent* e) {
    if (running_) {
        startStop();   // stop first; close again once it has stopped
        e->ignore();
        return;
    }
    e->accept();
}
