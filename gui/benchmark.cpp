#include "benchmark.h"

#include <QApplication>
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
// a raised plate with a bright thin rim (like the edge-on edges of thin plate crystals), per-frame
// defocus that grows with the distance to the frame's depth, sensor noise, and a little wobble.
// Deterministic, so two machines running it are comparable.
namespace {
const int kSampleW = 4000, kSampleH = 2667, kSampleN = 32;   // 2 GB of frames: runs of a few seconds, not sub-second noise
const int kQuickFrames = 40;                                 // the quick benchmark's subset of the open stack
const float kDefocusPxPerFrame = 1.0f;                       // the sample's blur growth (sigma) per frame of distance

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
    progress("generating the sample stack (once): scene");
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
    // the truth, so outputs can be scored against it (FINDINGS 31): the depth per pixel in frames (grey,
    // value = z / (N - 1), 16-bit) and the all-in-focus image, in a truth/ subfolder so no stacker reads them
    {
        QString td = QDir(dir).filePath("truth");
        QDir().mkpath(td);
        std::vector<float> z3(n * 3), sharp(n * 3);
        for (size_t i = 0; i < n; ++i) {
            float v = Z[i] / (N - 1);
            z3[i * 3] = z3[i * 3 + 1] = z3[i * 3 + 2] = v;
            sharp[i * 3] = tintR[i] * T[i]; sharp[i * 3 + 1] = tintG[i] * T[i]; sharp[i * 3 + 2] = tintB[i] * T[i];
        }
        write_tiff_rgb16(QDir(td).filePath("depth_frames_over_N-1.tif").toStdString(), z3, W, H, {});
        write_tiff_rgb16(QDir(td).filePath("sharp.tif").toStdString(), sharp, W, H, {});
        QFile f(QDir(td).filePath("README.txt"));
        if (f.open(QIODevice::WriteOnly | QIODevice::Text))
            f.write(QString("xLamina synthetic sample: %1 frames of %2 x %3. Frame k is in focus where depth = k; defocus sigma = %5 px per frame of distance.\n"
                            "depth_frames_over_N-1.tif: grey = depth / %4 (16-bit). sharp.tif: the all-in-focus scene. Frame k is shifted by\n"
                            "(round(1.5 sin 0.9k), round(1.2 cos 0.7k)) px; frame 0's shift is (0, 1). Plate: ellipse centre (0.6 W, 0.5 H), radii (0.25 W, 0.30 H),\n"
                            "depth %4 x (0.82 + 0.08 ex), bright rim where |ex^2 + ey^2 - 1| < 0.012; matrix depth %4 x 0.45 x (0.5 + 0.5 sin(2 pi 1.3 x/W + 0.7) cos(2 pi 0.9 y/H + 1.9)).\n")
                        .arg(N).arg(W).arg(H).arg(N - 1).arg(kDefocusPxPerFrame).toUtf8());
    }
    // defocus grows 1 px of sigma per frame of distance, about what a 20x / NA 0.4 objective does at
    // 0.468 um steps and 0.14 um/px (0.45 px/frame left the depth underdetermined at 1/4 resolution)
    const float sigmas[] = {0.f, 0.7f, 1.5f, 3.f, 5.f, 8.f, 12.f, 17.f, 24.f, 32.f};
    const int L = 10;
    std::vector<std::vector<float>> levels(L);
    std::vector<float> tmp;
    for (int l = 0; l < L; ++l) {
        progress(QString("generating the sample stack (once): defocus level %1 of %2").arg(l + 1).arg(L));
        gauss_blur(T, levels[l], tmp, W, H, sigmas[l]);
    }
    std::vector<float>().swap(tmp);
    std::vector<float> rgb(n * 3);
    for (int k = 0; k < N; ++k) {
        progress(QString("generating the sample stack (once): frame %1 of %2").arg(k + 1).arg(N));
        int dx = (int)std::lround(1.5 * std::sin(0.9 * k)), dy = (int)std::lround(1.2 * std::cos(0.7 * k));
        uint32_t rng = 12345u + 7919u * k;
        for (int y = 0; y < H; ++y) {
            int sy = std::clamp(y + dy, 0, H - 1);
            for (int x = 0; x < W; ++x) {
                int sx = std::clamp(x + dx, 0, W - 1);
                size_t s = (size_t)sy * W + sx, o = ((size_t)y * W + x) * 3;
                float d = kDefocusPxPerFrame * std::abs(Z[s] - k);
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
    resize(960, 560);
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
    mode_ = new QComboBox;
    mode_->addItem(QString("Quick (about 20 s): up to %1 frames at half resolution; the current settings, each GPU alone, frames streamed").arg(kQuickFrames), 0);
    mode_->addItem("Thorough (minutes): the whole stack at the Render panel's resolution, every setting varied, then the best of each combined", 1);
    mode_->addItem("Single run: the whole stack at the Render panel's resolution on the current settings only; Start again adds a run to compare", 2);
    mode_->setItemData(2, "For comparing the machine itself (an overclock, a driver, a cooling change): one render's timing and nothing else.\n"
                          "Change the machine, press Start again: the earlier runs stay in the table, each with its time of day.",
                       Qt::ToolTipRole);
    fl->addRow("Mode", mode_);
    plan_ = new QLabel;
    plan_->setWordWrap(true);
    fl->addRow("", plan_);
    vl->addWidget(what);
    connect(stack_, &QComboBox::currentIndexChanged, this, &BenchmarkDialog::updatePlan);
    connect(mode_, &QComboBox::currentIndexChanged, this, &BenchmarkDialog::updatePlan);

    table_ = new QTableWidget(0, 11);
    table_->setHorizontalHeaderLabels({"#", "Configuration", "Align s", "Fusion s", "Total s", "Frames/s", "Pass 1 s", "Depth s", "Refine s", "Compose s", "Finalize s"});
    table_->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);   // the numbers narrow, the label gets the rest
    table_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch);
    table_->verticalHeader()->hide();
    table_->setSelectionBehavior(QAbstractItemView::SelectRows);
    table_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    table_->setAlternatingRowColors(true);
    table_->setToolTip("Total is the wall time of the whole job (alignment, fusion, report; nothing is written to disk). A first run that\n"
                       "reads the frames from disk is marked cold and left out of the comparison. Bold: the fastest warm run.");
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

bool BenchmarkDialog::quick() const { return mode_->currentData().toInt() == 0; }
bool BenchmarkDialog::thorough() const { return mode_->currentData().toInt() == 1; }
bool BenchmarkDialog::single() const { return mode_->currentData().toInt() == 2; }
bool BenchmarkDialog::useSample() const { return stack_->currentData().toInt() == 1; }

// the quick benchmark takes every n-th frame of the open stack so that at most kQuickFrames remain
QStringList BenchmarkDialog::runFrames() const {
    const QStringList& all = useSample() ? sample_ : frames_;
    if (!quick() || all.size() <= kQuickFrames) return all;
    int step = (all.size() + kQuickFrames - 1) / kQuickFrames;
    QStringList out;
    for (int i = 0; i < all.size(); i += step) out << all[i];
    return out;
}

// Quick: the current settings (twice when the frames must first be read: the cold run is not compared),
// each GPU alone, the frames streamed instead of resident. Thorough: every setting varied in turn; the
// combined best is added at the end of the run.
void BenchmarkDialog::buildConfigs() {
    configs_.clear();
    auto add = [&](const QString& label, const QString& dim, PerfSettings p, bool cold = false) {
        for (const Config& c : configs_) if (!cold && !c.cold && c.perf == p) return;   // the same settings once (the cold run aside)
        configs_ << Config{label, dim, p, cold};
    };
    QStringList fr = runFrames();
    bool resident = perf_.cacheFrames && !fr.isEmpty() && worker_.resident(fr);
    if (perf_.cacheFrames && !resident) add("current settings, frames read from disk (cold)", "", perf_, true);
    add("current settings", "", perf_);
    if (single()) return;   // one render, to compare the machine against itself
    if (gpus_.size() > 1) {
        for (const GpuInfo& g : gpus_) {
            PerfSettings p = perf_;
            p.gpus = {g.index};
            add(QString("GPU %1 only (%2)").arg(g.index).arg(g.name), "gpus", p);
        }
        PerfSettings p = perf_;
        p.gpus.clear();
        add("all GPUs", "gpus", p);
    }
    if (thorough()) {
        for (int s : {1, 2, 3}) { PerfSettings p = perf_; p.streams = s; add(QString("%1 stream%2 per GPU").arg(s).arg(s == 1 ? "" : "s"), "streams", p); }
        for (int r : {8, 16, 32}) { PerfSettings p = perf_; p.readers = r; add(QString("%1 reader threads").arg(r), "readers", p); }
        for (int t : {2, 3, 4}) { PerfSettings p = perf_; p.pairThreads = t; add(QString("%1 alignment pairs in flight per GPU").arg(t), "pairs", p); }
    }
    {
        PerfSettings p = perf_;
        p.cacheFrames = !perf_.cacheFrames;
        add(p.cacheFrames ? "frames resident in host memory" : "frames streamed from disk every pass (no cache)", "cache", p);
    }
}

void BenchmarkDialog::updatePlan() {
    if (running_) return;
    buildConfigs();
    int n = runFrames().size();
    if (useSample() && sample_.isEmpty()) n = quick() ? std::min(kQuickFrames, kSampleN) : kSampleN;
    plan_->setText(QString("%1 run%6 of %2 frames at %3 resolution%4. %7: %5.")
                       .arg(configs_.size()).arg(n).arg(quick() ? "half" : base_.stack.po.g.scale == 0.5 ? "half" : "full")
                       .arg(thorough() ? ", plus the combined best if it differs" : "").arg(perf_.summary())
                       .arg(configs_.size() == 1 ? "" : "s").arg(single() ? "Settings" : "Starting from"));
}

void BenchmarkDialog::setRunning(bool on) {
    bool changed = running_ != on;
    running_ = on;
    if (changed) emit runningChanged(on);
    startBtn_->setText(on ? "Stop" : "Start");
    stack_->setEnabled(!on);
    mode_->setEnabled(!on);
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
    // single runs of the same stack accumulate, so that one machine setting can be compared with the next
    int key = single() ? (useSample() ? 2 : 1) : 0;
    if (key == 0 || key != keepKey_) {
        rows_.clear();
        table_->setRowCount(0);
    }
    keepKey_ = key;
    stopping_ = false;
    combinedDone_ = false;
    next_ = 0;
    setRunning(true);
    sweepClock_.start();
    if (useSample() && sample_.isEmpty()) makeSample();
    else { buildConfigs(); beginSweep(); }
}

void BenchmarkDialog::makeSample() {
    stage_->setText("Generating the sample stack (once)…");
    progress_->setRange(0, 0);
    if (sampleThread_.joinable()) sampleThread_.join();
    sampleThread_ = std::thread([this] {
        QStringList files = generate_sample(sampleDir_, [this](const QString& s) {
            QMetaObject::invokeMethod(this, [this, s] { stage_->setText(s); }, Qt::QueuedConnection);
        });
        QMetaObject::invokeMethod(this, [this, files] {
            sample_ = files;
            progress_->setRange(0, 1000);
            buildConfigs();
            sweepClock_.start();   // the generation is not part of the benchmark
            beginSweep();
        }, Qt::QueuedConnection);
    });
}

void BenchmarkDialog::beginSweep() {
    if (!running_) return;
    // the open stack's frames would only take the budget (the sample's own, left by an earlier run, stay)
    if (useSample() && base_.stack.cache_frames && !worker_.resident(runFrames())) worker_.releaseCache();
    runNext();
}

void BenchmarkDialog::runNext() {
    if (stopping_ || next_ >= configs_.size()) { finishSweep(); return; }
    const Config& c = configs_[next_];
    JobOptions jo = base_;
    c.perf.apply(jo);
    if (quick()) jo.stack.po.g.scale = 0.5;
    jo.stack.write_outputs = false;
    jo.out = QDir(QStandardPaths::writableLocation(QStandardPaths::TempLocation)).filePath("xLamina-benchmark").toStdString();
    jo.align.paths.clear();
    for (const QString& f : runFrames()) jo.align.paths.push_back(f.toStdWString());
    if (useSample()) {
        jo.align.dir = sampleDir_.toStdWString();
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
    if (single()) r.cfg.label += QDateTime::currentDateTime().toString(", HH:mm:ss");   // which run was which overclock
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

// After a thorough sweep: take the best value of every varied setting and run them together, unless that
// combination has been run already.
void BenchmarkDialog::finishSweep() {
    if (thorough() && !stopping_ && !combinedDone_ && rows_.size() > 1) {
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
    double took = sweepClock_.elapsed() / 1000.0;
    if (best >= 0 && single()) {
        // every row has the same settings: report the last run against the fastest so far
        const Row& b = rows_[best];
        int last = -1, warm = 0;
        for (int i = 0; i < rows_.size(); ++i) if (rows_[i].rc == 0 && !rows_[i].cfg.cold) { last = i; ++warm; }
        const Row& l = rows_[last];
        QString s = QString("This run: %1 s, %2 frames/s").arg(l.total, 0, 'f', 2).arg(l.fps, 0, 'f', 1);
        if (warm > 1 && last == best) s += QString("; the fastest of %1 runs").arg(warm);
        else if (warm > 1) s += QString("; %1 % slower than the fastest of %2 runs (%3 s%4)").arg(100 * (l.total / b.total - 1), 0, 'f', 1).arg(warm).arg(b.total, 0, 'f', 2).arg(b.cfg.label.mid(b.cfg.label.indexOf(',')));
        stage_->setText(s + (stopping_ ? ". Stopped." : "."));
    } else if (best >= 0) {
        const Row& b = rows_[best];
        double cur = -1;
        for (const Row& r : rows_) if (r.rc == 0 && !r.cfg.cold && r.cfg.dimension.isEmpty()) cur = r.total;
        QString s = QString("Fastest: %1, %2 s").arg(b.cfg.label).arg(b.total, 0, 'f', 2);
        if (cur > 0 && b.cfg.perf != perf_) s += QString(" (current settings %1 s, %2 % slower)").arg(cur, 0, 'f', 2).arg(100 * (cur / b.total - 1), 0, 'f', 0);
        else if (cur > 0) s += "; the current settings are already the fastest";
        stage_->setText(s + QString(". Benchmark took %1 s%2").arg(took, 0, 'f', 0).arg(stopping_ ? ", stopped." : "."));
    } else {
        stage_->setText(stopping_ ? "Stopped." : "No run finished; see the log.");
    }
    progress_->setValue(0);
    startBtn_->setEnabled(true);
    setRunning(false);
    updatePlan();   // the frames are resident now: the next Start has no cold run
    if (!autoReport_.isEmpty() && --autoStarts_ > 0) {
        QMetaObject::invokeMethod(this, &BenchmarkDialog::startStop, Qt::QueuedConnection);
        return;
    }
    if (!autoReport_.isEmpty()) {
        QFile f(autoReport_);
        if (f.open(QIODevice::WriteOnly | QIODevice::Text)) f.write(reportText().toUtf8());
        grab().save(autoReport_ + ".png");
        autoReport_.clear();
        QMetaObject::invokeMethod(this, [this] { close(); }, Qt::QueuedConnection);
    }
}

void BenchmarkDialog::autoRun(bool sample, const QString& reportPath) {
    autoReport_ = reportPath;
    // XLAMINA_BENCH_MODE = 0 quick (default), 1 thorough, 2 single run; XLAMINA_BENCH_STARTS = presses of Start
    int mode = qEnvironmentVariableIntValue("XLAMINA_BENCH_MODE");
    for (int i = 0; i < mode_->count(); ++i) if (mode_->itemData(i).toInt() == mode) mode_->setCurrentIndex(i);
    autoStarts_ = std::max(1, qEnvironmentVariableIntValue("XLAMINA_BENCH_STARTS"));
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
    if (useSample()) ts << "Stack: built-in sample, " << kSampleN << " frames of " << kSampleW << " x " << kSampleH;
    else ts << "Stack: " << stackLabel_ << ", " << frames_.size() << " frames";
    ts << "; " << (quick() ? "quick" : single() ? "single-run" : "thorough") << " mode, " << runFrames().size() << " frames per run at "
       << ((quick() || base_.stack.po.g.scale == 0.5) ? "half" : "full") << " resolution; outputs";
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
