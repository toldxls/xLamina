#include "sweepdialog.h"

#include <QDialogButtonBox>
#include <QDir>
#include <QFileDialog>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QPushButton>
#include <QToolButton>
#include <QVBoxLayout>

#include <iterator>

using stackcore::StackOptions;
using stackcore::SweepVariant;

namespace {
// The sets, from the 2026-11-08 parameter search scored against the source frames (FINDINGS 44). Detail in B is
// decided by the composition: the soft-label width is the dial (1 px = +10 % fine detail but seams, 6 = a quarter
// lost; Travis kept 3 over the sharper 2 at 1:1), the refinement's range and window and the B radius move it a
// little, aggregation 0 / 16 draw a blotch. Every depth knob (volume resolution, the ramp, the priors, SGM) leaves
// detail unchanged and only shifts the background, which the flat-region knobs tune on their own: blur width,
// blend width, the quiet end. The 2026-11-07 set (ramp, dark reach, halo prior, aggregation) is gone: within noise.
struct Preset { const char* name; std::vector<std::pair<std::string, std::string>> options; };
const Preset kQuick[] = {
    {"soft-labels-2", {{"--soft-px", "2"}}},
    {"soft-labels-1.5", {{"--soft-px", "1.5"}}},
    {"refine-range-24", {{"--refine-r", "24"}}},
    {"bg-0.6.21", {{"--flat-blur", "128"}, {"--flat-blend", "16"}}},
    {"bg-blur-512", {{"--flat-blur", "512"}}},
    {"bg-blend-64", {{"--flat-blend", "64"}}},
};
const Preset kThoroughExtra[] = {
    {"soft-labels-4", {{"--soft-px", "4"}}},
    {"refine-window-5", {{"--refine-fm-px", "5"}}},
    {"refine-window-15", {{"--refine-fm-px", "15"}}},
    {"b-radius-32", {{"--b-radius", "32"}}},
    {"bg-blend-off", {{"--flat-blend", "0"}}},
    {"bg-quiet-end", {{"--flat-end", "quiet"}}},
    {"volume-quarter", {{"--depth-fraction", "0.25"}}},
};
// variants that need their own pass 1 (fold + volume): flat-blur (two in the quick set), b-radius, depth-fraction
int pass1_groups(bool thorough) { return thorough ? 1 + 2 + 2 : 1 + 2; }
}  // namespace

SweepDialog::SweepDialog(const StackOptions&, const QString& defaultOut, QWidget* parent) : QDialog(parent) {
    setWindowTitle("Run sweep");
    resize(560, 360);
    auto* lay = new QVBoxLayout(this);
    auto* intro = new QLabel("Renders B under the settings that measurably change it, each on its own against the panel's settings, "
                             "over one alignment. The two-up viewer then opens with the variants listed by how much they differ "
                             "from the base, so the first rows are the ones worth looking at.");
    intro->setWordWrap(true);
    lay->addWidget(intro);

    auto* g = new QGroupBox("Set");
    auto* gl = new QVBoxLayout(g);
    quick_ = new QRadioButton("Quick: soft labels 2 / 1.5, refine range 24, the 0.6.21 background, background blur 512, background blend 64");
    quick_->setChecked(true);
    thorough_ = new QRadioButton("Thorough: the quick set plus soft labels 4, refine window 5 / 15, B radius 32, blend off, quiet end, 1/4 volume");
    gl->addWidget(quick_);
    gl->addWidget(thorough_);
    lay->addWidget(g);
    connect(quick_, &QRadioButton::toggled, this, [this] { updateSummary(); });

    auto* outRow = new QHBoxLayout;
    outRow->addWidget(new QLabel("Output folder:"));
    out_ = new QLineEdit(QDir::toNativeSeparators(defaultOut));
    out_->setToolTip("Each variant writes its own subfolder here; sweep.json lists them for the two-up viewer.");
    outRow->addWidget(out_, 1);
    auto* browse = new QToolButton;
    browse->setText("…");
    connect(browse, &QToolButton::clicked, this, [this] {
        QString d = QFileDialog::getExistingDirectory(this, "Sweep output folder", out_->text());
        if (!d.isEmpty()) out_->setText(QDir::toNativeSeparators(d));
    });
    outRow->addWidget(browse);
    lay->addLayout(outRow);

    summary_ = new QLabel;
    summary_->setWordWrap(true);
    lay->addWidget(summary_);
    lay->addStretch(1);

    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Cancel);
    QPushButton* run = buttons->addButton("Run sweep", QDialogButtonBox::AcceptRole);
    run->setDefault(true);
    connect(buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
    lay->addWidget(buttons);
    updateSummary();
}

std::vector<SweepVariant> SweepDialog::variants() const {
    std::vector<SweepVariant> all;
    all.push_back({"base", {}});
    for (const Preset& p : kQuick) all.push_back({p.name, p.options});
    if (thorough_->isChecked())
        for (const Preset& p : kThoroughExtra) all.push_back({p.name, p.options});
    return all;
}

void SweepDialog::updateSummary() {
    bool th = thorough_->isChecked();
    size_t n = 1 + std::size(kQuick) + (th ? std::size(kThoroughExtra) : 0);
    int groups = pass1_groups(th);
    double secs = 7.0 + 5.0 * groups + 7.0 * n;   // alignment, a pass 1 + volume per group, a B per variant (this machine)
    summary_->setText(QString("%1 variants in %2 pass-1 groups, about %3 min on this machine.").arg(n).arg(groups).arg(secs / 60.0, 0, 'f', 1));
}
