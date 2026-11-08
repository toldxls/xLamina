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
// The sets, from the 2026-11-07 sensitivity sweep on the real stack (share of B's pixels changed by > 2 %):
// depth volume 1/4: 7.9 %, background blur off: 6.9 %, confidence ramp high: 4.7 %, background blur 256: 3.5 %,
// confidence ramp low: 2.5 %, B radius 32: 2.1 %, aggregation off / 64: 2.1 / 1.6 %, dark reach 48 / 0: 1.5 / 1.4 %,
// halo prior off: 0.9 %. SGM P1 / P2, soft label width, end vote, fine-focus range and window, dark prior strength
// and C radius all moved less than 0.5 % of the pixels and are not worth a look.
struct Preset { const char* name; std::vector<std::pair<std::string, std::string>> options; };
const Preset kQuick[] = {
    {"bg-blur-off", {{"--flat-blur", "0"}}},
    {"trust-focus-less", {{"--conf-lo", "0.55"}, {"--conf-hi", "0.75"}}},
    {"trust-focus-more", {{"--conf-lo", "0.35"}, {"--conf-hi", "0.55"}}},
    {"volume-quarter", {{"--depth-fraction", "0.25"}}},
    {"dark-prior-everywhere", {{"--dark-reach", "0"}}},
};
const Preset kThoroughExtra[] = {
    {"bg-blur-256", {{"--flat-blur", "256"}}},
    {"aggregation-off", {{"--vol-agg-px", "0"}}},
    {"aggregation-64", {{"--vol-agg-px", "64"}}},
    {"dark-reach-48", {{"--dark-reach", "48"}}},
    {"halo-prior-off", {{"--halo-prior", "0"}}},
    {"b-radius-32", {{"--b-radius", "32"}}},
};
// variants that need their own pass 1 (fold + volume): flat-blur, depth-fraction, vol-agg-px, b-radius
int pass1_groups(bool thorough) { return thorough ? 1 + 1 + 1 + 1 + 2 + 1 : 1 + 1 + 1; }
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
    quick_ = new QRadioButton("Quick: background blur off, focus trusted less / more, 1/4 volume, dark prior everywhere");
    quick_->setChecked(true);
    thorough_ = new QRadioButton("Thorough: the quick set plus background blur 256, aggregation off / 64, dark reach 48, halo prior off, B radius 32");
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
