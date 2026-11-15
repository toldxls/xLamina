#include "sweepdialog.h"

#include <QDialogButtonBox>
#include <QDir>
#include <QFileDialog>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QPushButton>
#include <QStringList>
#include <QToolButton>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>

using stackcore::StackOptions;
using stackcore::SweepVariant;

namespace {
std::string num(double v) { return QString::number(v, 'g', 4).toStdString(); }
QString depthName(double f) { return f >= 0.99 ? "full" : f >= 0.49 ? "half" : "quarter"; }
QString depthWords(double f) { return f >= 0.99 ? "Full" : f >= 0.49 ? "1/2" : "1/4"; }
}  // namespace

// The sets are made from the panel's values: each variant changes one of the settings that decide what B looks like
// where it goes wrong (FINDINGS 56, 57): the depth resolution (1/2 and 1/4 lose faint detail in different places),
// the fine focus's range and window (blobs around specks on bright faces, soft depth edges), Radius (C), which also
// guides B, and Radius (B) (outlines and bands beside strong edges). The earlier sets (soft labels, the background's
// blur, blend and end) changed little that a user could act on.
SweepDialog::SweepDialog(const StackOptions& base, const QString& defaultOut, QWidget* parent) : QDialog(parent) {
    setWindowTitle("Run sweep");
    resize(600, 380);
    QStringList quickWords, extraWords;
    auto add = [](std::vector<SweepVariant>& set, QStringList& words, const QString& name, const QString& said,
                  std::vector<std::pair<std::string, std::string>> options) {
        set.push_back({name.toStdString(), std::move(options)});
        words << said;
    };
    // ---- depth resolution: the other of 1/4 and 1/2 in the quick set, Full in the thorough one
    const double df = base.po.depth_fraction;
    const double other = df >= 0.49 && df < 0.99 ? 0.25 : 0.5;
    add(quickSet_, quickWords, "depth-" + depthName(other), "depth resolution " + depthWords(other), {{"--depth-fraction", num(other)}});
    ++quickGroups_;
    if (df < 0.99) { add(extraSet_, extraWords, "depth-full", "depth resolution Full (small stacks only)", {{"--depth-fraction", "1"}}); ++extraGroups_; }
    else { add(extraSet_, extraWords, "depth-quarter", "depth resolution 1/4", {{"--depth-fraction", "0.25"}}); ++extraGroups_; }
    // ---- the fine focus: half and double the range, double the window; off, three times and half the window as extras
    const int r = std::max(1, base.refine_r);
    const int rHalf = std::max(2, r / 2), rDouble = r * 2;
    add(quickSet_, quickWords, QString("fine-range-%1").arg(rHalf), QString("fine focus range %1").arg(rHalf), {{"--refine-r", num(rHalf)}});
    add(quickSet_, quickWords, QString("fine-range-%1").arg(rDouble), QString("%1").arg(rDouble), {{"--refine-r", num(rDouble)}});
    const double w = base.refine_fm_px > 0 ? base.refine_fm_px : 9.0;
    add(quickSet_, quickWords, QString("fine-window-%1").arg(w * 2, 0, 'g', 3), QString("fine focus window %1 px").arg(w * 2, 0, 'g', 3), {{"--refine-fm-px", num(w * 2)}});
    add(extraSet_, extraWords, "fine-focus-off", "fine focus off", {{"--refine-r", "0"}});
    add(extraSet_, extraWords, QString("fine-range-%1").arg(r * 3), QString("range %1").arg(r * 3), {{"--refine-r", num(r * 3)}});
    add(extraSet_, extraWords, QString("fine-window-%1").arg(w / 2, 0, 'g', 3), QString("window %1 px").arg(w / 2, 0, 'g', 3), {{"--refine-fm-px", num(w / 2)}});
    // ---- Radius (C): half and double (16 and 32 from 0); off as an extra. Each is a pass 1 of its own
    const double c = base.po.c_radius_px;
    const double c1 = c > 0 ? std::round(c / 2) : 16, c2 = c > 0 ? c * 2 : 32;
    add(quickSet_, quickWords, QString("c-radius-%1").arg(c1), QString("Radius (C) %1").arg(c1), {{"--c-radius", num(c1)}});
    add(quickSet_, quickWords, QString("c-radius-%1").arg(c2), QString("%1").arg(c2), {{"--c-radius", num(c2)}});
    quickGroups_ += 2;
    if (c > 0) { add(extraSet_, extraWords, "c-radius-0", "Radius (C) 0", {{"--c-radius", "0"}}); ++extraGroups_; }
    // ---- Radius (B): 16 and 32 from off, else half and double; a larger one as an extra
    const double b = base.po.b_radius_px;
    const double b1 = b > 0 ? std::round(b / 2) : 16, b2 = b > 0 ? b * 2 : 32;
    add(quickSet_, quickWords, QString("b-radius-%1").arg(b1), QString("Radius (B) %1").arg(b1), {{"--b-radius", num(b1)}});
    add(quickSet_, quickWords, QString("b-radius-%1").arg(b2), QString("%1").arg(b2), {{"--b-radius", num(b2)}});
    quickGroups_ += 2;
    add(extraSet_, extraWords, QString("b-radius-%1").arg(b2 * 2), QString("Radius (B) %1").arg(b2 * 2), {{"--b-radius", num(b2 * 2)}});
    ++extraGroups_;

    auto* lay = new QVBoxLayout(this);
    auto* intro = new QLabel("Renders B several times, each with one setting changed from the Render panel's: the depth resolution, "
                             "the fine focus, and the two radii. The compare window then opens with the variants listed by how much "
                             "they differ from the panel's B, so the first rows are the ones worth looking at.");
    intro->setWordWrap(true);
    lay->addWidget(intro);

    auto* g = new QGroupBox("Set");
    auto* gl = new QVBoxLayout(g);
    quick_ = new QRadioButton("Quick: " + quickWords.join(", "));
    quick_->setChecked(true);
    thorough_ = new QRadioButton("Thorough: the quick set plus " + extraWords.join(", "));
    gl->addWidget(quick_);
    gl->addWidget(thorough_);
    lay->addWidget(g);
    connect(quick_, &QRadioButton::toggled, this, [this] { updateSummary(); });

    auto* outRow = new QHBoxLayout;
    outRow->addWidget(new QLabel("Output folder:"));
    out_ = new QLineEdit(QDir::toNativeSeparators(defaultOut));
    out_->setToolTip("Each variant writes its own subfolder here; sweep.json lists them for the compare window.");
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
    all.insert(all.end(), quickSet_.begin(), quickSet_.end());
    if (thorough_->isChecked()) all.insert(all.end(), extraSet_.begin(), extraSet_.end());
    return all;
}

void SweepDialog::updateSummary() {
    const bool th = thorough_->isChecked();
    const size_t n = 1 + quickSet_.size() + (th ? extraSet_.size() : 0);
    const int groups = quickGroups_ + (th ? extraGroups_ : 0);
    const double secs = 7.0 + 5.0 * groups + 7.0 * n;   // alignment, a pass 1 + volume per group, a B per variant (this machine)
    summary_->setText(QString("%1 variants in %2 pass-1 groups, about %3 min on this machine.").arg(n).arg(groups).arg(secs / 60.0, 0, 'f', 1));
}
