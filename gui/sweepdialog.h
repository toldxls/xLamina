// "Run sweep…": render B under a set of alternative settings, each changed on its own from the Render panel's, over
// one alignment and one pass 1 per group. The sets are the settings a user reaches for when B has a fault: the depth
// resolution, the fine focus's range and window, and the two radii. The user picks the quick or the thorough set and
// the output folder; the two-up viewer opens on the result.
#pragma once

#include <QDialog>
#include <QLabel>
#include <QLineEdit>
#include <QRadioButton>

#include <vector>

#include "stackcore/pipeline.h"

class SweepDialog : public QDialog {
    Q_OBJECT
public:
    SweepDialog(const stackcore::StackOptions& base, const QString& defaultOut, QWidget* parent = nullptr);
    std::vector<stackcore::SweepVariant> variants() const;   // the base first
    QString outDir() const { return out_->text(); }

private:
    void updateSummary();
    // the two sets, made from the panel's own values (half and double of what is set), and the pass-1 groups of each
    std::vector<stackcore::SweepVariant> quickSet_, extraSet_;
    int quickGroups_ = 1, extraGroups_ = 0;
    QRadioButton* quick_ = nullptr;
    QRadioButton* thorough_ = nullptr;
    QLineEdit* out_ = nullptr;
    QLabel* summary_ = nullptr;
};
