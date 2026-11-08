// "Run sweep…": render B under a fixed set of alternative settings, chosen from what measurably changes B on real
// stacks (FINDINGS 35: a one-at-a-time sweep over every depth parameter), over one alignment and one pass 1 per
// group. The user picks the quick or the thorough set and the output folder; the two-up viewer opens on the result.
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
    QRadioButton* quick_ = nullptr;
    QRadioButton* thorough_ = nullptr;
    QLineEdit* out_ = nullptr;
    QLabel* summary_ = nullptr;
};
