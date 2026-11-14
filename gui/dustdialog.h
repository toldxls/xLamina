// The dust map's window: the map with its contrast raised, what a setting finds in it marked on top, and the two
// settings that decide that: how much darker than its surroundings a pixel must be to count as dust, and how far
// around it the correction reaches. The marks follow the sliders, so the place where real specks end and the
// image's own noise begins can be seen instead of guessed.
#pragma once

#include <QDialog>
#include <QLabel>
#include <QPushButton>
#include <QSlider>
#include <QSpinBox>
#include <QTimer>

#include <memory>

#include "imageview.h"
#include "stackcore/dust.h"

class DustDialog : public QDialog {
    Q_OBJECT
public:
    // reads the map (a second or two for a 45 Mpx frame); loaded() is false, with error(), when it cannot be used
    DustDialog(const QString& path, float darker, int margin, QWidget* parent = nullptr);
    bool loaded() const { return loaded_; }
    QString error() const { return error_; }
    std::shared_ptr<const stackcore::DustFix> fix() const { return fix_; }   // what the settings found, once accepted
    float darker() const;   // 0.03 = 3 %
    int margin() const;

private:
    void detect();
    stackcore::DustMapData map_;
    std::shared_ptr<stackcore::DustFix> fix_;
    bool loaded_ = false;
    QString error_;
    ImageView* view_ = nullptr;
    QSlider* sens_ = nullptr;
    QLabel* sensValue_ = nullptr;
    QSpinBox* margin_ = nullptr;
    QLabel* info_ = nullptr;
    QPushButton* use_ = nullptr;
    QTimer timer_;
};
