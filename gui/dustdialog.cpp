#include "dustdialog.h"

#include <QApplication>
#include <QDialogButtonBox>
#include <QDir>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QPainter>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>
#include <vector>

DustDialog::DustDialog(const QString& path, float darker, int margin, QWidget* parent) : QDialog(parent) {
    setWindowTitle("Dust map — " + QFileInfo(path).fileName());
    setWindowFlag(Qt::WindowContextHelpButtonHint, false);
    setWindowFlag(Qt::WindowMaximizeButtonHint, true);
    resize(1200, 820);

    QApplication::setOverrideCursor(Qt::WaitCursor);
    std::string why;
    loaded_ = stackcore::load_dust_map(path.toStdWString(), map_, &why);
    QApplication::restoreOverrideCursor();
    if (!loaded_) { error_ = QString::fromStdString(why); return; }

    QVBoxLayout* l = new QVBoxLayout(this);
    QLabel* hint = new QLabel("The map is shown with its contrast raised. <b>Red</b> is what would be corrected in every frame; each dust spot has a yellow ring. "
                              "Raise the sensitivity until the faint specks are caught. When rings, bands or scattered dots light up, it has reached the "
                              "image's own noise: go back a step.");
    hint->setWordWrap(true);
    l->addWidget(hint);
    view_ = new ImageView;
    l->addWidget(view_, 1);

    // the map as a picture: each pixel's share of its surroundings' light, 80 % .. 105 % over black .. white, so a
    // speck of a few percent shows (in the photograph itself it is a faint grey dot on white)
    {
        QImage img(map_.w, map_.h, QImage::Format_Grayscale8);
        for (int y = 0; y < map_.h; ++y) {
            uchar* row = img.scanLine(y);
            const float* r = map_.ratio.data() + (size_t)y * map_.w;
            for (int x = 0; x < map_.w; ++x) row[x] = (uchar)std::lround(std::clamp((r[x] - 0.80f) / 0.25f, 0.f, 1.f) * 255.f);
        }
        view_->setImage(img, false);
        view_->setLabel("dust map (contrast raised)");
    }

    QHBoxLayout* row = new QHBoxLayout;
    row->addWidget(new QLabel("Sensitivity: darker than its surroundings by"));
    sens_ = new QSlider(Qt::Horizontal);
    sens_->setRange(2, 20);   // half percents: 1 % .. 10 %
    sens_->setInvertedAppearance(true);   // to the right = more sensitive = a smaller difference counts
    sens_->setFixedWidth(260);
    sens_->setValue(std::clamp((int)std::lround(darker * 200.0), 2, 20));
    sens_->setToolTip("How much darker than its surroundings a pixel of the map must be to count as dust.\n"
                      "To the right = more sensitive: fainter specks are caught, and in the end the image's own noise.\n"
                      "3 % is the default. Dust is stronger in the frames than in a white frame, so a faint speck here can still leave a trail.");
    row->addWidget(sens_);
    sensValue_ = new QLabel;
    sensValue_->setMinimumWidth(70);
    row->addWidget(sensValue_);
    row->addSpacing(20);
    row->addWidget(new QLabel("Margin"));
    margin_ = new QSpinBox;
    margin_->setRange(2, 30);
    margin_->setSuffix(" px");
    margin_->setValue(std::clamp(margin, 2, 30));
    margin_->setToolTip("How far around each dust pixel the correction reaches: the shadow's soft edge, and the few pixels a speck\n"
                        "sits away from where the map shows it. 9 px is the default; larger for a map taken at another time than the stack.");
    row->addWidget(margin_);
    row->addSpacing(20);
    info_ = new QLabel;
    row->addWidget(info_, 1);
    l->addLayout(row);

    QDialogButtonBox* bb = new QDialogButtonBox(QDialogButtonBox::Cancel);
    use_ = bb->addButton("Use this dust map", QDialogButtonBox::AcceptRole);
    use_->setDefault(true);
    connect(bb, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(bb, &QDialogButtonBox::rejected, this, &QDialog::reject);
    l->addWidget(bb);

    // the marks follow the controls a moment after they stop moving (a detection takes a few tenths of a second)
    timer_.setSingleShot(true);
    timer_.setInterval(150);
    connect(&timer_, &QTimer::timeout, this, [this] { detect(); });
    connect(sens_, &QSlider::valueChanged, this, [this](int v) { sensValue_->setText(QString("%1 %").arg(v / 2.0, 0, 'f', 1)); timer_.start(); });
    connect(margin_, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int) { timer_.start(); });
    sensValue_->setText(QString("%1 %").arg(sens_->value() / 2.0, 0, 'f', 1));
    detect();
}

float DustDialog::darker() const { return sens_ ? sens_->value() / 200.f : 0.03f; }
int DustDialog::margin() const { return margin_ ? margin_->value() : 9; }

// What the settings find, drawn over the map at a quarter of its size: red where a pixel would be corrected, and a
// ring round each spot so that a speck of a few pixels can be found with the whole map in view.
void DustDialog::detect() {
    QApplication::setOverrideCursor(Qt::WaitCursor);
    auto fix = std::make_shared<stackcore::DustFix>();
    std::string why;
    const bool ok = stackcore::detect_dust(map_, darker(), margin(), *fix, &why);
    QApplication::restoreOverrideCursor();
    if (!ok) {
        fix_.reset();
        view_->setOverlay(QImage());
        info_->setText("<span style=\"color:#e06060\">" + QString::fromStdString(why).toHtmlEscaped() + "</span>");
        use_->setEnabled(false);
        return;
    }
    fix_ = fix;
    const int s = 4, ow = (map_.w + s - 1) / s, oh = (map_.h + s - 1) / s;
    std::vector<uint8_t> m((size_t)ow * oh, 0);
    for (uint32_t p : fix->px) m[(size_t)((p / map_.w) / s) * ow + (p % map_.w) / s] = 1;
    QImage ov(ow, oh, QImage::Format_ARGB32_Premultiplied);
    ov.fill(Qt::transparent);
    for (int y = 0; y < oh; ++y) {
        QRgb* row = reinterpret_cast<QRgb*>(ov.scanLine(y));
        for (int x = 0; x < ow; ++x) if (m[(size_t)y * ow + x]) row[x] = qPremultiply(qRgba(255, 40, 40, 150));
    }
    // the spots: connected groups of the marked blocks, a ring round each one's middle
    {
        QPainter p(&ov);
        p.setRenderHint(QPainter::Antialiasing);
        p.setPen(QPen(QColor(255, 220, 40, 230), 2.0));
        std::vector<uint32_t> stack;
        for (int y0 = 0; y0 < oh; ++y0)
            for (int x0 = 0; x0 < ow; ++x0) {
                if (m[(size_t)y0 * ow + x0] != 1) continue;
                double sx = 0, sy = 0, cnt = 0;
                int xmin = x0, xmax = x0, ymin = y0, ymax = y0;
                stack.push_back((uint32_t)((size_t)y0 * ow + x0));
                m[(size_t)y0 * ow + x0] = 2;
                while (!stack.empty()) {
                    const uint32_t q = stack.back();
                    stack.pop_back();
                    const int x = (int)(q % ow), y = (int)(q / ow);
                    sx += x; sy += y; ++cnt;
                    xmin = std::min(xmin, x); xmax = std::max(xmax, x); ymin = std::min(ymin, y); ymax = std::max(ymax, y);
                    for (int dy = -1; dy <= 1; ++dy)
                        for (int dx = -1; dx <= 1; ++dx) {
                            const int xx = x + dx, yy = y + dy;
                            if (xx < 0 || yy < 0 || xx >= ow || yy >= oh || m[(size_t)yy * ow + xx] != 1) continue;
                            m[(size_t)yy * ow + xx] = 2;
                            stack.push_back((uint32_t)((size_t)yy * ow + xx));
                        }
                }
                const double r = std::max(12.0, 0.5 * std::hypot(xmax - xmin + 1, ymax - ymin + 1) + 6.0);
                p.drawEllipse(QPointF(sx / cnt + 0.5, sy / cnt + 0.5), r, r);
            }
    }
    view_->setOverlayAt(ov, QPointF(0, 0), s);
    const double share = 100.0 * fix->px.size() / ((double)map_.w * map_.h);
    info_->setText(fix->spots == 0 ? QString("No dust found at this setting.")
                                   : QString("%1 spot%2, %3 pixels in and around them (%4 % of the image)").arg(fix->spots).arg(fix->spots == 1 ? "" : "s").arg(fix->px.size()).arg(share, 0, 'f', 3));
    use_->setEnabled(fix->spots > 0);
}
