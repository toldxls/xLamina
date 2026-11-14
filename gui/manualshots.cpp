#include "manualshots.h"

#include <QAbstractButton>
#include <QApplication>
#include <QDir>
#include <QDockWidget>
#include <QMouseEvent>
#include <QStatusBar>
#include <QFile>
#include <QFileInfo>
#include <QFormLayout>
#include <QJsonArray>
#include <QJsonDocument>
#include <QMenuBar>
#include <QSlider>
#include <QTabWidget>
#include <QToolBar>

#include "mainwindow.h"

namespace {
QString plain(QString s) { return s.remove('&').remove(QChar(0x2026)).trimmed(); }   // no mnemonics, no ellipsis

// a window is polished before it is first shown: the moment to say that it stays off the screen
class OffScreen : public QObject {
public:
    using QObject::QObject;
    bool eventFilter(QObject* o, QEvent* e) override {
        if (e->type() == QEvent::Polish)
            if (QWidget* w = qobject_cast<QWidget*>(o); w && w->isWindow()) w->setAttribute(Qt::WA_DontShowOnScreen);
        return false;
    }
};
}

void ManualShots::keepOffScreen() { qApp->installEventFilter(new OffScreen(qApp)); }

ManualShots::ManualShots(MainWindow& w, const QString& dir) : w_(w), dir_(dir) {
    QDir().mkpath(dir_);
    w_.resize(1500, 900);

    // while it renders: the split view, one grab per thing being built (the last of each stays)
    connect(&live_, &QTimer::timeout, this, [this] {
        if (!w_.rendering() || w_.liveView_->isHidden()) return;
        const QString what = w_.liveView_->toolTip();
        const QString shot = what.startsWith("depth") ? "live_depth" : what.startsWith("detail") ? "live_detail" : "live_compose";
        rects_.remove(shot);
        grab(&w_, shot);
        mark(shot, "left pane", QRect(w_.view_->mapTo(&w_, QPoint(0, 0)), w_.view_->size()), w_.size());
        mark(shot, "right pane", QRect(w_.liveView_->mapTo(&w_, QPoint(0, 0)), w_.liveView_->size()), w_.size());
    });
    live_.start(350);

    auto tab = [this](const QString& name) {
        for (int i = 0; i < w_.tabs_->count(); ++i) if (w_.tabs_->tabText(i) == name) w_.tabs_->setCurrentIndex(i);
    };
    auto dock = [this](const QString& object) { return w_.findChild<QDockWidget*>(object); };

    steps_ = {
        [this, tab, dock] { w_.setDeveloper(false); w_.resizeDocks({w_.framesDock_}, {390}, Qt::Horizontal); dock("geometry")->raise(); tab("B"); },
        [this, tab, dock] { w_.view_->fit(); },
        [this, tab, dock] { grab(&w_, "main"); },
        [this, tab, dock] {   // the list with a flagged frame in sight, when the stack has one
            QListWidgetItem* f = !w_.warped_.isEmpty() ? w_.warped_.first() : !w_.soft_.isEmpty() ? w_.soft_.first() : !w_.shifted_.isEmpty() ? w_.shifted_.first() : nullptr;
            if (f) w_.frameList_->scrollToItem(f, QAbstractItemView::PositionAtCenter);
        },
        [this, tab, dock] {
            grab(w_.framesDock_, "frames");
            // the rows the manual points at: the first one wholly in sight, and the flagged one
            QListWidget* l = w_.frameList_;
            auto row = [&](const QString& name, QListWidgetItem* it) {
                if (!it) return;
                const QRect r = l->visualItemRect(it).translated(l->viewport()->mapTo(w_.framesDock_, QPoint(0, 0)));
                mark("frames", name, r, w_.framesDock_->size());
            };
            QListWidgetItem* top = l->itemAt(QPoint(20, 4));
            if (top && l->visualItemRect(top).top() < 0) top = l->item(l->row(top) + 1);
            QListWidgetItem* flagged = !w_.warped_.isEmpty() ? w_.warped_.first() : !w_.soft_.isEmpty() ? w_.soft_.first() : !w_.shifted_.isEmpty() ? w_.shifted_.first() : nullptr;
            if (top && top == flagged) top = l->item(l->row(top) + 1);
            row("first row", top);
            row("flagged row", flagged);
            grab(dock("render"), "render_panel");
        },
        [this, tab, dock] { dock("geometry")->raise(); },
        [this, tab, dock] { grab(dock("geometry"), "geometry"); dock("log")->raise(); },
        [this, tab, dock] { grab(dock("log"), "log"); dock("hardware")->raise(); },
        [this, tab, dock] { grab(dock("hardware"), "hardware"); dock("geometry")->raise(); },
        // an output at 1:1
        [this, tab, dock] { tab("C"); w_.view_->actualSize(); w_.view_->centerOn(w_.view_->image().width() * 0.50, w_.view_->image().height() * 0.44); },
        [this, tab, dock] { grab(&w_, "main_1to1"); w_.view_->fit(); tab("B"); },
        // measuring: two lines drawn across crystals, as the mouse would draw them
        [this, tab, dock] { w_.measureAct_->setChecked(true); w_.view_->addMeasure(QLineF(0.36, 0.32, 0.47, 0.41)); w_.view_->addMeasure(QLineF(0.22, 0.66, 0.34, 0.60)); },
        [this, tab, dock] { grab(&w_, "measure"); w_.view_->clearMeasures(); w_.measureAct_->setChecked(false); },
        // preferences: General and Performance (System names this machine's folders, so it is described, not shown)
        [this, tab, dock] { w_.preferences(); },
        [this, tab, dock] { grab(w_.prefsDlg_, "prefs_general"); if (auto* t = w_.prefsDlg_->findChild<QTabWidget*>()) t->setCurrentIndex(1); },
        [this, tab, dock] { grab(w_.prefsDlg_, "prefs_performance"); if (auto* t = w_.prefsDlg_->findChild<QTabWidget*>()) t->setCurrentIndex(0); w_.prefsDlg_->hide(); },
        [this, tab, dock] { modal("batch", [this] { w_.batchDialog({w_.folder_}); }); },
        [this, tab, dock] { modal("sweep", [this] { w_.runSweep(); }); },
        [this, tab, dock] { w_.benchmark(); },
        [this, tab, dock] { if (w_.bench_) { grab(w_.bench_, "benchmark"); w_.bench_->close(); } },
        // the two-up viewer: B against C, the difference, retouching
        [this, tab, dock] { w_.twoUp(); w_.twoUp_->resize(1500, 900); w_.twoUp_->select(0, "B"); w_.twoUp_->select(1, "C"); },
        [this, tab, dock] { for (ImageView* v : w_.twoUp_->findChildren<ImageView*>()) { v->fit(); break; } },
        [this, tab, dock] { grab(w_.twoUp_, "compare"); },
        // measuring in the two-up viewer: a line drawn in the left pane shows in both
        [this, tab, dock] {
            for (QAction* a : w_.twoUp_->findChildren<QAction*>()) if (plain(a->text()) == "Measure") a->setChecked(true);
            if (ImageView* v = w_.twoUp_->findChildren<ImageView*>().value(0)) v->addMeasure(QLineF(0.36, 0.32, 0.47, 0.41));
        },
        [this, tab, dock] {
            grab(w_.twoUp_, "compare_measure");
            for (ImageView* v : w_.twoUp_->findChildren<ImageView*>()) v->clearMeasures();
            for (QAction* a : w_.twoUp_->findChildren<QAction*>()) if (plain(a->text()) == "Measure") a->setChecked(false);
        },
        [this, tab, dock] {
            ImageView* v = w_.twoUp_->findChildren<ImageView*>().value(0);
            v->actualSize();
            v->centerOn(v->image().width() * 0.50, v->image().height() * 0.44);
            for (QAction* a : w_.twoUp_->findChildren<QAction*>()) if (plain(a->text()) == "Difference") a->setChecked(true);
        },
        [this, tab, dock] { grab(w_.twoUp_, "compare_difference"); for (QAction* a : w_.twoUp_->findChildren<QAction*>()) if (plain(a->text()) == "Difference") a->setChecked(false); },
        [this, tab, dock] { w_.twoUp_->setRetouch(true); },
        [this, tab, dock] {   // the brush circle follows the mouse: a move over the left pane
            QWidget* vp = w_.twoUp_->findChildren<ImageView*>().value(0)->viewport();
            const QPointF pos(vp->width() * 0.5, vp->height() * 0.5);
            QMouseEvent e(QEvent::MouseMove, pos, vp->mapToGlobal(pos), Qt::NoButton, Qt::NoButton, Qt::NoModifier);
            QCoreApplication::sendEvent(vp, &e);
        },
        [this, tab, dock] { grab(w_.twoUp_, "compare_retouch"); w_.twoUp_->setRetouch(false); for (ImageView* v : w_.twoUp_->findChildren<ImageView*>()) { v->fit(); break; } },
        // the developer controls: the Render panel with them, the depth map with the frame under the cursor
        [this, tab, dock] { w_.setDeveloper(true); },
        [this, tab, dock] { grab(dock("render"), "render_panel_developer"); tab("depth"); },
        [this, tab, dock] { w_.probeHover(0.5, 0.6); },
        [this, tab, dock] { grab(&w_, "depth"); grab(w_.depthBar_, "depth_bar"); },
        // a layer held (as a click on it holds it), then two steps on with the keys
        [this, tab, dock] { w_.probeHover(-1, -1); w_.lockDepthFrame(24); w_.stepDepthFrame(1); w_.stepDepthFrame(1); },
        [this, tab, dock] { grab(&w_, "depth_layer"); w_.lockDepthFrame(-1); },
        [this, tab, dock] {   // bands: the palette repeating every 6 frames, without the highlight
            DepthStyle st = w_.depthStyle_;
            st.palette = 4; st.bands = 6; st.highlight = false; st.popup = false;
            w_.setDepthStyle(st, false);
            w_.probeHover(0.5, 0.6);
        },
        [this, tab, dock] { grab(w_.view_, "depth_bands"); DepthStyle st; w_.setDepthStyle(st, false); },
        // the depth map in the two-up viewer, over B
        [this, tab, dock] { w_.twoUp(); w_.twoUp_->select(0, "B"); w_.twoUp_->select(1, "depth"); },
        [this, tab, dock] { if (QSlider* s = w_.twoUp_->findChildren<QSlider*>().value(2)) s->setValue(50); },
        [this, tab, dock] { grab(w_.twoUp_, "compare_depth"); },
        // the dust map's window, when the run names a dust map (XLAMINA_MANUAL_DUST: any white-background frame): as it
        // opens, and with the sensitivity raised to 1.5 %. Dismissed without setting the map.
        [this, tab, dock] {
            const QString map = qEnvironmentVariable("XLAMINA_MANUAL_DUST");
            if (map.isEmpty() || !QFileInfo::exists(map)) return;
            QTimer::singleShot(1200, this, [this] {
                QWidget* d = QApplication::activeModalWidget();
                if (!d) return;
                grab(d, "dust_map");
                if (QSlider* s = d->findChild<QSlider*>()) s->setValue(3);
                QTimer::singleShot(2500, this, [this] {
                    QWidget* d2 = QApplication::activeModalWidget();
                    if (!d2) return;
                    grab(d2, "dust_map_sensitive");
                    if (auto* q = qobject_cast<QDialog*>(d2)) q->reject();
                });
            });
            w_.setDustMap(map, false);   // returns when the window has been dismissed
        },
        [this, tab, dock] { finish(); },
    };
    connect(&run_, &QTimer::timeout, this, [this] {
        if (w_.rendering() || !w_.result_) return;   // the render first
        live_.stop();
        if (next_ < steps_.size()) steps_[next_++]();
    });
    run_.start(900);
}

void ManualShots::mark(const QString& shot, const QString& name, const QRect& r, const QSize& full) {
    if (name.isEmpty() || r.isEmpty() || full.isEmpty()) return;
    QJsonObject s = rects_.value(shot).toObject();
    if (s.contains(name)) return;   // the first of a name is the one on top of the window
    auto f = [](double v) { return std::round(v * 10000.0) / 10000.0; };
    s[name] = QJsonArray{f(r.x() / (double)full.width()), f(r.y() / (double)full.height()), f(r.width() / (double)full.width()), f(r.height() / (double)full.height())};
    rects_[shot] = s;
}

// every control with a name: buttons and check boxes by their text, groups and docks by their title, form rows by
// their label (the label and its field together), tabs, toolbar buttons and menus
void ManualShots::markAll(const QString& shot, QWidget* root) {
    auto rectOf = [root](QWidget* c) { return QRect(c->mapTo(root, QPoint(0, 0)), c->size()); };
    auto shown = [root](QWidget* c) { return c && c->window() == root->window() && c->isVisibleTo(root) && !c->visibleRegion().isEmpty(); };   // not the controls of another window that is this one's child
    for (QToolBar* tb : root->findChildren<QToolBar*>())
        for (QAction* a : tb->actions())
            if (QWidget* b = tb->widgetForAction(a); shown(b) && !a->text().isEmpty()) mark(shot, plain(a->text()), rectOf(b), root->size());
    for (QMenuBar* mb : root->findChildren<QMenuBar*>())
        for (QAction* a : mb->actions()) mark(shot, "menu " + plain(a->text()), mb->actionGeometry(a).translated(mb->mapTo(root, QPoint(0, 0))), root->size());
    for (QAbstractButton* b : root->findChildren<QAbstractButton*>()) if (shown(b)) mark(shot, plain(b->text()), rectOf(b), root->size());
    for (QGroupBox* g : root->findChildren<QGroupBox*>()) if (shown(g)) mark(shot, "group " + plain(g->title()).section(' ', 0, 0), rectOf(g), root->size());
    for (QDockWidget* d : root->findChildren<QDockWidget*>()) if (shown(d)) mark(shot, "panel " + d->windowTitle().section(" (", 0, 0), rectOf(d), root->size());
    for (QFormLayout* f : root->findChildren<QFormLayout*>())
        for (int r = 0; r < f->rowCount(); ++r) {
            QLayoutItem* li = f->itemAt(r, QFormLayout::LabelRole);
            QLayoutItem* fi = f->itemAt(r, QFormLayout::FieldRole);
            QLabel* label = li ? qobject_cast<QLabel*>(li->widget()) : nullptr;
            if (!shown(label) || !fi || !shown(fi->widget())) continue;
            mark(shot, "row " + plain(label->text()), rectOf(label).united(rectOf(fi->widget())), root->size());
        }
    for (QTabBar* t : root->findChildren<QTabBar*>())
        if (shown(t)) for (int i = 0; i < t->count(); ++i) mark(shot, "tab " + plain(t->tabText(i)), t->tabRect(i).translated(t->mapTo(root, QPoint(0, 0))), root->size());
    int n = 0, sliders = 0;
    QList<ImageView*> views = root->findChildren<ImageView*>();   // left to right: "view", then "view 2"
    std::sort(views.begin(), views.end(), [&](ImageView* x, ImageView* y) { return rectOf(x).x() < rectOf(y).x(); });
    for (ImageView* v : views) if (shown(v)) mark(shot, n++ == 0 ? "view" : "view 2", rectOf(v), root->size());
    for (QListWidget* l : root->findChildren<QListWidget*>()) if (shown(l)) mark(shot, "list", rectOf(l), root->size());
    for (QTableWidget* l : root->findChildren<QTableWidget*>()) if (shown(l)) mark(shot, "table", rectOf(l), root->size());
    for (QProgressBar* p : root->findChildren<QProgressBar*>()) if (shown(p)) mark(shot, "progress", rectOf(p), root->size());
    for (QStatusBar* s : root->findChildren<QStatusBar*>()) if (shown(s)) mark(shot, "status bar", rectOf(s), root->size());
    for (DepthBar* d : root->findChildren<DepthBar*>()) if (shown(d)) mark(shot, "depth bar", rectOf(d), root->size());
    for (QSlider* s : root->findChildren<QSlider*>()) if (shown(s)) mark(shot, QString("slider %1").arg(++sliders), rectOf(s), root->size());
}

void ManualShots::grab(QWidget* root, const QString& shot) {
    if (!root) return;
    // a whole window is mostly photograph: JPEG there (a fifth of the PNG's size), PNG for the small panels and dialogs
    // two pixels per screen pixel, so the manual's zoom view has something to show: text and controls are drawn
    // at that scale, and a viewer's image from twice as many of its own pixels
    QPixmap px(root->size() * 2);
    px.setDevicePixelRatio(2.0);
    px.fill(root->palette().color(QPalette::Window));
    root->render(&px);
    const bool photo = root->width() * root->height() > 900000;
    QFile::remove(QDir(dir_).filePath(shot + (photo ? ".png" : ".jpg")));
    px.save(QDir(dir_).filePath(shot + (photo ? ".jpg" : ".png")), photo ? "JPG" : "PNG", photo ? 88 : -1);
    markAll(shot, root);
}

void ManualShots::modal(const QString& shot, const std::function<void()>& open) {
    QTimer::singleShot(700, this, [this, shot] {
        QWidget* d = QApplication::activeModalWidget();
        if (!d) return;
        grab(d, shot);
        if (auto* q = qobject_cast<QDialog*>(d)) q->reject(); else d->close();
    });
    open();   // returns when the dialog has been dismissed
}

void ManualShots::finish() {
    run_.stop();
    QFile f(QDir(dir_).filePath("rects.js"));
    if (f.open(QIODevice::WriteOnly)) {
        f.write("// where each control sits in each grab (x, y, width, height as fractions); written by xLamina --manual\nwindow.RECTS = ");
        f.write(QJsonDocument(rects_).toJson(QJsonDocument::Indented));
        f.write(";\n");
    }
    QCoreApplication::quit();
}
