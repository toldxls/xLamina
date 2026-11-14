#include <QApplication>
#include <QDir>
#include <QIcon>
#include <QTimer>

#include <QMessageBox>

#include "diaglog.h"
#include "mainwindow.h"
#include "manualshots.h"
#include "prefs.h"
#include "updater.h"

// xLamina [FOLDER] [--render] [--shots DIR [--every S]] [--bench REPORT.txt [--sample]] [--self-update] [--batch DIR DIR ...]
//   --batch   every argument after it is a folder: render them one after the other with the saved settings, then quit
//   --render  start a render of FOLDER at once
//   --render-in S       developer aid: start the render S seconds after opening (the thumbnails' cache fill first)
//   --shots   developer aid: save a grab of the window to DIR every S seconds (default 3) while a render runs
//   --bench   developer aid: run the benchmark unattended (on FOLDER, or the built-in sample with --sample),
//             write REPORT.txt and REPORT.txt.png, then quit
//   --self-update       developer aid: install the latest release without asking and restart
//   --updated-from V    passed by the updater to the new instance, which says so in the status bar
//   --manual DIR        developer aid: with --render, save the manual's screen grabs and rects.js to DIR, then quit
//                       (manualshots.h; tools\manual.cmd runs it off screen)
//   --report DIR        save the problem report (Help > Report a problem) into DIR and quit
//   --hover FX,FY       developer aid: after the render, show the last tab and move a synthetic cursor to that
//                       fraction of the view (the depth map's note and highlight, for an unattended grab)
// An exception that reaches the event loop (a slot that ran out of memory converting a whole image, say) ended the
// program without a word. It is logged and said instead, and the window stays.
class App : public QApplication {
public:
    using QApplication::QApplication;
    bool notify(QObject* o, QEvent* e) override {
        try { return QApplication::notify(o, e); }
        catch (const std::bad_alloc&) { report("this PC ran out of memory (RAM)"); }
        catch (const std::exception& x) { report(QString::fromLocal8Bit(x.what())); }
        catch (...) { report("an unknown error"); }
        return false;
    }

private:
    void report(const QString& what) {
        DiagLog::write("unexpected error in the window: " + what);
        if (busy_) return;   // one box, not one per event
        busy_ = true;
        QMessageBox::warning(activeWindow(), "xLamina", "Something went wrong: " + what + ".\n\nxLamina keeps running, but it is safest to save your "
                                                        "work and restart it. Help > Report a problem saves the details to send.");
        busy_ = false;
    }
    bool busy_ = false;
};

int main(int argc, char** argv) {
    App app(argc, argv);
    app.setApplicationName("xLamina");
    app.setOrganizationName("xLamina");
    DiagLog::open();   // the session's log file, with the machine's details (diaglog.h)
    Updater::removeLeftovers();                              // the files an update renamed aside
    QTimer::singleShot(5000, &Updater::removeLeftovers);     // again, once the replaced exe has surely exited
    QIcon icon;
    for (int s : {16, 24, 32, 48, 64, 128, 256}) icon.addFile(QString(":/xlamina-%1.png").arg(s), QSize(s, s));
    app.setWindowIcon(icon);
    applyTheme(appSettings().value("ui/theme", "dark").toString());
    for (int i = 1; i < argc; ++i) if (QString::fromLocal8Bit(argv[i]) == "--manual") ManualShots::keepOffScreen();
    for (int i = 1; i + 1 < argc; ++i)
        if (QString::fromLocal8Bit(argv[i]) == "--report") {   // the report Help > Report a problem saves, into DIR, with no window
            QString why;
            const QString p = DiagLog::makeReport(QString::fromLocal8Bit(argv[i + 1]), "made with --report", &why);
            DiagLog::write(p.isEmpty() ? "report: " + why : "report saved: " + p);
            return p.isEmpty() ? 1 : 0;
        }
    MainWindow w;
    w.show();
    bool autoRender = false, sample = false, selfUpdate = false;
    QStringList batchDirs;
    QString folder, shots, bench, updatedFrom, hover, manual, report;
    double every = 3, renderIn = -1;
    for (int i = 1; i < argc; ++i) {
        QString a = QString::fromLocal8Bit(argv[i]);
        if (a == "--batch") { for (++i; i < argc; ++i) batchDirs << QString::fromLocal8Bit(argv[i]); break; }
        else if (a == "--render") autoRender = true;
        else if (a == "--render-in" && i + 1 < argc) renderIn = atof(argv[++i]);
        else if (a == "--sample") sample = true;
        else if (a == "--self-update") selfUpdate = true;
        else if (a == "--shots" && i + 1 < argc) shots = QString::fromLocal8Bit(argv[++i]);
        else if (a == "--bench" && i + 1 < argc) bench = QString::fromLocal8Bit(argv[++i]);
        else if (a == "--every" && i + 1 < argc) every = atof(argv[++i]);
        else if (a == "--updated-from" && i + 1 < argc) updatedFrom = QString::fromLocal8Bit(argv[++i]);
        else if (a == "--hover" && i + 1 < argc) hover = QString::fromLocal8Bit(argv[++i]);
        else if (a == "--manual" && i + 1 < argc) manual = QString::fromLocal8Bit(argv[++i]);
        else if (a == "--report" && i + 1 < argc) report = QString::fromLocal8Bit(argv[++i]);
        else folder = a;
    }
    if (!updatedFrom.isEmpty()) w.notifyUpdated(updatedFrom);
    if (selfUpdate) QMetaObject::invokeMethod(&w, &MainWindow::selfUpdate, Qt::QueuedConnection);
    if (!folder.isEmpty()) w.openFolder(folder);
    if (autoRender) QMetaObject::invokeMethod(&w, "render", Qt::QueuedConnection);
    if (!batchDirs.isEmpty()) {
        QMetaObject::invokeMethod(&w, [&w, batchDirs] { w.startBatch(batchDirs); }, Qt::QueuedConnection);
        QTimer* bt = new QTimer(&w);   // quit when the queue has run out
        QObject::connect(bt, &QTimer::timeout, &w, [&w, bt] { if (!w.batchRunning()) { bt->stop(); QTimer::singleShot(1500, &w, &QWidget::close); } });
        QTimer::singleShot(1000, bt, [bt] { bt->start(500); });
    }
    if (renderIn >= 0) QTimer::singleShot((int)(renderIn * 1000), &w, &MainWindow::render);
    if (!bench.isEmpty()) QMetaObject::invokeMethod(&w, [&w, sample, bench] { w.autoBenchmark(sample, bench); }, Qt::QueuedConnection);
    if (!manual.isEmpty()) new ManualShots(w, manual);   // grabs for docs/manual, then quits (manualshots.h)
    if (hover.contains(',')) {   // once the render is done, and again every 2 s (the window may still be laying out)
        const double fx = hover.section(',', 0, 0).toDouble(), fy = hover.section(',', 1, 1).toDouble();
        QTimer* ht = new QTimer(&w);
        QObject::connect(ht, &QTimer::timeout, &w, [&w, fx, fy] { if (!w.rendering()) w.probeHover(fx, fy); });
        ht->start(2000);
    }
    if (!shots.isEmpty()) {
        QDir().mkpath(shots);
        QTimer* t = new QTimer(&w);
        int* n = new int(0);
        QObject::connect(t, &QTimer::timeout, &w, [&w, shots, n] {
            w.grab().save(QString("%1/shot_%2.png").arg(shots).arg(++*n, 2, 10, QChar('0')));
            if (!w.rendering() && *n > 1) static_cast<QTimer*>(w.findChild<QTimer*>("shots"))->stop();
        });
        t->setObjectName("shots");
        t->start((int)(every * 1000));
    }
    return app.exec();
}
