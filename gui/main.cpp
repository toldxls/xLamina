#include <QApplication>
#include <QDir>
#include <QIcon>
#include <QTimer>

#include "mainwindow.h"
#include "prefs.h"
#include "updater.h"

// xLamina [FOLDER] [--render] [--shots DIR [--every S]] [--bench REPORT.txt [--sample]] [--self-update]
//   --render  start a render of FOLDER at once
//   --render-in S       developer aid: start the render S seconds after opening (the thumbnails' cache fill first)
//   --shots   developer aid: save a grab of the window to DIR every S seconds (default 3) while a render runs
//   --bench   developer aid: run the benchmark unattended (on FOLDER, or the built-in sample with --sample),
//             write REPORT.txt and REPORT.txt.png, then quit
//   --self-update       developer aid: install the latest release without asking and restart
//   --updated-from V    passed by the updater to the new instance, which says so in the status bar
int main(int argc, char** argv) {
    QApplication app(argc, argv);
    app.setApplicationName("xLamina");
    app.setOrganizationName("xLamina");
    Updater::removeLeftovers();                              // the files an update renamed aside
    QTimer::singleShot(5000, &Updater::removeLeftovers);     // again, once the replaced exe has surely exited
    QIcon icon;
    for (int s : {16, 24, 32, 48, 64, 128, 256}) icon.addFile(QString(":/xlamina-%1.png").arg(s), QSize(s, s));
    app.setWindowIcon(icon);
    applyTheme(appSettings().value("ui/theme", "dark").toString());
    MainWindow w;
    w.show();
    bool autoRender = false, sample = false, selfUpdate = false;
    QString folder, shots, bench, updatedFrom;
    double every = 3, renderIn = -1;
    for (int i = 1; i < argc; ++i) {
        QString a = QString::fromLocal8Bit(argv[i]);
        if (a == "--render") autoRender = true;
        else if (a == "--render-in" && i + 1 < argc) renderIn = atof(argv[++i]);
        else if (a == "--sample") sample = true;
        else if (a == "--self-update") selfUpdate = true;
        else if (a == "--shots" && i + 1 < argc) shots = QString::fromLocal8Bit(argv[++i]);
        else if (a == "--bench" && i + 1 < argc) bench = QString::fromLocal8Bit(argv[++i]);
        else if (a == "--every" && i + 1 < argc) every = atof(argv[++i]);
        else if (a == "--updated-from" && i + 1 < argc) updatedFrom = QString::fromLocal8Bit(argv[++i]);
        else folder = a;
    }
    if (!updatedFrom.isEmpty()) w.notifyUpdated(updatedFrom);
    if (selfUpdate) QMetaObject::invokeMethod(&w, &MainWindow::selfUpdate, Qt::QueuedConnection);
    if (!folder.isEmpty()) w.openFolder(folder);
    if (autoRender) QMetaObject::invokeMethod(&w, "render", Qt::QueuedConnection);
    if (renderIn >= 0) QTimer::singleShot((int)(renderIn * 1000), &w, &MainWindow::render);
    if (!bench.isEmpty()) QMetaObject::invokeMethod(&w, [&w, sample, bench] { w.autoBenchmark(sample, bench); }, Qt::QueuedConnection);
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
