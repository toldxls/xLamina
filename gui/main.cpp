#include <QApplication>
#include <QDir>
#include <QIcon>
#include <QTimer>

#include "mainwindow.h"

// xLamina [FOLDER] [--render] [--shots DIR [--every S]]
//   --render  start a render of FOLDER at once
//   --shots   developer aid: save a grab of the window to DIR every S seconds (default 3) while a render runs
int main(int argc, char** argv) {
    QApplication app(argc, argv);
    app.setApplicationName("xLamina");
    app.setOrganizationName("xLamina");
    QIcon icon;
    for (int s : {16, 24, 32, 48, 64, 128, 256}) icon.addFile(QString(":/xlamina-%1.png").arg(s), QSize(s, s));
    app.setWindowIcon(icon);
    MainWindow w;
    w.show();
    bool autoRender = false;
    QString folder, shots;
    double every = 3;
    for (int i = 1; i < argc; ++i) {
        QString a = QString::fromLocal8Bit(argv[i]);
        if (a == "--render") autoRender = true;
        else if (a == "--shots" && i + 1 < argc) shots = QString::fromLocal8Bit(argv[++i]);
        else if (a == "--every" && i + 1 < argc) every = atof(argv[++i]);
        else folder = a;
    }
    if (!folder.isEmpty()) w.openFolder(folder);
    if (autoRender) QMetaObject::invokeMethod(&w, "render", Qt::QueuedConnection);
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
