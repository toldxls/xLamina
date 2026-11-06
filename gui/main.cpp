#include <QApplication>

#include "mainwindow.h"

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    app.setApplicationName("xLamina");
    app.setOrganizationName("xLamina");
    MainWindow w;
    w.show();
    bool autoRender = false;
    QString folder;
    for (int i = 1; i < argc; ++i) {
        QString a = QString::fromLocal8Bit(argv[i]);
        if (a == "--render") autoRender = true;
        else folder = a;
    }
    if (!folder.isEmpty()) w.openFolder(folder);
    if (autoRender) QMetaObject::invokeMethod(&w, "render", Qt::QueuedConnection);
    return app.exec();
}
