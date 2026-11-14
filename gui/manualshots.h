// Developer aid (xLamina FOLDER --render --manual DIR): the screen grabs of the manual (docs/manual). While the
// folder renders and afterwards it walks through the windows and dialogs the manual shows, saves each as DIR/<shot>.png
// and writes DIR/rects.js: where every named control sits in every grab, as fractions of the grab, which the manual's
// page uses to place its numbered marks and its cursor. Then xLamina quits. Nothing appears on the
// screen during the run; tools\manual.cmd does all of it.
#pragma once

#include <QJsonObject>
#include <QObject>
#include <QString>
#include <QTimer>

#include <functional>
#include <vector>

class MainWindow;
class QWidget;

class ManualShots : public QObject {
public:
    ManualShots(MainWindow& w, const QString& dir);
    // call before the first window is shown: every window and dialog of the run is laid out and painted as usual but
    // never put on the screen (the offscreen platform plugin has no fonts on Windows, so it is not an option)
    static void keepOffScreen();

private:
    void grab(QWidget* root, const QString& shot);          // the picture and the controls in it
    void mark(const QString& shot, const QString& name, const QRect& r, const QSize& full);
    void markAll(const QString& shot, QWidget* root);
    void modal(const QString& shot, const std::function<void()>& open);   // a modal dialog: grabbed, then dismissed
    void finish();

    MainWindow& w_;
    QString dir_;
    QJsonObject rects_;
    std::vector<std::function<void()>> steps_;
    size_t next_ = 0;
    QTimer live_, run_;
};
