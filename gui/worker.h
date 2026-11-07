// Runs a stacking job on a background thread and reports through signals.
#pragma once

#include <QImage>
#include <QObject>
#include <QThread>

#include <atomic>
#include <memory>

#include "stackcore/pipeline.h"

struct JobResultData {
    stackcore::StackOptions::Result result;
    stackcore::AlignOut align;
    QString reportJson;
    QString outDir;
    int rc = 0;
};

class Worker : public QObject {
    Q_OBJECT
public:
    explicit Worker(QObject* parent = nullptr);
    ~Worker() override;
    bool running() const { return running_; }
    void start(const stackcore::JobOptions& jo, int liveDown = 6);   // live views at 1/liveDown of the output
    void cancel() { cancel_.store(true); }

signals:
    void progress(QString stage, double fraction);
    void preview(QString name, QImage image);
    void frame(int k, QImage image);
    void logLine(QString line);
    void finished(std::shared_ptr<JobResultData> data);

private:
    QThread thread_;
    std::atomic<bool> cancel_{false};
    bool running_ = false;
};

// Reads the frames one by one on a thread and emits a thumbnail of each.
class ThumbLoader : public QObject {
    Q_OBJECT
public:
    explicit ThumbLoader(QObject* parent = nullptr);
    ~ThumbLoader() override;
    void start(const QStringList& files, int width);
    void stop();

signals:
    void thumbnail(int index, QImage image);

private:
    QThread thread_;
    std::atomic<bool> stop_{false};
    std::atomic<int> generation_{0};
};
