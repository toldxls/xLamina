#include "worker.h"

#include <QMetaObject>

#include "imageview.h"

using namespace stackcore;

Worker::Worker(QObject* parent) : QObject(parent) {}

Worker::~Worker() {
    cancel();
    thread_.quit();
    thread_.wait();
}

void Worker::start(const JobOptions& jo_in) {
    if (running_) return;
    running_ = true;
    cancel_.store(false);
    auto jo = std::make_shared<JobOptions>(jo_in);
    auto data = std::make_shared<JobResultData>();
    jo->stack.result = &data->result;
    QObject* worker = new QObject();
    worker->moveToThread(&thread_);
    thread_.start();
    QMetaObject::invokeMethod(worker, [this, jo, data, worker] {
        set_log([this](const std::string& s) { emit logLine(QString::fromStdString(s)); });
        Hooks hooks;
        hooks.cancel = &cancel_;
        hooks.preview_down = 6;
        hooks.preview_every = 6;
        hooks.progress = [this](const std::string& stage, double f) { emit progress(QString::fromStdString(stage), f); };
        hooks.preview = [this](const std::string& name, const std::vector<float>& rgb, int w, int h) {
            emit preview(QString::fromStdString(name), toQImage(rgb.data(), w, h));
        };
        try {
            data->rc = run_job(*jo, data->align, hooks);
            data->reportJson = QString::fromStdString(report_json(jo->align, data->align));
            data->outDir = QString::fromStdString(jo->stack.out);
        } catch (const std::exception& e) {
            emit logLine(QString("failed: %1").arg(e.what()));
            data->rc = 1;
        }
        set_log(nullptr);
        running_ = false;
        emit finished(data);
        worker->deleteLater();
    }, Qt::QueuedConnection);
}
