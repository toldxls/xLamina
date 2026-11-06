#pragma once

#include <QCheckBox>
#include <QColorSpace>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMainWindow>
#include <QMap>
#include <QPlainTextEdit>
#include <QProgressBar>
#include <QPushButton>
#include <QTabBar>

#include <memory>

#include "imageview.h"
#include "worker.h"

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent = nullptr);
    void openFolder(const QString& dir);

protected:
    void dragEnterEvent(QDragEnterEvent* e) override;
    void dropEvent(QDropEvent* e) override;
    void closeEvent(QCloseEvent* e) override;

public slots:
    void render();

private slots:
    void chooseFolder();
    void cancel();
    void saveCurrent();
    void showFrame(int row);
    void showOutput(int tab);
    void onProgress(QString stage, double fraction);
    void onPreview(QString name, QImage image);
    void onFrame(int k, QImage image);
    void onThumbnail(int index, QImage image);
    void onLog(QString line);
    void onFinished(std::shared_ptr<JobResultData> data);
    void dropFlagged();

private:
    void buildUi();
    void labelWobble(const stackcore::AlignOut& A);
    void setBusy(bool busy);
    void setStatusGeometry(const JobResultData& d);
    QImage outputImage(const QString& name);
    QImage forDisplay(QImage img) const;   // the frames' colour space -> sRGB

    QString folder_;
    QStringList frames_;
    QList<int> frameRows_;                 // list row of each frame
    QList<int> jobFrames_;                 // the frames (indices into frames_) of the current / last render
    QList<int> flagged_;                   // frames whose wobble is an outlier
    QString wobble_;                       // the wobble summary line
    QAction* dropFlagged_ = nullptr;
    QColorSpace colorSpace_;               // the frames' ICC profile (invalid = assume sRGB)
    bool selecting_ = false;
    QLabel* frameView_ = nullptr;
    ThumbLoader thumbs_;
    std::shared_ptr<JobResultData> result_;
    QMap<QString, QImage> cache_;        // 8-bit versions of outputs and previews
    QString current_;                    // name of the image on screen

    ImageView* view_ = nullptr;
    QListWidget* frameList_ = nullptr;
    QTabBar* tabs_ = nullptr;
    QPlainTextEdit* log_ = nullptr;
    QProgressBar* progress_ = nullptr;
    QLabel* stage_ = nullptr;
    QLabel* status_ = nullptr;
    QLabel* geometry_ = nullptr;
    QPushButton* renderBtn_ = nullptr;
    QPushButton* cancelBtn_ = nullptr;
    QPushButton* saveBtn_ = nullptr;
    QComboBox* scale_ = nullptr;
    QComboBox* ref_ = nullptr;
    QCheckBox* lockScale_ = nullptr;
    QDoubleSpinBox* fov_ = nullptr;
    QDoubleSpinBox* smoothing_ = nullptr;
    QLineEdit* outDir_ = nullptr;
    QMap<QString, QCheckBox*> outputs_;
    Worker worker_;
};
