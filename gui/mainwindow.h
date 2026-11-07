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
#include <QSplitter>
#include <QTabBar>

#include <memory>

#include "imageview.h"
#include "worker.h"

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent = nullptr);
    void openFolder(const QString& dir);
    bool rendering() const { return worker_.running(); }

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
    void onFullFrame(int index, QImage image);
    void onLog(QString line);
    void onFinished(std::shared_ptr<JobResultData> data);
    void dropFlagged();
    void frameMenu(const QPoint& pos);

private:
    void buildUi();
    void labelWobble(const stackcore::AlignOut& A);
    void setBusy(bool busy);
    void setStatusGeometry(const JobResultData& d);
    QImage outputImage(const QString& name);
    QImage forDisplay(QImage img) const;   // the frames' colour space -> sRGB

    QString folder_;
    QStringList frames_;
    // rows carry the frame index (into frames_) in Qt::UserRole, -1 for files left out; the list order is
    // the stack order and a frame may be listed more than once, so jobs refer to rows, not frames
    QList<QListWidgetItem*> jobItems_;     // the rows of the current / last render, in stack order
    QList<QListWidgetItem*> flagged_;      // rows whose wobble is an outlier
    QString wobble_;                       // the wobble summary line
    QAction* dropFlagged_ = nullptr;
    QColorSpace colorSpace_;               // the frames' ICC profile (invalid = assume sRGB)
    bool selecting_ = false;
    ImageView* liveView_ = nullptr;        // right pane while rendering
    QSplitter* split_ = nullptr;
    ThumbLoader thumbs_;
    ThumbLoader frameLoader_;              // one frame at full resolution
    int pendingFrame_ = -1;
    QComboBox* liveRes_ = nullptr;
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
    QDoubleSpinBox* cradius_ = nullptr;
    QDoubleSpinBox* bradius_ = nullptr;
    QLineEdit* outDir_ = nullptr;
    QMap<QString, QCheckBox*> outputs_;
    Worker worker_;
};
