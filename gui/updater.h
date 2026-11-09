// In-app updates: the latest GitHub release compared with this build; its zip downloaded, unpacked with
// Windows' tar, and swapped into the application folder. A file in use (the running exe, its loaded DLLs)
// is renamed aside first, which Windows allows where overwriting is refused; the leftovers are deleted at
// the next start. The settings file is untouched, so the new version starts as the old one was set.
#pragma once

#include <QNetworkAccessManager>
#include <QObject>
#include <QString>
#include <QStringList>

class QNetworkReply;

struct ReleaseInfo {
    QString version;   // "0.6.21"
    QString tag;       // "v0.6.21"
    QString notes;     // the release's markdown body
    QString page;      // the release's web page
    QString url;       // the win64 zip
    qint64 size = 0;   // the zip's size in bytes, 0 = unknown
    bool valid() const { return !version.isEmpty() && !url.isEmpty(); }
};

class Updater : public QObject {
    Q_OBJECT
public:
    explicit Updater(QObject* parent = nullptr);
    void check();                          // -> available(release) | upToDate(latest) | failed(why)
    void download(const ReleaseInfo& r);   // -> downloadProgress .. downloaded(zip) | failed(why)
    void abort();                          // a running request; it ends in failed()
    bool busy() const { return reply_ != nullptr; }

    // Unpacks the zip and swaps its files into the application folder. Returns an empty string, or why it
    // failed; then the folder is as it was.
    static QString install(const QString& zip);
    static bool canWriteAppDir();                   // the folder the exe runs from accepts new files
    static void removeLeftovers();                  // the replaced version's files, renamed aside by install()
    static bool restart(const QStringList& args);   // a new instance of this exe; the caller quits
    static int compareVersions(const QString& a, const QString& b);   // "0.6.21" vs "0.6.9": > 0

signals:
    void available(ReleaseInfo release);
    void upToDate(QString latest);
    void failed(QString why);
    void downloadProgress(qint64 received, qint64 total);
    void downloaded(QString zipPath);

private:
    QNetworkAccessManager net_;
    QNetworkReply* reply_ = nullptr;
};
