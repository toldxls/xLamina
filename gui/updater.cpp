#include "updater.h"

#include <QCoreApplication>
#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QProcess>
#include <QSaveFile>
#include <QTemporaryFile>
#include <QUrl>
#include <algorithm>

#include "version.h"

namespace {
const char* kFeed = "https://api.github.com/repos/toldxls/xLamina/releases/latest";
const char* kAside = ".old";   // a replaced file's suffix until the next start

QString feedUrl() {
    // developer override: XLAMINA_UPDATE_FEED=file:///C:/x/latest.json (the GitHub JSON shape, pointing at a local
    // zip) rehearses an update without a release
    QString f = qEnvironmentVariable("XLAMINA_UPDATE_FEED");
    return f.isEmpty() ? QString(kFeed) : f;
}

QNetworkRequest request(const QUrl& url) {
    QNetworkRequest rq(url);
    rq.setRawHeader("User-Agent", QByteArray("xLamina/") + XLAMINA_VERSION);   // GitHub's API refuses requests without one
    rq.setRawHeader("Accept", "application/vnd.github+json, application/octet-stream;q=0.9, */*;q=0.8");
    rq.setAttribute(QNetworkRequest::RedirectPolicyAttribute, QNetworkRequest::NoLessSafeRedirectPolicy);   // the assets live on another host
    rq.setTransferTimeout(30000);   // no bytes for 30 s ends the request
    return rq;
}

QString appDir() { return QCoreApplication::applicationDirPath(); }
}   // namespace

Updater::Updater(QObject* parent) : QObject(parent) {}

int Updater::compareVersions(const QString& a, const QString& b) {
    auto parts = [](QString s) {
        if (s.startsWith('v')) s.remove(0, 1);
        QList<int> v;
        for (const QString& p : s.split('.')) v.push_back(p.toInt());
        return v;
    };
    QList<int> x = parts(a), y = parts(b);
    for (int i = 0; i < std::max(x.size(), y.size()); ++i) {
        int xi = i < x.size() ? x[i] : 0, yi = i < y.size() ? y[i] : 0;
        if (xi != yi) return xi < yi ? -1 : 1;
    }
    return 0;
}

void Updater::abort() {
    if (reply_) reply_->abort();
}

void Updater::check() {
    if (reply_) return;
    QNetworkReply* r = net_.get(request(QUrl(feedUrl())));
    reply_ = r;
    connect(r, &QNetworkReply::finished, this, [this, r] {
        if (reply_ == r) reply_ = nullptr;
        r->deleteLater();
        if (r->error() != QNetworkReply::NoError) {
            int http = r->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
            emit failed(http == 403 ? QString("GitHub is rate-limiting update checks from this address; try again in an hour.")
                                    : "Could not reach GitHub: " + r->errorString());
            return;
        }
        QJsonParseError pe;
        QJsonDocument doc = QJsonDocument::fromJson(r->readAll(), &pe);
        if (!doc.isObject()) { emit failed("The release feed is not JSON: " + pe.errorString()); return; }
        QJsonObject o = doc.object();
        ReleaseInfo rel;
        rel.tag = o.value("tag_name").toString();
        rel.version = rel.tag.startsWith('v') ? rel.tag.mid(1) : rel.tag;
        rel.notes = o.value("body").toString();
        rel.page = o.value("html_url").toString();
        for (const QJsonValue& a : o.value("assets").toArray()) {
            QJsonObject ao = a.toObject();
            QString name = ao.value("name").toString();
            if (name.startsWith("xLamina-") && name.endsWith("-win64.zip")) {
                rel.url = ao.value("browser_download_url").toString();
                rel.size = (qint64)ao.value("size").toDouble();
                break;
            }
        }
        if (rel.version.isEmpty()) { emit failed("The release feed names no version."); return; }
        if (compareVersions(rel.version, XLAMINA_VERSION) <= 0) { emit upToDate(rel.version); return; }
        if (rel.url.isEmpty()) { emit failed(QString("xLamina %1 is out, but its release has no Windows zip yet.").arg(rel.version)); return; }
        emit available(rel);
    });
}

void Updater::download(const ReleaseInfo& rel) {
    if (reply_) return;
    QString path = QDir::temp().filePath(QString("xLamina-%1-win64.zip").arg(rel.version));
    QSaveFile* file = new QSaveFile(path, this);   // lands under its name only when complete
    if (!file->open(QIODevice::WriteOnly)) {
        delete file;
        emit failed("Cannot write " + QDir::toNativeSeparators(path));
        return;
    }
    QNetworkReply* r = net_.get(request(QUrl(rel.url)));
    reply_ = r;
    connect(r, &QNetworkReply::readyRead, this, [r, file] { file->write(r->readAll()); });
    connect(r, &QNetworkReply::downloadProgress, this, &Updater::downloadProgress);
    connect(r, &QNetworkReply::finished, this, [this, r, file, path, size = rel.size] {
        if (reply_ == r) reply_ = nullptr;
        r->deleteLater();
        if (r->error() != QNetworkReply::NoError) {
            file->cancelWriting();
            delete file;
            emit failed(r->error() == QNetworkReply::OperationCanceledError ? QString("The download was cancelled.") : "The download failed: " + r->errorString());
            return;
        }
        file->write(r->readAll());
        qint64 got = file->size();
        bool ok = file->commit();
        delete file;
        if (!ok) { emit failed("Cannot finish writing " + QDir::toNativeSeparators(path)); return; }
        if (size > 0 && got != size) {
            QFile::remove(path);
            emit failed(QString("The download is incomplete: %1 of %2 bytes.").arg(got).arg(size));
            return;
        }
        emit downloaded(path);
    });
}

bool Updater::canWriteAppDir() {
    QTemporaryFile t(QDir(appDir()).filePath("write-test-XXXXXX"));
    return t.open();
}

QString Updater::install(const QString& zip) {
    QDir staging(QDir::temp().filePath("xLamina-update"));
    staging.removeRecursively();
    if (!QDir().mkpath(staging.path())) return "Cannot create " + QDir::toNativeSeparators(staging.path());
    // Windows 10 1803+ ships bsdtar, which reads zips (and the backslash paths PowerShell's Compress-Archive writes)
    QString tar = QDir(qEnvironmentVariable("SystemRoot", "C:/Windows")).filePath("System32/tar.exe");
    QProcess p;
    p.start(tar, {"-xf", QDir::toNativeSeparators(zip), "-C", QDir::toNativeSeparators(staging.path())});
    if (!p.waitForFinished(120000) || p.exitStatus() != QProcess::NormalExit || p.exitCode() != 0) {
        QString err = QString::fromLocal8Bit(p.readAllStandardError()).trimmed();
        return "Unpacking the zip failed" + (err.isEmpty() ? QString(" (is " + QDir::toNativeSeparators(tar) + " missing?).") : ": " + err);
    }
    if (!QFileInfo::exists(staging.filePath("xLamina.exe"))) return "The zip holds no xLamina.exe.";

    // the swap: every file of the new version replaces the old one. A file in use is renamed aside first
    // (Windows lets a running exe and its mapped DLLs be renamed, never overwritten). Any failure puts the
    // renamed files back, so the folder is whole either way.
    struct Done { QString target, aside; bool copied; };
    QList<Done> done;
    auto rollback = [&done] {
        for (int i = done.size() - 1; i >= 0; --i) {
            if (done[i].copied) QFile::remove(done[i].target);
            if (!done[i].aside.isEmpty()) QFile::rename(done[i].aside, done[i].target);
        }
    };
    QString app = appDir();
    QDirIterator it(staging.path(), QDir::Files, QDirIterator::Subdirectories);
    while (it.hasNext()) {
        QString src = it.next();
        QString target = QDir(app).filePath(staging.relativeFilePath(src));
        QDir().mkpath(QFileInfo(target).path());
        Done d{target, {}, false};
        if (QFileInfo::exists(target)) {
            d.aside = target + kAside;
            QFile::remove(d.aside);   // a leftover of an earlier update that is still in use gets a numbered name
            for (int n = 2; QFileInfo::exists(d.aside) && n < 10; ++n) d.aside = target + kAside + QString::number(n);
            if (!QFile::rename(target, d.aside)) {
                rollback();
                return "Cannot replace " + QDir::toNativeSeparators(target) + " (is another xLamina running from this folder?).";
            }
        }
        done.push_back(d);
        if (!QFile::copy(src, target)) {
            rollback();
            return "Cannot write " + QDir::toNativeSeparators(target) + ".";
        }
        done.last().copied = true;
    }
    staging.removeRecursively();
    QFile::remove(zip);
    return {};
}

void Updater::removeLeftovers() {
    QDirIterator it(appDir(), {QString("*") + kAside + "*"}, QDir::Files, QDirIterator::Subdirectories);
    while (it.hasNext()) QFile::remove(it.next());   // the old exe may still be exiting: the caller tries again later
}

bool Updater::restart(const QStringList& args) {
    return QProcess::startDetached(QCoreApplication::applicationFilePath(), args, appDir());
}
