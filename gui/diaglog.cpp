#include "diaglog.h"

#include <cuda_runtime.h>
#include <windows.h>

#include <QCoreApplication>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QMutex>
#include <QProcess>
#include <QSettings>
#include <QSysInfo>

#include <thread>

#include "prefs.h"
#include "version.h"

namespace {
QMutex g_m;
QFile g_file;
QString g_dir, g_stage, g_what, g_started, g_last, g_unfinished;
qint64 g_bytes = 0;
const qint64 kMaxBytes = 5 << 20;   // a session's file stops growing here (a batch of hundreds of folders)
const int kKeep = 10;

QString marker() { return QDir(g_dir).filePath("running.json"); }

void writeMarker() {   // under g_m
    QFile f(marker());
    if (!f.open(QIODevice::WriteOnly | QIODevice::Truncate)) return;
    QJsonObject o{{"version", XLAMINA_VERSION}, {"started", g_started}, {"what", g_what}, {"stage", g_stage},
                  {"pid", (qint64)QCoreApplication::applicationPid()}};
    f.write(QJsonDocument(o).toJson(QJsonDocument::Compact));
}

// the NVIDIA driver's version, from the driver's own library (System32 only); empty when it is not there
QString driverVersion() {
    HMODULE lib = LoadLibraryExA("nvml.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!lib) return {};
    typedef int (*NoArg)();
    typedef int (*Ver)(char*, unsigned int);
    NoArg init = (NoArg)GetProcAddress(lib, "nvmlInit_v2"), shut = (NoArg)GetProcAddress(lib, "nvmlShutdown");
    Ver ver = (Ver)GetProcAddress(lib, "nvmlSystemGetDriverVersion");
    QString v;
    if (init && shut && ver && init() == 0) {
        char buf[96] = {0};
        if (ver(buf, sizeof buf) == 0) v = QString::fromLatin1(buf);
        shut();
    }
    FreeLibrary(lib);
    return v;
}

QString cpuName() {
    QSettings reg("HKEY_LOCAL_MACHINE\\HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", QSettings::NativeFormat);
    return reg.value("ProcessorNameString").toString().simplified();
}
}  // namespace

namespace DiagLog {

QString dir() { return g_dir; }
QString path() { return g_file.fileName(); }

QString redact(QString text) {
    // the profile folder in both spellings, and the bare user name where a path was cut
    const QString home = QDir::toNativeSeparators(QDir::homePath());
    if (home.size() > 3) {
        text.replace(home, "%USERPROFILE%", Qt::CaseInsensitive);
        text.replace(QDir::fromNativeSeparators(home), "%USERPROFILE%", Qt::CaseInsensitive);
    }
    return text;
}

QString systemDetails() {
    QStringList l;
    l << QString("xLamina %1 (Qt %2)").arg(XLAMINA_VERSION, qVersion());
    l << QString("Windows: %1, build %2").arg(QSysInfo::prettyProductName(), QSysInfo::kernelVersion());
    l << QString("CPU: %1, %2 logical cores").arg(cpuName()).arg(std::thread::hardware_concurrency());
    l << hostSummary();
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) { n = 0; l << "GPUs: none usable (" + QString::fromLatin1(cudaGetErrorString(cudaGetLastError())) + ")"; }
    for (int i = 0; i < n; ++i) {
        cudaDeviceProp p{};
        if (cudaGetDeviceProperties(&p, i) != cudaSuccess) { cudaGetLastError(); continue; }
        l << QString("GPU %1: %2, %3 GB, compute capability %4.%5").arg(i).arg(QString::fromLatin1(p.name)).arg(p.totalGlobalMem / 1073741824.0, 0, 'f', 1).arg(p.major).arg(p.minor);
    }
    const QString drv = driverVersion();
    int rt = 0, dv = 0;
    cudaRuntimeGetVersion(&rt);
    if (cudaDriverGetVersion(&dv) != cudaSuccess) { dv = 0; cudaGetLastError(); }
    l << QString("NVIDIA driver %1; CUDA runtime %2.%3, the driver's CUDA %4.%5").arg(drv.isEmpty() ? "unknown" : drv)
             .arg(rt / 1000).arg(rt % 1000 / 10).arg(dv / 1000).arg(dv % 1000 / 10);
    return l.join('\n');
}

static QString readUnfinished();

QString takeUnfinished() {
    QMutexLocker lk(&g_m);
    const QString u = g_unfinished;
    g_unfinished.clear();
    return u;
}

void open() {
    QMutexLocker lk(&g_m);
    if (g_file.isOpen()) return;
    const QString ini = qEnvironmentVariable("XLAMINA_SETTINGS");
    g_dir = !ini.isEmpty() ? QFileInfo(ini).absoluteDir().filePath("logs")
                           : QDir(qEnvironmentVariable("LOCALAPPDATA", QDir::homePath())).filePath("xLamina/logs");
    QDir().mkpath(g_dir);
    // the newest files stay
    QFileInfoList old = QDir(g_dir).entryInfoList({"xlamina-*.log"}, QDir::Files, QDir::Name | QDir::Reversed);
    for (int i = kKeep - 1; i < old.size(); ++i) QFile::remove(old[i].absoluteFilePath());
    g_file.setFileName(QDir(g_dir).filePath("xlamina-" + QDateTime::currentDateTime().toString("yyyyMMdd-HHmmss") + ".log"));
    g_unfinished = readUnfinished();
    if (!g_file.open(QIODevice::WriteOnly | QIODevice::Append | QIODevice::Text)) return;
    lk.unlock();
    write("=== session " + QDateTime::currentDateTime().toString(Qt::ISODate));
    for (const QString& s : systemDetails().split('\n')) write(s);
}

void write(const QString& line) {
    QMutexLocker lk(&g_m);
    if (!g_file.isOpen() || g_bytes > kMaxBytes) return;
    const QByteArray b = (QDateTime::currentDateTime().toString("HH:mm:ss.zzz ") + redact(line) + "\n").toUtf8();
    g_bytes += b.size();
    g_file.write(g_bytes > kMaxBytes ? QByteArray("(the log is full for this session)\n") : b);
    g_file.flush();
    if (g_last.size() < (1 << 20)) g_last += redact(line) + "\n";
}

void runStart(const QString& what, const QStringList& settings) {
    {
        QMutexLocker lk(&g_m);
        g_what = what;
        g_stage = "starting";
        g_started = QDateTime::currentDateTime().toString(Qt::ISODate);
        g_last.clear();
        writeMarker();
    }
    write("=== " + what + " " + QDateTime::currentDateTime().toString(Qt::ISODate));
    for (const QString& s : settings) write("  " + s);
}

void runStage(const QString& stage) {
    // "pass 1: frame 12" and "depth map: volume to the GPU, 1.2 of 4.9 GB" move all the time: the name is the part
    // before the first comma or digit run that follows a space
    QString name = stage.section(',', 0, 0).trimmed();
    {
        QMutexLocker lk(&g_m);
        if (name == g_stage || g_what.isEmpty()) return;
        g_stage = name;
        writeMarker();
    }
    write("stage: " + name);
}

void runEnd(const QString& outcome) {
    write("=== " + outcome);
    QMutexLocker lk(&g_m);
    g_what.clear();
    QFile::remove(marker());
}

QString lastRun() {
    QMutexLocker lk(&g_m);
    return g_last;
}

// read when the session's file is opened, before a render of this session can write its own marker
static QString readUnfinished() {   // under g_m
    QFile f(marker());
    if (!f.open(QIODevice::ReadOnly)) return {};
    const QJsonObject o = QJsonDocument::fromJson(f.readAll()).object();
    f.close();
    // another xLamina that is rendering right now wrote it: not a run that died, and its marker stays
    if (const DWORD pid = (DWORD)o.value("pid").toInteger(); pid && pid != GetCurrentProcessId()) {
        if (HANDLE h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid)) {
            DWORD code = 0;
            const bool alive = GetExitCodeProcess(h, &code) && code == STILL_ACTIVE;
            CloseHandle(h);
            if (alive) return {};
        }
    }
    QFile::remove(marker());
    if (o.isEmpty()) return {};
    const QDateTime t = QDateTime::fromString(o.value("started").toString(), Qt::ISODate);
    return QString("%1 (started %2, xLamina %3) stopped during \"%4\"")
        .arg(o.value("what").toString("a render"), t.isValid() ? t.toString("d MMM HH:mm") : QString("earlier"), o.value("version").toString("?"), o.value("stage").toString("?"));
}

QString makeReport(const QString& folder, const QString& note, QString* why) {
    const QString name = "xLamina-report-" + QDateTime::currentDateTime().toString("yyyyMMdd-HHmm");
    const QString work = QDir(folder).filePath(name);
    QDir(work).removeRecursively();
    if (!QDir().mkpath(work)) { if (why) *why = "cannot create " + QDir::toNativeSeparators(work); return {}; }
    auto put = [&](const QString& file, const QString& text) {
        QFile f(QDir(work).filePath(file));
        if (f.open(QIODevice::WriteOnly | QIODevice::Text)) f.write(redact(text).toUtf8());
    };
    put("system.txt", systemDetails() + "\n\nnow: " + hostSummary() + "\n" + (note.isEmpty() ? QString() : "\n" + note + "\n"));
    put("last-render.txt", lastRun().isEmpty() ? QString("(no render in this session)\n") : lastRun());
    // the settings, with every value that holds a folder cut to its last part
    {
        QSettings s = appSettings();
        QStringList lines;
        for (const QString& k : s.allKeys()) {
            QVariant v = s.value(k);
            QString t = v.typeId() == QMetaType::QStringList ? v.toStringList().join("; ") : v.typeId() == QMetaType::QByteArray ? QString("(%1 bytes)").arg(v.toByteArray().size()) : v.toString();
            if (t.contains('\\') || t.contains('/')) {
                QStringList parts;
                for (const QString& p : t.split("; ")) parts << ".../" + QFileInfo(p).fileName();
                t = parts.join("; ");
            }
            lines << k + " = " + t;
        }
        put("settings.txt", lines.join('\n') + "\n");
    }
    {
        QMutexLocker lk(&g_m);
        g_file.flush();
    }
    int n = 0;
    for (const QFileInfo& fi : QDir(g_dir).entryInfoList({"xlamina-*.log"}, QDir::Files, QDir::Name | QDir::Reversed)) {
        if (n++ >= 3) break;
        QFile::copy(fi.absoluteFilePath(), QDir(work).filePath(fi.fileName()));   // (written redacted already)
    }
    // one zip, by Windows' own tar (as the updater unpacks with it); the folder stays when that does not work
    const QString zip = QDir(folder).filePath(name + ".zip");
    QFile::remove(zip);
    const QString tar = QDir(qEnvironmentVariable("SystemRoot", "C:/Windows")).filePath("System32/tar.exe");
    if (QFileInfo::exists(tar)) {
        QProcess p;
        p.setWorkingDirectory(folder);
        p.start(tar, {"-a", "-c", "-f", name + ".zip", name});
        if (p.waitForFinished(20000) && p.exitStatus() == QProcess::NormalExit && p.exitCode() == 0 && QFileInfo(zip).size() > 0) {
            QDir(work).removeRecursively();
            return zip;
        }
        QFile::remove(zip);
    }
    return work;
}

}  // namespace DiagLog
