// The session's log file and what goes with it: a header that names the machine, a block per render with its
// settings and how it ended, a marker that says a render was under way (so the next start can tell that the last
// one did not finish), and the report a user can send: one zip with the logs, the machine's details and the
// settings, no image data, the user's folder names taken out.
//
// %LOCALAPPDATA%\xLamina\logs\xlamina-YYYYMMDD-HHMMSS.log, one per session, the newest ten kept. With
// XLAMINA_SETTINGS set (a test instance on its own settings file) the logs go beside that file instead.
#pragma once

#include <QString>
#include <QStringList>

namespace DiagLog {

void open();                                 // start this session's file and write the header; call once, early
void write(const QString& line);             // any thread; on the disk before it returns
QString dir();
QString path();                              // this session's file
QString systemDetails();                     // the header's lines: version, Windows, CPU, RAM, GPUs, driver
QString redact(QString text);                // the user's profile folder -> %USERPROFILE%

// a render (or sweep, or benchmark run): its settings at the start, the stage as it moves, how it ended
void runStart(const QString& what, const QStringList& settings);
void runStage(const QString& stage);         // cheap: writes only when the stage's name changes
void runEnd(const QString& outcome);
QString lastRun();                           // the last run's block of this session, for the report

// At start: what the previous session was doing if it ended inside a render ("pass 2: fine focus of a render
// started 14:02"), else empty. The marker is cleared by the call.
QString takeUnfinished();

// The report: <folder>/xLamina-report-YYYYMMDD-HHMM.zip (a plain folder of that name when it cannot be zipped).
// Returns its path, or an empty string with `why` set.
QString makeReport(const QString& folder, const QString& note, QString* why);

}  // namespace DiagLog
