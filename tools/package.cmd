@echo off
rem Builds and zips a release: dist\xLamina-<version>-win64.zip with xLamina.exe (GUI), stackcli.exe,
rem the Qt and VC++ runtime files, README and LICENSE.
setlocal
cd /d "%~dp0.."
call tools\build.cmd
if errorlevel 1 exit /b 1
if "%VS%"=="" (
  if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Redist" set "VS=%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools"
  if exist "%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Redist" set "VS=%ProgramFiles%\Microsoft Visual Studio\2022\Community"
)
if "%QT_DIR%"=="" (
  for /d %%d in ("C:\Qt\6.*") do if exist "%%d\msvc2022_64\bin\windeployqt.exe" set "QT_DIR=%%d\msvc2022_64"
)
for /f "tokens=3" %%v in ('findstr /c:"XLAMINA_VERSION" cli\version.h') do set VER=%%~v
set NAME=xLamina-%VER%-win64
if exist dist\%NAME% rmdir /s /q dist\%NAME%
mkdir dist\%NAME%
copy /y build\cli\stackcli.exe dist\%NAME%\ >nul
rem LibRaw (raw and DNG frames), when the build has it
if exist build\cli\raw.dll copy /y build\cli\raw.dll dist\%NAME%\ >nul
if exist build\gui\xLamina.exe (
  copy /y build\gui\xLamina.exe dist\%NAME%\ >nul
  "%QT_DIR%\bin\windeployqt.exe" --release --no-translations --no-opengl-sw --no-system-d3d-compiler --compiler-runtime "dist\%NAME%\xLamina.exe" >nul
  if errorlevel 1 ( echo windeployqt failed & exit /b 2 )
)
rem the VC++ runtime (windeployqt --compiler-runtime finds it only in a full Visual Studio)
for /d %%r in ("%VS%\VC\Redist\MSVC\14.*") do if exist "%%r\x64\Microsoft.VC143.CRT\vcruntime140.dll" set "REDIST=%%r\x64\Microsoft.VC143.CRT"
if not "%REDIST%"=="" (
  copy /y "%REDIST%\vcruntime140.dll" dist\%NAME%\ >nul
  copy /y "%REDIST%\vcruntime140_1.dll" dist\%NAME%\ >nul
  copy /y "%REDIST%\msvcp140.dll" dist\%NAME%\ >nul
  copy /y "%REDIST%\msvcp140_1.dll" dist\%NAME%\ >nul
  copy /y "%REDIST%\msvcp140_2.dll" dist\%NAME%\ >nul
) else ( echo VC++ redist not found: the zip will need the runtime installed & exit /b 3 )
copy /y README.md dist\%NAME%\ >nul
copy /y LICENSE dist\%NAME%\ >nul
rem the icon as a file: a shortcut that points at it keeps a sharp icon across updates (Explorer's cache serves a
rem stale small image for a while after the exe it reads the icon from is replaced)
copy /y gui\icon\xlamina.ico dist\%NAME%\ >nul
if exist dist\%NAME%.zip del dist\%NAME%.zip
powershell -NoProfile -Command "Compress-Archive -Path 'dist\%NAME%\*' -DestinationPath 'dist\%NAME%.zip'"
echo dist\%NAME%.zip
rem the zip's SHA-256, for the release notes (GitHub states the same digest for the uploaded asset; the updater checks it)
certutil -hashfile dist\%NAME%.zip SHA256 | findstr /v ":"
