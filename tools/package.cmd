@echo off
rem Builds and zips a release: dist\xLamina-<version>-win64.zip, one folder xLamina with xLamina.exe (GUI),
rem stackcli.exe, the Qt and VC++ runtime files, README and LICENSE. dist\xLamina-<version>-win64\ keeps the same
rem files unzipped, ready to run.
setlocal
cd /d "%~dp0.."
call tools\build.cmd
if errorlevel 1 exit /b 1
if "%VS%"=="" (
  if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Redist" set "VS=%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools"
  if exist "%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Redist" set "VS=%ProgramFiles%\Microsoft Visual Studio\2022\Community"
)
if "%QT_DIR%"=="" (
  for /f "delims=" %%d in ('powershell -NoProfile -Command "Get-ChildItem C:\Qt -Directory -Filter 6.* | Where-Object { Test-Path ($_.FullName + '\msvc2022_64\bin\windeployqt.exe') } | Sort-Object { [version]$_.Name } | Select-Object -Last 1 -ExpandProperty FullName"') do set "QT_DIR=%%d\msvc2022_64"
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
  rem only the JPEG plugin, which Save as writes through (image files are read by the engine or by Windows, never by Qt)
  "%QT_DIR%\bin\windeployqt.exe" --release --no-translations --no-opengl-sw --no-system-d3d-compiler --compiler-runtime --exclude-plugins qsvg,qsvgicon,qgif,qico,qtuiotouchplugin "dist\%NAME%\xLamina.exe" >nul
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
rem the manual (Help > Manual opens manual\index.html beside the program)
xcopy /e /q /y /i docs\manual dist\%NAME%\manual >nul
copy /y README.md dist\%NAME%\ >nul
copy /y LICENSE dist\%NAME%\ >nul
rem the icon as a file: a shortcut that points at it keeps a sharp icon across updates (Explorer's cache serves a
rem stale small image for a while after the exe it reads the icon from is replaced)
copy /y gui\icon\xlamina.ico dist\%NAME%\ >nul
if exist dist\%NAME%.zip del dist\%NAME%.zip
rem the zip holds one folder, xLamina, with the files in it: unpacked "here" in Downloads it makes its own folder
rem instead of putting the program among the user's files (the updater of 1.0.2 and later reads this layout; 1.0.1
rem and earlier need the files at the top of the zip)
if exist dist\_zip rmdir /s /q dist\_zip
mkdir dist\_zip\xLamina
xcopy /e /q /y dist\%NAME%\* dist\_zip\xLamina\ >nul
powershell -NoProfile -Command "Compress-Archive -Path 'dist\_zip\xLamina' -DestinationPath 'dist\%NAME%.zip'"
rmdir /s /q dist\_zip
echo dist\%NAME%.zip
rem the zip's SHA-256, for the release notes (GitHub states the same digest for the uploaded asset; the updater checks it)
certutil -hashfile dist\%NAME%.zip SHA256 | findstr /v ":"
