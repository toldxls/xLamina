@echo off
rem Builds and zips a release: dist\xLamina-<version>-win64.zip with xLamina.exe (GUI), stackcli.exe,
rem the Qt and VC++ runtime files, README and LICENSE.
setlocal
cd /d "%~dp0.."
call tools\build.cmd
if errorlevel 1 exit /b 1
if "%QT_DIR%"=="" (
  for /d %%d in ("C:\Qt\6.*") do if exist "%%d\msvc2022_64\bin\windeployqt.exe" set "QT_DIR=%%d\msvc2022_64"
)
for /f "tokens=3" %%v in ('findstr /c:"XLAMINA_VERSION" cli\version.h') do set VER=%%~v
set NAME=xLamina-%VER%-win64
if exist dist\%NAME% rmdir /s /q dist\%NAME%
mkdir dist\%NAME%
copy /y build\cli\stackcli.exe dist\%NAME%\ >nul
if exist build\gui\xLamina.exe (
  copy /y build\gui\xLamina.exe dist\%NAME%\ >nul
  "%QT_DIR%\bin\windeployqt.exe" --release --no-translations --no-opengl-sw --no-system-d3d-compiler --compiler-runtime "dist\%NAME%\xLamina.exe" >nul
  if errorlevel 1 ( echo windeployqt failed & exit /b 2 )
)
copy /y README.md dist\%NAME%\ >nul
copy /y LICENSE dist\%NAME%\ >nul
if exist dist\%NAME%.zip del dist\%NAME%.zip
powershell -NoProfile -Command "Compress-Archive -Path 'dist\%NAME%\*' -DestinationPath 'dist\%NAME%.zip'"
echo dist\%NAME%.zip
