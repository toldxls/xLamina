@echo off
rem Builds and zips a release: dist\xLamina-<version>-win64.zip with stackcli.exe, README and LICENSE.
setlocal
cd /d "%~dp0.."
call tools\build.cmd
if errorlevel 1 exit /b 1
for /f "tokens=3" %%v in ('findstr /c:"XLAMINA_VERSION" cli\version.h') do set VER=%%~v
set NAME=xLamina-%VER%-win64
if exist dist\%NAME% rmdir /s /q dist\%NAME%
mkdir dist\%NAME%
copy /y build\cli\stackcli.exe dist\%NAME%\ >nul
copy /y README.md dist\%NAME%\ >nul
copy /y LICENSE dist\%NAME%\ >nul
if exist dist\%NAME%.zip del dist\%NAME%.zip
powershell -NoProfile -Command "Compress-Archive -Path 'dist\%NAME%\*' -DestinationPath 'dist\%NAME%.zip'"
echo dist\%NAME%.zip
