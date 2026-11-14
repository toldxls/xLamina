@echo off
rem Remakes the manual's screen grabs: tools\manual.cmd <folder of frames> [<dust map>]
rem The dust map (any photograph of a white background, of any size) is for the grabs of the dust map's window; without
rem it those two grabs are left as they are.
rem Renders the folder in build\gui\xLamina.exe with every window kept off the screen and writes the grabs and
rem rects.js (where each control sits in each grab) to docs\manual\img. The folder's path shows in some grabs, so
rem use one you are content to publish (a copy or hard links under C:\Users\Public, say). About 50 frames is plenty.
rem The run uses its own settings file, so the installed program's settings are untouched.
setlocal
if "%~1"=="" ( echo usage: tools\manual.cmd FOLDER & exit /b 1 )
cd /d "%~dp0.."
if not exist build\gui\xLamina.exe ( echo build\gui\xLamina.exe not found: run tools\build.cmd first & exit /b 2 )
if "%QT_DIR%"=="" (
  for /f "delims=" %%d in ('powershell -NoProfile -Command "Get-ChildItem C:\Qt -Directory -Filter 6.* | Where-Object { Test-Path ($_.FullName + '\msvc2022_64\bin\Qt6Core.dll') } | Sort-Object { [version]$_.Name } | Select-Object -Last 1 -ExpandProperty FullName"') do set "QT_DIR=%%d\msvc2022_64"
)
set "PATH=%QT_DIR%\bin;%PATH%"
set "QT_PLUGIN_PATH=%QT_DIR%\plugins"
set "XLAMINA_SETTINGS=%TEMP%\xlamina_manual.ini"
(
  echo [ui]
  echo developer=false
  echo theme=dark
  echo [render]
  echo fov=1.49
) > "%XLAMINA_SETTINGS%"
set "XLAMINA_MANUAL_DUST=%~2"
if not exist docs\manual\img mkdir docs\manual\img
start "" /wait build\gui\xLamina.exe "%~1" --render --manual docs\manual\img
del "%XLAMINA_SETTINGS%"
dir /b docs\manual\img
