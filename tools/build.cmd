@echo off
rem Builds xLamina with Visual Studio 2022 Build Tools (or a full VS) and the CUDA toolkit.
rem Override the locations with the VS and CUDA_PATH environment variables.
setlocal
if "%VS%"=="" (
  if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" set "VS=%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools"
  if exist "%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" set "VS=%ProgramFiles%\Microsoft Visual Studio\2022\Community"
  if exist "%ProgramFiles%\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" set "VS=%ProgramFiles%\Microsoft Visual Studio\2022\Professional"
)
if "%CUDA_PATH%"=="" (
  for /d %%d in ("%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v13.*") do set "CUDA_PATH=%%d"
)
if "%VS%"=="" ( echo Visual Studio 2022 not found; set VS & exit /b 10 )
if "%CUDA_PATH%"=="" ( echo CUDA toolkit not found; set CUDA_PATH & exit /b 11 )
call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 10
set "PATH=%VS%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;%VS%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%CUDA_PATH%\bin;%PATH%"
if "%QT_DIR%"=="" (
  for /d %%d in ("C:\Qt\6.*") do if exist "%%d\msvc2022_64\lib\cmake\Qt6" set "QT_DIR=%%d\msvc2022_64"
)
cd /d "%~dp0.."
rem extra arguments go to the configure step, e.g. tools\build.cmd -DXLAMINA_RAW=OFF
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="%QT_DIR%" %*
if errorlevel 1 exit /b 20
cmake --build build
