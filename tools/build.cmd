@echo off
set VS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.5
call "%VS%\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 10
set PATH=%VS%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;%VS%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%CUDA_PATH%\bin;%PATH%
cd /d "C:\Users\user\Desktop\Stacker_handoff_2026-11-06\Stacker_handoff_2026-11-06\Stacker"
where cl.exe
where nvcc.exe
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
if errorlevel 1 exit /b 20
cmake --build build
