cmake_minimum_required(VERSION 3.24)
project(xLamina LANGUAGES CXX CUDA)

# dynamic C runtime (Qt's prebuilt libraries use it); the release ships the two VC++ runtime DLLs
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Ampere (3060 Ti / 3070 / 3080 Ti). Add 89 / 120 + PTX when other cards appear (docs/DESIGN.md).
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

find_package(CUDAToolkit REQUIRED)

# CCCL (thrust/cub) wants MSVC's conforming preprocessor
if(MSVC)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>)
  add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
endif()

add_subdirectory(core)
add_subdirectory(cli)
add_subdirectory(tools/m0)

# the GUI needs Qt 6 (Widgets, Network for the update check); set CMAKE_PREFIX_PATH to its install, e.g. C:/Qt/6.8.3/msvc2022_64
find_package(Qt6 COMPONENTS Widgets Network QUIET)
if(Qt6_FOUND)
  add_subdirectory(gui)
else()
  message(STATUS "Qt 6 not found: building without the GUI")
endif()

install(TARGETS stackcli RUNTIME DESTINATION .)
install(FILES README.md LICENSE DESTINATION .)
