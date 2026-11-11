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

# Camera raw and DNG frames (on hold, off by default: written but never built or tested, FINDINGS Open items):
# -DXLAMINA_RAW=ON fetches LibRaw (LGPL 2.1 / CDDL 1.0) at configure time and builds it as raw.dll, which
# then ships beside the executables. Without it raw files are not picked up.
option(XLAMINA_RAW "Read camera raw and DNG frames through LibRaw" OFF)
if(XLAMINA_RAW)
  include(FetchContent)
  FetchContent_Declare(libraw_src
    GIT_REPOSITORY https://github.com/LibRaw/LibRaw.git
    GIT_TAG 0.22.3
    GIT_SHALLOW TRUE)
  FetchContent_Declare(libraw_cmake
    GIT_REPOSITORY https://github.com/LibRaw/LibRaw-cmake.git
    GIT_TAG eb98e4325aef2ce85d2eb031c2ff18640ca616d3
    EXCLUDE_FROM_ALL)
  FetchContent_MakeAvailable(libraw_src)   # sources only: LibRaw has no CMakeLists of its own
  set(LIBRAW_PATH "${libraw_src_SOURCE_DIR}" CACHE STRING "" FORCE)
  set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)      # LibRaw as a DLL (our own libraries name their type)
  set(ENABLE_OPENMP OFF CACHE BOOL "" FORCE)         # the reader threads decode one frame each
  set(ENABLE_LCMS OFF CACHE BOOL "" FORCE)
  set(ENABLE_JASPER OFF CACHE BOOL "" FORCE)
  set(ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(libraw_cmake)
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
