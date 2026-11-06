cmake_minimum_required(VERSION 3.24)
project(Stacker LANGUAGES CXX CUDA)

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

add_subdirectory(core)
add_subdirectory(cli)
add_subdirectory(tools/m0)
