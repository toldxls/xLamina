add_library(stackcore STATIC
    src/tiff.cpp
    src/pinned_pool.cpp
    src/streamer.cpp
    src/convert.cu
    src/warp.cu
    src/imgops.cu
    src/fold.cu
    src/volume.cu
    src/depth.cu)
target_include_directories(stackcore PUBLIC include)
# static CUDA runtime: the executables need only the NVIDIA driver, no cudart DLL beside them
target_link_libraries(stackcore PUBLIC CUDA::cudart_static)
target_compile_definitions(stackcore PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN)
