add_library(stackcore STATIC
    src/tiff.cpp
    src/pinned_pool.cpp
    src/streamer.cpp
    src/frame_cache.cpp
    src/convert.cu
    src/warp.cu
    src/imgops.cu
    src/fold.cu
    src/volume.cu
    src/depth.cu
    src/compose.cu
    src/align.cu
    src/pipeline.cpp)
target_include_directories(stackcore PUBLIC include)
# static CUDA runtime: the executables need only the NVIDIA driver, no CUDA DLLs beside them
# (no cuFFT: Windows ships it only as a DLL; the alignment's small DFT is done by twiddle matmuls)
target_link_libraries(stackcore PUBLIC CUDA::cudart_static)
target_compile_definitions(stackcore PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN)
