add_library(stackcore STATIC
    src/tiff.cpp
    src/pinned_pool.cpp
    src/streamer.cpp
    src/convert.cu
    src/warp.cu)
target_include_directories(stackcore PUBLIC include)
target_link_libraries(stackcore PUBLIC CUDA::cudart)
target_compile_definitions(stackcore PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN)
