add_library(stackcore STATIC
    src/tiff.cpp
    src/raw.cpp
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
# Windows Imaging Component decodes JPEG and compressed TIFF frames (tiff.cpp): part of Windows, nothing to ship
target_link_libraries(stackcore PUBLIC CUDA::cudart_static windowscodecs ole32)
target_compile_definitions(stackcore PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN)
# camera raw and DNG frames (raw.cpp): LibRaw, as raw.dll beside the executables
if(TARGET raw)
  target_link_libraries(stackcore PRIVATE raw)
  target_compile_definitions(stackcore PRIVATE XLAMINA_HAVE_LIBRAW)
endif()

# copies the DLLs stackcore needs beside an executable target
function(stackcore_copy_dlls tgt)
  if(TARGET raw)
    add_custom_command(TARGET ${tgt} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:raw> $<TARGET_FILE_DIR:${tgt}>)
  endif()
endfunction()
