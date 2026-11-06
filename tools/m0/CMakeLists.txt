add_executable(m0_bench m0_bench.cu)
target_link_libraries(m0_bench PRIVATE CUDA::cudart_static CUDA::nvml)
target_compile_definitions(m0_bench PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
