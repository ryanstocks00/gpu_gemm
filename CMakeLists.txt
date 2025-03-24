cmake_minimum_required(VERSION 3.19)

project(GemmTest CUDA)

enable_language(CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)
FetchContent_Declare(
    cutlass
    GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
    GIT_TAG v3.8.0
)

FetchContent_GetProperties(cutlass)
if(NOT cutlass_POPULATED)
    FetchContent_Populate(cutlass)
endif()

set(CMAKE_CUDA_FLAGS_DEBUG "-G -g")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

message(STATUS "cutlass_SOURCE_DIR: ${cutlass_SOURCE_DIR}")
add_compile_options("-I ${cutlass_SOURCE_DIR}/include")
add_compile_options("-I ${cutlass_SOURCE_DIR}/tools/util/include")
add_compile_options(--expt-relaxed-constexpr)
set(SOURCE_FILES gemm_test.cu)
add_executable(gemm_test ${SOURCE_FILES})
target_include_directories(gemm_test PRIVATE ${cutlass_SOURCE_DIR}/include)
set_target_properties(gemm_test PROPERTIES CUDA_ARCHITECTURES native)

add_executable(my_gemm_test my_gemm_test.cu)
target_include_directories(my_gemm_test PRIVATE ${cutlass_SOURCE_DIR}/include)
set_target_properties(my_gemm_test PROPERTIES CUDA_ARCHITECTURES native)
target_link_libraries(my_gemm_test CUDA::curand CUDA::cublas)

add_executable(gemm_grouped gemm_grouped.cu)
target_include_directories(gemm_grouped PRIVATE ${cutlass_SOURCE_DIR}/include)
target_include_directories(gemm_grouped PRIVATE ${cutlass_SOURCE_DIR}/tools/util/include)
set_target_properties(gemm_grouped PROPERTIES CUDA_ARCHITECTURES native)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CUDA_STANDARD 17)
