cmake_minimum_required(VERSION 3.19)

project(GemmTest CUDA)

enable_language(CXX CUDA)

find_package(CUDA REQUIRED)

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


set(SOURCE_FILES gemm_test.cu)
add_executable(gemm_test ${SOURCE_FILES})
target_include_directories(gemm_test PUBLIC ${cutlass_SOURCE_DIR}/include)
set_target_properties(gemm_test PROPERTIES CUDA_ARCHITECTURES native)


set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CUDA_STANDARD 17)
