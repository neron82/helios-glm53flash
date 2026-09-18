cmake_minimum_required(VERSION 3.22)
project(aux_kernels LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CUDA_INCLUDE_DIR "/usr/local/cuda-13.0/targets/x86_64-linux/include")
set(CUDA_LIB_DIR "/usr/local/cuda-13.0/targets/x86_64-linux/lib")

add_compile_options(-O3 -Wall)
include_directories(${CUDA_INCLUDE_DIR})

add_library(aux STATIC
    activation.cu
    add.cu
    dsa_topk.cu
    gdn.cu
    hc_mix.cu
    norm.cu
    routing.cu
)

target_include_directories(aux PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(aux PRIVATE
    -L${CUDA_LIB_DIR}
    -lcudart
)

# Parity tests
add_executable(test_aux test/test_aux.cpp)
target_link_libraries(test_aux PRIVATE aux -L${CUDA_LIB_DIR} -lcudart)
target_include_directories(test_aux PRIVATE ${CUDA_INCLUDE_DIR})