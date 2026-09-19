cmake_minimum_required(VERSION 3.24)
project(helios CXX CUDA)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=x86-64-v3 -Wall")
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -lineinfo")

# Kernel libraries. These live in standalone projects (each with its own project() and its own
# test executables, usable on their own), but they are added here as subprojects so that a clean
# checkout builds with one cmake invocation instead of three. Linking targets rather than archive
# paths also lets CMake order the dependencies itself.
add_subdirectory(src/cuda/quant ${CMAKE_BINARY_DIR}/quant)
add_subdirectory(src/cuda/aux   ${CMAKE_BINARY_DIR}/aux)
add_subdirectory(src/cuda/attn  ${CMAKE_BINARY_DIR}/attn)

add_executable(helios
  src/main.cpp
  src/core/safetensors.cpp
  src/core/device.cpp
  src/core/model.cpp
  src/engine/cache.cpp
  src/engine/slotmgr.cpp
  src/engine/sampler.cpp
  src/engine/glue.cu
  src/engine/glue2.cu
  src/engine/runner.cpp
  src/engine/server.cpp
  src/core/chat.cpp
  src/tokenizer/tokenizer.cpp
)
target_include_directories(helios PRIVATE src third_party
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/attn
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/aux
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/quant)
target_link_libraries(helios PRIVATE CUDA::cudart CUDA::cuda_driver Threads::Threads
  exl3_quant_core aux attn_core)

add_executable(helios_utf8_test test/test_utf8.cpp)
target_include_directories(helios_utf8_test PRIVATE src)

# Pure policy test: no CUDA, no model, so it runs anywhere and pins the reuse decision itself.
add_executable(helios_prefix_test src/engine/test_prefix.cpp)
target_include_directories(helios_prefix_test PRIVATE src)

add_executable(helios_chat_test src/core/test_chat.cpp src/core/chat.cpp src/tokenizer/tokenizer.cpp)
target_include_directories(helios_chat_test PRIVATE src third_party)
target_link_libraries(helios_chat_test PRIVATE Threads::Threads)

add_executable(helios_tokenizer_test src/tokenizer/test_tokenizer.cpp src/tokenizer/tokenizer.cpp)
target_include_directories(helios_tokenizer_test PRIVATE src third_party)
target_link_libraries(helios_tokenizer_test PRIVATE Threads::Threads)
