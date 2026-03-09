cmake_minimum_required(VERSION 3.16)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set the CUDA compiler (use nvcc from PATH, or set CMAKE_CUDA_COMPILER to your nvcc path)
if(NOT DEFINED CMAKE_CUDA_COMPILER OR NOT EXISTS "${CMAKE_CUDA_COMPILER}")
  find_program(NVCC_EXECUTABLE nvcc PATHS /usr/bin /usr/local/cuda/bin ENV PATH)
  if(NVCC_EXECUTABLE)
    set(CMAKE_CUDA_COMPILER "${NVCC_EXECUTABLE}" CACHE FILEPATH "CUDA compiler")
  endif()
  unset(NVCC_EXECUTABLE CACHE)
endif()
# Avoid "Failed to detect a default CUDA architecture" when running from IDE or headless
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "53;75;80" CACHE STRING "CUDA architectures to build for")
endif()

project(StillFantasy CXX CUDA)
include(cmake/cuda.cmake)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)

option(LLAMA3_SUPPORT ON)
if (LLAMA3_SUPPORT)
  message(STATUS "LLAMA3 SUPPORT")
  add_definitions(-DLLAMA3_SUPPORT)
endif()
option(QWEN2_SUPPORT OFF)
if (QWEN2_SUPPORT)
  message(STATUS "QWEN2 SUPPORT")
  add_definitions(-DQWEN2_SUPPORT)
endif()

option(QWEN3_SUPPORT OFF)
if (QWEN3_SUPPORT)
  message(STATUS "QWEN3 SUPPORT")
  add_definitions(-DQWEN3_SUPPORT)
endif()
# ---- Add dependencies via CPM ----
# see https://github.com/TheLartians/CPM.cmake for more info
option(USE_CPM "Use CPM for dependency management" OFF)

if(USE_CPM)
  # Use CPM to manage dependencies
  include(cmake/CPM.cmake)

  CPMAddPackage(
    NAME GTest
    GITHUB_REPOSITORY google/googletest
    VERSION 1.15.0
  )

  CPMAddPackage(
    NAME glog
    GITHUB_REPOSITORY google/glog
    VERSION 0.7.1
    OPTIONS "BUILD_TESTING Off"
  )

  CPMAddPackage(
    NAME Armadillo
    GITLAB_REPOSITORY conradsnicta/armadillo-code
    GIT_TAG 14.0.1
  )

  CPMAddPackage(
    NAME sentencepiece
    GITHUB_REPOSITORY google/sentencepiece
    VERSION 0.2.0
  )
  find_package(sentencepiece REQUIRED)

  if (LLAMA3_SUPPORT OR QWEN2_SUPPORT OR QWEN3_SUPPORT)
    CPMAddPackage(
        NAME absl
        GITHUB_REPOSITORY abseil/abseil-cpp
        GIT_TAG 20240722.0
        OPTIONS "BUILD_TESTING Off" "ABSL_PROPAGATE_CXX_STD ON" "ABSL_ENABLE_INSTALL ON"
    )
    CPMAddPackage(
        NAME re2
        GITHUB_REPOSITORY google/re2
        GIT_TAG 2024-07-02
    )
    CPMAddPackage(
        NAME nlohmann_json
        GITHUB_REPOSITORY nlohmann/json
        VERSION 3.11.3
    )
  endif()
endif()

# ---------------------------------
find_package(GTest REQUIRED)
find_package(glog REQUIRED)
find_package(Armadillo REQUIRED)

aux_source_directory(stillfantasy/source/tensor/ DIR_TENSOR)
aux_source_directory(stillfantasy/source/base/ DIR_BASE)
aux_source_directory(stillfantasy/source/op/ DIR_OP)
aux_source_directory(stillfantasy/source/model/ DIR_MODEL)
aux_source_directory(stillfantasy/source/engine/ DIR_ENGINE)
aux_source_directory(stillfantasy/source/op/kernels/cpu DIR_KERNEL_CPU)
aux_source_directory(stillfantasy/source/op/kernels/cuda DIR_KERNEL_CUDA)
aux_source_directory(stillfantasy/source/op/kernels/ DIR_KERNEL)
aux_source_directory(stillfantasy/source/sampler DIR_SAMPLE)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(StillFantasy SHARED ${DIR_TENSOR} ${DIR_BASE} ${DIR_OP} ${DIR_KERNEL} ${DIR_MODEL} ${DIR_ENGINE} ${DIR_KERNEL_CPU} ${DIR_KERNEL_CUDA} ${DIR_KERNEL} ${DIR_SAMPLE})
target_link_libraries(StillFantasy sentencepiece glog::glog gtest gtest_main pthread cudart armadillo)
target_link_directories(StillFantasy PUBLIC ${CMAKE_CUDA_COMPILER_LIBRARY_ROOT}/lib64)

target_include_directories(StillFantasy PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(StillFantasy PUBLIC ${PROJECT_SOURCE_DIR}/stillfantasy/include)
target_include_directories(StillFantasy PUBLIC ${Armadillo_INCLUDE_DIR})
target_include_directories(StillFantasy PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

if(USE_CPM)
  # Add sentencepiece include directory
  target_include_directories(StillFantasy PUBLIC ${sentencepiece_SOURCE_DIR}/src)
endif()

set_target_properties(StillFantasy PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_subdirectory(test)
add_subdirectory(demo)
