cmake_minimum_required(VERSION 3.18)

# 1. MUST BE BEFORE project()
if(DEFINED ENV{CONDA_PREFIX})
    set(CMAKE_CUDA_COMPILER "$ENV{CONDA_PREFIX}/bin/nvcc")
    set(CUDA_TOOLKIT_ROOT_DIR "$ENV{CONDA_PREFIX}/targets/x86_64-linux")
endif()

# 2. Tell CMake to generate the mapping file for your IDE (IntelliSense/Clangd)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 3. NOW initialize the project
project(GeoCuTool LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES 52 61 70 75 80 86 89)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/geocutool/csrc 
    ${CUDA_TOOLKIT_ROOT_DIR}/include
)

add_library(primitive_gs SHARED geocutool/csrc/primitive/gs.cu)
target_link_libraries(primitive_gs PRIVATE cudart)

add_library(primitive_pgs SHARED geocutool/csrc/primitive/pgs.cu)
target_link_libraries(primitive_pgs PRIVATE cudart)

add_library(kdtree SHARED geocutool/csrc/data_structure/kdtree.cu)
target_link_libraries(kdtree PRIVATE cudart)