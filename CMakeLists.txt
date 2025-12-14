cmake_minimum_required(VERSION 3.21)

project(hip_simple_kernels LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # Set the CMAKE_PREFIX_PATH to where HIP is installed only on Linux
    set(CMAKE_PREFIX_PATH "/opt/rocm" CACHE STRING "Path to ROCm installation" FORCE)
endif()

# Find HIP package
find_package(hip REQUIRED)

# Add subdirectories (ch01, ch02, etc.)
add_subdirectory(ch01)
add_subdirectory(ch02)
add_subdirectory(ch03)
add_subdirectory(ch04)
add_subdirectory(gemm)
add_subdirectory(convolution)

set(CMAKE_CXX_COMPILER "/opt/rocm/bin/amdclang++")
set(CMAKE_HIP_COMPILER "/opt/rocm/bin/amdclang++")
add_definitions(-D__HIP_PLATFORM_AMD__)
