cmake_minimum_required(VERSION 3.21)

project(hip_simple_kernels LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 14)
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
add_subdirectory(ch02)  # Add other subdirectories as needed
add_subdirectory(ch03)
