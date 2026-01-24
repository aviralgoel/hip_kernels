cmake_minimum_required(VERSION 3.21)

# Try to find ROCm installation
if(EXISTS "/opt/rocm")
    set(ROCM_PATH "/opt/rocm")
elseif(EXISTS "/opt/rocm-7.3.0")
    set(ROCM_PATH "/opt/rocm-7.3.0")
else()
    # Try to find any rocm installation
    file(GLOB ROCM_DIRS "/opt/rocm-*")
    if(ROCM_DIRS)
        list(SORT ROCM_DIRS)
        list(REVERSE ROCM_DIRS)
        list(GET ROCM_DIRS 0 ROCM_PATH)
    else()
        set(ROCM_PATH "/opt/rocm")
    endif()
endif()

message(STATUS "Using ROCm installation at: ${ROCM_PATH}")

# Set compilers BEFORE project() call
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(CMAKE_CXX_COMPILER "${ROCM_PATH}/bin/amdclang++")
    set(CMAKE_HIP_COMPILER "${ROCM_PATH}/bin/amdclang++")
endif()

# Set the CMAKE_PREFIX_PATH to where HIP is installed
set(CMAKE_PREFIX_PATH "${ROCM_PATH}" CACHE STRING "Path to ROCm installation" FORCE)
list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}/lib/cmake/hip")

project(hip_simple_kernels LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_definitions(-D__HIP_PLATFORM_AMD__)

# Find HIP package (lowercase 'hip' is correct for ROCm)
find_package(hip REQUIRED CONFIG)

# Add subdirectories (ch01, ch02, etc.)
add_subdirectory(ch01)
add_subdirectory(ch02)
add_subdirectory(ch03)
add_subdirectory(ch04)
add_subdirectory(gemm)
add_subdirectory(convolution)
add_subdirectory(reduction)