cmake_minimum_required(VERSION 3.21)

project(hip_simple_kernels LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if( $ENV{HIP_PLATFORM} MATCHES "nvidia" )
  message(STATUS "HIP_PLATFORM: nvidia")
else()
  message(STATUS "HIP_PLATFORM: amd")
endif()
# Find HIP package
find_package(hip REQUIRED)

# Add subdirectories (ch01, ch02, etc.)
add_subdirectory(ch01)
add_subdirectory(ch02)  # Add other subdirectories as needed
