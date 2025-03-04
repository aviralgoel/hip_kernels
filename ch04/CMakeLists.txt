cmake_minimum_required( VERSION 3.21 )

project( hip_simple_kernels LANGUAGES CXX HIP )

set( CMAKE_CXX_STANDARD 14 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )
set( CMAKE_CXX_EXTENSIONS OFF )

if( $ENV{HIP_PLATFORM} MATCHES "nvidia" )
  message( STATUS "HIP_PLATFORM: nvidia" )
else()
  message( STATUS "HIP_PLATFORM: amd" )
endif()

find_package( hip REQUIRED )

set( example_list
  04_tiledMatrixMultiplication
  )

foreach( example ${example_list} )
  add_executable( ${example}  ${example}.cpp )
  
  # Since the files don't end in .hip (or .cu), we need to tell the
  # compiler that it should look for GPU kernels.
  set_source_files_properties(${example}.cpp PROPERTIES LANGUAGE HIP)

  if( NOT CMAKE_CXX_COMPILER MATCHES ".*hipcc")
    target_link_libraries( ${example} hip::device hip::host )
  endif()
endforeach()