# Define the local example list for ch01
set(example_list
    naive_gemm 
    tiled_gemm
    online_softmax
    mfma_gemm
)

# Create executables for each example
foreach(example ${example_list})
  add_executable(${example} ${example}.cpp)
  set_source_files_properties(${example}.cpp PROPERTIES LANGUAGE HIP)

  # Add OpenMP support for mfma_gemm
  if(${example} STREQUAL "mfma_gemm")
    target_compile_options(${example} PRIVATE -fopenmp)
    target_link_libraries(${example} -fopenmp)  # Removed PRIVATE to match
    message(STATUS "Enabling OpenMP for ${example}")
  endif()

  if(NOT CMAKE_CXX_COMPILER MATCHES ".*hipcc")
    target_link_libraries(${example} hip::device hip::host)
  endif()
  # print the executable
  message(STATUS "Executable: ${example}")
endforeach()
