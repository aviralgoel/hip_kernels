# Define the local example list for reduction
set(example_list
    01_reduction_atomic
    02_reduction_shared
    03_reduction_sequential
    04_reduction_contiguous
    05_warp_reduce
    06_grid_reduction
)

# Create executables for each example
foreach(example ${example_list})
  add_executable(${example} ${example}.cpp)
  set_source_files_properties(${example}.cpp PROPERTIES LANGUAGE HIP)

  if(NOT CMAKE_CXX_COMPILER MATCHES ".*hipcc")
    target_link_libraries(${example} hip::device hip::host)
  endif()
  # print the executable
  message(STATUS "Reduction executable: ${example}")
endforeach()
