# Define the local example list for rms normalization
set(example_list
    01_rms_normalization
)

# Create executables for each example
foreach(example ${example_list})
  add_executable(${example} ${example}.cpp)
  set_source_files_properties(${example}.cpp PROPERTIES LANGUAGE HIP)

  if(NOT CMAKE_CXX_COMPILER MATCHES ".*hipcc")
    target_link_libraries(${example} hip::device hip::host)
  endif()
  message(STATUS "RMS executable: ${example}")
endforeach()
