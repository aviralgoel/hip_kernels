# Define the local example list for ch01
set(example_list
basic_convolution_Fcmem
naive_convolution
)

# Create executables for each example
foreach(example ${example_list})
  add_executable(${example} ${example}.cpp)
  set_source_files_properties(${example}.cpp PROPERTIES LANGUAGE HIP)

  if(NOT CMAKE_CXX_COMPILER MATCHES ".*hipcc")
    target_link_libraries(${example} hip::device hip::host)
  endif()
  # print the executable
  message(STATUS "Executable: ${example}")
endforeach()
