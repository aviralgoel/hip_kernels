# Define the local example list for ch03
set(example_list
  03_colorToGrayscale
  03_imageBlur
  03_matrixMultiplication
)

# Create executables for each example
foreach(example ${example_list})
  add_executable(${example} ${example}.cpp)
  set_source_files_properties(${example}.cpp PROPERTIES LANGUAGE HIP)

  if(NOT CMAKE_CXX_COMPILER MATCHES ".*hipcc")
    target_link_libraries(${example} hip::device hip::host)
  endif()
endforeach()
