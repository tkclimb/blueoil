set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BUILD_TESTS_DIR})
add_subdirectory(generic)

if(USE_IPP)
  add_subdirectory(intel_hls)
endif()
