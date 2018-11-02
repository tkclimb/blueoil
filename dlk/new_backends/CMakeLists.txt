cmake_minimum_required(VERSION 3.1)
project(conv_sample CXX)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE Debug)

get_filename_component(ROOT_DIR ${CMAKE_SOURCE_DIR} ABSOLUTE)
get_filename_component(SRC_DIR ${ROOT_DIR}/src ABSOLUTE)
get_filename_component(INCLUDE_DIR "${ROOT_DIR}/include" ABSOLUTE)
get_filename_component(TESTS_DIR ${ROOT_DIR}/tests ABSOLUTE)
get_filename_component(SRC_INTERFACE_DIR ${SRC_DIR}/interface ABSOLUTE)
get_filename_component(SRC_BACKEND_DIR ${SRC_DIR}/backends ABSOLUTE)
get_filename_component(SRC_BACKEND_GENERIC_DIR ${SRC_DIR}/generic ABSOLUTE)
get_filename_component(SRC_BACKEND_INTEL_HLS_DIR ${SRC_DIR}/intel_hls ABSOLUTE)
get_filename_component(BUILD_DIR ${CMAKE_BINARY_DIR} ABSOLUTE)
get_filename_component(BUILD_TESTS_DIR ${BUILD_DIR}/tests ABSOLUTE)

if(WIN32)
  # for Windows operating system in general
  message(FATAL_ERROR "Windows32 is not supported...")
elseif(MSVC OR MSYS OR MINGW)
  # for detecting Windows compilers
  message(FATAL_ERROR "Windows is not supported...")
elseif(APPLE)
  # for MacOS X
  message("Run on macOS")
  message("COMPILER: ${CMAKE_CXX_COMPILER}")
elseif(UNIX AND NOT APPLE)
  # for Linux, BSD, Solaris, Minix
  message("Run on Linux")
  message("COMPILER: ${CMAKE_CXX_COMPILER}")
else()
endif()


option(BUILD_TESTS "Build tests." ON)
option(USE_IPP "Use intel HLS compiler (i++)." OFF)


include_directories(${INCLUDE_DIR})
add_compile_options(-std=c++0x)

if(BUILD_TESTS)
  message("here")
  enable_testing()
  add_subdirectory(${TESTS_DIR})
endif()