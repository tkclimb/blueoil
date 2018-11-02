set(CMAKE_CXX_COMPILER i++)
set(HLS_COMPONENT --component intel_hls_qconv3x3_impl)

# for C++ software model simulation
add_executable(sim ${SRC_DIR}/)
target_compile_options(sim PRIVATE -march=x86 ${HLS_COMPONENT})
target_include_directories(syn PRIVATE ${HLS_INCLUDE_DIR})

# for HDL simulation with modelsim
add_executable(syn ${SRC} ${INTEL_HLS_SRC})
target_compile_options(syn PRIVATE -march=5CSEBA6U23I7 ${HLS_COMPONENT} -ghdl --clock 50MHz)
target_include_directories(syn PRIVATE ${HLS_INCLUDE_DIR})