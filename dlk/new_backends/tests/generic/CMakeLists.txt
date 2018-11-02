add_executable(test_generic test_generic.cpp)
target_compile_options(test_generic PRIVATE -Wall
                                            -Wextra
                                            -Werror
                                            -Wno-unknown-pragmas
                                            -Wno-unused-function
                                            -Wno-unused-parameter
                                            -Wno-unused-variable
                                            -Wno-sign-compare)

add_test(NAME test_generic
        COMMAND $<TARGET_FILE:test_generic>
        WORKING_DIRECTORY ${BUILD_TEST_DIR}/tests/generic)