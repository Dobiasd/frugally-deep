message(STATUS "Building Unit Tests ${UNITTEST}")

find_package(doctest REQUIRED)

macro(_add_test _NAME)
    add_executable(${_NAME} ${_NAME}.cpp)
    add_test(NAME ${_NAME} COMMAND ${_NAME})

    target_include_directories(${_NAME} SYSTEM PUBLIC ${doctest_INCLUDE_DIR})
    target_link_libraries(${_NAME} ${CMAKE_THREAD_LIBS_INIT})
endmacro()

_add_test(test_model_full_test)

add_custom_target(unittest test_model_full_test
                        COMMENT "Running unittests\n\n"
                        VERBATIM
                        )
