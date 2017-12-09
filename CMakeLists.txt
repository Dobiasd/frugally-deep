cmake_minimum_required(VERSION 3.2)

project(frugally-deep)
set(PROJECT_VERSION 0.1)

message(STATUS "===( ${PROJECT_NAME} ${PROJECT_VERSION} )===")

option(UNITTEST "Build unit tests" OFF)
message(STATUS "Building Unit Tests ${UNITTEST}")

set(COMPILE_OPTIONS -Wall
                    -Wextra
                    -pedantic
                    -Werror
                    -Weffc++
                    -Wconversion
                    -Wsign-conversion
                    -Wctor-dtor-privacy
                    -Wreorder
                    -Wold-style-cast
                    -Wparentheses
                    )

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

add_library(fdeep INTERFACE)
target_include_directories(fdeep INTERFACE
                       $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                       $<INSTALL_INTERFACE:include>
                        )

if( UNITTEST )
    enable_testing()
    add_subdirectory(test)
endif()

install(TARGETS fdeep EXPORT fdeep-config DESTINATION include)
install(DIRECTORY "${CMAKE_SOURCE_DIR}/include/" DESTINATION include)
install(EXPORT fdeep-config DESTINATION share/fdeep/cmake)