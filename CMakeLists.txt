cmake_minimum_required(VERSION 3.2)

set(FDEEP_TOP_DIR ${CMAKE_CURRENT_LIST_DIR})

list(APPEND CMAKE_MODULE_PATH "${FDEEP_TOP_DIR}/cmake")

include(cmake/hunter.cmake) # default off

project(frugally-deep VERSION 0.7.6)

message(STATUS "===( ${PROJECT_NAME} ${PROJECT_VERSION} )===")

option(FDEEP_BUILD_UNITTEST "Build unit tests" OFF)
option(FDEEP_USE_TOOLCHAIN "Use external toolchain" OFF)
option(FDEEP_USE_DOUBLE "Use double precision" OFF)

if(NOT FDEEP_USE_TOOLCHAIN)
  include(cmake/toolchain.cmake)
endif()

add_library(fdeep INTERFACE)
target_include_directories(fdeep INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
  )

if(FDEEP_USE_DOUBLE)
  target_compile_definitions(fdeep INTERFACE FDEEP_FLOAT_TYPE=double)
endif()

find_package(Threads REQUIRED)
target_link_libraries(fdeep INTERFACE Threads::Threads)

hunter_add_package(FunctionalPlus) # default noop
find_package(FunctionalPlus CONFIG REQUIRED)
target_link_libraries(fdeep INTERFACE FunctionalPlus::fplus)

hunter_add_package(Eigen) # default noop
find_package(Eigen3 CONFIG REQUIRED)
target_link_libraries(fdeep INTERFACE Eigen3::Eigen)

hunter_add_package(nlohmann_json) # default noop
find_package(nlohmann_json CONFIG REQUIRED)
target_link_libraries(fdeep INTERFACE nlohmann_json)

if(FDEEP_BUILD_UNITTEST)
    enable_testing()
    subdirs(test)
endif()

# pkgconfig installation:
include(cmake/pkgconfig.cmake)
