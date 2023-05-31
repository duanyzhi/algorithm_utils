cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(algorithm_utils LANGUAGES CXX C)

find_package(OpenCV REQUIRED)

include_directories(${PROJECT_SOURCE_DIR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")


include_directories(${CMAKE_CURRENT_BINARY_DIR})

include_directories(interface)

set(alu_interface
  "interface/api.h"
  "interface/common.h"
  "interface/alu.h"
)

set(alu_srcs
  "core/tensor.h"
  "core/tensor.cc"
  "core/alu_test.cc"

  "common/common.cc"
  "common/logging.h"

  "arithmetic/mathematic.h"
  "arithmetic/mathematic.cc"

  "cv/core/methods.h"
  "cv/core/methods.cc"
)

set(alulibs alu)
add_library(${alulibs}
    SHARED
        ${alu_interface}
        ${alu_srcs}
)

set(PROJECT_VERSION 0.1.0)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set_target_properties(${alulibs}
    PROPERTIES
    CXX_EXTENSIONS ON
    LINKER_LANGUAGE CXX POSITION_INDEPENDENT_CODE ON CUDA_SEPARABLE_COMPILATION ON
    COMPILE_DEFINITIONS BUILDING_DLL
    IMPORTED_LOCATION ${OpenCV_DIR}/x64/vc15/lib
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
)

add_subdirectory(test)
