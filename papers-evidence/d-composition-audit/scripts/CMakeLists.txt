cmake_minimum_required(VERSION 3.20)

project(composed_bounds_measure LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT DEFINED BOYS_SOURCE_DIR)
    message(FATAL_ERROR "pass -DBOYS_SOURCE_DIR=<path to the boys v5.0.0 source>")
endif()

if(NOT EXISTS "${BOYS_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "BOYS_SOURCE_DIR does not look like the boys source: ${BOYS_SOURCE_DIR}")
endif()

# A measurement build, not the library's test build.
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(BOYS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_CUDA OFF CACHE BOOL "" FORCE)
set(BOYS_BUILD_C_API OFF CACHE BOOL "" FORCE)
set(BOYS_BUILD_PROBE OFF CACHE BOOL "" FORCE)

# The shipped four-lane surface is the object of the measurement. The library's
# default is ON, but a stale cache from a closed-seam configure would silently
# drop fp16 and bf16 and turn the 576-cell design into a 288-cell one.
set(BOYS_FP16_SEAM ON CACHE BOOL "" FORCE)

add_subdirectory("${BOYS_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/boys-lib" EXCLUDE_FROM_ALL)

add_executable(measure-interaction measure_interaction.cpp)
target_link_libraries(measure-interaction PRIVATE boys::boys)
