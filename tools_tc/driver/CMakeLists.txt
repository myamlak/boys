# The cross-check driver, built on its own against a configured boys build.
# Build the tree first (cmake -S . -B build), then:
#   cmake -S tools_tc/driver -B build/tools_tc-driver
#   cmake --build build/tools_tc-driver
# Point BOYS_BUILD at a different configured build directory if the library
# was built there instead.
cmake_minimum_required(VERSION 3.20)
project(dump LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 23)

set(BOYS_BUILD "${CMAKE_CURRENT_SOURCE_DIR}/../../build" CACHE PATH
    "boys build directory holding the built boys library")
find_library(BOYS_LIB boys PATHS "${BOYS_BUILD}" "${BOYS_BUILD}/Release"
    "${BOYS_BUILD}/Debug" NO_DEFAULT_PATH)
if (NOT BOYS_LIB)
    message(FATAL_ERROR
        "no boys library under ${BOYS_BUILD}; configure and build the tree first")
endif()

add_executable(dump_values ../dump_values.cpp)
target_include_directories(dump_values PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_link_libraries(dump_values PRIVATE ${BOYS_LIB})
