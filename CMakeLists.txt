cmake_minimum_required(VERSION 3.25)

# boys — self-contained Boys-function kernel.
#
# Layout: public headers under include/boys/, sources under src/,
# tests under tests/ (gtest, pinned submodule), benchmark drivers under
# benchmarks/, the regeneration script under tools/. The library exports
# only the m = 1 instantiations of the templated entries (extern-template
# architecture, see src/boys.cpp); the CUDA lane is an optional second
# target that needs the CUDA toolkit (device pins are a documented local
# gate — CI has no GPU leg and never builds it). No PCH in this tree.
#
# Consumers add this tree with add_subdirectory and link boys::boys: the
# library's only requirement is C++20 (the double NTTP kAccuracyMultiplier),
# propagated through the target; tests/benchmarks are OFF unless this tree
# is the top-level project (or the consumer opts in), and a missing
# gtest/benchmark submodule is a warning — not a FATAL — when the feature
# that needs it is OFF. See README.md, "Consuming as a submodule".

project(boys VERSION 1.1.0 LANGUAGES CXX)

if (CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(BOYS_TOP_LEVEL ON)
else()
    set(BOYS_TOP_LEVEL OFF)
endif()

option(BOYS_BUILD_TESTS
    "Build the boys test suite (needs the pinned googletest submodule)"
    ${BOYS_TOP_LEVEL})
option(BUILD_BENCHMARKS
    "Build the CPU benchmark drivers; the timed runs themselves are local-only"
    ${BOYS_TOP_LEVEL})
option(BUILD_CUDA
    "Build the CUDA lane (needs the CUDA toolkit; a documented local gate — never CI)"
    OFF)

if (BUILD_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 75 86 CACHE STRING
            "CUDA architectures for the Boys kernels" FORCE)
    endif()
endif()

# Pinned submodules, cloned explicitly — no FetchContent, no network at
# configure time (googletest v1.17.0, benchmark v1.9.5). A missing
# submodule is a FATAL only when the feature that needs it is ON.
if (BOYS_BUILD_TESTS)
    if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/googletest/CMakeLists.txt")
        message(FATAL_ERROR "third_party/googletest is missing. Clone with "
            "--recurse-submodules (pinned v1.17.0).")
    endif()
    set(gtest_force_shared_crt ON CACHE BOOL "Force gtest to use the shared MSVC CRT" FORCE)
    add_subdirectory(third_party/googletest)
elseif (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/googletest/CMakeLists.txt")
    message(WARNING "third_party/googletest is missing; BOYS_BUILD_TESTS is OFF, "
        "so the test suite is not built (informational)")
endif()

# --- Library: the Boys-function kernel --------------------------------------
# BoysFp16 gates the fp16/bf16 API entries in the public headers, so it must
# be PUBLIC: consumers compile against the same seam the library was built
# with. The fp16/bf16 I/O types themselves (f16.hpp) are unconditional.
# The headers need >= C++20 (the double NTTP kAccuracyMultiplier); the
# requirement is propagated through the target, so consumers need no
# global standard flags.
add_library(boys
    src/boys.cpp
    src/boys_c.cpp
    src/boys_simd.cpp)
add_library(boys::boys ALIAS boys)
target_include_directories(boys PUBLIC include PRIVATE src)
target_compile_features(boys PUBLIC cxx_std_20)
target_compile_definitions(boys PUBLIC BoysFp16=1)
if (MSVC)
    # AVX2 intrinsics live in the SIMD TU only (F16C comes with /arch:AVX2);
    # the entry points are runtime-checked via BoysAvx2Available().
    set_source_files_properties(src/boys_simd.cpp PROPERTIES
        COMPILE_OPTIONS "/arch:AVX2")
else()
    # GCC does not imply F16C via -mavx2.
    set_source_files_properties(src/boys_simd.cpp PROPERTIES
        COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
endif()

# --- Tests (gtest; the reference grid is located via BoysDataDir) -----------
if (BOYS_BUILD_TESTS)
    include(CTest)

    add_executable(boys-tests
        tests/boys_test.cpp
        tests/boys_boundary_test.cpp
        tests/boys_accuracy_test.cpp
        tests/boys_fixed_n_test.cpp
        tests/boys_c_test.cpp
        tests/f16_test.cpp)
    target_link_libraries(boys-tests PRIVATE boys gtest gtest_main)
    target_include_directories(boys-tests PRIVATE src)
    target_compile_features(boys-tests PRIVATE cxx_std_23)
    target_compile_definitions(boys-tests PRIVATE
        BoysDataDir="${CMAKE_CURRENT_SOURCE_DIR}/tests/data")

    # The BOUNDARY-CHECK standalone (tests/boys_boundary_standalone.cpp)
    # reproduces the manuscript's measured recursion-boundary cells from the
    # committed reference grid. It is deliberately self-contained (standard
    # library only — the CI legs compile it standalone with plain -O2) and
    # reads boys_reference.csv from its working directory, so ctest runs it
    # with the working directory set to tests/data.
    add_executable(boys-boundary-standalone tests/boys_boundary_standalone.cpp)

    add_test(NAME boys-tests COMMAND boys-tests)
    add_test(NAME boys-boundary-standalone COMMAND boys-boundary-standalone)
    set_tests_properties(boys-boundary-standalone PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/tests/data")
endif()

# --- CUDA lane (optional; needs the toolkit) --------------------------------
if (BUILD_CUDA)
    # The host wrapper (boys_cuda.cpp) and the kernels (boys_cuda.cu) are one
    # target: the .cpp compiles as C++23, the .cu as C++20 with a CUDA-safe
    # include list (see the .cu preamble). Entry points take raw device
    # pointers + a stream and report through boys::BoysStatus.
    add_library(boys-cuda
        src/boys_cuda.cpp
        src/boys_cuda.cu)
    target_include_directories(boys-cuda PUBLIC include PRIVATE src)
    target_compile_definitions(boys-cuda PUBLIC BoysFp16=1)

    if (BOYS_BUILD_TESTS)
        add_executable(boys-cuda-tests tests/boys_cuda_test.cpp)
        target_link_libraries(boys-cuda-tests PRIVATE boys-cuda boys
            gtest gtest_main)
        target_include_directories(boys-cuda-tests PRIVATE src)
        add_test(NAME boys-cuda-tests COMMAND boys-cuda-tests)
    endif()
endif()

# --- Benchmark drivers -------------------------------------------------------
if (BUILD_BENCHMARKS)
    # boys_benchmark.cpp is a Google Benchmark driver; boys_unsorted_simd_
    # benchmark.cpp is a runs-log protocol driver with its own main(). The
    # timed runs are local-only by design (machine/clock/commit recorded per
    # the paper's methodology) and never part of CI.
    if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/benchmark/CMakeLists.txt")
        message(FATAL_ERROR "BUILD_BENCHMARKS=ON needs third_party/benchmark. "
            "Clone with --recurse-submodules (pinned v1.9.5).")
    endif()
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
    add_subdirectory(third_party/benchmark)

    add_executable(boys-benchmark benchmarks/boys_benchmark.cpp)
    target_link_libraries(boys-benchmark PRIVATE boys benchmark::benchmark)
    target_include_directories(boys-benchmark PRIVATE src)
    target_compile_definitions(boys-benchmark PRIVATE BoysFp16=1)
    target_compile_features(boys-benchmark PRIVATE cxx_std_23)

    add_executable(boys-unsorted-simd-benchmark
        benchmarks/boys_unsorted_simd_benchmark.cpp)
    target_link_libraries(boys-unsorted-simd-benchmark PRIVATE boys)
    target_include_directories(boys-unsorted-simd-benchmark PRIVATE src)
    target_compile_definitions(boys-unsorted-simd-benchmark PRIVATE BoysFp16=1)
    target_compile_features(boys-unsorted-simd-benchmark PRIVATE cxx_std_23)
    if (MSVC)
        # The mixed per-vector kernel compiles AVX2 intrinsics in its own TU.
        set_source_files_properties(benchmarks/boys_unsorted_simd_benchmark.cpp
            PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    else()
        set_source_files_properties(benchmarks/boys_unsorted_simd_benchmark.cpp
            PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
    endif()

    if (BUILD_CUDA)
        # CUDA benchmark driver: the kernels TU compiles as C++20 alongside
        # the C++ driver TU. Timed runs local-only.
        add_executable(boys-cuda-benchmark
            benchmarks/boys_cuda_benchmark.cpp
            benchmarks/boys_cuda_benchmark_kernels.cu
            benchmarks/boys_cuda_benchmark_kernels.hpp)
        target_include_directories(boys-cuda-benchmark PRIVATE include src)
        target_link_libraries(boys-cuda-benchmark PRIVATE boys-cuda boys)
        target_compile_definitions(boys-cuda-benchmark PRIVATE BoysFp16=1)
    endif()
elseif (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/benchmark/CMakeLists.txt")
    message(WARNING "third_party/benchmark is missing; BUILD_BENCHMARKS is OFF, "
        "so the benchmark drivers are not built (informational)")
endif()

# --- Generator byte-identity gate (top-level convenience targets) -----------
# --check regenerates the Chebyshev tables and the reference grid into a temp
# and byte-compares against the committed files (default paths: src/boys_
# coefficients.hpp and tests/data/boys_reference.csv — spelled out here and
# in CI). The generator is deterministic (mpmath pinned in
# requirements-boys.txt), so --check over the committed files is the
# identity proof; CI additionally runs the no-args in-place regeneration
# plus git diff --exit-code.
if (BOYS_TOP_LEVEL)
    find_package(Python3 COMPONENTS Interpreter)
    if (Python3_FOUND)
        add_custom_target(boys-generator-check
            COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_boys_coefficients.py --check
                --header ${CMAKE_CURRENT_SOURCE_DIR}/src/boys_coefficients.hpp
                --reference ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/boys_reference.csv
            COMMENT "Boys table/reference byte-identity check (needs mpmath)"
            VERBATIM)
    endif()
endif()

# --- Docs (zero-warning Doxygen gate, top-level only) ------------------------
if (BOYS_TOP_LEVEL)
    find_package(Doxygen QUIET)
    if (DOXYGEN_FOUND)
        add_custom_target(boys-docs
            COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Generate boys API documentation (HTML, zero warnings)")
    endif()
endif()
