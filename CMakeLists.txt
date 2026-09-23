cmake_minimum_required(VERSION 3.25)

# boys — self-contained Boys-function kernel.
#
# Layout: public headers under include/boys/, sources under src/,
# tests under tests/ (gtest, vendored), benchmark drivers under
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
# gtest/benchmark tree is a warning — not a FATAL — when the feature
# that needs it is OFF. See README.md, "Consuming as a submodule".

project(boys VERSION 2.0.0 LANGUAGES CXX)

if (CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(BOYS_TOP_LEVEL ON)
else()
    set(BOYS_TOP_LEVEL OFF)
endif()

option(BOYS_BUILD_TESTS
    "Build the boys test suite (needs the vendored googletest)"
    ${BOYS_TOP_LEVEL})
option(BUILD_BENCHMARKS
    "Build the CPU benchmark drivers; the timed runs themselves are local-only"
    ${BOYS_TOP_LEVEL})
option(BUILD_CUDA
    "Build the CUDA lane (needs the CUDA toolkit; a documented local gate — never CI)"
    OFF)

# Warnings in this tree's own code are build failures. ON by default: a gate
# that is off until someone opts in is not a gate, and a consumer embedding
# this tree inherits the same standard. Turning it off keeps the warning
# levels — it drops only the promotion (/WX, -Werror), so a consumer with a
# compiler whose warning noise this tree has not been made clean for loses
# the failure, never the diagnosis. Third-party is never gated (below).
option(BOYS_WERROR "Treat warnings in boys' own code as errors" ON)

# The gate, applied BY TARGET and never by directory. add_compile_options()
# is directory-scoped and CMake applies a directory's options to every
# subdirectory added after it, so a bare top-level call would also land on
# the pinned GoogleTest and Google Benchmark trees configured below and turn
# someone else's warning noise into a failed build here — the exact
# inversion this gate exists to avoid. Third-party code is not this
# project's to gate, and gtest/gtest_main/benchmark* keep the flags their
# own build gives them.
#
# The COMPILE_LANGUAGE:CXX wrapper keeps the set on the host C++ compiler:
# a target that also carries CUDA sources (boys-cuda and its drivers, OFF by
# default) hands its .cu to nvcc, which does not speak /W4. Its C++
# translation units are gated like any other; its device ones are not.
#
# Every first-party target in this file is named boys* and every one of them
# is handed to this function; the guard at the end of the file makes a
# missed one a configure error rather than a silent hole in the gate.
function(boys_strict_warnings target)
    if (MSVC)
        set(flags /W4)
        if (BOYS_WERROR)
            list(APPEND flags /WX)
        endif()
    else()
        set(flags -Wall -Wextra)
        if (BOYS_WERROR)
            list(APPEND flags -Werror)
        endif()
    endif()
    foreach (flag IN LISTS flags)
        target_compile_options(${target} PRIVATE
            "$<$<COMPILE_LANGUAGE:CXX>:${flag}>")
    endforeach()
    set_property(DIRECTORY APPEND PROPERTY BOYS_STRICT_TARGETS ${target})
endfunction()
set_property(DIRECTORY PROPERTY BOYS_STRICT_TARGETS "")

if (BUILD_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 75 86 CACHE STRING
            "CUDA architectures for the Boys kernels" FORCE)
    endif()
endif()

# Vendored dependencies, committed in full — no FetchContent, no network
# at configure time (googletest v1.17.0, benchmark v1.9.5). A missing tree
# is a FATAL only when the feature that needs it is ON.
if (BOYS_BUILD_TESTS)
    if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/googletest/CMakeLists.txt")
        message(FATAL_ERROR "third_party/googletest (v1.17.0, vendored) is "
            "missing from this source tree.")
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
    src/boys_half_native.cpp
    src/boys_simd.cpp
    src/boys_transform.cpp)
add_library(boys::boys ALIAS boys)
target_include_directories(boys PUBLIC include PRIVATE src)
target_compile_features(boys PUBLIC cxx_std_20)
target_compile_definitions(boys PUBLIC BoysFp16=1)
boys_strict_warnings(boys)

# --- Architecture gate (the AVX2 tier is x86_64-only) ------------------------
# The probe is a configure-time compiler check rather than a lookup on
# CMAKE_SYSTEM_PROCESSOR: the compiler is the thing that has to accept
# -mavx2 / /arch:AVX2, so it is the thing to ask. One decision, and
# src/boys_simd.cpp sees it as BOYS_SIMD_X86, so the flags and the guard in
# that TU cannot drift apart.
#
# On a non-x86_64 target the SIMD TU still compiles: the intrinsics and the
# flag set are gated out and the region kernels are defined against the
# certified scalar lanes instead (see the guard in src/boys_simd.cpp).
# BoysAvx2Available() reports false there.
include(CheckCXXSourceCompiles)
check_cxx_source_compiles("
#if !defined(__x86_64__) && !defined(_M_X64)
#  error not an x86_64 target
#endif
int main() { return 0; }
" BOYS_ARCH_X86_64)

if (BOYS_ARCH_X86_64)
    target_compile_definitions(boys PRIVATE BOYS_SIMD_X86=1)
    if (MSVC)
        # AVX2 intrinsics live in the SIMD TU only (F16C comes with /arch:AVX2);
        # the entry points are runtime-checked via BoysAvx2Available(), which
        # requires the FMA feature bit because the kernels are FMA chains.
        set_source_files_properties(src/boys_simd.cpp PROPERTIES
            COMPILE_OPTIONS "/arch:AVX2")
    else()
        # GCC does not imply F16C via -mavx2.
        set_source_files_properties(src/boys_simd.cpp PROPERTIES
            COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
    endif()
else()
    target_compile_definitions(boys PRIVATE BOYS_SIMD_X86=0)
    message(STATUS "boys: non-x86_64 target — the AVX2 tier is not built; the "
        "region kernels fall back to the certified scalar lanes and "
        "BoysAvx2Available() reports false.")
endif()

# --- Tests (gtest; the reference grid is located via BoysDataDir) -----------
if (BOYS_BUILD_TESTS)
    include(CTest)

    add_executable(boys-tests
        tests/boys_test.cpp
        tests/boys_boundary_test.cpp
        tests/boys_accuracy_test.cpp
        tests/boys_tier_test.cpp
        tests/boys_fixed_n_test.cpp
        tests/boys_all_n_test.cpp
        tests/boys_c_test.cpp
        tests/f16_test.cpp
        tests/boys_half_native_test.cpp
        tests/boys_transform_test.cpp)
    target_link_libraries(boys-tests PRIVATE boys gtest gtest_main)
    target_include_directories(boys-tests PRIVATE src)
    target_compile_features(boys-tests PRIVATE cxx_std_23)
    # BoysTierReference is the accuracy gate lane's 80-digit grid when the
    # tree carries it; the tier tests fall back to BoysDataDir's grid at run
    # time and print which reference they measured against either way.
    target_compile_definitions(boys-tests PRIVATE
        BoysDataDir="${CMAKE_CURRENT_SOURCE_DIR}/tests/data"
        BoysTierReference="${CMAKE_CURRENT_SOURCE_DIR}/tests/data/boys_accuracy_gate_reference.csv")
    boys_strict_warnings(boys-tests)

    # The ACCURACY GATE (tests/boys_accuracy_gate.cpp) measures every
    # documented per-lane, per-region bound against the independent
    # high-precision reference grid (tests/data/boys_accuracy_gate_reference.csv,
    # regenerated by tools/gen_boys_accuracy_gate_reference.py) over a dense
    # sweep that runs past the suite's grid, and reports each claim as a ratio
    # of delivered error to documented bound with the location of the maximum.
    # It exits non-zero on any exceeded claim; passes that hold only because
    # the returned value is below what the format can hold are counted apart.
    #
    # The gate prints the revision it measured, read from the tree at configure
    # time, so "verified at this revision" names one: a result pasted into a
    # report months later would otherwise be attributable to nothing.
    find_package(Git QUIET)
    if(GIT_FOUND)
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" rev-parse --short HEAD
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            OUTPUT_VARIABLE BOYS_GATE_REVISION
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET)
    endif()
    if(NOT BOYS_GATE_REVISION)
        set(BOYS_GATE_REVISION "unknown")
    endif()
    add_executable(boys-accuracy-gate tests/boys_accuracy_gate.cpp)
    target_link_libraries(boys-accuracy-gate PRIVATE boys)
    target_include_directories(boys-accuracy-gate PRIVATE src)
    target_compile_features(boys-accuracy-gate PRIVATE cxx_std_23)
    target_compile_definitions(boys-accuracy-gate PRIVATE
        BoysDataDir="${CMAKE_CURRENT_SOURCE_DIR}/tests/data"
        BoysGateRevision="${BOYS_GATE_REVISION}")
    boys_strict_warnings(boys-accuracy-gate)

    # The run-time accuracy tier (BoysAllOrdersAtTier / QueryTier) arrives on a
    # branch of its own, so the gate compiles its tier claims only where the
    # headers this revision ships declare those entries. Asked of the compiler,
    # not of a version string: a revision without the tier reports those claims
    # as evidence-absent instead of failing to build, and a revision that has
    # them cannot slip through unmeasured.
    #
    # The probe is a STATIC_LIBRARY try-compile, not an executable one, and
    # that is the whole question: it compiles the entries' declarations and does
    # not link them, because a probe that links tests whether THIS build links
    # the library's definitions, and answers no for a revision that merely keeps
    # the tier's code in a translation unit the probe does not compile. The
    # definitions are tested where they belong --- the gate target links the
    # library, so a declaration without a definition fails that link loudly
    # instead of quietly dropping three claims into evidence-absent.
    #
    # The result is unset first so the probe runs on every configure: a cached
    # no, left behind by an earlier revision, would be the silent direction.
    include(CheckCXXSourceCompiles)
    unset(BOYS_GATE_HAS_TIER CACHE)
    set(_boys_probe_target_type "${CMAKE_TRY_COMPILE_TARGET_TYPE}")
    set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
    set(CMAKE_REQUIRED_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/include")
    if (MSVC)
        set(CMAKE_REQUIRED_FLAGS "/std:c++20 /EHsc")
    else()
        set(CMAKE_REQUIRED_FLAGS "-std=c++20")
    endif()
    check_cxx_source_compiles("
#include <boys/boys.hpp>
int main() {
    double out[33];
    boys::BoysAllOrdersAtTier(boys::AccuracyTier::kRelaxed64, 32, 30.0, out);
    const boys::TierCoverage c = boys::QueryTier(
        boys::AccuracyTier::kRelaxed64, boys::AccuracyRegion::kC, 1e-12);
    return (c.meets || c.reachable > 0.0) ? 0 : 1;
}
" BOYS_GATE_HAS_TIER)
    set(CMAKE_TRY_COMPILE_TARGET_TYPE "${_boys_probe_target_type}")
    unset(_boys_probe_target_type)
    unset(CMAKE_REQUIRED_FLAGS)
    unset(CMAKE_REQUIRED_INCLUDES)

    if (BOYS_GATE_HAS_TIER)
        target_compile_definitions(boys-accuracy-gate PRIVATE BOYS_GATE_TIER=1)
        message(STATUS "boys: the accuracy gate will measure the run-time accuracy tier")
    else()
        message(STATUS "boys: this revision has no run-time accuracy tier; the gate's tier "
            "claims are reported as evidence-absent")
    endif()

    # The BOUNDARY-CHECK standalone (tests/boys_boundary_standalone.cpp)
    # reproduces the measured recursion-boundary cells from the
    # committed reference grid. It is deliberately self-contained (standard
    # library only — the CI legs compile it standalone with plain -O2) and
    # reads boys_reference.csv from its working directory, so ctest runs it
    # with the working directory set to tests/data.
    add_executable(boys-boundary-standalone tests/boys_boundary_standalone.cpp)
    boys_strict_warnings(boys-boundary-standalone)

    add_test(NAME boys-tests COMMAND boys-tests)
    add_test(NAME boys-boundary-standalone COMMAND boys-boundary-standalone)
    # --strict: the gate fails on a claim whose evidence is not in the tree as
    # well as on one it measures exceeded, so a green ctest never means "every
    # documented claim is met at this revision" while a claim rest on a
    # measurement this revision cannot re-run.
    add_test(NAME boys-accuracy-gate COMMAND boys-accuracy-gate --strict)
    set_tests_properties(boys-boundary-standalone PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/tests/data")

    # The dispatch assertion: each CI leg states what BoysAvx2Available() MUST
    # report for its architecture (1 on x86_64, 0 on arm64) and the probe fails
    # the leg if it disagrees. It is worth more than the rest of the matrix
    # combined, because the SIMD correctness tests soft-skip on that same
    # predicate: a wrong CPUID bit would make every one of them SKIP while CI
    # reported green with the whole vector tier dead — the failure this tree
    # already suffered once.
    #
    # It is a *target that links boys*, not a hand-compiled snippet, so it
    # inherits the configuration's CRT (MSVC Debug = /MDd), its sanitizer flags
    # and its architecture from one source. Each leg supplies the expectation
    # as an option at configure time; leaving it empty (every developer build)
    # builds no probe and adds no test.
    set(BOYS_EXPECT_AVX2 "" CACHE STRING
        "CI dispatch assertion: 1 on an x86_64 leg, 0 on an arm64 leg; empty = not built")
    if (NOT BOYS_EXPECT_AVX2 STREQUAL "")
        add_executable(boys-avx2-probe tests/boys_avx2_probe.cpp)
        target_link_libraries(boys-avx2-probe PRIVATE boys)
        target_compile_definitions(boys-avx2-probe PRIVATE
            BOYS_EXPECT_AVX2=${BOYS_EXPECT_AVX2})
        boys_strict_warnings(boys-avx2-probe)
        add_test(NAME boys-avx2-probe COMMAND boys-avx2-probe)
    endif()
endif()

# --- CUDA lane (optional; needs the toolkit) --------------------------------
if (BUILD_CUDA)
    # The host wrapper (boys_cuda.cpp) and the kernels (boys_cuda.cu) are one
    # target: the .cpp takes the library target's own minimum standard, the .cu
    # is pinned at C++20 with a CUDA-safe include list (see the .cu preamble).
    # Entry points take raw device pointers + a stream and report through
    # boys::BoysStatus.
    add_library(boys-cuda
        src/boys_cuda.cpp
        src/boys_cuda.cu)
    target_include_directories(boys-cuda PUBLIC include PRIVATE src)
    target_compile_definitions(boys-cuda PUBLIC BoysFp16=1)
    # Stated because this target links nothing that carries a standard into it:
    # without it the host TU compiles at the compiler's default (C++14 on
    # MSVC) and the library's headers do not build.
    target_compile_features(boys-cuda PRIVATE cxx_std_20)
    boys_strict_warnings(boys-cuda)

    if (BOYS_BUILD_TESTS)
        add_executable(boys-cuda-tests tests/boys_cuda_test.cpp)
        target_link_libraries(boys-cuda-tests PRIVATE boys-cuda boys
            gtest gtest_main)
        target_include_directories(boys-cuda-tests PRIVATE src)
        boys_strict_warnings(boys-cuda-tests)
        add_test(NAME boys-cuda-tests COMMAND boys-cuda-tests)
    endif()
endif()

# --- Benchmark drivers -------------------------------------------------------
if (BUILD_BENCHMARKS)
    # boys_benchmark.cpp is a Google Benchmark driver; boys_unsorted_simd_
    # benchmark.cpp is a runs-log protocol driver with its own main(). The
    # timed runs are local-only by design (machine/clock/commit recorded per
    # run) and never part of CI.
    if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/benchmark/CMakeLists.txt")
        message(FATAL_ERROR "BUILD_BENCHMARKS=ON needs third_party/benchmark "
            "(v1.9.5, vendored), which is missing from this source tree.")
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
    boys_strict_warnings(boys-benchmark)

    # The batch entry's own driver: it reports the entry against the
    # per-argument loop and separates the sort's cost from the win, with
    # ratios as the result and a protocol that survives a busy machine (see
    # the file preamble). Portable: it calls no intrinsics of its own.
    add_executable(boys-all-n-benchmark benchmarks/boys_all_n_benchmark.cpp)
    target_link_libraries(boys-all-n-benchmark PRIVATE boys)
    target_compile_definitions(boys-all-n-benchmark PRIVATE BoysFp16=1)
    target_compile_features(boys-all-n-benchmark PRIVATE cxx_std_23)
    boys_strict_warnings(boys-all-n-benchmark)

    # The AVX2 timing harness is x86_64-only, like the tier it measures. It is
    # a local-only target (never a CI leg), so on a non-x86_64 host it is
    # reported and skipped rather than half-built: the measured numbers would
    # be the scalar lanes' under a SIMD name.
    if (BOYS_ARCH_X86_64)
        add_executable(boys-unsorted-simd-benchmark
            benchmarks/boys_unsorted_simd_benchmark.cpp)
        target_link_libraries(boys-unsorted-simd-benchmark PRIVATE boys)
        target_include_directories(boys-unsorted-simd-benchmark PRIVATE src)
        target_compile_definitions(boys-unsorted-simd-benchmark PRIVATE BoysFp16=1)
        target_compile_features(boys-unsorted-simd-benchmark PRIVATE cxx_std_23)
        boys_strict_warnings(boys-unsorted-simd-benchmark)
        if (MSVC)
            # The mixed per-vector kernel compiles AVX2 intrinsics in its own TU.
            set_source_files_properties(benchmarks/boys_unsorted_simd_benchmark.cpp
                PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
        else()
            set_source_files_properties(benchmarks/boys_unsorted_simd_benchmark.cpp
                PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
        endif()
    else()
        message(STATUS "boys: skipping boys-unsorted-simd-benchmark — its AVX2 "
            "kernels are x86_64-only (this is a local-only target)")
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
        boys_strict_warnings(boys-cuda-benchmark)
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

# --- Gate: no first-party target can silently miss the strict set ------------
# Structural, not a reading of this file. BUILDSYSTEM_TARGETS lists the
# targets this directory created; every first-party target that compiles
# anything is named boys* (the third-party subdirectories own gtest,
# gtest_main, gmock and benchmark* — names the prefix cannot match; the
# custom targets of this tree, boys-generator-check and boys-docs, compile
# nothing and are skipped by type), and each of those must have been through
# boys_strict_warnings. A target added later that forgets the call fails the
# configure here — naming every offender at once, rather than the first —
# instead of quietly compiling unguarded.
get_property(boysStrictTargets DIRECTORY PROPERTY BOYS_STRICT_TARGETS)
get_property(boysAllTargets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
set(boysUngatedTargets "")
foreach (target IN LISTS boysAllTargets)
    if (NOT target MATCHES "^boys")
        continue()
    endif()
    get_target_property(targetType ${target} TYPE)
    if (targetType STREQUAL "UTILITY")
        continue()
    endif()
    if (NOT target IN_LIST boysStrictTargets)
        list(APPEND boysUngatedTargets ${target})
    endif()
endforeach()
if (boysUngatedTargets)
    message(FATAL_ERROR "boys: first-party targets without the strict warning "
        "set: ${boysUngatedTargets}. Call boys_strict_warnings(<target>) right "
        "after each one is created (see BOYS_WERROR).")
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
