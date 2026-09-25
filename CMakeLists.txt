cmake_minimum_required(VERSION 3.25)

# boys — self-contained Boys-function kernel.
#
# Layout: public headers under include/boys/, sources under src/,
# tests under tests/ (gtest, vendored), benchmark drivers under
# benchmarks/, the regeneration script under tools/. The templated entries'
# kernel ships in the headers, so a consumer instantiates whichever accuracy
# multiplier it names; the library itself exports the default m = 1
# instantiations and the sampled rungs (extern-template architecture, see
# include/boys/boys.hpp and src/boys.cpp). The CUDA lane is an optional second
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

# The header-only consumer check proves that the entries' template definitions
# are reachable from the public headers, and it proves it by NOT linking the
# library: with a library on its link line it would still compile, still run,
# and prove nothing. Assert the absence here, so a later edit that links it
# fails the configure on every leg instead of quietly retiring the check.
function(boys_assert_no_linked_library target)
    get_target_property(linked ${target} LINK_LIBRARIES)
    if (linked)
        message(FATAL_ERROR "boys: ${target} must link nothing - what it proves is that "
            "the public headers alone suffice - but it links: ${linked}. Remove the link, "
            "or the check stops proving what it exists to prove.")
    endif()
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
    # The toolkit and the target architectures, as text, for the device probe's
    # report. A device figure whose toolkit and target architecture are unstated
    # cannot be reproduced, and both are configure-time facts that the compiled
    # binary cannot otherwise recover: nvcc's own version is not a runtime query,
    # and the architectures are what this build was told to emit, not what the
    # card happens to be. Spelled out as strings here so the probe's report and
    # its text never have to guess.
    string(REPLACE ";" "," BOYS_CUDA_ARCHITECTURES_TEXT
        "${CMAKE_CUDA_ARCHITECTURES}")
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
    src/boys_orders_simd.cpp
    src/boys_probe.cpp
    src/boys_simd.cpp
    src/boys_transform.cpp)
add_library(boys::boys ALIAS boys)
target_include_directories(boys PUBLIC include PRIVATE src)
target_compile_features(boys PUBLIC cxx_std_20)
target_compile_definitions(boys PUBLIC BoysFp16=1)
boys_strict_warnings(boys)

# --- The multiply-add route -------------------------------------------------
# The scalar lanes' fused multiply-add is an instruction where the target has
# one and a call into the C runtime where it does not, and a Clenshaw
# evaluation costs deg + 1 fused steps — so on a plain x86-64 build that leaves
# the fused route in force, every value the table regions return pays that many
# calls, 19 to 25 of them for the double lane and 11 for the float lane. The
# packed lanes are unaffected: src/boys_simd.cpp is compiled with the fused
# instruction regardless of the global flags. OFF is the certified arithmetic:
# the published bounds are the fused route's, and the fused route is what every
# release, every CI leg and the accuracy gate are judged at.
#
# ON selects the separate route instead: the product rounds and then the sum
# rounds, two roundings rather than one, written so that no target pays a call
# for it. It is a different arithmetic with its own delivered figures, which
# the accuracy gate prints per lane; docs/lane-contract.md carries them beside
# the fused route's. The definition is PUBLIC because every entry is
# header-defined: a consumer that instantiates a kernel in its own translation
# unit compiles the same arithmetic the library reports, and a consumer that
# does not gets it from the library's own instantiations.
option(BOYS_MULADD_SEPARATE
    "Build the scalar multiply-add at the two-rounding route, which costs no \
call on a target without the fused instruction"
    OFF)

if (BOYS_MULADD_SEPARATE)
    target_compile_definitions(boys PUBLIC BOYS_MULADD_SEPARATE=1)
    message(STATUS "boys: scalar multiply-add route = separate (two roundings, "
        "no call); the published bounds are the fused route's — see "
        "docs/lane-contract.md for this route's measured figures.")
else()
    message(STATUS "boys: scalar multiply-add route = fused (one rounding)")
endif()

# --- What this target does with a bare product-plus-add ----------------------
# Whether `a * b + c` written bare rounds once or twice follows from the target
# and the flags rather than from the source, and it is not a detail: it decides
# whether the multiply-add route above is one arithmetic or two.
# aarch64 has the fused instruction in its baseline, x86-64 does not unless
# -mfma is passed, MSVC never contracts under its default /fp:precise, and gcc
# and clang contract by default — so the same source is a different arithmetic
# on two machines, and a bound is a statement about one of them.
#
# Measured rather than inferred from the architecture name. The probe evaluates
# the bare step the library's own run-time measurement uses —
# (1 + 2^-27)^2 - (1 + 2^-26), which is 2^-54 when the step is fused and 0 when
# it is two roundings — and asks whether the bare form reached the fused value.
# Both readings are volatile so the expression is evaluated rather than folded,
# and the addend is chosen so the two cannot coincide: a probe whose own two
# references agree has measured nothing, and reports the answer that claims
# least rather than the one it cannot see. The claims that answer decides are
# checked against the run-time measurement in the consumer check, so a configure
# that got it wrong is a failing test rather than a quiet one.
#
# It carries the build's own flags, the configuration's included.
# -ffp-contract is a licence rather than an instruction and gcc acts on it from
# -O2 up, so a probe compiled at -O1 would report no contraction for a Release
# build that contracts — a wrong answer in the direction that hides the
# difference. It compiles and runs for the same reason: what the target's own
# arithmetic does is not a question a compile-only check can answer, and a
# cross-compiling configure cannot answer it at all.
#
# One decision, read as BOYS_SCALAR_CONTRACTS. The definition is PUBLIC for the
# reason BoysFp16 is: every scalar entry is header-defined, so a consumer that
# instantiates a kernel in its own translation unit compiles its own arithmetic,
# and the claims that arithmetic carries are then compiled against the answer
# this configure measured rather than against an assumption about the host. It
# is left undefined where no answer was measured, so a build that has not
# measured contraction is a build that claims the least.
if (NOT CMAKE_CROSSCOMPILING)
include(CheckCXXSourceRuns)
unset(BOYS_SCALAR_CONTRACTS CACHE)
set(_boys_contract_flags "${CMAKE_CXX_FLAGS} -std=c++20")
if (CMAKE_BUILD_TYPE)
    string(TOUPPER "${CMAKE_BUILD_TYPE}" _boys_contract_config)
    string(APPEND _boys_contract_flags " ${CMAKE_CXX_FLAGS_${_boys_contract_config}}")
    unset(_boys_contract_config)
endif()
set(_boys_contract_saved_flags "${CMAKE_REQUIRED_FLAGS}")
set(CMAKE_REQUIRED_FLAGS "${_boys_contract_flags}")
check_cxx_source_runs("
#include <cmath>

int main() {
    volatile double a = 1.0 + 0x1p-27;
    volatile double b = 1.0 + 0x1p-27;
    volatile double c = -(1.0 + 0x1p-26);
    volatile double product = a * b;
    const double bare = a * b + c;
    const double fused = std::fma(a, b, c);
    const double separate = product + c;

    if (fused == separate) {
        return 1;
    }

    return bare == fused ? 0 : 1;
}
" BOYS_SCALAR_CONTRACTS)
set(CMAKE_REQUIRED_FLAGS "${_boys_contract_saved_flags}")
unset(_boys_contract_saved_flags)
unset(_boys_contract_flags)
endif()

# 1 and 0 are the two measured answers and both are defined, so a reader asks the
# value rather than whether the name exists. Neither is defined where no answer
# was measured: a build that has not measured contraction does not get to assert
# the claims the answer decides, and the tri-state is what keeps it from
# inheriting the one that would be wrong rather than the one that claims least.
if (BOYS_SCALAR_CONTRACTS)
    target_compile_definitions(boys PUBLIC BOYS_SCALAR_CONTRACTS=1)
    message(STATUS "boys: this target contracts a bare product-plus-add (one rounding) — "
        "the scalar lanes' two multiply-add routes are one arithmetic here, and the "
        "claims a build carries are compiled against that answer.")
elseif (NOT CMAKE_CROSSCOMPILING)
    target_compile_definitions(boys PUBLIC BOYS_SCALAR_CONTRACTS=0)
    message(STATUS "boys: this target does not contract a bare product-plus-add (two "
        "roundings) — the scalar lanes' two routes are two arithmetics here, and the "
        "claims a build carries are compiled against that answer.")
else()
    message(STATUS "boys: contraction is not measured in a cross-compiling configure — "
        "no claims are compiled against it and BoysBackends() reports it at run time.")
endif()

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
        set_source_files_properties(src/boys_simd.cpp src/boys_orders_simd.cpp PROPERTIES
            COMPILE_OPTIONS "/arch:AVX2")
    else()
        # GCC does not imply F16C via -mavx2.
        set_source_files_properties(src/boys_simd.cpp src/boys_orders_simd.cpp PROPERTIES
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
        tests/boys_across_orders_test.cpp
        tests/boys_c_test.cpp
        tests/f16_test.cpp
        tests/boys_half_native_test.cpp
        tests/boys_backend_test.cpp
        tests/boys_probe_test.cpp
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

    # The route-refusal probe. The many-argument and fixed-order entries do not
    # carry the rational minimax route: their region bodies evaluate the shipped
    # seed and the shipped per-order fits directly, so the route would be
    # accepted and the shipped values returned under its name. They refuse it
    # where the call is named, which is the one kind of property this gate
    # cannot measure by calling - the call it would make does not build. It is
    # measured by making the call in a probe instead, so "refused" is a result
    # the gate reports rather than a sentence in a header, and a revision that
    # stopped refusing fails the row that reports it.
    #
    # STATIC_LIBRARY again, for the same reason as the tier probe: this asks
    # whether the declarations accept the policy, not whether this build links
    # the definitions.
    unset(BOYS_GATE_BATCH_ACCEPTS_ROUTE CACHE)
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
    double out[33] = {};
    const double x = 1.0;
    boys::BoysAllN<1.0, boys::EvalPolicy<boys::FitRoute::kRationalMinimax>>(1, &x, out, 1);
    boys::BoysAllN<1.0, boys::EvalPolicy<boys::FitRoute::kRationalMinimax>>(
        1, &x, out, 1, boys::BoysSortedArgs{});
    boys::BoysFixedN<1.0, boys::EvalPolicy<boys::FitRoute::kRationalMinimax>>(0, &x, out, 1);
    return 0;
}
" BOYS_GATE_BATCH_ACCEPTS_ROUTE)
    set(CMAKE_TRY_COMPILE_TARGET_TYPE "${_boys_probe_target_type}")
    unset(_boys_probe_target_type)
    unset(CMAKE_REQUIRED_FLAGS)
    unset(CMAKE_REQUIRED_INCLUDES)

    if (NOT BOYS_GATE_BATCH_ACCEPTS_ROUTE)
        target_compile_definitions(boys-accuracy-gate PRIVATE BOYS_GATE_BATCH_REFUSES_ROUTE=1)
        message(STATUS "boys: the batch and fixed-order entries refuse a route they do not carry")
    else()
        message(STATUS "boys: a route the batch or fixed-order entry does not carry is ACCEPTED")
    endif()

    # The orders axis's two limits, each measured the same way: by compiling the
    # call and finding that it does not build. Both are properties of a call
    # shape rather than of a value, so no run can show them - what a run would
    # show is that the library answered, which is the failure mode.
    #
    #   the fixed-order entry  a packed lane keeps four ORDERS of one argument,
    #                          and a fixed-order call produces exactly one order
    #                          at every argument: there are not four orders to
    #                          fill a lane with, so the axis cannot be formed
    #                          there at all.
    #   a relaxed multiplier   the packed orders lane evaluates every stored fit
    #                          at its full degree and reads no effective-degree
    #                          table, so it carries no rung.
    #
    # One probe per limit, and deliberately not one probe for both: a single
    # call that fails to build would report the same thing whether one limit
    # held or the other did, so a revision that lifted one of them would keep
    # the other's row green for the wrong reason.
    unset(BOYS_GATE_FIXEDN_ACCEPTS_ORDERS CACHE)
    unset(BOYS_GATE_ORDERS_RUNG_ACCEPTED CACHE)
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
    double out[33] = {};
    const double x = 1.0;
    using OrdersAxis = boys::EvalPolicy<boys::FitRoute::kChebyshev, boys::EvalScheme::kSplitClenshaw,
                                        boys::BoysBudget::kFloat, boys::PackAxis::kOrders>;
    boys::BoysFixedN<1.0, OrdersAxis>(0, &x, out, 1);
    return 0;
}
" BOYS_GATE_FIXEDN_ACCEPTS_ORDERS)
    check_cxx_source_compiles("
#include <boys/boys.hpp>
int main() {
    double out[33] = {};
    const double x = 1.0;
    using OrdersAxis = boys::EvalPolicy<boys::FitRoute::kChebyshev, boys::EvalScheme::kSplitClenshaw,
                                        boys::BoysBudget::kFloat, boys::PackAxis::kOrders>;
    boys::BoysAllN<64.0, OrdersAxis>(1, &x, out, 1);
    return 0;
}
" BOYS_GATE_ORDERS_RUNG_ACCEPTED)
    set(CMAKE_TRY_COMPILE_TARGET_TYPE "${_boys_probe_target_type}")
    unset(_boys_probe_target_type)
    unset(CMAKE_REQUIRED_FLAGS)
    unset(CMAKE_REQUIRED_INCLUDES)

    if (NOT BOYS_GATE_FIXEDN_ACCEPTS_ORDERS)
        target_compile_definitions(boys-accuracy-gate PRIVATE BOYS_GATE_FIXEDN_REFUSES_ORDERS=1)
    endif()

    if (NOT BOYS_GATE_ORDERS_RUNG_ACCEPTED)
        target_compile_definitions(boys-accuracy-gate PRIVATE BOYS_GATE_ORDERS_REFUSES_RUNG=1)
    endif()

    if (BOYS_GATE_FIXEDN_ACCEPTS_ORDERS)
        message(STATUS "boys: the orders axis on BoysFixedN is ACCEPTED by the probe — a call "
            "shape with no orders to pack takes an orders-axis policy")
    else()
        message(STATUS "boys: the orders axis on BoysFixedN is REFUSED, measured by compiling "
            "the call")
    endif()

    if (BOYS_GATE_ORDERS_RUNG_ACCEPTED)
        message(STATUS "boys: the orders axis at a relaxed multiplier is ACCEPTED by the probe")
    else()
        message(STATUS "boys: the orders axis at a relaxed multiplier is REFUSED, measured by "
            "compiling the call")
    endif()

    if (BOYS_GATE_RATIONAL_RUNG_ACCEPTED)
        message(STATUS "boys: the rational route at a relaxed rung is ACCEPTED by the probe")
    else()
        message(STATUS "boys: the rational route at a relaxed rung is REFUSED, measured by "
            "compiling the call")
    endif()

    if (BOYS_GATE_F32_PAIR_ACCEPTED)
        message(STATUS "boys: the single-precision engines accept another route or scheme")
    else()
        message(STATUS "boys: the single-precision engines REFUSE another route or scheme, "
            "measured by compiling the call")
    endif()

    if (NOT BOYS_GATE_ORDERS_ROUTE_ACCEPTED)
        target_compile_definitions(boys-accuracy-gate PRIVATE BOYS_GATE_ORDERS_REFUSES_ROUTE=1)
        message(STATUS "boys: the orders axis with a route other than the shipped one is REFUSED, "
            "measured by compiling the call")
    else()
        message(STATUS "boys: the orders axis ACCEPTS another route")
    endif()

    # The other two refusals this gate's record carries, each measured by
    # compiling the call it names rather than by quoting the assertion that
    # states it. A refusal the library asserts and cannot show is the same
    # sentence the record forbids: citing our own refusal as the reason a bound
    # is not owed. Both of these are refusals the gate reports as owed work, so
    # what the probe establishes is narrower than for the two above - it says
    # the call does not build, not that it cannot.
    unset(BOYS_GATE_RATIONAL_RUNG_ACCEPTED CACHE)
    unset(BOYS_GATE_F32_PAIR_ACCEPTED CACHE)
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
    double out[33] = {};
    boys::BoysAllOrders<64.0, boys::EvalPolicy<boys::FitRoute::kRationalMinimax>>(1, 1.0, out);
    return 0;
}
" BOYS_GATE_RATIONAL_RUNG_ACCEPTED)
    check_cxx_source_compiles("
#include <boys/boys.hpp>
int main() {
    float out[33] = {};
    boys::BoysAllOrdersF32<1.0, boys::EvalPolicy<boys::FitRoute::kRationalMinimax>>(1, 1.0f, out);
    boys::BoysAllOrdersF32<1.0,
                           boys::EvalPolicy<boys::FitRoute::kChebyshev, boys::EvalScheme::kHorner>>(
        1, 1.0f, out);
    return 0;
}
" BOYS_GATE_F32_PAIR_ACCEPTED)
    unset(BOYS_GATE_ORDERS_ROUTE_ACCEPTED CACHE)
    check_cxx_source_compiles("
#include <boys/boys.hpp>
int main() {
    double out[33] = {};
    using OrdersRational = boys::EvalPolicy<boys::FitRoute::kRationalMinimax,
                                            boys::EvalScheme::kSplitClenshaw,
                                            boys::BoysBudget::kFloat,
                                            boys::PackAxis::kOrders>;
    boys::BoysAllOrders<1.0, OrdersRational>(1, 1.0, out);
    return 0;
}
" BOYS_GATE_ORDERS_ROUTE_ACCEPTED)
    set(CMAKE_TRY_COMPILE_TARGET_TYPE "${_boys_probe_target_type}")
    unset(_boys_probe_target_type)
    unset(CMAKE_REQUIRED_FLAGS)
    unset(CMAKE_REQUIRED_INCLUDES)

    if (NOT BOYS_GATE_RATIONAL_RUNG_ACCEPTED)
        target_compile_definitions(boys-accuracy-gate PRIVATE BOYS_GATE_REFUSES_RATIONAL_RUNG=1)
    endif()

    if (NOT BOYS_GATE_F32_PAIR_ACCEPTED)
        target_compile_definitions(boys-accuracy-gate PRIVATE BOYS_GATE_REFUSES_F32_PAIR=1)
    endif()

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

    # The CONSUMER CHECKS (tests/consumer_umbrella.cpp and
    # tests/consumer_header_only.cpp). The suite above includes headers from
    # src/ and links the library, so it cannot see the two defects a consumer
    # meets first: a public header the umbrella does not reach, and an entry
    # whose definition never left a .cpp file (a link error for every
    # multiplier the library does not pre-instantiate). These two programs are
    # consumers, and each is built so that its own defect has somewhere to
    # show:
    #
    #   boys-consumer-umbrella  includes ONLY <boys/boys.hpp>, first include,
    #     no internal header, and links the library. It exercises every entry
    #     the public headers document - all three product modes, both QueryTier
    #     overloads, both many-argument shapes, the fp16/bf16 lanes, the native
    #     half lane - at multipliers the library does not pre-instantiate, and
    #     judges every returned value against the bound its entry states, with
    #     the committed 45-digit grid as the reference.
    #
    #   boys-consumer-header-only  proves the definitions really are reachable
    #     from the headers: it names non-rung multipliers and links NOTHING.
    #     Its meaning is the empty link line, so the target's link libraries
    #     are asserted empty below rather than trusted.
    #
    # Neither target adds src/ to its include path (the library publishes
    # include/ as PUBLIC and src/ as PRIVATE), and both say so: they carry an
    # __has_include probe that fails the compile if the internal directory ever
    # reaches their include path. The include path and the compile definitions
    # of the header-only check are read from the library's own interface - the
    # same ones a consumer inherits from linking boys::boys - so the two cannot
    # drift apart.
    add_executable(boys-consumer-umbrella tests/consumer_umbrella.cpp)
    target_link_libraries(boys-consumer-umbrella PRIVATE boys)
    target_compile_definitions(boys-consumer-umbrella PRIVATE
        BoysConsumerReference="${CMAKE_CURRENT_SOURCE_DIR}/tests/data/boys_reference.csv")
    boys_strict_warnings(boys-consumer-umbrella)

    get_target_property(boysConsumerIncludes boys INTERFACE_INCLUDE_DIRECTORIES)
    get_target_property(boysConsumerDefines boys INTERFACE_COMPILE_DEFINITIONS)
    if (NOT boysConsumerIncludes OR NOT boysConsumerDefines)
        message(FATAL_ERROR "boys: the library publishes its consumer surface through "
            "INTERFACE_INCLUDE_DIRECTORIES and INTERFACE_COMPILE_DEFINITIONS; the "
            "header-only consumer check reads both and found one empty.")
    endif()
    add_executable(boys-consumer-header-only tests/consumer_header_only.cpp)
    target_include_directories(boys-consumer-header-only PRIVATE ${boysConsumerIncludes})
    target_compile_definitions(boys-consumer-header-only PRIVATE ${boysConsumerDefines}
        BoysConsumerReference="${CMAKE_CURRENT_SOURCE_DIR}/tests/data/boys_reference.csv")
    target_compile_features(boys-consumer-header-only PRIVATE cxx_std_20)
    boys_strict_warnings(boys-consumer-header-only)
    boys_assert_no_linked_library(boys-consumer-header-only)

    add_test(NAME boys-tests COMMAND boys-tests)
    add_test(NAME boys-boundary-standalone COMMAND boys-boundary-standalone)
    add_test(NAME boys-consumer-umbrella COMMAND boys-consumer-umbrella)
    add_test(NAME boys-consumer-header-only COMMAND boys-consumer-header-only)
    # --strict: the gate fails on a claim whose evidence is not in the tree as
    # well as on one it measures exceeded, so a green ctest never means "every
    # documented claim is met at this revision" while a claim rest on a
    # measurement this revision cannot re-run.
    add_test(NAME boys-accuracy-gate COMMAND boys-accuracy-gate --strict)
    set_tests_properties(boys-boundary-standalone PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/tests/data")

    # The option probe's driver: measures this build's evaluation options on
    # the machine it is run on and prints what it found, refusal included (see
    # the file preamble). Not a test and not registered as one — its result is
    # a property of the host and the clock, and a leg that failed because the
    # runner was busy would be a leg that failed for the wrong reason. Built
    # with the tests so that a tree which builds them still exercises the
    # probe's compilability; the numbers come from a local run.
    add_executable(boys-option-probe benchmarks/boys_option_probe.cpp)
    target_link_libraries(boys-option-probe PRIVATE boys)
    target_compile_features(boys-option-probe PRIVATE cxx_std_20)
    boys_strict_warnings(boys-option-probe)

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

    # --- Build-fact probe ----------------------------------------------------
    # tests/boys_build_facts.cpp states what this build is: the instruction sets
    # it targets, whether a bare `a * b + c` written in it is a single rounding,
    # whether the library's own objects call an out-of-line fma, and the
    # arithmetic backends the build carries. docs/build-facts.md records the
    # answers per CI leg and says what a consumer should read into them.
    #
    # Every fact is deterministic, which is why the probe takes no clock
    # readings: a duration taken on a shared runner is a distribution across
    # ephemeral machines, not a fact, and none of these facts needs one. The
    # two paths this target is handed are what make the binary fact readable
    # from inside the build - the artifact to inspect, and the tool that reads
    # its symbol table. Both are decided here rather than looked up at run time
    # because the compiler is what knows them: link.exe carries dumpbin's /dump
    # mode on every MSVC install, and a GNU or Clang toolchain names its own nm
    # or objdump. A toolchain that names none leaves both empty, and the probe
    # reports the route as unestablished instead of guessing it.
    set(BOYS_FACTS_CONFIG "${CMAKE_BUILD_TYPE}")
    if (NOT BOYS_FACTS_CONFIG)
        set(BOYS_FACTS_CONFIG "$<CONFIG>")
    endif()
    if (MSVC)
        set(BOYS_FACTS_SYMBOL_TOOL "${CMAKE_LINKER}")
        set(BOYS_FACTS_SYMBOL_MODE "msvc-dump")
    else()
        find_program(BOYS_FACTS_SYMBOL_BINARY NAMES llvm-nm nm objdump)
        if (BOYS_FACTS_SYMBOL_BINARY)
            set(BOYS_FACTS_SYMBOL_TOOL "${BOYS_FACTS_SYMBOL_BINARY}")
            get_filename_component(BOYS_FACTS_SYMBOL_NAME
                "${BOYS_FACTS_SYMBOL_BINARY}" NAME)
            if (BOYS_FACTS_SYMBOL_NAME MATCHES "objdump")
                set(BOYS_FACTS_SYMBOL_MODE "objdump-symbols")
            else()
                set(BOYS_FACTS_SYMBOL_MODE "nm-undefined")
            endif()
        else()
            set(BOYS_FACTS_SYMBOL_TOOL "")
            set(BOYS_FACTS_SYMBOL_MODE "")
        endif()
    endif()

    add_executable(boys-build-facts tests/boys_build_facts.cpp)
    target_link_libraries(boys-build-facts PRIVATE boys)
    target_compile_features(boys-build-facts PRIVATE cxx_std_23)
    target_compile_definitions(boys-build-facts PRIVATE
        BOYS_FACTS_CONFIG="${BOYS_FACTS_CONFIG}"
        BOYS_FACTS_LIB="$<TARGET_FILE:boys>"
        BOYS_FACTS_SYMBOL_TOOL="${BOYS_FACTS_SYMBOL_TOOL}"
        BOYS_FACTS_SYMBOL_MODE="${BOYS_FACTS_SYMBOL_MODE}")
    boys_strict_warnings(boys-build-facts)

    # Registered as a ctest case too, so a plain `ctest` runs it on every leg
    # that runs the suite - the sanitizer leg included, where it is the probe's
    # own file and process handling that is under the sanitizers. It is the
    # recorded-table check (--check) that the legs run as its own step: that one
    # needs the table's path, and its verdict is about the table rather than
    # about this program.
    add_test(NAME boys-build-facts COMMAND boys-build-facts)
endif()

# --- CUDA lane (optional; needs the toolkit) --------------------------------
if (BUILD_CUDA)
    # The host wrapper (boys_cuda.cpp) and the kernels (boys_cuda.cu) are one
    # target: the .cpp takes the library target's own minimum standard, the .cu
    # is pinned at C++20 with a CUDA-safe include list (see the .cu preamble).
    # Entry points take raw device pointers + a stream and report through
    # boys::BoysStatus.
    #
    # The device option probe (boys_cuda_probe.cpp + boys_cuda_probe_kernels.cu)
    # is in this target and not in a target of its own: it measures this lane's
    # entries by calling this lane's launchers, so it has to be built against the
    # same kernels the figures are about, and a consumer reaches it through
    # include/boys/boys_cuda_probe.hpp either way.
    add_library(boys-cuda
        src/boys_cuda.cpp
        src/boys_cuda.cu
        src/boys_cuda_probe.cpp
        src/boys_cuda_probe_kernels.cu)
    target_include_directories(boys-cuda PUBLIC include PRIVATE src)
    target_compile_definitions(boys-cuda PUBLIC BoysFp16=1)
    # Stated because this target links nothing that carries a standard into it:
    # without it the host TU compiles at the compiler's default (C++14 on
    # MSVC) and the library's headers do not build.
    target_compile_features(boys-cuda PRIVATE cxx_std_20)
    # The probe's report states the toolkit that compiled these kernels and the
    # architectures they were compiled for. Both are configure-time facts, so
    # they are passed as text: nothing at run time can recover what nvcc was
    # told, and a figure that cannot be reproduced is not a measurement.
    target_compile_definitions(boys-cuda PRIVATE
        BOYS_CUDA_TOOLKIT_TEXT="${CMAKE_CUDA_COMPILER_VERSION}"
        BOYS_CUDA_ARCHITECTURES_TEXT="${BOYS_CUDA_ARCHITECTURES_TEXT}")
    boys_strict_warnings(boys-cuda)

    if (BOYS_BUILD_TESTS)
        add_executable(boys-cuda-tests tests/boys_cuda_test.cpp)
        target_link_libraries(boys-cuda-tests PRIVATE boys-cuda boys
            gtest gtest_main)
        target_include_directories(boys-cuda-tests PRIVATE src)
        boys_strict_warnings(boys-cuda-tests)
        add_test(NAME boys-cuda-tests COMMAND boys-cuda-tests)

        # The DEVICE OPTION PROBE's own test. It asserts nothing about a cost -
        # those are the card's - and asserts instead what a test can own about
        # the surface: that an absent device is a status, that the card the
        # figures are about is named in the returned data, and that a verdict
        # names only an entry the run measured.
        add_executable(boys-cuda-probe-tests tests/boys_cuda_probe_test.cpp)
        target_link_libraries(boys-cuda-probe-tests PRIVATE boys-cuda boys
            gtest gtest_main)
        target_include_directories(boys-cuda-probe-tests PRIVATE src)
        target_compile_features(boys-cuda-probe-tests PRIVATE cxx_std_20)
        boys_strict_warnings(boys-cuda-probe-tests)
        add_test(NAME boys-cuda-probe-tests COMMAND boys-cuda-probe-tests)

        # The DEVICE GATE (tests/boys_cuda_accuracy_gate.cpp) measures every
        # documented device bound against the CPU gate's own committed
        # high-precision reference - not against the CPU lane, which is what
        # the suite's cross-lane comparison does and where two errors can
        # cancel. It reports the CPU gate's row shape and names the card it
        # measured, so a figure pasted into a report says which arithmetic
        # produced it. It is a local gate for the same reason the CUDA lane is
        # one: it needs the device.
        # tests/boys_cuda_device_demo.cu carries the consumer side of the
        # device-callable entries (boys_cuda_device.hpp): kernels that form x
        # per thread and call the entries from inside themselves. It is built
        # into this gate rather than into the library because it is a consumer
        # of the public headers, and it is the only thing here that proves a
        # consumer's own kernel can reach the arithmetic.
        add_executable(boys-cuda-accuracy-gate
            tests/boys_cuda_accuracy_gate.cpp
            tests/boys_cuda_device_demo.cu)
        target_link_libraries(boys-cuda-accuracy-gate PRIVATE boys-cuda boys)
        target_include_directories(boys-cuda-accuracy-gate PRIVATE src)
        target_compile_features(boys-cuda-accuracy-gate PRIVATE cxx_std_23)
        target_compile_definitions(boys-cuda-accuracy-gate PRIVATE
            BoysDataDir="${CMAKE_CURRENT_SOURCE_DIR}/tests/data"
            BoysGateRevision="${BOYS_GATE_REVISION}")
        boys_strict_warnings(boys-cuda-accuracy-gate)
        add_test(NAME boys-cuda-accuracy-gate COMMAND boys-cuda-accuracy-gate)
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

    # The across-orders lane's driver. Portable: it calls the lane's private
    # header and no intrinsic of its own, so the same source answers on the
    # Linux lanes, where the platform's instruction counter is available and
    # a count per call can be read without a clock.
    add_executable(boys-across-orders-benchmark
        benchmarks/boys_across_orders_benchmark.cpp)
    target_link_libraries(boys-across-orders-benchmark PRIVATE boys)
    target_include_directories(boys-across-orders-benchmark PRIVATE src)
    target_compile_definitions(boys-across-orders-benchmark PRIVATE BoysFp16=1)
    target_compile_features(boys-across-orders-benchmark PRIVATE cxx_std_23)
    boys_strict_warnings(boys-across-orders-benchmark)

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

        # The device option probe's driver: argument parsing and printing only.
        # The measurement is boys::RunDeviceOptionProbe in the library, which is
        # the surface a consumer calls; this tool exists so a person can get the
        # same report at a terminal without writing a program. Local-only by
        # construction: it needs the device.
        add_executable(boys-device-probe benchmarks/boys_device_probe.cpp)
        target_link_libraries(boys-device-probe PRIVATE boys-cuda)
        target_compile_features(boys-device-probe PRIVATE cxx_std_20)
        boys_strict_warnings(boys-device-probe)
    endif()
elseif (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/benchmark/CMakeLists.txt")
    message(WARNING "third_party/benchmark is missing; BUILD_BENCHMARKS is OFF, "
        "so the benchmark drivers are not built (informational)")
endif()

# --- Generator byte-identity gate (top-level convenience targets) -----------
# --check regenerates the Chebyshev tables and the reference grid into a temp
# and byte-compares against the committed files (default paths: include/boys/
# boys_coefficients.hpp and tests/data/boys_reference.csv — spelled out here and
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
                --header ${CMAKE_CURRENT_SOURCE_DIR}/include/boys/boys_coefficients.hpp
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
