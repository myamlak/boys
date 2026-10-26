cmake_minimum_required(VERSION 3.25)

# boys — self-contained Boys-function kernel.
#
# Layout: public headers under include/boys/, sources under src/,
# tests under tests/ (gtest, vendored), benchmark drivers under
# benchmarks/, the regeneration script under tools/. The entries' kernels ship
# in the headers, the default entries included: a consumer instantiates
# whichever policy it names, and a consumer that links no archive still compiles
# and links a call that names no policy (see include/boys/boys.hpp and
# include/boys/kernels.hpp). The CUDA lane is an optional second
# target that needs the CUDA toolkit (device pins are a documented local
# gate — CI has no GPU leg and never builds it). No PCH in this tree.
#
# Consumers add this tree with add_subdirectory and link boys::boys, or boys::boys-tier
# for the headers alone with no archive on the link line: the library's only requirement
# is C++20;
# propagated through the target; tests/benchmarks are OFF unless this tree
# is the top-level project (or the consumer opts in), and a missing
# gtest/benchmark tree is a warning — not a FATAL — when the feature
# that needs it is OFF. See README.md, "Consuming as a submodule".

project(boys VERSION 5.1.1 LANGUAGES CXX)

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
option(BUILD_EXAMPLES
    "Build the examples: one small program per question a new caller asks"
    ${BOYS_TOP_LEVEL})
# One file per question, in the order a caller asks them. The list is read
# twice below - once to build each program, once to register it with ctest -
# so an example is added in one place and is then both built and run.
set(boysExamples
    00_first_call
    01_one_order
    02_all_orders
    03_arguments_array
    05_policy
    06_what_it_guarantees
    07_option_probe
    08_shell_quartet_batch)
option(BUILD_CUDA
    "Build the CUDA lane (needs the CUDA toolkit; CI compiles it and runs no kernel, so the device figures stay a local gate)"
    OFF)

# Two of the library's translation units answer a surface a calculation does not call: the C
# linkage surface and the option probe (one instantiation per cell of the option space, which is
# what makes its report an account of the space). Each is a member of `boys` when its option is
# ON, and a consumer that asks for neither compiles neither - a submodule consumer's default,
# since only the top-level build carries the suite, the examples and the recorded runs that reach
# them.
option(BOYS_BUILD_C_API
    "Build the C linkage surface (src/boys_c.cpp) into the library"
    ${BOYS_TOP_LEVEL})
option(BOYS_BUILD_PROBE
    "Build the option probe (src/boys_probe.cpp) into the library"
    ${BOYS_TOP_LEVEL})

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

# The header-only consumer check proves that the entries' definitions are
# reachable from the public headers alone, at either tier, and it proves it by
# NOT linking the library: with a library on its link line it would still
# compile, still run, and prove nothing. What it proves is the header-only
# contract itself - a call site compiles the body it calls, whatever its flags.
# Assert the absence here, so a later edit that links it fails the configure on
# every leg instead of quietly retiring the check.
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
    # The runtime target the device lane's public interface carries (CUDA::cudart):
    # enabling the language names the compiler, not the runtime's imported targets,
    # and the export hands one of them to every consumer of the lane.
    find_package(CUDAToolkit REQUIRED)
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
# The headers need >= C++20; the
# requirement is propagated through the target, so consumers need no
# global standard flags.
#
# The seam is a build's to close: every half entry sits behind `#if BoysFp16`, so OFF compiles a
# smaller surface rather than a different arithmetic - the float lanes the half entries share an
# engine with are untouched. ON is the shipped surface, and the profile the certified figures
# belong to (docs/build-facts.md).
option(BOYS_FP16_SEAM
    "Compile the half/bfloat entries into the surface (BoysFp16=1); OFF closes the seam across \
the whole tree and refuses a build that also asks for the test suite, which names those entries"
    ON)

# The library's members, with the two optional units behind their options. One list, because a
# source cannot be taken back out of a target once it has been named in it.
set(boysLibrarySources
    src/boys.cpp
    src/boys_defaults.cpp
    src/boys_orders_simd.cpp
    src/boys_simd.cpp)
if (BOYS_BUILD_C_API)
    list(APPEND boysLibrarySources src/boys_c.cpp)
endif()
if (BOYS_BUILD_PROBE)
    list(APPEND boysLibrarySources src/boys_probe.cpp)
endif()
add_library(boys ${boysLibrarySources})
unset(boysLibrarySources)
add_library(boys::boys ALIAS boys)
# Two spellings of one directory: the source tree's absolute path for a consumer
# that adds this tree, and `include` relative to the install prefix for the
# exported package, which has no source directory of its own.
target_include_directories(boys
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE src)
target_compile_features(boys PUBLIC cxx_std_20)
if (BOYS_FP16_SEAM)
    target_compile_definitions(boys PUBLIC BoysFp16=1)
endif()
boys_strict_warnings(boys)

# The option probe's own translation unit carries one instantiation per cell of
# the option space — every combination of every class the library declares, which
# is what makes its report an account of the space rather than of a sample of it —
# and each instantiation is a function the object format gives a section of its
# own. The count passes what a COFF object holds by default, so the compiler is
# told to use the large-address form of the same file. It is MSVC's spelling only
# because COFF is where the limit is: the ELF tools this project also builds with
# carry no such ceiling and need no flag. It is set on this source rather than on
# the target, because nothing else here comes near the limit and a flag spread
# over targets that do not need it is a fact about the build nobody can trace back
# to a measurement.
if (MSVC AND BOYS_BUILD_PROBE)
    set_source_files_properties(src/boys_probe.cpp PROPERTIES COMPILE_OPTIONS "/bigobj")
endif()

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

# --- What an unnamed call resolves to ----------------------------------------
# The choices a call site that names no policy gets — the fit route, the
# evaluation scheme, the packing axis, the division form and the fit granularity
# on a host class, and the device lane's own division form and region-B
# exponential beside them, because the two lanes' figures are two sets — are
# compile-time values, read through include/boys/boys_build_defaults.hpp, whose
# committed file carries the committed choices. This option points a build at a
# header carrying its own seven, and the build then compiles those choices into
# every entry that names none. Nothing is paid for this at run time and nothing
# is dispatched: the choice was a template argument before and is one still, so
# the compiler resolves exactly the call it resolved before, at the values this
# build named.
#
# What such a header carries is stated in that header's own comment: the seven
# names, and the machine, the date and the option probe's own figures beside
# them, because a choice made there is a measurement taken on one host and a
# reader has to be able to tell a tuned build from the committed one.
#
# NOT IN THIS TREE: the step that WRITES such a header. Running boys-option-probe
# at configure time, reading the ranking it reports and emitting the header with
# this host's figures is what this option exists to be written through, and it is
# deliberately absent rather than stubbed: the option below takes a header the
# build already has, and the message says which header is in force. A
# cross-compiling configure cannot run the probe on the machine it is compiling
# for at all, so it stays on the committed header and says so here.
set(BOYS_BUILD_DEFAULTS "" CACHE FILEPATH
    "Header carrying this build's default fit route, scheme, packing axis, \
division form and fit granularity, and the device lane's own division form and \
region-B exponential; empty builds the committed choices")

if (BOYS_BUILD_DEFAULTS)
    if (NOT EXISTS "${BOYS_BUILD_DEFAULTS}")
        message(FATAL_ERROR "boys: BOYS_BUILD_DEFAULTS names "
            "\"${BOYS_BUILD_DEFAULTS}\", which does not exist.")
    endif()

    # Copied under the name the headers include, into a directory that comes
    # BEFORE include/ on the search path: the compiler then reads it as
    # boys/boys_build_defaults.hpp and never reaches the committed file. The
    # copy is a configure_file dependency, so editing the header reconfigures
    # the build instead of leaving a stale copy behind.
    set(BOYS_BUILD_DEFAULTS_DIR "${CMAKE_CURRENT_BINARY_DIR}/boys-build-defaults")
    configure_file("${BOYS_BUILD_DEFAULTS}"
        "${BOYS_BUILD_DEFAULTS_DIR}/boys/boys_build_defaults.hpp" COPYONLY)
    target_include_directories(boys BEFORE PUBLIC "$<BUILD_INTERFACE:${BOYS_BUILD_DEFAULTS_DIR}>")
    # PUBLIC, so every target that links this one inherits it: a translation unit
    # that reaches the committed header instead of the replacement says which
    # build it is in and refuses to compile, rather than resolving one default
    # while the library's own instantiations resolve another. BUILD_INTERFACE
    # because the directory is this configure's: exported as it stands it makes
    # every consumer of the installed package read a path under a build tree
    # that no longer exists, which CMake refuses at generate time. What an
    # installed consumer reads instead is the seam the install rules below put
    # beside the headers - the one this build was compiled against.
    #
    # The seam's identity travels with the replacement: the sha256 of the file the
    # option named. A build input that changes delivered bits has to be able to say
    # which file it was, or a consumer whose regression baseline moved has no way
    # to see that the seam is why. The path above is this configure's and dies with
    # its build directory; a digest is content and travels, which is what makes two
    # builds of one seam comparable across machines
    # (include/boys/version.hpp, BuildDefaultsSeamIdentity()).
    #
    # Hashed here and not at build time because a configure-time fact is what this
    # file's other build facts are, and because the copy above makes the file a
    # dependency of the build system: editing it reconfigures, so the digest cannot
    # go stale behind an edit. Read from the path resolved the way configure_file
    # resolves it - relative to this source directory - so a configure run from
    # another working directory hashes the file it copied and not a stranger.
    get_filename_component(BOYS_BUILD_DEFAULTS_FILE "${BOYS_BUILD_DEFAULTS}" ABSOLUTE)
    file(SHA256 "${BOYS_BUILD_DEFAULTS_FILE}" BOYS_BUILD_DEFAULTS_SEAM_SHA256)
    # The same digest again as an identifier, because the front door names the namespace its
    # default entries are compiled into from the seam, and a namespace is named by a token
    # rather than by a string: two translation units that read two seams then define two
    # different entries rather than one entry with two bodies. The string above is what a build
    # reports as the seam it read (`include/boys/version.hpp`, BuildDefaultsSeamIdentity()).
    set(BOYS_BUILD_DEFAULTS_SEAM_IDENT "seam_${BOYS_BUILD_DEFAULTS_SEAM_SHA256}")
    target_compile_definitions(boys PUBLIC
        BOYS_BUILD_DEFAULTS_REPLACED=1
        BOYS_BUILD_DEFAULTS_SEAM_SHA256="${BOYS_BUILD_DEFAULTS_SEAM_SHA256}"
        BOYS_BUILD_DEFAULTS_SEAM_IDENT=${BOYS_BUILD_DEFAULTS_SEAM_IDENT})
    message(STATUS "boys: build defaults = ${BOYS_BUILD_DEFAULTS} — the seven choices an "
        "unnamed call resolves to are that header's, and they are a measurement on the machine "
        "it was taken on. sha256 ${BOYS_BUILD_DEFAULTS_SEAM_SHA256}")
elseif (CMAKE_CROSSCOMPILING)
    message(STATUS "boys: build defaults = the committed header (the shipped choices); this is "
        "a cross-compiling configure, which cannot measure the machine it compiles for.")
else()
    message(STATUS "boys: build defaults = the committed header (the shipped choices)")
endif()

# --- The contraction the library's arithmetic is compiled at -----------------
# A bare `a * b + c` in the scalar arithmetic is one rounding or two according
# to the target and the flags, and the two are different numbers. The figures
# this tree publishes are the two-rounding arithmetic's: the accuracy gate holds
# its bounds on a build that does not contract and reports three of them
# exceeded on one that does, so that is the arithmetic the library is compiled
# at here — stated, rather than left to whatever the host's compiler happens to
# do.
#
# It takes a statement because the default is not the same on two machines.
# aarch64 has the fused step in its baseline and its gcc contracts by default;
# x86-64 has no fused step unless -mfma is passed, so the same source and the
# same flags deliver two arithmetics, and the accuracy gate said so in its own
# report: the x86 legs read `bare product-plus-add does not contract here` and
# the arm64 legs `contracts here`. The lane that showed it is the half lane,
# which runs the float engine's downward recursion — a bare product-plus-add
# per step — against a budget of 1e-7 plus a half-ULP, which is one fp32
# rounding wide.
#
# The value named is the two-rounding model and not the fused one. Naming the
# fused model would ask for an instruction the x86-64 baseline does not have,
# and on a target that does have it the fused arithmetic is already what the
# fused route asks for, one call site at a time by name. This way each of the
# two routes means one arithmetic on every host, and a bound measured on one is
# a statement about the others.
#
# It reaches a consumer as well, where the arithmetic is compiled: every entry
# is header-defined, so a translation unit that instantiates a kernel compiles
# its own copy of these steps. The language guard keeps the option off C and
# CUDA translation units, which do not compile the C++ arithmetic and whose
# host libraries have their own flag vocabulary.
#
# MSVC's spelling is /fp:precise, which is its default and does not contract;
# /fp:contract is not named because not every MSVC this library builds on
# accepts it. Contraction there is opt-in, so the statement's work on that
# toolchain is to refuse a project-level /fp:fast rather than to change a
# default.
if (MSVC)
    set(_boys_contract_option "/fp:precise")
else()
    set(_boys_contract_option "-ffp-contract=off")
endif()
# Stated where the probe below measures, not only where the sources compile: a
# build that states its arithmetic in one place and measures it in another is
# measuring a different build than the one it ships.
string(APPEND CMAKE_CXX_FLAGS " ${_boys_contract_option}")
target_compile_options(boys PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${_boys_contract_option}>")

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

# One build fact, two carriers: the tier flags and the measured contraction answer
# are properties of the PROGRAM, and two targets carry translation units of one
# program - `boys` (the host library) and `boys-cuda` (the device lane, a target of
# its own because it needs the toolkit). An interface target carries both facts
# once and each library links it, so neither can be built at one tier beside the
# other's other tier - which the guard in include/boys/isa_tiers.hpp refuses at a
# program's start-up rather than letting the linker merge one body silently.
add_library(boys-tier INTERFACE)
# One name for the link line on both routes: the installed package hands this target out as
# boys::boys-tier through its namespace, and an add_subdirectory consumer reaches the same
# spelling through this alias rather than the bare target name (`boys` has its own above).
add_library(boys::boys-tier ALIAS boys-tier)
# An interface target has no translation units of its own to warn about; what it
# needs is the audit's bookkeeping, which boys_strict_warnings cannot give it (it
# sets PRIVATE options). The property is set directly, for the same reason.
set_property(DIRECTORY APPEND PROPERTY BOYS_STRICT_TARGETS boys-tier)

# This target is also the headers-only consumer's whole interface: the include
# path, the C++20 floor, the fp16 seam and the contraction pin are the rest of
# what a unit is compiled with, so a target that links this one alone compiles
# the arithmetic the library was built at, with no archive on its link line
# (README.md, "Consuming as a submodule"). `boys` spells the include path and
# the standard out again because the header-only consumer check reads both off
# the library target itself.
target_include_directories(boys-tier
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
target_compile_features(boys-tier INTERFACE cxx_std_20)
if (BOYS_FP16_SEAM)
    target_compile_definitions(boys-tier INTERFACE BoysFp16=1)
endif()
# The replacement seam the block above puts on `boys`, on the headers-only target too: the
# comment above claims this target alone is a consumer's whole interface, and the seam is part
# of that interface - a unit that links only this one must read the replacement and refuse the
# committed header, exactly as one that links `boys`. The variables are that block's.
if (BOYS_BUILD_DEFAULTS)
    target_include_directories(boys-tier BEFORE INTERFACE
        "$<BUILD_INTERFACE:${BOYS_BUILD_DEFAULTS_DIR}>")
    target_compile_definitions(boys-tier INTERFACE
        BOYS_BUILD_DEFAULTS_REPLACED=1
        BOYS_BUILD_DEFAULTS_SEAM_SHA256="${BOYS_BUILD_DEFAULTS_SEAM_SHA256}"
        BOYS_BUILD_DEFAULTS_SEAM_IDENT=${BOYS_BUILD_DEFAULTS_SEAM_IDENT})
endif()
# The contraction pin the section above states on `boys`, stated again on the
# target a headers-only consumer links: its units compile the entries' own
# steps, and a target whose fused step is in the baseline (aarch64) contracts a
# bare product-plus-add at the compiler's default.
target_compile_options(boys-tier INTERFACE
    "$<$<COMPILE_LANGUAGE:CXX>:${_boys_contract_option}>")
unset(_boys_contract_option)

# 1 and 0 are the two measured answers and both are defined, so a reader asks the
# value rather than whether the name exists. Neither is defined where no answer
# was measured: a build that has not measured contraction does not get to assert
# the claims the answer decides, and the tri-state is what keeps it from
# inheriting the one that would be wrong rather than the one that claims least.
if (BOYS_SCALAR_CONTRACTS)
    target_compile_definitions(boys-tier INTERFACE BOYS_SCALAR_CONTRACTS=1)
    message(STATUS "boys: this target contracts a bare product-plus-add (one rounding) — "
        "the scalar lanes' two multiply-add routes are one arithmetic here, and the "
        "claims a build carries are compiled against that answer.")
elseif (NOT CMAKE_CROSSCOMPILING)
    target_compile_definitions(boys-tier INTERFACE BOYS_SCALAR_CONTRACTS=0)
    message(STATUS "boys: this target does not contract a bare product-plus-add (two "
        "roundings) — the scalar lanes' two routes are two arithmetics here, and the "
        "claims a build carries are compiled against that answer.")
else()
    message(STATUS "boys: contraction is not measured in a cross-compiling configure — "
        "no claims are compiled against it and BoysBackends() reports it at run time.")
endif()
target_link_libraries(boys PUBLIC boys-tier)

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

option(BOYS_ISA_AVX2
    "compile the library - and, through the target, every consumer that links it - at the \
AVX2+FMA+F16C tier; OFF builds the scalar tier across the whole tree"
    ON)

if (BOYS_ARCH_X86_64)
    target_compile_definitions(boys PRIVATE BOYS_SIMD_X86=1)

    # The instruction-set tier is a property of the PROGRAM, not of a file: the
    # entries are the caller's to compile now, and two translation units at two
    # tiers cannot share one definition of them - the guard in
    # include/boys/isa_tiers.hpp aborts such a program at startup rather than
    # letting the linker merge one body silently. So the tier is set on the
    # target and propagated PUBLIC, and a consumer that links `boys` inherits
    # the one flag set this build was measured under; a consumer that needs the
    # portable scalar tier configures -DBOYS_ISA_AVX2=OFF.
    #
    # The flag set is two things at once, and the second is why the contraction
    # pin rides beside it. It lets the AVX2/FMA/F16C intrinsics compile - and, on
    # GCC and Clang, -mfma is also a licence to fuse a bare `a * b + c` into one
    # rounding, which the rest of the build's flags do not give it. That licence
    # is not this flag set's to spend: measured on the two-build differential, a
    # vector-tier unit built with the compiler's default contraction returns a
    # last-bit-different value from the certified lane (29 rows gcc, 23 clang),
    # while the same build with the pin returns the certified bits exactly
    # (466 of 466 identical). A consumer that re-spells the flags itself, header
    # only, owns that same pairing - docs/build-facts.md states the profile the
    # figures belong to.
    if (BOYS_ISA_AVX2)
        if (MSVC)
            # AVX2 intrinsics come with /arch:AVX2 (F16C included); no
            # contraction flag is added because MSVC contracts only under
            # /fp:fast, which is not what the measurement above runs under.
            target_compile_options(boys-tier INTERFACE /arch:AVX2)
            # The CUDA language does not take the C++ flag set: nvcc hands its own
            # switches to the host pass, so a translation unit compiled by nvcc
            # would build its shared inline bodies at the scalar tier beside the
            # C++ units' vector ones - a mixed program, which the tier guard in
            # include/boys/isa_tiers.hpp refuses at start-up (measured here: the
            # CUDA-on gate aborted on exactly that). The flags travel to the host
            # pass through -Xcompiler, one spelling per driver.
            target_compile_options(boys-tier INTERFACE
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/arch:AVX2>)
        else()
            # GCC does not imply F16C via -mavx2.
            target_compile_options(boys-tier INTERFACE -mavx2 -mfma -mf16c)
            if (DEFINED BOYS_SCALAR_CONTRACTS)
                if (BOYS_SCALAR_CONTRACTS)
                    target_compile_options(boys-tier INTERFACE -ffp-contract=fast)
                    target_compile_options(boys-tier INTERFACE
                        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffp-contract=fast>)
                else()
                    target_compile_options(boys-tier INTERFACE -ffp-contract=off)
                    target_compile_options(boys-tier INTERFACE
                        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffp-contract=off>)
                endif()
            endif()
            target_compile_options(boys-tier INTERFACE
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-mavx2,-mfma,-mf16c>)
        endif()
    else()
        message(STATUS "boys: BOYS_ISA_AVX2=OFF - the whole tree compiles the "
            "scalar tier, and the vector-tier-only test and benchmark targets "
            "do not apply to it")
    endif()
else()
    target_compile_definitions(boys PRIVATE BOYS_SIMD_X86=0)
    message(STATUS "boys: non-x86_64 target — the AVX2 tier is not built; the "
        "region kernels fall back to the certified scalar lanes and "
        "BoysAvx2Available() reports false.")
endif()

# The content a record's stamp carries: every file the program reads from, hashed at configure
# time by the tool that recomputes those digests at check time. One implementation is what makes
# the two comparable - a second hashing written here would have to agree with the first over
# every byte of every file, and nothing would say when it did not.
#
# The version is not read here. A program prints the library's own, from the header it compiles
# against, so that cell cannot disagree with the code that prints it; the content it was built
# from is what a program cannot know by itself, and that is what these cells carry.
#
# The interpreter is asked for and not required, and its absence is the fallback below: a tree
# configured without one carries the version and no digest, and the stamp says so visibly rather
# than naming content it did not read. An interpreter that is here and a read that fails is the
# opposite case and stops the configure: a record that cannot date itself by its content is
# refused by the checker after the run, when the run is spent.
#
# The read is this tree's own: the two cells land in the stamp of the accuracy gate's record and
# the option probe's report, and both programs are registered under BOYS_BUILD_TESTS. So the
# recorder runs where this tree is top-level, and a consumer's configure runs neither the tool nor
# its FATAL - both cells read "unknown", the fallback a tree without an interpreter already
# carries, and no target of a consumer's build reads them.
find_package(Python3 COMPONENTS Interpreter QUIET)
foreach(_boys_record IN ITEMS probe gate)
    string(TOUPPER "${_boys_record}" _boys_RECORD)
    set(BOYS_${_boys_RECORD}_INPUTS "unknown")

    if(BOYS_TOP_LEVEL AND Python3_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tools/record_stamp.py")
        execute_process(
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/record_stamp.py"
                    --inputs ${_boys_record} --root "${CMAKE_CURRENT_SOURCE_DIR}"
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            OUTPUT_VARIABLE _boys_cell
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _boys_status
            ERROR_VARIABLE _boys_output)

        if(NOT _boys_status STREQUAL "0")
            string(STRIP "${_boys_output}" _boys_output)
            message(FATAL_ERROR
                "boys: this source tree carries tools/record_stamp.py and an interpreter, and "
                "the files the ${_boys_record} record reads from could not be read. The digest "
                "the record would carry would name content this tree does not have, and the "
                "check that reads the record refuses one it cannot hold to a tree - after the "
                "run.\n"
                "  command: \"${Python3_EXECUTABLE}\" tools/record_stamp.py --inputs "
                "${_boys_record} --root ${CMAKE_CURRENT_SOURCE_DIR}\n"
                "  exit:    ${_boys_status}\n"
                "  output:  ${_boys_output}\n"
                "A source tree without the tool or without an interpreter is not this case: its "
                "digest is \"unknown\" and its stamp says so.")
        endif()

        set(BOYS_${_boys_RECORD}_INPUTS "${_boys_cell}")
    endif()

    unset(_boys_cell)
    unset(_boys_status)
    unset(_boys_output)
endforeach()

# --- Feature pairs a build cannot split -------------------------------------
# The suite, the examples and the option probe's drivers call the surfaces the three options above
# can leave out, so a configure that asks for a driver and not for the surface it calls is refused
# here, with both options named, rather than by a link error after a build.
if (BOYS_BUILD_TESTS AND NOT BOYS_BUILD_C_API)
    message(FATAL_ERROR "boys: BOYS_BUILD_TESTS builds tests/boys_c_test.cpp, which calls the C "
        "linkage surface; set BOYS_BUILD_C_API=ON, or BOYS_BUILD_TESTS=OFF.")
endif()
if (BOYS_BUILD_TESTS AND NOT BOYS_BUILD_PROBE)
    message(FATAL_ERROR "boys: BOYS_BUILD_TESTS builds tests/boys_probe_test.cpp and the "
        "boys-option-probe driver, which call the option probe; set BOYS_BUILD_PROBE=ON, or "
        "BOYS_BUILD_TESTS=OFF.")
endif()
if (BUILD_EXAMPLES AND NOT BOYS_BUILD_PROBE)
    message(FATAL_ERROR "boys: BUILD_EXAMPLES builds examples/07_option_probe.cpp, which calls "
        "the option probe; set BOYS_BUILD_PROBE=ON, or BUILD_EXAMPLES=OFF.")
endif()
if (BOYS_BUILD_TESTS AND NOT BOYS_FP16_SEAM)
    message(FATAL_ERROR "boys: the test suite builds tests/boys_accuracy_gate.cpp, which names "
        "the half entries through a policy on a path it does not guard; set BOYS_FP16_SEAM=ON, "
        "or BOYS_BUILD_TESTS=OFF.")
endif()

# --- Tests (gtest; the reference grid is located via BoysDataDir) -----------
if (BOYS_BUILD_TESTS)
    include(CTest)

    add_executable(boys-tests
        tests/boys_test.cpp
        tests/boys_boundary_test.cpp
        tests/boys_accuracy_test.cpp
        tests/boys_fixed_n_test.cpp
        tests/boys_all_n_test.cpp
        tests/boys_across_orders_test.cpp
        tests/boys_orders_f32_test.cpp
        tests/boys_c_test.cpp
        tests/f16_test.cpp
        tests/boys_half_native_test.cpp
        tests/boys_backend_test.cpp
        # The multiply-add route's liveness: a lane that stops reading the build's
        # route keeps compiling and keeps every suite that names the route green,
        # so what these hold is the values each lane delivers, against the
        # library's own summations at a named route (the reference arithmetic is
        # tests/boys_muladd_route_reference.hpp, shared by the two).
        tests/boys_muladd_route_test.cpp
        tests/boys_muladd_route_simd_test.cpp
        tests/boys_fixture_pins_test.cpp
        tests/boys_probe_test.cpp
        tests/boys_build_defaults_test.cpp
        # The uniform grid's transposed twin, held to the interval-major table by the
        # one translation unit that indexes both: a lane header cannot, because a
        # translation unit opens only the group headers its policies name.
        tests/boys_coefficients_pair_check.cpp
        tests/boys_transform_test.cpp
        # The front door's coefficient set: the nine groups <boys/boys.hpp> opens,
        # asserted present and the other twelve absent, and every host entry
        # called so a default path reading a group it does not open is a build
        # failure here rather than a link failure at a caller.
        tests/boys_default_tables_test.cpp
        # The refusal's reason: every combination QueryCombination can be asked
        # about, and every member of the device option space, held to the reason
        # its own field documents. The device arm of this file is compiled in by
        # the CUDA lane's target below; this is the host arm.
        tests/boys_refusal_reason_test.cpp)
    target_link_libraries(boys-tests PRIVATE boys gtest gtest_main)
    target_include_directories(boys-tests PRIVATE src)
    target_compile_features(boys-tests PRIVATE cxx_std_23)
    # The directory the suite's reference grids are read from, the accuracy gate's
    # 80-digit grid among them; a test that reads one skips when it is absent.
    target_compile_definitions(boys-tests PRIVATE
        BoysDataDir="${CMAKE_CURRENT_SOURCE_DIR}/tests/data")
    boys_strict_warnings(boys-tests)

    # tests/boys_muladd_route_simd_test.cpp includes the private header the
    # packed backends are declared in, so this unit carries the intrinsics' flag
    # set the way src/boys_simd.cpp does - the same set and the same pinning of
    # the measured contraction answer, and for the same reason: the reference
    # arithmetic it holds those backends to has to be the build's arithmetic,
    # not this unit's own reading of a bare product-plus-add.
    if (BOYS_ARCH_X86_64)
        if (MSVC)
            set_source_files_properties(tests/boys_muladd_route_simd_test.cpp PROPERTIES
                COMPILE_OPTIONS "/arch:AVX2")
        else()
            set(_boys_simd_test_flags "-mavx2;-mfma;-mf16c")
            if (DEFINED BOYS_SCALAR_CONTRACTS)
                if (BOYS_SCALAR_CONTRACTS)
                    list(APPEND _boys_simd_test_flags "-ffp-contract=fast")
                else()
                    list(APPEND _boys_simd_test_flags "-ffp-contract=off")
                endif()
            endif()
            set_source_files_properties(tests/boys_muladd_route_simd_test.cpp PROPERTIES
                COMPILE_OPTIONS "${_boys_simd_test_flags}")
            unset(_boys_simd_test_flags)
        endif()
    endif()

    # The ACCURACY GATE (tests/boys_accuracy_gate.cpp) measures every
    # documented per-lane, per-region bound against the independent
    # high-precision reference grid (tests/data/boys_accuracy_gate_reference.csv,
    # regenerated by tools/gen_boys_accuracy_gate_reference.py) over a dense
    # sweep that runs past the suite's grid, and reports each claim as a ratio
    # of delivered error to documented bound with the location of the maximum.
    # It exits non-zero on any exceeded claim; passes that hold only because
    # the returned value is below what the format can hold are counted apart.
    # Its far-representation book reads a second grid beside that one
    # (tests/data/boys_far_gate_reference.csv, regenerated by
    # tools/gen_boys_far_gate_reference.py), from the same tree directory and
    # located the same way, because the far region's three arms dispatch on
    # boundaries the first grid does not resolve.
    #
    add_executable(boys-accuracy-gate tests/boys_accuracy_gate.cpp)
    target_link_libraries(boys-accuracy-gate PRIVATE boys)
    target_include_directories(boys-accuracy-gate PRIVATE src)
    target_compile_features(boys-accuracy-gate PRIVATE cxx_std_23)
    target_compile_definitions(boys-accuracy-gate PRIVATE
        BoysDataDir="${CMAKE_CURRENT_SOURCE_DIR}/tests/data"
        # The same directory as the tree spells it, beside the absolute one the gate reads
        # through: the reference line the gate prints is kept in the recorded run, and a record
        # is read by people who do not have the machine it was made on.
        BoysDataDirRelative="tests/data"
        # The content the run was taken from: the stamp's identity is the version the gate
        # prints and this cell, and the checker holds the record to the digests.
        BoysGateInputs="${BOYS_GATE_INPUTS}")
    boys_strict_warnings(boys-accuracy-gate)

    # The external-row application: when ON, the gate measures Boost.Math's documented Airy
    # sentence as two rows of its own book. Requires Boost >= 1.92; OFF leaves the build unchanged.
    option(BOYS_GATE_EXTERNAL_BOOST
        "Measure Boost.Math's documented Airy accuracy claim as rows of the gate's book"
        OFF)
    if (BOYS_GATE_EXTERNAL_BOOST)
        # A package config if present, otherwise a header tree named by BOOST_ROOT.
        find_package(Boost 1.92 CONFIG QUIET)
        if (NOT Boost_FOUND)
            find_path(BOYS_GATE_BOOST_INCLUDE boost/version.hpp
                HINTS "${BOOST_ROOT}/include" "$ENV{BOOST_ROOT}/include"
                      "${BOOST_ROOT}" "$ENV{BOOST_ROOT}"
                DOC "Boost headers for the gate's external row")
            if (NOT BOYS_GATE_BOOST_INCLUDE)
                message(FATAL_ERROR
                    "boys: BOYS_GATE_EXTERNAL_BOOST is ON and no Boost >= 1.92 was found. "
                    "Install that Boost with its CMake package, or point BOOST_ROOT at its "
                    "source tree.")
            endif()
            set(Boost_INCLUDE_DIRS "${BOYS_GATE_BOOST_INCLUDE}")
            file(STRINGS "${BOYS_GATE_BOOST_INCLUDE}/boost/version.hpp" _boys_boost_version
                REGEX "#define BOOST_LIB_VERSION")
            string(REGEX REPLACE ".*\"([^\"]+)\".*" "\\1" Boost_VERSION_STRING
                "${_boys_boost_version}")
            string(REPLACE "_" "." Boost_VERSION_STRING "${Boost_VERSION_STRING}")
        endif()
        if (NOT Boost_INCLUDE_DIRS AND Boost_INCLUDE_DIR)
            set(Boost_INCLUDE_DIRS "${Boost_INCLUDE_DIR}")
        endif()
        if (NOT Boost_INCLUDE_DIRS)
            message(FATAL_ERROR
                "boys: Boost was found but names no include directory; the external row "
                "cannot be compiled")
        endif()
        if (Boost_VERSION_STRING VERSION_LESS "1.92")
            message(FATAL_ERROR
                "boys: BOYS_GATE_EXTERNAL_BOOST is stated at Boost 1.92; this tree is "
                "Boost ${Boost_VERSION_STRING}")
        endif()
        # Both the entry under test and its reference must be present.
        list(GET Boost_INCLUDE_DIRS 0 _boys_boost_include)
        foreach (_boys_boost_stem
                 math/special_functions/airy
                 math/special_functions/relative_difference
                 multiprecision/cpp_bin_float)
            if (NOT EXISTS "${_boys_boost_include}/boost/${_boys_boost_stem}.hpp")
                message(FATAL_ERROR
                    "boys: ${_boys_boost_include} has no boost/${_boys_boost_stem}.hpp; "
                    "BOYS_GATE_EXTERNAL_BOOST needs Boost.Math and Boost.Multiprecision "
                    "(Boost >= 1.92)")
            endif()
        endforeach()
        target_compile_definitions(boys-accuracy-gate PRIVATE BOYS_GATE_EXTERNAL_BOOST=1)
        # Boost's headers are not this tree's to keep warning-clean; C4714 comes at code generation.
        target_include_directories(boys-accuracy-gate SYSTEM PRIVATE ${Boost_INCLUDE_DIRS})
        if (MSVC)
            target_compile_options(boys-accuracy-gate PRIVATE
                "$<$<COMPILE_LANGUAGE:CXX>:/wd4714>")
        endif()
        message(STATUS "boys: the external Boost row is ON - Boost ${Boost_VERSION_STRING} from "
            "${Boost_INCLUDE_DIRS}")
    endif()

    # The DEVICE LANE's arm of that cross. The gate measures the device lane
    # where the build carries it, through the CUDA surface's own entries, so
    # this target links boys-cuda and states BOYS_GATE_CUDA in exactly the
    # configuration that defines boys-cuda: the two conditions are one, and a
    # build without the lane gets neither the arm nor the link. What the arm
    # needs beyond the sources is what boys-cuda already carries - the toolkit's
    # include directories (PUBLIC on that target) and the CUDA runtime the
    # target's own objects are linked against - so nothing about the toolkit is
    # repeated here.
    # The device-callable half of that arm (tests/boys_accuracy_gate_fast_device.cu) is
    # a translation unit of its own because a kernel that calls a __device__ entry may
    # only be compiled as CUDA, and the gate's own extension is not. It is added here
    # in the same configuration the two lines above state, so a build without the lane
    # is a build with no kernel in this target and one compilation language fewer.
    if (BUILD_CUDA)
        target_link_libraries(boys-accuracy-gate PRIVATE boys-cuda)
        target_compile_definitions(boys-accuracy-gate PRIVATE BOYS_GATE_CUDA=1)
        target_sources(boys-accuracy-gate PRIVATE tests/boys_accuracy_gate_fast_device.cu)
    endif()

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
    # STATIC_LIBRARY again: this asks whether the declarations accept the
    # policy, not whether this build links the definitions.
    include(CheckCXXSourceCompiles)
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
#include <boys/boys.hpp>\n#include <boys/kernels.hpp>   // the policy-taking spelling these probes ask about
int main() {
    double out[33] = {};
    const double x = 1.0;
    boys::BoysAllN<boys::SeamPolicy<boys::FitRoute::kRationalMinimax>>(1, &x, out, 1);
    boys::BoysAllN<boys::SeamPolicy<boys::FitRoute::kRationalMinimax>>(
        1, &x, out, 1, boys::BoysSortedArgs{});
    boys::BoysFixedN<boys::SeamPolicy<boys::FitRoute::kRationalMinimax>>(0, &x, out, 1);
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

    # The orders axis's remaining limit, measured the same way the ones above
    # are: by compiling the call and finding that it does not build. It is a
    # property of a call shape rather than of a value, so no run can show it -
    # what a run would show is that the library answered, which is the failure
    # mode.
    #
    #   the fixed-order entry  a packed lane keeps four ORDERS of one argument,
    #                          and a fixed-order call produces exactly one order
    #                          at every argument: there are not four orders to
    #                          fill a lane with, so the axis cannot be formed
    #                          there at all.
    #
    # The axis's other limit - a route other than the shipped one - was built
    # rather than argued away, so it carries no probe: the packing-axis rows of
    # the gate measure every such combination instead, which is the reading that
    # says the axis answers them.
    unset(BOYS_GATE_FIXEDN_ACCEPTS_ORDERS CACHE)
    set(_boys_probe_target_type "${CMAKE_TRY_COMPILE_TARGET_TYPE}")
    set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
    set(CMAKE_REQUIRED_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/include")
    if (MSVC)
        set(CMAKE_REQUIRED_FLAGS "/std:c++20 /EHsc")
    else()
        set(CMAKE_REQUIRED_FLAGS "-std=c++20")
    endif()
    check_cxx_source_compiles("
#include <boys/boys.hpp>\n#include <boys/kernels.hpp>   // the policy-taking spelling these probes ask about
int main() {
    double out[33] = {};
    const double x = 1.0;
    using OrdersAxis = boys::SeamPolicy<boys::FitRoute::kChebyshev, boys::EvalScheme::kSplitClenshaw,
                                        boys::BoysBudget::kFloat, boys::PackAxis::kOrders>;
    boys::BoysFixedN<OrdersAxis>(0, &x, out, 1);
    return 0;
}
" BOYS_GATE_FIXEDN_ACCEPTS_ORDERS)
    set(CMAKE_TRY_COMPILE_TARGET_TYPE "${_boys_probe_target_type}")
    unset(_boys_probe_target_type)
    unset(CMAKE_REQUIRED_FLAGS)
    unset(CMAKE_REQUIRED_INCLUDES)

    if (NOT BOYS_GATE_FIXEDN_ACCEPTS_ORDERS)
        target_compile_definitions(boys-accuracy-gate PRIVATE BOYS_GATE_FIXEDN_REFUSES_ORDERS=1)
    endif()

    if (BOYS_GATE_FIXEDN_ACCEPTS_ORDERS)
        message(STATUS "boys: the orders axis on BoysFixedN is ACCEPTED by the probe — a call "
            "shape with no orders to pack takes an orders-axis policy")
    else()
        message(STATUS "boys: the orders axis on BoysFixedN is REFUSED, measured by compiling "
            "the call")
    endif()

    # Every refusal below is measured by compiling the call it names rather than
    # by quoting the assertion that states it. A refusal the library asserts and
    # cannot show is the same defect as citing a refusal
    # as the reason a bound is not owed. A probe establishes something narrower
    # than the refusal it backs - it says the call does not build, not that it
    # cannot - and each verdict is printed after the probe that produced it,
    # because a statement placed above its probe reads a variable not yet set.
    unset(BOYS_GATE_F32_ACCEPTS_ORDERS CACHE)
    set(_boys_probe_target_type "${CMAKE_TRY_COMPILE_TARGET_TYPE}")
    set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
    set(CMAKE_REQUIRED_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/include")
    if (MSVC)
        set(CMAKE_REQUIRED_FLAGS "/std:c++20 /EHsc")
    else()
        set(CMAKE_REQUIRED_FLAGS "-std=c++20")
    endif()
    # The orders axis on the single-precision engines: the orders entry and the
    # many-argument entry, both at the shipped route and scheme.
    #
    # Only the shapes that CAN carry the axis are probed here. A one-order entry
    # has one order to pack and refuses the axis for that reason alone, which is
    # a property of the call and not of the axis; asking it here would make this
    # probe's verdict a fact about the shape while the gate reads it as a fact
    # about the axis. That shape has the probe below, and its verdict is booked
    # as the shape limit it is.
    check_cxx_source_compiles("
#include <boys/boys.hpp>\n#include <boys/kernels.hpp>   // the policy-taking spelling these probes ask about
int main() {
    float in[33] = {};
    float out[33] = {};
    using OrdersAxis = boys::SeamPolicy<boys::FitRoute::kChebyshev, boys::EvalScheme::kSplitClenshaw,
                                        boys::BoysBudget::kFloat, boys::PackAxis::kOrders>;
    boys::BoysAllOrdersF32<OrdersAxis>(32, 1.0f, out);
    boys::BoysAllNF32<OrdersAxis>(1, in, out, 1);
    return 0;
}
" BOYS_GATE_F32_ACCEPTS_ORDERS)
    # The one-order shape on that lane, which is the shape the probe above used
    # to be failing on. Refused is the expected verdict: there is one order here
    # and a packing axis has nothing to pack.
    unset(BOYS_GATE_F32_SINGLE_ACCEPTS_ORDERS CACHE)
    check_cxx_source_compiles("
#include <boys/boys.hpp>\n#include <boys/kernels.hpp>   // the policy-taking spelling these probes ask about
int main() {
    using OrdersAxis = boys::SeamPolicy<boys::FitRoute::kChebyshev, boys::EvalScheme::kSplitClenshaw,
                                        boys::BoysBudget::kFloat, boys::PackAxis::kOrders>;
    boys::BoysSingleF32<OrdersAxis>(1, 1.0f);
    return 0;
}
" BOYS_GATE_F32_SINGLE_ACCEPTS_ORDERS)
    set(CMAKE_TRY_COMPILE_TARGET_TYPE "${_boys_probe_target_type}")
    unset(_boys_probe_target_type)
    unset(CMAKE_REQUIRED_FLAGS)
    unset(CMAKE_REQUIRED_INCLUDES)

    # Each statement below reads a probe measured above.
    if (BOYS_GATE_F32_ACCEPTS_ORDERS)
        message(STATUS "boys: the orders packing axis on the single-precision engines is ACCEPTED, "
            "measured by compiling the call")
    else()
        target_compile_definitions(boys-accuracy-gate PRIVATE BOYS_GATE_F32_REFUSES_ORDERS=1)
        message(STATUS "boys: the orders packing axis on the single-precision engines is REFUSED, "
            "measured by compiling the call")
    endif()

    if (NOT BOYS_GATE_F32_SINGLE_ACCEPTS_ORDERS)
        target_compile_definitions(boys-accuracy-gate PRIVATE
            BOYS_GATE_F32_SINGLE_REFUSES_ORDERS=1)
        message(STATUS "boys: the one-order entry on the single-precision engines is REFUSED the "
            "orders axis, measured by compiling the call")
    else()
        message(STATUS "boys: the one-order entry on the single-precision engines ACCEPTS the "
            "orders axis")
    endif()

    # The uniform partition on the batched double-precision entries. Six probes
    # stood here, one per cell of the entry book in tests/boys_accuracy_gate.cpp:
    # the plane entry and its sorted overload on the arguments axis of the shipped
    # route, and the fixed-order entry, at both schemes. Each compiled one cell's
    # call and found that it did not build - the entries' RefuseUniformPartition -
    # and the gate printed the refusal for that cell.
    #
    # The six cells are served at this revision: each entry hands a policy naming
    # the grid to the per-argument path, whose body reads the grid, and the gate's
    # entry book measures them over the whole committed grid. The probes came out
    # with the refusals they measured, and nothing replaced them, because a row
    # that measures a cell calls it: a revision that put one of those guards back
    # would fail to build the gate, which is a louder reading than a probe's and
    # one that names the file and line the call is in.

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
    # whose definition never left a .cpp file (a link error for every entry
    # the library does not pre-instantiate). These two programs are
    # consumers, and each is built so that its own defect has somewhere to
    # show:
    #
    #   boys-consumer-umbrella  includes ONLY <boys/boys.hpp>, first include,
    #     no internal header, and links the library. It exercises every entry
    #     the public headers document - all three product modes, both
    #     many-argument shapes, the fp16/bf16 lanes, the native
    #     half lane - and judges every returned value against the bound its
    #     entry states, with the committed 45-digit grid as the reference.
    #
    #   boys-consumer-header-only  proves the definitions really are reachable
    #     from the headers alone, at either tier: it links NOTHING. Its meaning
    #     is the empty link line, so the target's link libraries are asserted
    #     empty below rather than trusted.
    #
    #   boys-consumer-defaults  is the shortcut's own check: the per-precision
    #     default each entry takes when the call site names no policy, written
    #     as a name from <boys/boys.hpp> alone and compared against the same
    #     entry with no argument, value by value over every order and a sweep
    #     of arguments, with the half lanes held against the policy their
    #     entries run. It prints the cells compared and the worst difference
    #     for each row.
    #
    #   boys-consumer-option-space  is the option space's own check: every
    #     combination the library's book reports as served is NAMED - a policy
    #     type, one arm per name, no string and no lookup - and every value that
    #     comes back is judged against the figure the book states for that
    #     combination and compared bit for bit with the same cell's
    #     compile-time spelling. It prints the combinations served and named
    #     per precision, so the bit-for-bit comparison is visible as one that
    #     could have failed.
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
        BoysConsumerReference="${CMAKE_CURRENT_SOURCE_DIR}/tests/data/boys_reference.csv"
        # The version the build was configured at, so the consumer check can
        # assert the header a caller reads agrees with project(boys VERSION ...)
        # instead of the two being two answers to one question.
        BoysExpectedVersion="${PROJECT_VERSION}")
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

    # The archive-free accuracy header (include/boys/boys_accuracy_here.hpp) answers the
    # measured figures from the headers alone; this target is how that is proven. It links
    # nothing - boys_assert_no_linked_library below holds that - and its unit includes that
    # header and no other of this library's, so a figure that needed the archive would fail
    # to link here. The include path and the compile definitions are the ones a headers-only
    # consumer inherits from boys-tier, so the header is compiled at this build's arithmetic.
    get_target_property(boysHereIncludes boys-tier INTERFACE_INCLUDE_DIRECTORIES)
    get_target_property(boysHereDefines boys-tier INTERFACE_COMPILE_DEFINITIONS)
    if (NOT boysHereIncludes)
        message(FATAL_ERROR "boys: boys-tier publishes no INTERFACE_INCLUDE_DIRECTORIES; the "
            "accuracy-here check reads that property and found it empty.")
    endif()
    add_executable(boys-accuracy-here tests/boys_accuracy_here_test.cpp)
    target_include_directories(boys-accuracy-here PRIVATE ${boysHereIncludes})
    if (boysHereDefines)
        target_compile_definitions(boys-accuracy-here PRIVATE ${boysHereDefines})
    endif()
    target_compile_features(boys-accuracy-here PRIVATE cxx_std_20)
    boys_strict_warnings(boys-accuracy-here)
    boys_assert_no_linked_library(boys-accuracy-here)

    add_executable(boys-consumer-defaults tests/consumer_defaults.cpp)
    target_link_libraries(boys-consumer-defaults PRIVATE boys)
    boys_strict_warnings(boys-consumer-defaults)

    # The default-versus-documentation check: the table a reader believes, against
    # the constants the call sites resolve to. boys-consumer-defaults above checks
    # that the named policy and the unnamed one are the same call, which is true
    # whichever value the constant holds; this is the one that notices when the
    # document stops describing the library.
    add_executable(boys-consumer-defaults-match-docs tests/consumer_defaults_match_docs.cpp)
    target_link_libraries(boys-consumer-defaults-match-docs PRIVATE boys)
    target_compile_definitions(boys-consumer-defaults-match-docs PRIVATE
        BoysSourceDir="${CMAKE_CURRENT_SOURCE_DIR}")
    boys_strict_warnings(boys-consumer-defaults-match-docs)

    add_executable(boys-consumer-option-space tests/consumer_option_space.cpp)
    target_link_libraries(boys-consumer-option-space PRIVATE boys)
    target_compile_definitions(boys-consumer-option-space PRIVATE
        BoysConsumerReference="${CMAKE_CURRENT_SOURCE_DIR}/tests/data/boys_reference.csv")
    boys_strict_warnings(boys-consumer-option-space)

    # --- What an unnamed call resolves to, and who compiles it ------------------
    #
    # A call site that names no policy compiles the entry where it is written: the default entries
    # are inline in the headers, so a consumer that links no archive still compiles and links its
    # call, and the consumer's own optimization level, target features and contraction setting are
    # the consumer's to choose. An earlier architecture kept the entries out of line and the
    # policy spellings out of the front door, and the owner ruled it out - the consumer compiles,
    # and their flags are their choice - so the front door now reaches the kernel definitions and
    # the gate below measures the new contract rather than stating it:
    #
    #  1. The front door alone carries a default call through compile AND link, with no archive on
    #     the link line. That is the header-only consumer's first requirement, and it takes the
    #     link step to measure: a header that declares an entry without defining it compiles, and
    #     the failure arrives afterwards, where this probe looks for it.
    #
    #  2. A call that NAMES a policy compiles from boys/kernels.hpp. That is the ambitious
    #     consumer's requirement, and the control for the first probe: a probe that failed for its
    #     own reasons - a typo, a missing include path, a standard the compiler was never handed -
    #     fails here too, so a refusal is read from the pair rather than from one measurement.
    #
    # Two statements the old comment made here are not made again, because the ruling moved their
    # premise: the entries are no longer values the library froze under its own flags, and the
    # seam no longer decides what links - an inline entry is satisfied by the translation unit
    # that reads it. The two registrations below are re-scoped to that contract: the flag check
    # pins one digest across the four profiles, the seam check each probe's own seam and rows.
    if (BOYS_BUILD_TESTS)
        unset(BOYS_FRONT_DOOR_COMPILES_A_DEFAULT_CALL CACHE)
        unset(BOYS_KERNELS_HEADER_COMPILES_A_NAMED_CALL CACHE)
        set(_boys_gate_type "${CMAKE_TRY_COMPILE_TARGET_TYPE}")
        set(CMAKE_TRY_COMPILE_TARGET_TYPE EXECUTABLE)
        set(CMAKE_REQUIRED_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/include")
        set(_boys_gate_std "-std=c++20")
        if (MSVC)
            set(_boys_gate_std "/std:c++20 /EHsc")
        endif()

        set(CMAKE_REQUIRED_FLAGS "${_boys_gate_std}")
        check_cxx_source_compiles("
#include <boys/boys.hpp>
int main() {
    double out[33] = {};
    boys::BoysAllOrders(0, 1.0, out);
    return out[0] == 0.0;
}
" BOYS_FRONT_DOOR_COMPILES_A_DEFAULT_CALL)

        set(CMAKE_REQUIRED_FLAGS "${_boys_gate_std}")
        check_cxx_source_compiles("
#include <boys/kernels.hpp>
int main() {
    double out[33] = {};
    boys::BoysAllOrders<boys::SeamPolicy<>>(0, 1.0, out);
    return out[0] == 0.0;
}
" BOYS_KERNELS_HEADER_COMPILES_A_NAMED_CALL)

        # A failed check reads as the empty string and a passed one as 1, so the two findings are
        # spelled as words before the message: a failure a reader has to decode from a blank is a
        # failure reported badly.
        if(BOYS_FRONT_DOOR_COMPILES_A_DEFAULT_CALL)
            set(_boys_front_door "true")
        else()
            set(_boys_front_door "false")
        endif()
        if(BOYS_KERNELS_HEADER_COMPILES_A_NAMED_CALL)
            set(_boys_kernels "true")
        else()
            set(_boys_kernels "false")
        endif()
        if(NOT BOYS_FRONT_DOOR_COMPILES_A_DEFAULT_CALL
                OR NOT BOYS_KERNELS_HEADER_COMPILES_A_NAMED_CALL)
            message(FATAL_ERROR
                "boys: a consumer that includes only boys/boys.hpp must compile AND link a "
                "default call with no archive, and a consumer that includes boys/kernels.hpp "
                "must compile a call that names a policy. Building boys::BoysAllOrders(0, 1.0, "
                "out) from boys/boys.hpp alone returned ${_boys_front_door}; building the "
                "named-policy call from boys/kernels.hpp returned ${_boys_kernels}. Both must be "
                "true: the default entries are the consumer's to instantiate, so nothing a "
                "consumer calls may be left for the archive to define.")
        endif()
        unset(_boys_front_door)
        unset(_boys_kernels)
        message(STATUS "boys: a default call compiles and links from boys/boys.hpp alone, with no "
            "archive on the link line, and a call naming a policy compiles from boys/kernels.hpp, "
            "measured by building both")

        unset(CMAKE_REQUIRED_FLAGS)
        unset(CMAKE_REQUIRED_INCLUDES)
        set(CMAKE_TRY_COMPILE_TARGET_TYPE "${_boys_gate_type}")
        unset(_boys_gate_type)

        if (MSVC)
            message(STATUS "boys: the flag-independence check is not built here — its variants "
                "are -O2, -O3, -O3 -march=native with contraction off, and -O2 -mavx2 -mfma "
                "-mf16c, and none of them is a spell this compiler takes. The entries are the "
                "caller's to compile under the ruling, and the four readings are one arithmetic "
                "source seen through four sets of flags; the check's premise is stated at its "
                "registration below.")
        else()
            find_package(Python3 COMPONENTS Interpreter QUIET)
            set(_boys_flag_variants o2 o3 native)
            if (BOYS_ARCH_X86_64)
                # The fourth reading is the SIMD flag set, and -mavx2 -mfma -mf16c are x86
                # spells: a non-x86 host has no such variant to build rather than a broken one
                # to skip. The checker compares whatever it is given, so three readings are a
                # check.
                list(APPEND _boys_flag_variants avx2)
            endif()
            set(_boys_flag_programs "")
            foreach (variant IN LISTS _boys_flag_variants)
                add_executable(boys-flag-independence-${variant}
                    tests/boys_flag_independence.cpp)
                target_link_libraries(boys-flag-independence-${variant} PRIVATE boys)
                boys_strict_warnings(boys-flag-independence-${variant})
                # The entries are the caller's to compile, so the four variants differ only in
                # what the caller chose and never in contraction: every one carries the
                # build's own -ffp-contract=off (inherited from the target's PUBLIC option, and
                # spelled again where a level is named). A variant that varied contraction
                # would read a different arithmetic and stop being a reading of this check.
                if (variant STREQUAL "o2")
                    # Stated last on the command line, so it replaces the build type's own level
                    # rather than joining it: this variant is the one a caller gets from -O2.
                    set(_boys_variant_flags -O2)
                elseif (variant STREQUAL "o3")
                    # The plain optimizer level a caller gets from -O3, so the two levels read
                    # the same header arithmetic they will compile into their own unit.
                    set(_boys_variant_flags -O3)
                elseif (variant STREQUAL "native")
                    set(_boys_variant_flags -O3 -march=native -ffp-contract=off)
                else()
                    set(_boys_variant_flags -O2 -mavx2 -mfma -mf16c)
                endif()
                # The -march=native variant can trip a clang diagnostic about the HOST CPU's own
                # feature set - measured 2026-10-11 on CI's ubuntu-24.04 runner, "invalid
                # feature combination: +avx10.1-256; will be promoted to avx10.1-512", an error
                # under -Werror. What that says is a property of the runner's processor and of
                # clang's feature table, never of this translation unit, which is what the
                # variant exists to read; the suppression is scoped to that variant for that
                # reason.
                if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND variant STREQUAL "native")
                    list(APPEND _boys_variant_flags -Wno-invalid-feature-combination)
                endif()
                target_compile_options(boys-flag-independence-${variant}
                    PRIVATE ${_boys_variant_flags})
                list(APPEND _boys_flag_programs $<TARGET_FILE:boys-flag-independence-${variant}>)
            endforeach()
            # The reading the four variants are held to: the entries are the caller's to compile,
            # so the four agree because the arithmetic is one source and the ISA tiers are
            # documented bit-equal to the scalar twins. A digest that differs across the variants
            # is a finding to report, never something to weaken the check for; the profile the
            # certified figures belong to is docs/build-facts.md's business rather than this
            # check's.
            if (Python3_FOUND)
                add_test(NAME boys-flag-independence
                    COMMAND ${Python3_EXECUTABLE}
                        "${CMAKE_CURRENT_SOURCE_DIR}/tools/check_flag_independence.py" --check
                        ${_boys_flag_programs})
            else()
                message(STATUS "boys: the flag-independence check is not registered — it reads "
                    "four programs' output, which needs a Python interpreter this configure did "
                    "not find")
            endif()

            # The seam's half, at build time because it is a LINK: a translation unit that reads a
            # seam compiles that seam's rows and says so. The tool compiles and links one probe
            # twice with NO archive on the link line - the headers alone, which is the contract now
            # that the default entries are inline - once against the committed table and once
            # against a replaced one, and each run prints the seam it compiled into, the table it
            # read and a digest of the default entries' bits over fixed points. What the check pins
            # is that each probe prints the seam it was built for, that the two digests differ -
            # the replaced rows moved the values - and that the replaced reading at F_3(1.25) is
            # the value the replaced table prints. The refusal half the old text stated was a
            # property of entries the library held: with the entries the caller's to compile, a
            # translation unit reading another seam satisfies its own call, so that reading is gone
            # and the reading above is the one this check carries.
            #
            # The replacement is the tuned fixture the example-transcript comment below quotes: it
            # is copied into a directory searched before include/ and read under the defines a
            # replaced configure carries, so the second probe really reads it. The value it is held
            # to at F_3(1.25) is the fixture's own move, 0.055476132923077695 against the committed
            # table's quoted 0.055476132923077584. Both seams are the tree's own files, so the
            # reading is the same one on every leg; a build configured with BOYS_BUILD_DEFAULTS
            # replaces neither, and its own seam is read by the suite's fixture pins and by the
            # accuracy gate on that leg rather than here.
            if (Python3_FOUND)
                # The tool compiles and links its probes itself, so it is handed this
                # configuration's own flags: the probes then compile the way this build compiles,
                # and the digests are this configuration's rather than the tool's.
                set(_boys_seam_flags "")
                foreach(_boys_seam_flag_var CMAKE_CXX_FLAGS CMAKE_EXE_LINKER_FLAGS)
                    separate_arguments(_boys_seam_flag_words UNIX_COMMAND
                        "${${_boys_seam_flag_var}}")
                    foreach(_boys_seam_flag_word IN LISTS _boys_seam_flag_words)
                        list(APPEND _boys_seam_flags "--flag=${_boys_seam_flag_word}")
                    endforeach()
                endforeach()
                add_test(NAME boys-seam-identity
                    COMMAND ${Python3_EXECUTABLE}
                        "${CMAKE_CURRENT_SOURCE_DIR}/tools/check_seam_identity.py" --check
                        --cxx "${CMAKE_CXX_COMPILER}"
                        "--std=-std=c++20"
                        "--include=${CMAKE_CURRENT_SOURCE_DIR}/include"
                        --seam-ident seam_committed
                        "--replacement=${CMAKE_CURRENT_SOURCE_DIR}/tests/build_defaults_tuned.hpp"
                        "--replaced-value=0.055476132923077695"
                        ${_boys_seam_flags})
            else()
                message(STATUS "boys: the seam-identity check is not registered — it compiles "
                    "and links two calls, which needs a Python interpreter this configure did not "
                    "find")
            endif()
        endif()
    endif()

    # The generator's split-Clenshaw evaluator against a direct Chebyshev sum:
    # the instrument the region-B Chebyshev route's delivered error is measured
    # with, held at both parities, because an odd degree's top coefficient is
    # the term a mis-seeded odd part drops while still returning a plausible
    # number. It reads the generator's pinned mpmath; a machine without it skips
    # with 77, the generator's own answer to a missing dependency.
    if (Python3_FOUND)
        add_test(NAME boys-generator-clenshaw-split
            COMMAND ${Python3_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/tests/gen_boys_clenshaw_split_test.py")
        set_tests_properties(boys-generator-clenshaw-split PROPERTIES SKIP_RETURN_CODE 77)
    else()
        message(STATUS "boys: the generator's split-Clenshaw test is not registered — it needs a "
            "Python interpreter this configure did not find")
    endif()

    add_test(NAME boys-tests COMMAND boys-tests)
    add_test(NAME boys-boundary-standalone COMMAND boys-boundary-standalone)
    add_test(NAME boys-consumer-umbrella COMMAND boys-consumer-umbrella)
    add_test(NAME boys-consumer-header-only COMMAND boys-consumer-header-only)
    add_test(NAME boys-accuracy-here COMMAND boys-accuracy-here)
    add_test(NAME boys-consumer-defaults COMMAND boys-consumer-defaults)
    add_test(NAME boys-consumer-defaults-match-docs COMMAND boys-consumer-defaults-match-docs)
    add_test(NAME boys-consumer-option-space COMMAND boys-consumer-option-space)
    # --strict: the gate fails on a claim whose evidence is not in the tree as
    # well as on one it measures exceeded, so a green ctest never means "every
    # documented claim is met at this revision" while a claim rest on a
    # measurement this revision cannot re-run. What it scopes over is the
    # claims the build carries: a build that closes the fp16 seam has rows
    # whose subject its sources do not declare at all, and those are reported
    # and named with their reason rather than counted as evidence this revision
    # failed to produce.
    add_test(NAME boys-accuracy-gate COMMAND boys-accuracy-gate --strict)
    # The external-row reading: same binary, its own script; the run must exit non-zero with
    # both rows exceeded.
    if (BOYS_GATE_EXTERNAL_BOOST)
        find_package(Python3 COMPONENTS Interpreter REQUIRED)
        add_test(NAME boys-external-boost-gate
            COMMAND ${Python3_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/tests/external_boost_gate_test.py"
                    $<TARGET_FILE:boys-accuracy-gate>)
    endif()
    set_tests_properties(boys-boundary-standalone PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/tests/data")

    # The examples, run rather than only built: each one already decides for
    # itself whether the number it printed is sane, so a program here that has
    # rotted into printing nonsense fails a leg. boys-example-07-option-probe
    # reports a timing, which is a property of the host and the clock rather
    # than of this revision - so it fails on having measured nothing at all and
    # never on which option came first (see the driver comment below).
    if (BUILD_EXAMPLES)
        foreach (example IN LISTS boysExamples)
            add_test(NAME boys-example-${example} COMMAND boys-example-${example})
        endforeach()

        # The documents quote what these programs print, and the loop above runs them and compares
        # nothing: a change that moves a printed digit used to leave README.md and
        # docs/getting-started.md quoting the old one with every leg green. This test is the
        # reading step - it runs each program and compares what it printed against the block the
        # documents quote, and it fails both on a block that no longer matches and on a block it
        # cannot associate with a program, which is how a check like this goes quiet.
        #
        # The programs come from the targets this build just made, through a generator expression,
        # rather than from a path spelled here - so the test reads this configuration's binaries on
        # whatever generator and platform is running it.
        #
        # Scoped to a build whose defaults are the committed ones. The quoted figures are what the
        # evaluation an unnamed call resolves to prints, so a build that replaced that resolution
        # through BOYS_BUILD_DEFAULTS prints other digits by construction - the tuned fixture moves
        # F_3(1.25) from the quoted 0.055476132923077584 to 0.055476132923077695 - and the claim is
        # not one that tree makes. Registering it there would be a leg that is red for being the
        # configuration it declares itself to be.
        #
        # The same reasoning excludes a build on the separate multiply-add route. The quoted figures
        # are the fused route's by construction, and a separate-route build prints the last digit
        # differently: measured, examples/08_shell_quartet_batch.cpp prints 0.052602850434881915
        # where docs/getting-started.md quotes 0.05260285043488188 - one token on one line, with
        # every other example matching.
        #
        # The delivery kernel is deliberately NOT excluded. An AVX2 build and a build forced onto
        # the scalar path were measured to print byte-identical stdout on all eight comparable
        # examples, so this test holds across the kernels the CI legs actually use, and narrowing it
        # to one kernel would give up that coverage for nothing.
        #
        # What IS excluded is a build whose compiler contracts the multiply-add. The figures are the
        # non-contracting build's, and this is measured rather than assumed: on CI, every Release leg
        # on gcc, clang and macOS failed this test on the last digit of F_3(1.25), while every Debug
        # leg passed and MSVC passed at both - Release contracts, Debug does not, and MSVC does not
        # at these settings. A document cannot be checked against a build it does not describe, so
        # the test runs where the figures came from. `docs/build-facts.md` records the same
        # distinction per leg as `contract.this-tu.fp64`.
        find_package(Python3 COMPONENTS Interpreter QUIET)
        if (Python3_FOUND AND MSVC AND NOT BOYS_BUILD_DEFAULTS AND NOT BOYS_MULADD_SEPARATE)
            list(GET boysExamples 0 boysFirstExample)
            add_test(NAME boys-example-transcripts
                COMMAND ${Python3_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/tools/check_example_transcripts.py" --check
                    --root "${CMAKE_CURRENT_SOURCE_DIR}"
                    --examples-dir "$<TARGET_FILE_DIR:boys-example-${boysFirstExample}>")
        elseif (NOT Python3_FOUND)
            message(STATUS "boys: the example-transcript check is not registered with ctest - "
                "no Python3 interpreter was found, and a check that is not run is not a check")
        endif()
    endif()

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
    # The content the report was taken from, read at configure time: a report read months later
    # is otherwise attributable to nothing, and nothing can hold one to the tree it describes.
    # 9688c97 renamed every option cell and the committed report did not follow, because it
    # named no revision for a check to disagree with. The driver prints the library's own
    # version beside this digest, read from the header it compiles against.
    target_compile_definitions(boys-option-probe PRIVATE
        BoysProbeInputs="${BOYS_PROBE_INPUTS}")
    boys_strict_warnings(boys-option-probe)

    # The defaults command: what this build's unnamed calls resolve to, class by
    # class, read from the table the build carries (see the file preamble of
    # tests/boys_defaults.cpp). Its answer is the same on every machine that
    # compiles the same build — nothing is evaluated and nothing is timed — so
    # unlike the two probes above it is registered as a ctest case as well as
    # built, and a plain `ctest` reads what this configure resolved.
    #
    # The seam file in force is named by the configure and not by the file: a
    # build that replaced the header must be able to say WHICH header produced
    # its choices, and a report that could not would be the defect the seam
    # exists to remove. So the path is a definition with a value this configure
    # owns, and the copy the compiler actually reads is named beside it where
    # there is one (the boys target copies the replacement before include/).
    set(BOYS_DEFAULTS_HEADER_PATH "${CMAKE_CURRENT_SOURCE_DIR}/include/boys/boys_build_defaults.hpp")
    set(BOYS_DEFAULTS_HEADER_COPY "")
    # The file itself, as a path a test can open at run time. It is the same file
    # the sha256 definition was taken of and the same one configure_file copied,
    # carried absolute because ctest runs from a build directory the option's own
    # path is not relative to (CI passes tests/build_defaults_tuned.hpp relative
    # to the source root). The defaults test reads it to hash it: the identity the
    # library reports is the seam file's sha256, and only an independent reading
    # of that file can hold the library to it
    # (tests/boys_build_defaults_test.cpp).
    set(BOYS_DEFAULTS_SEAM_FILE "${CMAKE_CURRENT_SOURCE_DIR}/include/boys/boys_build_defaults.hpp")
    if (BOYS_BUILD_DEFAULTS)
        set(BOYS_DEFAULTS_HEADER_PATH "${BOYS_BUILD_DEFAULTS}")
        set(BOYS_DEFAULTS_HEADER_COPY "${BOYS_BUILD_DEFAULTS_DIR}/boys/boys_build_defaults.hpp")
        set(BOYS_DEFAULTS_SEAM_FILE "${BOYS_BUILD_DEFAULTS_FILE}")
    endif()
    target_compile_definitions(boys-tests PRIVATE
        BOYS_DEFAULTS_SEAM_FILE="${BOYS_DEFAULTS_SEAM_FILE}")

    add_executable(boys-defaults tests/boys_defaults.cpp)
    target_link_libraries(boys-defaults PRIVATE boys)
    target_compile_features(boys-defaults PRIVATE cxx_std_20)
    target_compile_definitions(boys-defaults PRIVATE
        BOYS_DEFAULTS_HEADER_PATH="${BOYS_DEFAULTS_HEADER_PATH}"
        BOYS_DEFAULTS_HEADER_COPY="${BOYS_DEFAULTS_HEADER_COPY}")
    boys_strict_warnings(boys-defaults)

    add_test(NAME boys-defaults COMMAND boys-defaults)

    # The dispatch assertion: each CI leg states what BoysAvx2Available() MUST
    # report for its architecture (1 on x86_64, 0 on arm64) and the probe fails
    # the leg if it disagrees. It is worth more than the rest of the matrix
    # combined, because the SIMD correctness tests soft-skip on that same
    # predicate: a wrong CPUID bit would make every one of them SKIP while CI
    # reported green with the whole vector tier dead — the failure this
    # assertion exists to catch.
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

    # The checked entries' own check (tests/boys_checked_test.cpp): the five
    # double-precision shapes' Boys*Checked overloads against the entries they
    # guard. Each checked call is held to the unchecked entry's own bits, to
    # refusing what the unchecked entry runs with, to writing nothing when it
    # refuses, and to naming the first element a refused batch refused.
    add_executable(boys-checked-tests tests/boys_checked_test.cpp)
    target_link_libraries(boys-checked-tests PRIVATE boys)
    target_compile_features(boys-checked-tests PRIVATE cxx_std_20)
    boys_strict_warnings(boys-checked-tests)

    add_test(NAME boys-checked-tests COMMAND boys-checked-tests)
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
    # The alias every other target here carries, so a consumer's link line reads
    # boys::boys-cuda as the host targets read boys::boys - and so the export
    # below has a namespaced name to hand out like the others.
    add_library(boys::boys-cuda ALIAS boys-cuda)
    # The tier and the measured contraction answer: this target's units link into
    # the same programs as `boys`'s (the gate links both), so it carries the one
    # flag set the program is built at, contract define included.
    # The runtime rides the interface with the include path that reads it: a host-only
    # consumer links this archive and needs cudart's symbols and headers, and the
    # installed package answers both through the component's CUDAToolkit.
    target_link_libraries(boys-cuda PUBLIC boys-tier CUDA::cudart)
    # The CUDA driver targets below compile C++ translation units that include
    # CUDA headers (tests/boys_cuda_test.cpp, tests/boys_cuda_accuracy_gate.cpp
    # and the default check's host half). nvcc finds its own headers wherever the
    # toolkit lives; the host compiler does not, and a toolkit installed by
    # NVIDIA's own installer under /usr/local/cuda is not on its default search
    # path, so the toolkit's include directories - which CMake reports once the
    # CUDA language above is enabled - are named here, PUBLIC, because what links
    # this library is what reads them.
    # Wrapped, so this target exports like the host two do: the interface a package
    # consumer reads spells include/ (their own install), the toolkit's directories
    # are this configure's (an installed consumer's come from find_dependency), and
    # src/ is private and travels nowhere. Each toolkit directory is wrapped on its
    # own: one generator expression around the list drops the first entry's leading
    # slash and resolves it against this source directory instead.
    set(_boysCudaToolkitIncludes "")
    foreach (_boysCudaToolkitDir IN LISTS CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
        list(APPEND _boysCudaToolkitIncludes "$<BUILD_INTERFACE:${_boysCudaToolkitDir}>")
    endforeach()
    target_include_directories(boys-cuda
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
            ${_boysCudaToolkitIncludes}
        PRIVATE src)
    unset(_boysCudaToolkitIncludes)
    if (BOYS_BUILD_DEFAULTS_DIR)
        # A build's defaults are one seam and not a host half: the device lane's
        # rows are read from the same list (boys_cuda_options.hpp) and the device
        # lane names two choices of its own - the division form its entries take
        # and the region-B exponential its tables read (kDefaultDeviceDivisionForm,
        # kDefaultRegionBExp) - so a replacement carries all seven names, the
        # host's five and the device lane's two. One that leaves a device name out
        # is a compile error at that name and not a device default taken silently
        # from the committed file.
        #
        # The seam's identity travels with it, and it is the same digest the boys
        # target carries: the device lane reads the file the option named exactly
        # as the host does, so a build whose two lanes reported different seams
        # would be a build that had two. PUBLIC for the reason the definition
        # above is PUBLIC - a consumer's own .cu reaches the headers either way,
        # and boys_cuda.hpp includes boys.hpp, so both lanes are answered by the
        # one accessor (include/boys/version.hpp, BuildDefaultsSeamIdentity()).
        target_include_directories(boys-cuda BEFORE PUBLIC "$<BUILD_INTERFACE:${BOYS_BUILD_DEFAULTS_DIR}>")
        target_compile_definitions(boys-cuda PUBLIC
            BOYS_BUILD_DEFAULTS_REPLACED=1
            BOYS_BUILD_DEFAULTS_SEAM_SHA256="${BOYS_BUILD_DEFAULTS_SEAM_SHA256}"
            BOYS_BUILD_DEFAULTS_SEAM_IDENT=${BOYS_BUILD_DEFAULTS_SEAM_IDENT})
    endif()
    # The fp16 seam and the C++20 floor are carried by the tier target this one links, as the tier
    # and the contraction answer are: they are the program's facts, and this lane is one of its
    # units.
    # The multiply-add route is one build fact with two lanes behind it, and
    # this is the device's half: the kernels' translation unit reads the same
    # selection the host's does, so a build that selected the separate route
    # cannot leave the device arithmetic fused and report the host's route over
    # it. PUBLIC for the reason the boys target states at the option: every
    # device entry is header-defined, so a consumer's own .cu compiles the
    # arithmetic the library reports, and the batch kernels here compile it with
    # it. tests/boys_cuda_route_test.cu is what holds the two to each other.
    if (BOYS_MULADD_SEPARATE)
        target_compile_definitions(boys-cuda PUBLIC BOYS_MULADD_SEPARATE=1)
    endif()
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

        # The NEGATIVE CONTROL for the division-form check, and the same source
        # file: BOYS_CUDA_TEST_DROP_DIVISION_FORM makes every launch run
        # kExactDivision whatever form it asked for, so the check that demands the
        # plain reciprocal differ from exact division somewhere has no difference
        # to find. The control is registered on what that run PRINTS rather than on
        # its exit status, because the whole point of the fixture is that the check
        # fails: a run that matches the line below is a run in which the check was
        # shown to be falsifiable, and a crash or a hang matches nothing.
        add_executable(boys-cuda-tests-divform-control tests/boys_cuda_test.cpp)
        target_compile_definitions(boys-cuda-tests-divform-control
            PRIVATE BOYS_CUDA_TEST_DROP_DIVISION_FORM)
        target_link_libraries(boys-cuda-tests-divform-control PRIVATE boys-cuda boys
            gtest gtest_main)
        target_include_directories(boys-cuda-tests-divform-control PRIVATE src)
        boys_strict_warnings(boys-cuda-tests-divform-control)
        add_test(NAME boys-cuda-tests-divform-control
            COMMAND boys-cuda-tests-divform-control
                --gtest_filter=BoysCudaTest.DivisionFormsDifferByBits)
        set_tests_properties(boys-cuda-tests-divform-control PROPERTIES
            PASS_REGULAR_EXPRESSION "exact vs plain 0 of")

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

        # This suite holds two long poles and a timeout that reflects them, so a
        # hung run fails here instead of hanging. The tie test hunts a tie and
        # costs 52 minutes measured green on this card, 75 to 85 by the model
        # the file's own preamble states, and past 107 minutes on a machine with
        # other work on it; the refusal test was 35.8 minutes measured, and the
        # rest of the suite is about fifteen minutes. The four hours below are about 1.5x the upper end of
        # that, so the suite is not failed for running on a host other lanes are
        # using, and a run that has genuinely stopped is ended.
        #
        # The linux-cuda leg compiles this lane and runs none of it - it has no
        # device - so this is still a local-robustness bound and not a CI one.
        set_tests_properties(boys-cuda-probe-tests PROPERTIES TIMEOUT 14400)

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
            BoysDataDir="${CMAKE_CURRENT_SOURCE_DIR}/tests/data")
        boys_strict_warnings(boys-cuda-accuracy-gate)
        add_test(NAME boys-cuda-accuracy-gate COMMAND boys-cuda-accuracy-gate)

        # The axis-sensitivity check built and NOT registered as a test: for every row of
        # BoysDeviceOptions(), does the entry the row names read the axes the row states? It
        # reports the members of each axis the row carries more than one of that deliver
        # identical values - the shape a row whose entry ignores a stated axis has. It is built
        # so the tree compiles these two sources and so the pre-commit hook can check them; it
        # is not registered because it currently reports 120 cells on the scheme axis, every one
        # of them a pair this library documents as one arithmetic under two names, so a test
        # here would pin a red leg for a reason that is not the defect it exists to find. The
        # registration waits on the scheme axis being settled. A device is needed to run it, as
        # for the gate above, and `--collapse=<axis>` is its deliberate-defect control.
        add_executable(boys-cuda-axis-sensitivity
            tests/boys_cuda_axis_sensitivity.cpp
            tests/boys_cuda_axis_sensitivity_device.cu)
        target_link_libraries(boys-cuda-axis-sensitivity PRIVATE boys-cuda boys)
        target_include_directories(boys-cuda-axis-sensitivity PRIVATE src)
        target_compile_features(boys-cuda-axis-sensitivity PRIVATE cxx_std_23)
        boys_strict_warnings(boys-cuda-axis-sensitivity)

        # The CUDA consumer's own check of the lane's default, in two
        # translation units because the lane's headers split that way: the host
        # half includes <boys/boys_cuda.hpp> - the batch entry's home - and the
        # kernel half includes <boys/boys_cuda_device.hpp>, which is what a .cu
        # may include. Each half holds its entry's default to the name it is
        # documented under at compile time and runs both spellings of that entry
        # on the card, printing the cells compared and the worst difference
        # between them. It is a consumer rather than a test for the same reason
        # tests/boys_cuda_device_demo.cu is: only a caller of the public headers
        # can show what a caller gets.
        add_executable(boys-consumer-cuda-defaults
            tests/consumer_cuda_defaults_host.cpp
            tests/consumer_cuda_defaults.cu)
        target_link_libraries(boys-consumer-cuda-defaults PRIVATE boys-cuda boys)
        target_compile_features(boys-consumer-cuda-defaults PRIVATE cxx_std_20)
        boys_strict_warnings(boys-consumer-cuda-defaults)
        add_test(NAME boys-consumer-cuda-defaults COMMAND boys-consumer-cuda-defaults)

        # The multiply-add route's own check: the lane's arithmetic at both
        # routes and at the build's selection, held to the exact references a
        # host can state - the fused route to std::fma, the separate route to
        # round(round(a*b) + c) - and the build's selection held to the route
        # the lane reports. It is the one place the device's route is a measured
        # result rather than a sentence in a header, and the build that
        # -DBOYS_MULADD_SEPARATE=ON selects is the one where it can fail: there
        # the delivered arithmetic must be the two-rounding one, and a lane that
        # ran the fused step anyway reports the mismatch rather than passing.
        add_executable(boys-cuda-route-tests
            tests/boys_cuda_route_test.cpp
            tests/boys_cuda_route_test.cu)
        target_link_libraries(boys-cuda-route-tests PRIVATE boys-cuda boys)
        target_compile_features(boys-cuda-route-tests PRIVATE cxx_std_20)
        boys_strict_warnings(boys-cuda-route-tests)
        add_test(NAME boys-cuda-route-tests COMMAND boys-cuda-route-tests)

        # The refusal's reason, the device arm: the same source the suite builds as
        # tests/boys_refusal_reason_test.cpp, with the arm that walks
        # BoysDeviceOptions() compiled in. This configuration and no other is where
        # the device option space can be held to its own field comment - the rows
        # are defined by the lane (src/boys_cuda.cpp) - so the arm is stated here
        # rather than guarded in a header. What it needs beyond the sources is what
        # boys-cuda already carries, and it needs no device to run: it reads the
        # lane's own table and the host query, so a machine that can build this lane
        # can run this check.
        add_executable(boys-cuda-refusal-reason-tests
            tests/boys_refusal_reason_test.cpp)
        target_link_libraries(boys-cuda-refusal-reason-tests PRIVATE boys-cuda boys
            gtest gtest_main)
        target_compile_features(boys-cuda-refusal-reason-tests PRIVATE cxx_std_20)
        target_compile_definitions(boys-cuda-refusal-reason-tests
            PRIVATE BOYS_REFUSAL_REASON_DEVICE=1)
        boys_strict_warnings(boys-cuda-refusal-reason-tests)
        add_test(NAME boys-cuda-refusal-reason-tests
            COMMAND boys-cuda-refusal-reason-tests)
    endif()
endif()

# --- Benchmark drivers -------------------------------------------------------
if (BUILD_BENCHMARKS)
    # boys_benchmark.cpp is a Google Benchmark driver; boys_unsorted_simd_
    # benchmark.cpp is a driver of its own with its own main(). The
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
    target_compile_features(boys-benchmark PRIVATE cxx_std_23)
    boys_strict_warnings(boys-benchmark)

    # The batch entry's own driver: it reports the entry against the
    # per-argument loop and separates the sort's cost from the win, with
    # ratios as the result and a protocol that survives a busy machine (see
    # the file preamble). Portable: it calls no intrinsics of its own.
    add_executable(boys-all-n-benchmark benchmarks/boys_all_n_benchmark.cpp)
    target_link_libraries(boys-all-n-benchmark PRIVATE boys)
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
    target_compile_features(boys-across-orders-benchmark PRIVATE cxx_std_23)
    boys_strict_warnings(boys-across-orders-benchmark)

    # The region-A transform lane's driver: one row per (arithmetic mode, band)
    # over one argument set, the mode names read from the library
    # (BoysProductModes()) rather than written out here. Portable: it calls the
    # public header and no intrinsic of its own. This is the entry's cost
    # harness; the accuracy gate is what reads the same entry for accuracy.
    add_executable(boys-transform-benchmark benchmarks/boys_transform_benchmark.cpp)
    target_link_libraries(boys-transform-benchmark PRIVATE boys)
    target_compile_features(boys-transform-benchmark PRIVATE cxx_std_23)
    boys_strict_warnings(boys-transform-benchmark)

    # The AVX2 timing harness is x86_64-only, like the tier it measures. It is
    # a local-only target (never a CI leg), so on a non-x86_64 host it is
    # reported and skipped rather than half-built: the measured numbers would
    # be the scalar lanes' under a SIMD name.
    if (BOYS_ARCH_X86_64)
        add_executable(boys-unsorted-simd-benchmark
            benchmarks/boys_unsorted_simd_benchmark.cpp)
        target_link_libraries(boys-unsorted-simd-benchmark PRIVATE boys)
        target_include_directories(boys-unsorted-simd-benchmark PRIVATE src)
        target_compile_features(boys-unsorted-simd-benchmark PRIVATE cxx_std_23)
        boys_strict_warnings(boys-unsorted-simd-benchmark)
        # The mixed per-vector kernel compiles AVX2 intrinsics in its own TU -
        # this file's source together with the shipped src/boys_simd.cpp it
        # includes - so this flag set is what the shipped kernel's code is
        # compiled with here, as it is for the library's own copy of that unit.
        if (MSVC)
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
        boys_strict_warnings(boys-cuda-benchmark)

        # The device option probe's driver: argument parsing and printing only.
        # The measurement is boys::RunDeviceOptionProbe in the library, which is
        # the surface a consumer calls; this tool exists so a person can get the
        # same report at a terminal without writing a program. Local-only by
        # construction: it needs the device.
        #
        # It links the host library beside the device one because the report it
        # prints reads the host's own tables: the division forms a device row is
        # the member of are boys::BoysDivisionForms() (boys/boys.hpp), and a tool
        # that printed the space without them would be printing a projection of
        # this library rather than this library's report.
        add_executable(boys-device-probe benchmarks/boys_device_probe.cpp)
        target_link_libraries(boys-device-probe PRIVATE boys-cuda boys)
        target_compile_features(boys-device-probe PRIVATE cxx_std_20)
        boys_strict_warnings(boys-device-probe)
    endif()
elseif (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/benchmark/CMakeLists.txt")
    message(WARNING "third_party/benchmark is missing; BUILD_BENCHMARKS is OFF, "
        "so the benchmark drivers are not built (informational)")
endif()

# --- Examples: a consumer's first call, one question per file ---------------
# Each file answers one question a new caller asks, in the order they ask it,
# and prints the numbers that answer it. They are built here and REGISTERED
# WITH CTEST above (in the test section, where enable_testing() is in scope),
# so an example that stops compiling, or stops printing a number, fails a
# build instead of quietly teaching a call that no longer works. The README
# and docs/getting-started.md quote what these programs print and nothing
# else, which is what keeps a documented figure reproducible.
#
# They link boys::boys and nothing else, and take no private include
# directory: an example is the consumer's view, and it should break here if
# that view ever stops being reachable from the public surface.
if (BUILD_EXAMPLES)
    foreach (example IN LISTS boysExamples)
        add_executable(boys-example-${example} examples/${example}.cpp)
        target_link_libraries(boys-example-${example} PRIVATE boys::boys)
        boys_strict_warnings(boys-example-${example})
    endforeach()
elseif (NOT BOYS_TOP_LEVEL)
    message(STATUS "boys: examples are not built (BUILD_EXAMPLES is OFF)")
endif()

# --- Generator consistency gate (top-level convenience targets) -------------
# --check holds the committed tables to the record of the derivation that wrote
# them (tests/data/boys_coefficients_provenance.txt) and to their own read-back
# rules, and re-derives both artefacts into a temp and byte-compares them
# against the committed files whenever that record's generator, pins or
# environment does not match the tree (default paths: include/boys/
# boys_coefficients.hpp and tests/data/boys_reference.csv — spelled out here and
# in CI). The generator is deterministic (mpmath pinned in
# requirements-boys.txt), so the byte comparison over the committed files is the
# identity proof and the record is what says when it still applies. A
# regeneration writes the committed files and is local-only, which CONTRIBUTING.md
# states as a rule; CI runs the checks and never the writes.
if (BOYS_TOP_LEVEL)
    find_package(Python3 COMPONENTS Interpreter)
    if (Python3_FOUND)
        add_custom_target(boys-generator-check
            COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_boys_coefficients.py --check
                --header ${CMAKE_CURRENT_SOURCE_DIR}/include/boys/boys_coefficients.hpp
                --reference ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/boys_reference.csv
            COMMENT "Boys table/reference consistency check (needs mpmath)"
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

# --- Install: the archive, its headers, and the package a consumer finds -----
# Top-level only: a consumer that adds this tree already has the targets.
#
#     find_package(boys REQUIRED)
#     target_link_libraries(app PRIVATE boys::boys)
#
# The headers, the compile definitions and the C++20 requirement all arrive
# through that link, and boysConfig.cmake resolves its own location, so any
# prefix works. The version file carries project(boys VERSION ...) with the
# compatibility include/boys/version.hpp states.
#
# BOYS_BUILD_DEFAULTS is not installed: its replacement header is a build-tree
# seam, and the definition it publishes makes a package built with one refuse
# to compile rather than resolve other defaults silently.
if (BOYS_TOP_LEVEL)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)

    set(boysInstallCMakeDir "${CMAKE_INSTALL_LIBDIR}/cmake/boys")

    install(TARGETS boys boys-tier
        EXPORT boysTargets
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")

    # The device lane ships in the same package when it was built: a consumer
    # whose configure chose the CUDA path reaches it through the same
    # find_package call, under the component that asks for a toolkit.
    if (BUILD_CUDA)
        install(TARGETS boys-cuda
            EXPORT boysTargets
            ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
            LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
            RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
    endif()

    # The whole directory, not a pattern list: every header under include/boys is
    # public, and a list would silently drop the next one it was not told about.
    # `include` is spelled, not taken from CMAKE_INSTALL_INCLUDEDIR, so the
    # destination cannot disagree with the interface the target advertises.
    install(DIRECTORY include/boys
        DESTINATION include)

    # After the directory, so it replaces the committed seam with the one this
    # build was compiled against: a consumer of this package reads the choices
    # the library in it was built from, which is what the build facts beside the
    # headers already state. Installing the committed seam instead would hand a
    # consumer a header naming seven choices no object file in the package obeys.
    if (BOYS_BUILD_DEFAULTS)
        install(FILES "${BOYS_BUILD_DEFAULTS_DIR}/boys/boys_build_defaults.hpp"
            DESTINATION include/boys)
    endif()

    install(EXPORT boysTargets
        FILE boysTargets.cmake
        NAMESPACE boys::
        DESTINATION "${boysInstallCMakeDir}")

    # Read into the config by configure_package_config_file below: whether this package
    # carries the device component its find_package answers for.
    if (BUILD_CUDA)
        set(BOYS_CONFIG_HAS_CUDA TRUE)
    else()
        set(BOYS_CONFIG_HAS_CUDA FALSE)
    endif()

    configure_package_config_file(cmake/boysConfig.cmake.in
        "${CMAKE_CURRENT_BINARY_DIR}/boysConfig.cmake"
        INSTALL_DESTINATION "${boysInstallCMakeDir}")

    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/boysConfigVersion.cmake"
        VERSION "${PROJECT_VERSION}"
        COMPATIBILITY SameMajorVersion)

    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/boysConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/boysConfigVersion.cmake"
        DESTINATION "${boysInstallCMakeDir}")
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
