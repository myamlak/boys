# The Boys function kernel

[![CI](https://github.com/myamlak/boys/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/myamlak/boys/actions/workflows/ci.yml)
[![docs](https://github.com/myamlak/boys/actions/workflows/docs.yml/badge.svg)](https://github.com/myamlak/boys/actions/workflows/docs.yml)
[![license](https://img.shields.io/badge/license-BSD--3--Clause-blue)](LICENSE)
[![version](https://img.shields.io/github/v/tag/myamlak/boys?label=latest%20release%20tag)](https://github.com/myamlak/boys/tags)

F_n(x) = ∫₀¹ t^(2n) exp(−x t²) dt, for n = 0..32 and x ≥ 0 — the integral an electronic-structure
program evaluates at every order a shell quartet can ask for. This is a C++ library that evaluates
it, in double and single precision, on the CPU and optionally on a GPU. It depends on nothing outside
the C++ standard library and the optional CUDA toolkit.

This tree is **version 5.0.1**. The number is written down once, in the top-level `CMakeLists.txt`'s
`project(boys VERSION ...)` line, and `boys/version.hpp` carries it to a program built without
CMake: `boys::VersionString()` answers it.

## Your first call

Copy this into `try.cpp`, build it, run it. It prints the nine numbers below. Those numbers were
produced by a build that does not contract the multiply-add: a product and a sum stay two roundings,
not one. A build that does contract it evaluates different arithmetic and returns different bits;
`docs/lane-contract.md` prints the measured figure for each route. `docs/build-facts.md` records, per
CI leg, which arithmetic that leg compiles, and `docs/specification.md` says which calls a consumer's
own flags reach.

```cpp
#include <boys/boys_span.hpp>

#include <array>
#include <cstdio>

int main()
{
    // One order at one argument.
    std::printf("F_3(1.25)  = %.17g\n", boys::BoysSingle(3, 1.25));

    // Every order 0..6 at one argument. A shell quartet wants the whole
    // ladder, and this is the call that hands it over.
    std::array<double, boys::kMaxBoysOrder + 1> ladder{};
    boys::BoysAllOrders(6, 3.5, ladder);
    std::printf("F_0(3.5)   = %.17g\n", ladder[0]);
    std::printf("F_3(3.5)   = %.17g\n", ladder[3]);
    std::printf("F_6(3.5)   = %.17g\n", ladder[6]);

    // One order over an array of arguments. out[i] = F_2(x[i]).
    const std::array<double, 3> x{0.25, 4.0, 30.0};
    std::array<double, 3> out{};
    boys::BoysFixedN(2, x, out);
    std::printf("F_2(0.25)  = %.17g\n", out[0]);
    std::printf("F_2(4)     = %.17g\n", out[1]);
    std::printf("F_2(30)    = %.17g\n", out[2]);

    // Every order over an array of arguments - the batch shape an integral
    // engine actually calls, and the one entry that groups the arguments once
    // for the whole call rather than a ladder at a time. The result is
    // order-major: out[k * count + i] is F_k(x[i]), one plane per order.
    const std::array<double, 4> batch{0.0, 0.25, 4.0, 30.0};
    constexpr int kBatchTop = 6;
    std::array<double, batch.size() * (kBatchTop + 1)> grid{};
    boys::BoysAllN(kBatchTop, batch, grid);
    std::printf("F_0(30)    = %.17g\n", grid[0 * batch.size() + 3]);
    std::printf("F_6(30)    = %.17g\n", grid[6 * batch.size() + 3]);

    // F is positive and falls off with x; a batch of zeros or a negative
    // value would mean the call did not do what it says.
    const bool sane = out[0] > 0.0 && out[0] > out[1] && out[1] > out[2] && ladder[0] > ladder[6] &&
                      grid[0 * batch.size() + 3] > grid[6 * batch.size() + 3];
    if (!sane)
    {
        std::printf("FAIL: F is not positive and decreasing in x\n");
        return 1;
    }
    return 0;
}
```

`boys/boys_span.hpp` is the one header a C++ caller needs beyond `boys/boys.hpp`: it adds an
overload taking a `std::span` of each of the double lane's four array entries, so a container goes
straight in. Each overload compares the extents it was handed against what the call writes and
refuses a span that is short — a line naming the call and both extents, then an abort — and
`BoysAllNAtOrders` refuses a top-order span that does not run with the arguments the same way. What
the span spelling costs is that comparison, and what it buys is that a buffer too small is refused
rather than written past; a bare pointer carries no size, so the pointer entries cannot check, and a
call passing a pointer and a count still reaches the pointer entry exactly.

What a translation unit needs, by what it is for:

| A unit that ... | Includes | What that brings |
|---|---|---|
| declares a wrapper, names a type, or holds an entry as a value | `boys/boys_decl.hpp` | every declaration - the types, the entries, the accuracy queries, the table a report reads - and no arithmetic |
| calls an entry the default table answers for | `boys/boys.hpp` | the declarations, plus each default entry's own body inline at the row this build's table carries for its class: nothing else is needed, and no archive on the link line |
| names a policy, `BoysAllOrders<Policy>` rather than `BoysAllOrders` | `boys/boys.hpp` | the same header; the flags the unit is compiled with choose the ISA tier, and no flags is the scalar tier. One flag set per program is the contract the headers state |
| wants everything this library carries | `boys/boys_all.hpp` | the front door, plus the `std::span` overloads, the region-A transform lane and the option probe |
| is C | `boys/boys_c.h` | the C linkage surface; it is built in the archive, so link it |
| is CUDA | `boys/boys_cuda.hpp`, `boys/boys_cuda_device.hpp` | device calls that upload their tables on first use, or the same arithmetic written into your own kernel |

The archive is still needed for two things, and neither is computing F. One is the **reporting
queries** - the functions that answer what this build carries and what it guarantees: the fit routes,
the interval granularities, the evaluation schemes and their names, the division forms, the packing
axes, the far-region representations, the region-B exponentials, the lane contracts, the two accuracy
figures, and the tolerance query. They are declared in `boys/boys_decl.hpp` and defined in the
archive, so a translation unit that calls one links it. The other is the C ABI surface,
`boys/boys_c.h`. Everything else a call site names compiles from the headers.

**The measured figures do not need the archive.** `boys/boys_accuracy_here.hpp` is a header of its
own — not in `boys/boys.hpp`'s chain — that states each lane's published absolute figure and the
figure this build's arithmetic was measured to deliver, read at compile time from the macros the
build pins. It is **not the accuracy contract**: the contract is `BoysAccuracyGuaranteed`'s answer
at the `(n, x)` a calculation asks about — `max(1e-13·F_n(x), 1e-13·F_0(x))` on the double lane —
and that accessor is one of the archive's. Read the header as what this build measured, and the
accessor as what a calculation is promised.

    git clone https://github.com/myamlak/boys.git && cd boys
    cmake -S . -B build -DBOYS_BUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF
    cmake --build build --config Release

or, against a tree you have already built:

    c++ -std=c++20 -DBoysFp16=1 -mavx2 -mfma -mf16c -ffp-contract=off \
        -I include try.cpp -L build -lboys -o try     # -I and -L as your build laid them out
    ./try

On Windows with MSVC, from a Developer Command Prompt or a shell that has run `vcvars64.bat`, the
same line is spelled with its own switches. `/arch:AVX2` is the whole tier there — MSVC takes F16C
with it — and no contraction pin rides beside it, because MSVC fuses only under `/fp:fast`, which
this build does not use:

    cl /nologo /std:c++20 /EHsc /MD /DBoysFp16=1 /arch:AVX2 /I include try.cpp
    .\try.exe

`/MD` is the runtime the CMake build uses, and `cl`'s own default is `/MT`: a program that links
the archive — `build\Release\boys.lib` on the command line — and pulls an object out of it without
`/MD` stops at `LNK2038: mismatch detected for 'RuntimeLibrary'`. This first program references
nothing in the archive and does not need it on the link line at all, so it builds and runs either
way.

`-DBoysFp16=1` and the tier flags are the ones the recipe above put on the library, and
`docs/build-facts.md` states the profile they spell. Every entry is compiled where it is called,
so a unit that leaves either out builds a program the headers refuse at start-up in a program that
links the archive. The refusal is a disagreement between two announcements and needs both: the
archive's own units are the second one, and a program built from the headers alone makes the one
announcement and runs.

```
F_3(1.25)  = 0.055476132923076911
F_0(3.5)   = 0.46984703520162335
F_3(3.5)   = 0.011831383583716588
F_6(3.5)   = 0.0040954447623731041
F_2(0.25)  = 0.16753319090734875
F_2(4)     = 0.017525782161992999
F_2(30)    = 0.00013483513281472914
F_0(30)    = 0.16180215937964007
F_6(30)    = 3.6049669032736188e-08
```

The three `OFF`s are there so this builds the library and nothing else — the option probe and the C
linkage surface are units of the archive itself and not targets beside it — which needs no compiler
past C++20: the suite and the benchmark drivers are the targets that want C++23, and each is behind
one of those flags. Drop them to build all three as well; that is the contributor's build. This
build carries no test target, so the accuracy gate and the option probe are not in it:
`cmake -S . -B build -DBOYS_BUILD_TESTS=ON` on the directory you have is what adds them, and *Check
the figures yourself* in [the specification](docs/specification.md#accuracy-contract) gives the
command that then runs the gate.

On Windows with the Visual Studio generator the library lands in `build/Release/`. This program is
[`examples/00_first_call.cpp`](examples/00_first_call.cpp); the contributor's build compiles it and
`ctest` runs it, along with every other program this README and the guide quote, so the transcript
above cannot rot silently.

**[docs/getting-started.md](docs/getting-started.md)** takes these four calls one at a time, then
the questions that follow them: how to name a specific evaluation, how to ask what the library
guarantees before you rely on it, and how to find out which of the available options is fastest on
your machine. Every figure in it is the
output of a program in [`examples/`](examples/), so you can reproduce any of them.

## Consuming as a submodule

As a submodule, this tree builds the library and nothing else: the test suite, the examples and the
benchmark drivers are OFF unless this tree is the top-level project, and so are the archive's two
optional units - the C linkage surface and the option probe.

    git submodule add https://github.com/myamlak/boys external/boys

```cmake
add_subdirectory(external/boys boys)
target_link_libraries(app PRIVATE boys::boys)
```

`boys::boys` hands your translation units everything the library was built with: the include
directory, C++20, the tier flags, the contraction answer and the fp16 seam. For the calculations
alone, with no archive on the link line, link `boys::boys-tier` instead - the same interface, and
the entries' bodies come inline from the headers.

Installed, from a top-level build:

    cmake --install build --prefix /where/you/want/it     # or set the prefix at configure time

That writes the headers to `<prefix>/include/boys`, the archive to `<prefix>/lib`, and the package
below to `<prefix>/lib/cmake/boys`, as `boysConfig.cmake` and its version file beside
`boysTargets.cmake`. It exports `boys::boys` and `boys::boys-tier`, and `boys::boys-cuda` when the
build carried the device lane; the version file accepts any `5.x` package for a request of `5.0.1`.
The install rules are written at the top level only, so a consumer that adds this tree as a
subdirectory already has the targets and needs no package:

```cmake
find_package(boys REQUIRED)
target_link_libraries(app PRIVATE boys::boys)     # or boys::boys-tier, for the headers alone
```

With the test suite on, the vendored GoogleTest's own install rules run as part of the top-level
install: its headers and its four libraries land in the same prefix, and an install from a directory
whose suite targets were never built stops on `gmock.lib`. Build the tree you install, or configure
with `-DBOYS_BUILD_TESTS=OFF` as the first recipe does.

Each optional, named one at a time:

    cmake -S . -B build -DBOYS_ISA_AVX2=OFF                          # the portable scalar tier
    cmake -S . -B build -DBOYS_FP16_SEAM=OFF -DBOYS_BUILD_TESTS=OFF  # no half/bfloat entries
    cmake -S . -B build -DBOYS_MULADD_SEPARATE=ON                    # the two-rounding multiply-add route
    cmake -S . -B build -DBOYS_BUILD_C_API=ON                        # the C linkage surface, in the archive
    cmake -S . -B build -DBOYS_BUILD_PROBE=ON                        # the option probe, in the archive
    cmake -S . -B build -DBOYS_BUILD_DEFAULTS=my_seam.hpp            # your own default choices
    cmake -S . -B build -DBOYS_WERROR=OFF                            # drop -Werror
    cmake -S . -B build -DBUILD_CUDA=ON                              # the device lane
    cmake -S . -B build -DBOYS_BUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF

`BOYS_FP16_SEAM=OFF` closes the seam across the whole tree, which is why its line drops the test
suite too: the suite's accuracy gate names the half entries through a policy on an unguarded path,
so a configure that asks for both is refused rather than half-built. In a submodule, the same
option is a cache variable the parent sets before `add_subdirectory`.

    cmake --build build --target boys-option-probe     # needs the test suite (-DBOYS_BUILD_TESTS=ON); see the guide

Outside CMake the flags are yours to spell, and `docs/build-facts.md` states the profile the
figures belong to. A unit compiled with the other reading of the fp16 seam or of the tier builds a
program the headers refuse at start-up in a program that links the archive, and one that runs
against the headers alone.

    c++ -std=c++20 -DBoysFp16=1 -I include try.cpp -o try     # the headers alone, no archive

## Which entry do I call?

By what you want, not by what the library calls things:

| I want | Entry | One clause on when |
|---|---|---|
| F_n(x) for one order, one argument | `BoysSingle` | the simplest call; use it when you have one value |
| F_0..F_nmax at one argument | `BoysAllOrders` | a shell quartet's ladder; this is the common case |
| F_n at every argument of an array | `BoysFixedN` | when the order is fixed and the arguments vary; takes an output stride |
| F_0..F_nmax at every argument of an array | `BoysAllN` | the batch shape of an integral engine; the arguments may arrive in any order |
| ...with each argument's own top order | `BoysAllNAtOrders` | a shell-quartet batch, where no argument is padded to a common order |
| ...when you know the arguments are already sorted | `BoysAllN` with `BoysSortedArgs{}`, pointer form only | skips the internal sort when your loop already produces a non-decreasing array |
| ...when the arguments are not known to be in domain | `BoysSingleChecked`, `BoysAllOrdersChecked`, `BoysFixedNChecked`, `BoysAllNChecked`, `BoysAllNAtOrdersChecked` | the checked twin of every double entry above; it refuses with a status rather than answering |
| any of the above in single precision | `BoysSingleF32`, `BoysAllOrdersF32`, `BoysFixedNF32`, `BoysAllNF32`, `BoysAllNAtOrdersF32` | when the rest of your kernel is `float` |
| half-precision storage | `BoysSingleF16`, `BoysAllOrdersF16`, `BoysFixedNF16`, `BoysAllNF16`, `BoysAllNAtOrdersF16`, `BoysSingleBf16`, `BoysAllOrdersBf16` | 16-bit I/O around the single-precision engine |
| an answer on a GPU | `boys/boys_cuda.hpp` | device arrays; uploads its tables on first use, so warm the path before measuring |
| the same arithmetic inside your own CUDA kernel | `boys/boys_cuda_device.hpp` | when a round trip through global memory would cost more than the evaluation |

The complete list, with every overload, is the entry-point table in the
[API reference](https://myamlak.github.io/boys/).

**Choosing among the evaluation settings?** A `SeamPolicy` carries five option axes — the fit route,
the evaluation scheme, the packing axis, the interval granularity and the division form — and
[docs/getting-started.md](docs/getting-started.md#what-does-each-setting-change) tabulates what each
one changes, what the tree has measured about it, and where its figures are. Which member is fastest
is a property of your machine and its flags: the option probe ranks them where they will run.

**Calling it on a GPU?** The last two rows are the two routes onto a card, and
[docs/gpu-guide.md](docs/gpu-guide.md) is written for you: which route answers which problem, what a
device call looks like, what you must allocate and pass, what the library refuses and why that is a
feature, and how to read the bound you got.

**What the bound is, and what it is not.** For every `n <= 32` and every `x >= 0` the error of a
returned value is at or below **`1e-13 · F_0(x)`** — the ladder's first rung scaled by `1e-13`, not
the value the caller asked for. F is decreasing in the order, so the `1e-13 · F_n(x)` arm of the
pair is never the wider one, and where the order's own value is tiny the figure the bound is stated
in is a large part of it. Read against `F_n(x)`'s own value the bound is therefore not uniform: on
the committed grid the delivered error is 3.496e-14 of `F_n(x)` at the worst cell below the
region-C boundary, `x = 28.98933773882074`, and at `n = 32` it is under `1e-13` of the value again
above `x ≈ 93`; in between, the bound costs the value accuracy, and at the worst cell there,
`n = 32, x = 31`, the returned value is wrong by 140 % of its own. The bound holds above the lane
format's smallest normal, 2^-1022 in double and 2^-126 in float; below that the return is subnormal
or zero and no accuracy is claimed. Until 4.0.0 the guarantee instead carried a constant
floor, `5.5e-14`, and where F is small the two figures disagree: at
`n = 32, x = 29`, F is 1.93e-14 and the lane returned 6.87e-14 — an error of 4.94e-14, inside the
5.5e-14 floor those releases carried, and **2.55 times the true value** read relative to F. That
argument is served by the uniform grid — `FitGranularity::kUniform`, the table whose hi-bound is
`kFlatHi`, 31.0 in this build — and the ladder-scaled figure holds at every cell of it, at `0.288` of
the bound at worst.

**Two combinations the option space offers do not hold the bound read against `F_n(x)` alone.** The
double entries hold the ladder-scaled bound everywhere: the bare ladder's own worst cell is the low
end of the span it serves, `n = 32, x = 31`, at `0.288` of the bound — a `3.47x` margin. The two are
a lean partition's extended-band ladder (`FitGranularity::kCoarsest`, `kNarrow`, and `SeamPolicy<>`
as it resolves) and the rational-minimax route's high-order intervals; each holds the ladder-scaled
bound, at a worst measured `0.12` and `0.48` of it.
The float lane's figure carries a floor of its own for the same reason: binary32 arithmetic on
values of order one leaves about `1e-7` behind whatever the value is, so that lane states the
larger of `2^-22 · F_n(x)` and the `1.5e-7` it already published, plus `1e-7` in the
plain-reciprocal form. The [accuracy
contract](docs/specification.md#accuracy-contract) names the two combinations, the figure each holds
at its worst cell, and the command that measures the bound on your machine.
**And no entry checks its arguments**: outside `n` in `[0, 32]` and `x >= 0` the behaviour is
undefined, and a release build does not refuse the arguments — the scalar entries answer them.
A negative argument or an out-of-range order comes back as some value with no meaning; the exact
value is the build's own and moves with the compiler and the flags. A `NaN` argument comes back
`NaN`, and a `NaN` anywhere in a batch is worse: on this project's MSVC x64 Release build,
`BoysAllN` over `{1.0, NaN, 2.0}` is a segmentation fault. A build with assertions live aborts on
the argument instead. The five double
entries each have a `*Checked` twin — `BoysAllOrdersChecked` and its four siblings — that refuses
with a status rather than answering, and it is the call to make wherever the arguments are not known
to be in domain. The
[accuracy contract](docs/specification.md#accuracy-contract) carries the per-lane figures and the
command that measures the bound on your machine.

**How much accuracy do you need?** [docs/consumer-perspective.md](docs/consumer-perspective.md) —
what integral calculations actually require, and which of the library's evaluation paths that leaves
to choose between.

**Something wrong?** Open an issue at <https://github.com/myamlak/boys/issues> — a question, a bug, or
a proposal to relax the contract. When a returned value is the problem, quote the exact `(n, x)`, the
lane and the region: *Reporting a suspected violation* in
[docs/specification.md](docs/specification.md) says how to print the gate's own line for that cell,
so the issue can quote the line that failed rather than describe it. [CONTRIBUTING.md](CONTRIBUTING.md)
and its *Getting help* section say what to expect before writing code.

---

Everything above this line is what a first call needs. The specification — what each evaluation path
guarantees, the settings that select one, and the measurements behind the figures — is
**[docs/specification.md](docs/specification.md)**, and none of it is needed to make your first call.
The words that page uses — *lane*, *region*, *route*, *scheme*, *axis*, *gate* — are defined on the
documentation's landing page, together with the full API reference:
<https://myamlak.github.io/boys/>. **[docs/architecture.md](docs/architecture.md)** is the map for a
reader working in the tree rather than calling it: which file implements what, and which test holds
each thing to its contract.
