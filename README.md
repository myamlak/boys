# The Boys function kernel

[![CI](https://github.com/myamlak/boys/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/myamlak/boys/actions/workflows/ci.yml)
[![docs](https://github.com/myamlak/boys/actions/workflows/docs.yml/badge.svg)](https://github.com/myamlak/boys/actions/workflows/docs.yml)
[![license](https://img.shields.io/badge/license-BSD--3--Clause-blue)](LICENSE)
[![version](https://img.shields.io/github/v/tag/myamlak/boys?label=latest%20release%20tag)](https://github.com/myamlak/boys/tags)

F_n(x) = ∫₀¹ t^(2n) exp(−x t²) dt, for n = 0..32 and x ≥ 0 — the integral an electronic-structure
program evaluates at every order a shell quartet can ask for. This is a C++ library that evaluates
it, in double and single precision, on the CPU and optionally on a GPU. It depends on nothing outside
the C++ standard library and the optional CUDA toolkit.

## Your first call

Copy this into `try.cpp`, build it, run it. It prints the numbers below. They come from a build that
does not contract the multiply-add: a product and a sum stay two roundings rather than becoming one.
A compiler that does answers with values that differ inside this lane's bound. `docs/build-facts.md`
records per CI leg what each leg does, and `docs/specification.md` says which calls a consumer's own
flags reach.

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
overload of each many-argument entry taking a `std::span`, so a container goes straight in. It costs
nothing - the overload forwards to the entry and both spellings compile to the same call - and a call
passing a pointer and a count still reaches the pointer entry exactly.

What a translation unit needs, by what it is for:

| A unit that ... | Includes | What that brings |
|---|---|---|
| declares a wrapper, names a type, or holds an entry as a value | `boys/boys_decl.hpp` | every declaration - the types, the entries, the accuracy queries, the table a report reads - and no arithmetic |
| calls an entry the default table answers for | `boys/boys.hpp` | the declarations, plus each default entry's own body inline at the row this build's table carries for its class: nothing else is needed, and no archive on the link line |
| names a policy, `BoysAllOrders<Policy>` rather than `BoysAllOrders` | `boys/boys.hpp` | the same header; the flags the unit is compiled with choose the ISA tier, and no flags is the scalar tier. One flag set per program is the contract the headers state |
| wants everything this library carries | `boys/boys_all.hpp` | the front door, plus the `std::span` overloads, the region-A transform lane and the option probe |
| is C | `boys/boys_c.h` | the C linkage surface; it is built in the archive, so link it |
| is CUDA | `boys/boys_cuda.hpp`, `boys/boys_cuda_device.hpp` | device calls that upload their tables on first use, or the same arithmetic written into your own kernel |

The archive is still needed for the reporting queries - `BoysLaneContracts`, `BoysEvalSchemes`,
`BoysEvalSchemeFits`, `BoysEvalSchemeDelivered`, `BoysPackAxes`, `BoysDivisionForms`,
`BoysFarRepresentations`, `BoysRegionBExps`, `BoysFitGranularities`, `BoysFitRoutes`,
`BoysFitRoutesF32`, `EvalSchemeName`, `BoysAccuracyGuaranteed`, `BoysAccuracyDelivered`,
`QueryCombination`, `BoysAvx2Available` - and for the three entries the library defines out of
line: `BoysAllOrdersWithRoute`, `BoysAllOrdersWithFarRepresentation`, `BoysSingleF32WithRoute`.
Everything else a call site names compiles from the headers.

    git clone https://github.com/myamlak/boys.git && cd boys
    cmake -S . -B build -DBOYS_BUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF
    cmake --build build --config Release

or, against a tree you have already built:

    c++ -std=c++20 -I include try.cpp -L build -lboys -o try     # -I and -L as your build laid them out
    ./try

```
F_3(1.25)  = 0.055476132923077584
F_0(3.5)   = 0.46984703520162352
F_3(3.5)   = 0.01183138358371668
F_6(3.5)   = 0.0040954447623731674
F_2(0.25)  = 0.16753319090735061
F_2(4)     = 0.017525782161993072
F_2(30)    = 0.00013483513281472914
F_0(30)    = 0.16180215937964007
F_6(30)    = 3.6049669032736188e-08
```

The three `OFF`s are there so this builds the library and nothing else, which needs no compiler past
C++20 — the suite and the benchmark drivers are the targets that want C++23, and each is behind one
of those flags. Drop them to build all three as well; that is the contributor's build.

On Windows with the Visual Studio generator the library lands in `build/Release/`. This program is
[`examples/00_first_call.cpp`](examples/00_first_call.cpp); the contributor's build compiles it and
`ctest` runs it, along with every other program this README and the guide quote, so the transcript
above cannot rot silently.

**[docs/getting-started.md](docs/getting-started.md)** takes these four calls one at a time, then
the questions that follow them: how to name a specific evaluation, how to ask what the library
guarantees before you rely on it, and how to find out which of the available options is fastest on
your machine. Every figure in it is the
output of a program in [`examples/`](examples/), so you can reproduce any of them.

## Which entry do I call?

By what you want, not by what the library calls things:

| I want | Entry | One clause on when |
|---|---|---|
| F_n(x) for one order, one argument | `BoysSingle` | the simplest call; use it when you have one value |
| F_0..F_nmax at one argument | `BoysAllOrders` | a shell quartet's ladder; this is the common case |
| F_n at every argument of an array | `BoysFixedN` | when the order is fixed and the arguments vary; takes an output stride |
| F_0..F_nmax at every argument of an array | `BoysAllN` | the batch shape of an integral engine; the arguments may arrive in any order |
| ...with each argument's own top order | `BoysAllNAtOrders` | a shell-quartet batch, where no argument is padded to a common order |
| ...when you know the arguments are already sorted | `BoysAllN` with `BoysSortedArgs{}` | skips the internal sort when your loop already produces a non-decreasing array |
| any of the above in single precision | `BoysSingleF32`, `BoysAllOrdersF32`, `BoysFixedNF32`, `BoysAllNF32`, `BoysAllNAtOrdersF32` | when the rest of your kernel is `float` |
| half-precision storage | `BoysSingleF16`, `BoysAllOrdersF16`, `BoysFixedNF16`, `BoysAllNF16`, `BoysAllNAtOrdersF16`, `BoysSingleBf16`, `BoysAllOrdersBf16` | 16-bit I/O around the single-precision engine |
| an answer on a GPU | `boys/boys_cuda.hpp` | device arrays; uploads its tables on first use, so warm the path before measuring |
| the same arithmetic inside your own CUDA kernel | `boys/boys_cuda_device.hpp` | when a round trip through global memory would cost more than the evaluation |

The complete list, with every overload, is the entry-point table in the
[API reference](https://myamlak.github.io/boys/).

**Calling it on a GPU?** The last two rows are the two routes onto a card, and
[docs/gpu-guide.md](docs/gpu-guide.md) is written for you: which route answers which problem, what a
device call looks like, what you must allocate and pass, what the library refuses and why that is a
feature, and how to read the bound you got.

**What the bound is, and what it is not.** For the shipped build the guarantee is **relative**: the
error of a returned value is at or below **1e-13 · F_n(x)**, for every `n <= 32` and every
`x >= 0`. The bound holds above the lane format's smallest normal, 2^-1022 in double and 2^-126 in
float; below that the return is subnormal or zero and no accuracy is claimed. The figure was an
absolute `5.5e-14` until this release, and where F is small the two figures disagree: at
`n = 32, x = 29`, F is 1.93e-14 and the lane returned 6.87e-14 — an error of 4.94e-14, inside the
absolute figure of the day, and **2.55 times the true value** read relative to F.

**The figure has a floor, and two combinations are bounded by the floor instead.** The relative
figure is the one the shipped entries hold, with an order of margin: the double entries deliver at
worst `9.99e-15 · F_n(x)`, a tenth of it, and the worst cell is `n = 32, x = 29.04`. Two
combinations the option space offers do not hold the relative figure alone yet, and hold the absolute
figure these lanes published before it: a lean partition's extended-band ladder
(`FitGranularity::kCoarsest`, `kNarrow`, and `EvalPolicy<>` as it resolves) and the rational-minimax
route's high-order intervals.
The float lane's figure carries a floor of its own for the same reason: binary32 arithmetic on
values of order one leaves about `1e-7` behind whatever the value is, so that lane states the larger
of `2^-22 · F_n(x)` and the `1.5e-7` it already published. So the bound is **the larger of
`1e-13 · F_n(x)` and the lane's absolute figure** — `5.5e-14` in double, `1.5e-7` in float. The
[accuracy contract](docs/specification.md#accuracy-contract) names the two combinations, the figure
each holds at its worst cell, and what raises them to the relative one.
**And no entry checks its arguments**: outside `n` in `[0, 32]` and `x >= 0` the behaviour is
undefined, and a release build faults on `x < 0` or on a `NaN` anywhere in a batch, while `n > 32`
answers a wrong number with no diagnostic. The five double entries each have a `*Checked` twin —
`BoysAllOrdersChecked` and its four siblings — that refuses with a status instead of faulting. The
[accuracy contract](docs/specification.md#accuracy-contract) carries the per-lane figures and the
command that measures the bound on your machine.

**How much accuracy do you need?** [docs/consumer-perspective.md](docs/consumer-perspective.md) —
what integral calculations actually require, and which of the library's evaluation paths that leaves
to choose between.

---

Everything above this line is what a first call needs. The specification — what each evaluation path
guarantees, the settings that select one, and the measurements behind the figures — is
**[docs/specification.md](docs/specification.md)**, and none of it is needed to make your first call.
The words that page uses — *lane*, *region*, *route*, *scheme*, *axis*, *gate* — are defined on the
documentation's landing page, together with the full API reference:
<https://myamlak.github.io/boys/>. **[docs/architecture.md](docs/architecture.md)** is the map for a
reader working in the tree rather than calling it: which file implements what, and which test holds
each thing to its contract.
