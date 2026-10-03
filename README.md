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

Copy this into `try.cpp`, build it, run it. It prints the numbers below. They are a build that does not
contract the multiply-add; a compiler that does may differ in the last digit of a value and nowhere
else, which is what `docs/build-facts.md` records per CI leg.

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

    git clone https://github.com/myamlak/boys.git && cd boys
    cmake -S . -B build -DBOYS_BUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF
    cmake --build build

or, against a tree you have already built:

    c++ -std=c++20 -I include try.cpp -L build -lboys -o try     # -I and -L as your build laid them out
    ./try

```
F_3(1.25)  = 0.055476132923077535
F_0(3.5)   = 0.46984703520162352
F_3(3.5)   = 0.011831383583716678
F_6(3.5)   = 0.0040954447623731674
F_2(0.25)  = 0.1675331909073505
F_2(4)     = 0.017525782161993068
F_2(30)    = 0.00013483513281636802
F_0(30)    = 0.1618021593796416
F_6(30)    = 3.6049670926598394e-08
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
| any of the above in single precision | `BoysSingleF32`, `BoysAllOrdersF32`, `BoysAllNF32` | when the rest of your kernel is `float` |
| half-precision storage | `BoysSingleF16`, `BoysAllOrdersF16`, `BoysSingleBf16`, `BoysAllOrdersBf16` | 16-bit I/O around the single-precision engine |
| an answer on a GPU | `boys/boys_cuda.hpp` | device arrays; uploads its tables on first use, so warm the path before measuring |
| the same arithmetic inside your own CUDA kernel | `boys/boys_cuda_device.hpp` | when a round trip through global memory would cost more than the evaluation |

The complete list, with every overload, is the entry-point table in the
[API reference](https://myamlak.github.io/boys/).

**How much accuracy do you need?** [docs/consumer-perspective.md](docs/consumer-perspective.md) —
what integral calculations actually require, and which of the library's evaluation paths that leaves
to choose between. **Want the guarantee instead?** [What each path
guarantees](#accuracy-contract) — the bound, and the command that measures it on your machine.

---

Everything below this line is the specification: what each evaluation path guarantees, the settings
that select one, and the measurements behind the figures. None of it is needed to make your first
call. The words it uses — *lane*, *region*, *route*, *scheme*, *axis*, *gate* — are
defined on the documentation's landing page, together with the full API reference:
<https://myamlak.github.io/boys/>.

## Accuracy contract

The bound is |F̂_n(x) − F_n(x)| ≤ B, for every supported n, x and lane.

The library evaluates the function differently at different argument sizes, and one lane is tighter
over part of the range than over the rest. Every figure below holds for **all** x ≥ 0:

| Lane | Error bound |
|---|---|
| double single | ≤ 5.5e-14 everywhere; ≤ 3e-14 below x = 11.899848152108484; ≤ 1e-15 below about x = 1.0855 |
| double batch, whether the top order is the batch's or each argument's | ≤ 5.5e-14 |
| float single / batch | ≤ 1.5e-7, or ≤ 2.5e-7 in the plain-reciprocal form |
| fp16 / bf16 | ≤ 1.5e-7 + ½ ULP, or ≤ 2.5e-7 + ½ ULP in the plain-reciprocal form |
| native half, x ≥ 28.984375 | ≤ 8 ULP of the returned value |
| CUDA fp64 | the same budgets as the CPU double lanes |
| CUDA fp32, `RegionBExp::kAccurate` (the default) | the same budgets as the CPU float lanes |
| CUDA fp32, `RegionBExp::kFast` | ≤ 1.5e-7 + 8e-8 — the lane's budget plus the corrected seed's own contribution |

**One axis's members are not one figure.** `DivisionForm` is how every recurrence step divides, and
its three members are three arithmetics, not three spellings of one. Exact division rounds once per
step, the plain reciprocal rounds twice, and the refined reciprocal recovers the exact form's
rounding from the plain product.

On the **double** lane the three deliver one figure. Over the accuracy gate's reference grid,
56694 cells per form, the plain reciprocal leaves region A, region B and region
C where exact division has them and moves the extended band's worst from 3.22e-15 to 6.73e-15, inside
the 3e-14 that region publishes. The refined reciprocal is bit-identical to exact division in every
cell. So the double rows are figures under all three forms.

On the **single-precision** lanes the plain reciprocal costs accuracy, so those lanes publish a
figure for it beside their base: **1.5e-7 + 1e-7 = 2.5e-7**. Its measured worst is 1.7514e-07, at
n = 0, x = 9.74055, where the lane's exact and refined forms deliver 1.08354e-07 at worst. The fp16
and bf16 lanes run that arithmetic and round at the boundary, so the same term stands beside their
base, before the half digit above is added.

`BoysAccuracyGuaranteed` takes the form as an argument and answers the figure for the form you name.
The device lane carries no form to key one by: the axis is a host policy field the CUDA surface does
not name.

"ULP" is the last representable digit of the result in the format concerned.

**Choosing nothing.** A caller that has picked a precision and no axis writes one name: each precision
has a named default, and so does the device lane. [docs/lane-contract.md](docs/lane-contract.md#the-default-policy-per-precision-and-per-device)
states what each selects, the bound it carries, and the command that prints the name and the in-force
default as numbers. Two of the choices — the evaluation scheme and the interval partition — were
set from the option probe's own runs, and **neither was settled by them**. The partition was not
chosen against its axis at all: those runs are dated 2026-09-28, every partition-bearing row in them
is one partition, and the uniform partition reached the host lanes on 2026-09-29 and every lane on
2026-09-30, so re-deriving that default with the whole axis is owed. On the scheme axis the runs did
compare both rows and did not separate them.
The route and the packing axis carry the settings the library has always shipped and have not been
ranked against a timing. The option probe below is what ranks any of them, on the machine it is run on.
**Each default is a choice between two ways of computing one answer and not between two accuracies**, so
a version that moves one costs no accuracy at any call site that names nothing.

The rows above are bounds, and a bound is not the figure a lane delivers. Two lanes are delivered at
a different figure depending on one property of the build — whether the compiler fuses a bare
product-plus-add into a single rounding. **The architecture does not decide it**: of the six
configurations measured, gcc and AppleClang on arm64 contract one and MSVC on arm64 does not, so the
MSVC arm64 build delivers the x86-64 figures rather than its own architecture's. Against the
committed reference grid:

| lane | region | contracted | not contracted | bound |
|---|---|---|---|---|
| double, single | 1.0855 ≤ x < 11.8998 | 3.29e-15 | 3.22e-15 | 3e-14 |
| float, single | all arguments | 1.29e-07 | 1.06e-07 | 1.5e-07 |
| float, batch | all arguments | 1.29e-07 | 1.08e-07 | 1.5e-07 |

Those three rows are the whole of what moves: the double lane's other three regions deliver the same
worst cell, to the digit, on both arithmetics. The not-contracted column is measured on MSVC on
arm64, MSVC on x86-64, clang on x86-64 and AppleClang on x86-64; the contracted column on AppleClang
on arm64, and reproduced on x86-64 by building with `-mfma`.

Every bound holds either way, and the configure step **measures** which arithmetic a build runs by
compiling and running a bare product-plus-add rather than inferring it from the architecture name;
the report prints the answer. The `BoysFixedN` entry's agreement with `BoysSingle` is exact where the
build does not contract that form and inside the single lane's bound everywhere. A contracting build
may fuse at one call site and not at another, so the entry's report prints how many of its
comparisons were bit-for-bit equal, and [docs/lane-contract.md](docs/lane-contract.md) carries the
counts.

The fp16 and bf16 rows are fp16 *I/O* around the fp32 engine, so their error is the engine's.

The native half lane (`BoysAllOrdersHalf2`, `BoysAllNF16Native`) is a different thing. It evaluates
region C's ladder in packed binary16, one correctly rounded operation per step. That buys two
arguments to a register. It costs the accuracy its own bound names: ≤ 8 ULP of the returned value,
measured at a worst of 4.3 ULP.

Its output is 2^15 F_k(x). That exact scale keeps the ladder in the format's normal range down to
F_k(x) = 2^-29, which is x up to 359, 128 and 30 at orders 3, 4 and 8. **Past those arguments the
entry returns a subnormal half, then zero, by design, and no accuracy is claimed there.** A caller
that must be right at those orders and arguments wants the double or float lane. The entry covers
region C only, so x must be at or above the region-C boundary, and there is no fallback to the table
regions.

Its range shrinks steeply with the order, and above order 8 it has none at all. **Orders 9 to 32 have
no argument at which this lane returns a value inside its bound**, because the function is already
below 2^-29 at the smallest argument the entry accepts. Orders 0 and 1 cover the whole 16-bit range,
and orders 2 to 8 reach x of about 2636, 360, 129, 70, 47, 36 and 30.

The **region-A transform** (`BoysRegionAProduct`) is a separate entry rather than a lane. It computes
region A's fits as a matrix product instead of by the fitted recurrence — one product per band, all
orders at once, for a batch of arguments — in an arithmetic mode the caller names: `ProductMode::kFp64`,
`kTf32x3` or `kBf16x6`. It changes none of the lanes above. At fp64 it adds nothing measurable to the
coefficients' own truncation, 1.11e-16 over the band. The two split modes deliver 1.92e-07 and
1.95e-07, which is their 32-bit accumulator's floor rather than the operand split's: no amount of
operand precision reaches 64-bit-grade accuracy on a 32-bit accumulator. **Those two bounds are
arithmetic on a model of a 32-bit tensor-core accumulator, not a measurement of a card** — a tensor
core's fused sum is less accurate than the model, never more, and **no speed is claimed for any mode**.
A caller who seeds the downward recursion rather than reading values is held to a tighter
requirement: at fp64 only orders 0 to 2 and 25 to 32 may seed, and the two split modes may not seed
at all.

The **CUDA fp32 lane's single entries** — the batch one and the device-callable one — take a
certified choice of region-B exponential (`boys::RegionBExp`), and each choice carries its own
bound. `RegionBExp::kAccurate`, named `boys::kDefaultRegionBExp` because it is the lane's default,
is the default and the arithmetic the batch entries already run.
`RegionBExp::kFast` is the hardware approximation with its argument-scaling residual removed.

The correction is what the recurrence asks for, and the recurrence is what makes the choice
non-trivial: its condition number is 7.6e4 at the region-B boundary at order 32, so an error in the
e^{-x} it consumes is amplified by that much there, while an approximation whose relative error
grows with the argument carries fifteen ulp at the boundary and thirty-five at the far end of the
region. The bare approximation therefore fails the lane's budget over a band of region B at the
highest order — predicted at 2.30e-7 against the 1.5e-7 budget, measured at 2.57e-7 — and returns
the wrong sign there. It is not offered, because the failing band is interior to
region B and there is no certified sub-range to restrict it to. Corrected, the seed's relative error
is flat at a few ulp, its contribution to the value is capped at 8e-8 by that same amplification
(measured 5.0e-8), and the option returns the function's sign at every cell the device gate audits.

No speed is claimed for either option. The corrected form's kernel is the same size statically as the
accurate one — 944 instructions against 944 — and the exponential is evaluated once per
element outside the order loop; what separates the two options is the bound, not a measured time.

The GPU and CPU float lanes agree to within 3.5e-7. That is a cross-lane statement about
|GPU − CPU| and not a bound on either lane's distance from F_n(x): the default CUDA fp32 entry is
held to 1.5e-7 above, and the fast option's looser bound is not covered by the 3.5e-7 figure.

### Choosing a combination

The lanes above are one axis of five. A call is a lane, a fit route, an evaluation scheme, an interval
partition and a packing axis, and the library offers the product of all five:
2 routes × 2 schemes × 3 partitions × 2 axes, in 4 lanes — **96 combinations: none is refused, each
is certified and published or is a device cell a host without a CUDA device cannot run, and none
deliver outside the bound their lane publishes** (the gate command below prints that arithmetic).
`BoysAccuracyGuaranteed(...)` returns the bound a
combination carries — its lane's figure, plus the lane's own additive term where it
documents one — and `BoysAccuracyDelivered(...)` returns the figure it was measured to deliver,
which is the one to rank two combinations by. They are different questions, and the `reading` field
of the returned `AccuracyFigure` says which answer a figure is. A combination this revision does not
carry has no figure: both accessors say so and give the library's own reason rather than returning a
number.

**Choosing a combination is a name.** The four structural axes —
route, scheme, partition, packing axis — are the fields of an `EvalPolicy`, so a combination is a type
the call site writes once and the compiler resolves where it is written: 4 lanes × 24 axis
combinations, **96 combinations, each reachable as a name**. Naming one costs nothing at the call —
there is no table to look a combination up in, no string to match and no search at run time, which is
why those axes are not call arguments. All of them are named and all of them are evaluated by
`tests/consumer_option_space.cpp`, which prints the counts per precision.
A caller holding a route and a scheme as values names both at the call instead:
`BoysAllOrdersWithRoute(route, scheme, nmax, x, out)` on the ladder shape and
`BoysSingleF32WithRoute(route, scheme, n, x)` on the single-order shape.

**A caller that has a target rather than a comparison asks it directly.** `QueryCombination(...)`
takes the same axes and the absolute error the caller needs, and answers with a verdict beside
the numbers it was made on: `kGuaranteedInside` where the bound is at or below the tolerance,
`kDeliveredInside` where the bound is above it and the figure the combination's fits were measured
to deliver is at or below it, `kOutside` where neither is, and `kNotCarried` where this revision does not have
the combination — no verdict, no figure, and the library's own reason. The bound decides first, so
`kGuaranteedInside` is the state a calculation's safety can rest on and `kDeliveredInside` is
explicitly not that state. The request and both figures come back with the verdict, so an answer can
be checked against what decided it.

**The bound is the lane's, and no other axis moves it** — what the other axes change is what a call
delivers, and the gate prints every combination's measured figure beside its bound. The per-lane
counts, the reasons, the two figures side by side and the tolerance table are in
[docs/lane-contract.md](docs/lane-contract.md#every-combination-its-bound-its-delivered-figure-and-its-tolerance).

### Check the figures yourself

Every figure above is measured, and the measurement is in this tree. Build the gate and run it:

    cmake -S . -B build
    cmake --build build --target boys-accuracy-gate
    ./build/Release/boys-accuracy-gate --strict      # the config directory is your generator's

It sweeps the documented entries over the committed reference grid — 33 orders × 1718 arguments,
56,694 points — and prints, lane by lane and region by region, the worst error the lane delivered
beside the bound it claims. `--per-order` adds the same comparison order by order, naming the cells
the bound actually binds on; `--probe n x` prints one cell from every lane for one argument.

The run ends with a verdict line and an exit status: `PASS: every documented claim met at this
revision`, or the numbers of the claims that did not hold, with a non-zero status. `ctest` runs this
binary as one of its tests, but a passing `ctest` prints only how long the test took — the table is
in this binary's own output. CI runs this gate on every platform below and fails the build if a claim
does not hold. Of the two figure columns above, that holds the `Bound claimed` one: its figures are
the gate's own constants, and `tools/check_bound_transcripts.py`, which CI runs as well, reads this
table against them, so a bound that moves on one side and not the other is a red run. The two half
lanes are that column's one exception, and the checker names it: their figure is the half-quantum
the gate composes at run time from the format of the value it returned, so no constant states it.
The `Worst delivered` column is a run's own output, and it is held to one. `tests/data/` carries a run
of the gate kept as a file; `tools/check_bound_transcripts.py` reads this table against that run's
rows, and `tools/check_recorded_run.py` asks the repository what has moved since the revision the run
names. Both run in CI. So a change that moves a lane inside its bound leaves the gate green - the
bound still holds - and leaves the recorded run describing an older tree, which the second check
reports rather than passing over. The fix is a new run and a new transcription of it, never an edit to
the figure. A figure is one host's: your own run is the answer for your platform, your compiler and
your arithmetic, and the rows are compared cell by cell rather than as one number.

Run on the committed tree, it printed this (an excerpt; the full run carries one row per lane and
region):

| Lane | Region | Worst delivered | Bound claimed |
|---|---|---|---|
| double single | A | 2.22e-16 | 1e-15 |
| double single | band | 3.22e-15 | 3e-14 |
| double single | B | 7.52e-16 | 3e-14 |
| double single | C | 5e-14 | 5.5e-14 |
| float single | all | 6.36e-08 | 1.5e-07 |
| fp16 store-half | single | 0.000122 | 0.000122 |
| bf16 store-half | single | 0.000976 | 0.000977 |

and ended `PASS: every documented claim met at this revision`.

The gate's region labels are the ones above: `A` is the part of the double lane's range below
x = 1.0855252345349333, where it claims ≤ 1e-15; `band` is the rest of the range below
x = 11.899848152108484, where it claims ≤ 3e-14; and `B` and `C` are the two regions beyond that.

### Arguments outside the documented domain

Every entry takes n in [0, 32], x ≥ 0, and output spans of the documented size. What a caller gets
for stepping outside that is not the same on every surface:

| Surface | What a violation does |
|---|---|
| C++ (`boys::`) | The check is `assert` from `<cassert>` — the call aborts in a build with assertions enabled, and a build with them cleared has no check at all. Nothing in this tree defines or clears `NDEBUG`, so a Release build gets it from the toolchain's own release flags, and a caller who wants the check keeps it by compiling this library with `NDEBUG` undefined. There is no fallback value and no clamping: an order outside [0, 32] indexes the order-dependent tables out of bounds, which is why every bound above reads "for every supported n, x". |
| C (`boys_c.h`) | Every entry validates and returns a status instead: `BOYS_ERROR_INVALID_ARGUMENT` (1) for an order outside [0, 32], a negative or NaN x, or a null pointer. No abort and no undefined behaviour. |
| CUDA | The single, batch and all-N entries return `boys::BoysStatus`, whose `kInvalidArgument` is the same validation as the C surface's; the device-callable entries take their orders as template arguments, so an order outside the range does not compile. |
| Native half (`BoysAllOrdersHalf2`, `BoysAllNF16Native`) | Same `assert`, and one precondition of its own: the entry covers region C only, so an x below the region-C boundary is a violation rather than a fallback to the table regions. |
| Output spans | The caller sizes them. A span shorter than the call writes is an out-of-bounds write with no check on any surface, and no entry reallocates or truncates. |

### Fit routes

The double lane's stored fits come in two routes, and a caller picks one per call with
`BoysAllOrdersWithRoute`. `BoysFitRoutes()` reports them: for each route and region, the interval
its fit covers, the argument its selector takes over at, how many coefficients it stores, the worst
error it was measured to deliver, and the bar it is certified against. Both routes hold the same bar
over the same interval; they differ in what they store to do it. The routes are alternatives:
naming one changes only the fits that serve the intervals its rows report, and everywhere else
the entry runs the default route and returns its values bit for bit, so the lane's tighter per-order
1e-15 is untouched. A route name the build does not serve evaluates the default route rather than
returning something the caller did not ask for. **No speed is claimed for either route.**

The route composes with the evaluation scheme into one `EvalPolicy`, which every templated
double-precision entry takes. A caller holding a route and a scheme as values names both at the call
instead: `BoysAllOrdersWithRoute(route, scheme, ...)` on the ladder shape, and
`BoysSingleF32WithRoute(route, scheme, ...)` on the float lane's single-order shape.

[docs/lane-contract.md](docs/lane-contract.md#the-two-fit-routes) carries this in full: the table of
routes with each one's stored count, measured error and bar; the criterion each is accepted against;
which entries carry a route and what that carriage delivers; and why the rational route
buys the interval's values and not a recursion seed.

#### The float lane's routes

The float lane's region-A table and its region-B seed are its own, and they come in the same two
routes. They are selected at run time with `BoysSingleF32WithRoute`, which is `BoysSingleF32` with
one thing changed — which fit supplies the lane's region-A seed and its region-B seed — and
`BoysFitRoutesF32()` reports them the way `BoysFitRoutes()` reports the double lane's. Region C holds
no coefficient under either route, so naming one there changes nothing.

**The same choice is open at compile time, and there it carries the scheme as well.** The lane's
entries take the `EvalPolicy` above, so a caller who templates on it names the route and the scheme as
the second template argument instead of calling `BoysSingleF32WithRoute`. Both fields are read:
the route selects which family supplies the lane's fits, and the scheme selects which of
that family's stored forms is summed, so naming another pair there is answered rather than refused.
`BoysAllNF32` forwards its policy to the batch entry's body once per argument.

The float lane's rational route covers the whole of region A from zero: the lane reads every order
from its own fit over the region, so there is no band boundary for a selector to take over at and
the row states no `servesFrom` above zero.

| Route | Region | Interval | Stored | Measured | Bar |
|---|---|---|---|---|---|
| chebyshev (default) | A | [0, 11.899848152108484) | 1067 | 1.24e-07 | 1.5e-07 |
| rational minimax | A | [0, 11.899848152108484) | 465 | 1.44e-07 | 1.5e-07 |
| chebyshev (default) | B | [11.899848152108484, 28.98933773882074) | 11 | 3.14e-08 | 1.5e-07 |
| rational minimax | B | [11.899848152108484, 28.98933773882074) | 6 | 7.50e-08 | 1.5e-07 |

The measured column is the figure the generated header publishes for the row, and every one of them
is **the worse of the lane's two multiply-add routes**, because a build runs one of the two and a
figure that covers one of them understates the other; the gate prints each row's published figure
beside the worst it measures itself, on the entry the row names, every order 0..32 and every sample
of the row's interval. Both routes hold the lane's own bound, and the two region-A rows are the
comparison: the same interval, the same reference, the same arithmetic, 465 stored coefficients
against 1067. The rational route's pieces are its own cover of each order's interval rather than the
Chebyshev table's breaks, and both were accepted against the same criterion — the lane's own bar,
weighted by the downward recursion's gain, **on the coefficients as they are stored**, read in both
multiply-add routes with the worse taken, at every argument of the region's acceptance grid and at
every cell the accuracy gate sweeps. Half the lane's tolerance is not that criterion and is not one
at all: the rational route's reading sits above that half at 18 of the 33 orders under the fused
arithmetic and 19 under the separate one, so a search against it does not close.

**`FitGranularity::kNarrow` is served by this lane too, on both its routes.** The partition cuts
region A of this lane into 218 pieces at degree 6 and region B into four pieces at the same degree,
and each route stores its own fit on that cut: 1526 coefficients either way on the Chebyshev route
against the shipped table's 1067, and 1241 on the rational route against 465. Region B's seed is
where the narrowing is visible in the other direction: 28 stored on the Chebyshev route against the
shipped seed's 11, and 18 on the rational route against its 6. The gate's single-entry policy rows,
read in binary32 on the coefficients as stored, are 1.02681e-07 over region A and 4.51733e-08 over
region B for the Chebyshev route's narrow rows, and 1.15993e-07 over region A and 2.95023e-08 over
region B for the rational route's. Every narrow row is inside the lane's 1.5e-07 bar on the build
the gate ran, whose multiply-add route is the fused one. The figures the generated header publishes
for the narrow pieces are read the same way, at each multiply-add route: 1.19209e-07 for the narrow
Chebyshev pieces, and 1.14376e-07 fused against 1.10598e-07 separate over region A and 3.29858e-08
in both over region B for the rational ones. The route the build runs is the figure the gate judges
the row against.

**The Chebyshev route's fits are stored in both of the forms the two schemes read**, one monomial
coefficient per Chebyshev coefficient: 1067 stored either way in region A and 11 in region B, the same
pieces, intervals and degrees, so naming a scheme chooses a table and not a shape. Over region A the
worse route reads 1.24e-07 at the split Clenshaw reading's worst cell and 8.98e-08 of the Horner
reading, over region B 3.14e-08 and 2.19e-08, all inside the lane's 1.5e-07 bar. The second form costs
this lane nothing, and the reason is the length of its pieces: the monomial coefficients of any one
piece sum in absolute value to at most 1.6, so rounding the monomial table to binary32 moves a value
by at most 0.64 of the bar even if every coefficient rounds against it, and the measured reading is
0.6 of the bar.

**The stored counts are upper bounds and not minima.** The degree search is a scan that stops at the
first count holding the target, not an exhaustive minimax search over the family, so a cheaper cover
of the same intervals may exist; the counts above are what the search found, and a reader comparing
them is comparing what the two tables cost as generated.

**Region B is the other direction.** There the rational seed stores 6 coefficients against the
Chebyshev seed's 11 and delivers 7.50e-08 against 3.14e-08. Half the coefficients at more error is a
trade, not an improvement, and which side of it a caller wants depends on what their machine charges
for a coefficient fetch; **no speed is claimed in either direction**. A caller that needs the
accuracy should read the two measured figures and pick; the default is unchanged, value for value.

### The packing axis

A packed lane keeps four doubles in a register and the call has to supply four of something. That
something is a choice, and it is the fourth field of `EvalPolicy`: `PackAxis::kArguments`, the
shipped axis, puts four arguments at one order in a register, and `PackAxis::kOrders` puts four
orders at one argument there — which is the axis `BoysAllOrders(nmax, x, out)` actually has, since
that entry computes every order at a single argument. `BoysPackAxes()` reports both, with the
interval each one's packed lane evaluates. The orders axis covers region A at the same per-order fits
and the same bars the scalar region-A path holds, ≤ 1e-15 on those fits and ≤ 3e-14 on the
extended band, and naming it changes the region-A values a caller receives — the shipped entry reaches
most orders by a recursion from a seed where this lane evaluates each order's own fit — with both
inside the bound. Both partitions of region A are carried
on it: the double lane's shipped pieces are shared across the orders of a piece, which lets the lane
fetch one piece's coefficients at a fixed stride, and the narrow partition's are cut per order, which
the lane reaches with a gathered fetch that reads each of the four orders it holds its own piece.

[docs/lane-contract.md](docs/lane-contract.md#the-packing-axis-which-of-a-calls-values-share-a-vector)
carries the axis in full: which entries carry it, which two calls cannot form it and why they are
refused, what it gives up on the plane entry, and the measured count — the two coefficient fetches,
the counter that decides between them, and the command that reproduces the figures. **No time is
claimed for it, here or anywhere: a µop count is a property of the CPU it was taken on, and a time
taken on a loaded machine is not a measurement.**

One call cannot form the axis, and it is refused where the call is named rather than answered.
`BoysFixedN` computes exactly one order at every argument of an array: a packed lane keeps four
orders, and this call has one, so there are not four to fill a lane with — that is the entry's
signature and not a body nobody built.

The single-precision engines carry the same axis at **eight** orders to a register, because that is
what the float lane's own arithmetic path is: `avx2-fp32` holds eight floats where `avx2-fp64` holds
four doubles. The lane's tables are the ones that differ, and they are what sets the width rather
than the register: the float table gives each order its own cover, so one argument selects a
different piece in each of the eight lanes and the eight coefficient bases are fetched per lane
rather than stepped at a stride. The eight lanes share the degree the group is summed at, and a lane
whose own fit is cut shorter reads zeros above its own cut — which is that lane's own polynomial, and
down the sum it is that lane's own arithmetic, so the packed value is the per-order value and not a
value near it. Measured over region A and every order, the packed lane
and the per-order lane differ in **0 of 999240 values**; the gate carries a row for this axis beside
the per-order one, judged against the committed reference grid at the float lane's own 1.5e-7, and
its worst cell reads **1.06e-07**, a ratio of 0.705 to the bar.

**On this lane the axis is not the cheaper way to get the values.** It retires 3.06 times fewer
instructions and 2.61 times fewer retired slots than the per-order loop it replaces, which is what
filling the register buys — and 3.13 times *more* instructions and 3.71 times more retired slots than
the float lane's default entry, which serves all 33 orders from one seed fit and a downward
recurrence. A caller naming this axis is
choosing the per-order lane's shape, and the axis is served because it was named; the figures and the
command that reproduces them are in [docs/lane-contract.md](docs/lane-contract.md#the-same-axis-on-the-single-precision-engines-eight-orders-to-a-register).

Every other combination the axis names is built and measured. A route other than the shipped one is
answered by that route's own region-A fits, whose pieces
cover the same per-order intervals as the shipped table; and the narrow partition is answered by the
very fits the scalar entry reads there, fetched one order at a time instead of by a stride. The gate's
packing book carries a measured
row for each of them, on both routes, at both schemes, through both
entries that carry the axis, and names the worst cell of each.

**Reporting a suspected violation.** Open an issue with the exact `(n, x)`, the lane and the region,
and run `./build/Release/boys-accuracy-gate --strict` first — it prints the delivered error beside
the bound for every lane and region and ends in a verdict, so an issue can quote the line that
failed rather than describe it. If you suspect the generated tables,
`python3 tools/gen_boys_coefficients.py --check` re-derives them and the reference grid and reports
any drift, and `ctest --test-dir build --output-on-failure` runs the whole suite.

### Interval granularity

How narrowly the fitted domain is cut into pieces is the fifth field of `EvalPolicy`, and it is a
choice with a price on each side. A narrower piece needs a lower degree to hold the same bound —
halving a piece buys about `2^d` in the truncation, so **splitting is the lever and more degree is
not** — and the cost is that a table of narrow pieces stores more in total and needs a piece lookup
per call. Three partitions are offered and no spectrum between them. `FitGranularity::kCoarsest` is the
committed table the library has always carried. `FitGranularity::kNarrow` is a partition of region A
and of region B derived from the proved truncation bound below rather than placed by sampling, and it
is the default — the one a call site that names no partition reads. `FitGranularity::kUniform` is the
grid: one width for every interval, each interval fitted at its own degree, which is what lets a call
locate its piece by a multiply where the other two need a search.
`tools/gen_boys_coefficients.py --derive-partition` prints the design law's answer for all three.

| partition | stored coefficients | stored rows | read per evaluation |
| --- | --- | --- | --- |
| `FitGranularity::kCoarsest` | 1339 | 67 | 19 to 21 |
| `FitGranularity::kNarrow` | 3476 | 316 | 11 |

The grid is a third structure rather than a third row of that table: one width for every interval —
245 of them — with each interval fitted at its own degree, the double lane's held to degree 8 and the
float lane's to degree 4, so the coefficients a call reads are that interval's degree and not a
column of a partition-wide count. `tools/gen_boys_coefficients.py --derive-partition` prints its
ladder and the degrees it chose.

**Narrowing is a trade and not a saving.** The coefficients an evaluation reads fall from 19–21 to 11
on the default route and to 6–9 on the rational one, while the table a consumer carries grows from
1339 to 3476 on the default route, and the rational route's narrow table adds 2646 of its own over
the same 316 piece rows, with a piece lookup on every call. A consumer whose cost is per evaluation
gains; one whose cost is the table gains nothing and pays the lookup.

**Region A's pieces are held to a second reading that region B's are not.** The batch entry seeds its
downward recursion from the top order's piece, and that recursion carries the piece's error down to
F_0 multiplied by `w(n, b) = max(1, b^n / ∏(j + ½))` at the piece's right end `b` — 1.04436e5 at
order 12 and the region's right edge. A region-A cut must therefore hold `Δ(d')·w(b) ≤ target` as well
as its own size, so the walk places each piece under the tighter of the two: the 1e-15 bar the
single-order lane publishes for region A, and the batch lane's own `2.5e-14 / w(n, b)`. The gain
exceeds 25 on a trailing run of each order's pieces and tightens 127 of the partition's 311, the
tightest budget being 2.394e-19 at order 12's last piece, which ends at the region's right edge.

**The gain the walk holds is the envelope; the most a call in this revision reaches is 2.1711.** The
seeding fallback is taken only below the band's left edge, x < 1.0855, and what a call pays there is
its own top order's gain rather than the region's worst. For a fixed x the ratio x^n / ∏(j + ½) rises
with the order only until the order passes x − ½ and falls after, so below the band edge the largest
value it has at an order of 3 or more is order 3's 0.68: a call whose top order is 3 or more seeds at
gain 1. The two orders below that do amplify — 1.5712 at order 2 and 2.1711 at order 1, the second
only as x approaches the band's edge — a factor of 4.8e4 below the envelope the walk holds. Holding
the envelope rather than the reachable figure is deliberate, so that no piece's reading depends on
which order an entry happens to seed from, and the price it charges shows up as narrower pieces at
the high orders' right ends rather than as accuracy. The extended band is the same fit at either
granularity, because the upward recursion it feeds runs the other way and an error in it stays the
size it is.

**The member is certified.** Measured against the committed high-precision
reference over the interval its own pieces cover, `FitGranularity::kNarrow` delivers a worst absolute
error of 2.22e-16 over region A at region A's published 1e-15 bar — the shipped table's own figure —
and 7.21645e-16 over region B against the shipped seed's 9.9365e-15, a factor of 13.8. The gate's
granularity block carries its rows — one per partition, scheme and call shape, plus three for the
rational route over the narrow partition: its region-A pieces, its region-B
seed and the batch entry read through it — each judged against the bar the published table holds for
the cell it ran in, with the worst cell named. Those three rational rows
read 2.21663e-14 against the region's 3e-14, 4.12448e-14 against the seed's 5e-14 and 5e-14 against
the batch lane's 5.5e-14, which are the gate's own measured rows for those fits, on the run's
multiply-add route. The figures the generated header publishes for the two fitted rows are
2.42365e-14 over region A and 4.09566e-14 over region B, the same under both multiply-add routes,
and both are inside their bars as well. The block reports the trade above and the 4125493 of 5710087
axis cells (72.2%) that can discriminate, the rest carrying a bound at least as large as the value
itself.
Those rows are counted apart from every other book the gate reports, so nothing the library already
published moves.

`BoysFitGranularities()` states the interval
each partition's figures hold on, read off its own pieces and the fitted routes' domains: above that
interval the entry runs region C's asymptotic form, which no partition replaces, so a caller reading
one of these figures against a wider range would be matching the fitted tables' promise to an error
that is not theirs.

**Where a combination has no table or kernel it is refused where it is named**, with the reason,
rather than answered from another partition's fits. The rational minimax route has a narrow fit over
both regions and a fit over the shipped partition's; the single-precision lanes serve narrow tables on
both their routes; every partition is read from its own pieces, on either packing
axis, and the across-orders entries reach a per-order cut by fetching each order's own piece.

**Nothing of the axis cross is refused.** The rational route over the uniform grid was the last
member this space was owed — the grid's intervals are fixed by its width law rather than cut by a
criterion, so a rational pair over them was a fit to derive over the grid's own cells rather than a
table to cut — and it is now derived, emitted and served on every lane. **Every combination of the
axes is therefore either certified and published or a device cell a host without a CUDA device
cannot run**, which is the whole of the gate's arithmetic: none of them is refused.

**The entries are a further dimension, and the gate books them apart from the cross.** A call naming
the uniform grid through `BoysAllN`, `BoysAllNSorted`, `BoysAllNAtOrders` or `BoysFixedN` compiles
and is served: the entry hands a policy naming the grid to its per-argument path, which reads the
grid's own table rather than resolving the partition to a Chebyshev fit. The partitioned path is still never handed
the grid, and that is a contract rather than an omission — it reaches its values through a recursion
over the orders, which has no walk for a grid table, so `RefuseUniformPartition` names what that path
cannot answer. The gate crosses the entries in a book of its own, and that book reads 28 measured,
4 not applicable to the entry's shape and none refused, rather than letting the arithmetic above read
as a claim about the entries too.

A member that a *future* revision had not derived would still be refused where it is named, with the
reason, rather than answered from another partition's fits — which is what the refusals this library
carries used to say before each of them was served.

The partitions are different fits of the same function over the same interval, so a substitution
would return one partition's values under another's name — which is why the library refuses rather
than falling back, and why the packed lane carries the basis through rather than reusing a cut.

The proved bound is what the partition is derived from, and it needs no sampling: for this function
`|F_n(z)| ≤ F_n(Re z)` holds exactly, so the max modulus on a Bernstein ellipse is at most its value
at the ellipse's leftmost point, and Trefethen's interpolant bound ([Trefethen2019]) gives the
truncation. Its computation reproduces, to five significant figures, the two figures the generator's
own constant table carries (6.4676e-16 and 3.14e-18), and `docs/lane-contract.md` states the
derivation and the full trade curve, including why a degree-6 piece of the width a sevenfold split
gives misses a 1e-14 target: at that width the target needs degree 10, not degree 6.

## Version

This tree is **version 3.0.0**. That number is written down once, in `CMakeLists.txt`'s
`project(boys VERSION ...)`, and the test suite fails if anything else disagrees with it. A caller
reads it at run time through `boys::VersionString()`, or as the constants `boys::kVersionMajor`,
`boys::kVersionMinor` and `boys::kVersionPatch`, in `boys/version.hpp` (included by
`boys/boys.hpp`).

The version badge at the top of this file shows the latest *release tag*. It names the same version
when a release is cut, and can be older than the tree you are reading.

Public function signatures and supported domains are stable within a major version. Bitwise outputs
are not. Internal region thresholds, seed selection, recursion order and dispatch logic may change
between minor releases, as long as the bounds above hold. Byte-for-byte reproducibility requires
pinning the release tag, the compiler and the build flags.

## Building and consuming

Requires CMake (>= 3.25), git, and a C++23 compiler. The leg-by-leg record of what is built and
tested is the "Supported platforms" table below.

Optional builds: `-DBUILD_BENCHMARKS=ON` for the CPU benchmark drivers, which are local-only by
design, and `-DBUILD_CUDA=ON`, which needs the CUDA toolkit.

`-DBOYS_MULADD_SEPARATE=ON` moves the scalar multiply-add off the platform's fused call. On a target
with the fused instruction there is nothing to move and the option changes no result; on a target
without one the fused step is a library call per recurrence step, and this option removes it. It is a
**different arithmetic** — two roundings rather than one — and it has its own measured bound, which
`docs/lane-contract.md` states lane by lane. `BoysBackends()` reports which route is in force.
`-DBOYS_WERROR=OFF` drops `-WX` for a consumer whose compiler warning noise this tree has not been
made clean for. `-DBOYS_BUILD_DEFAULTS=<header>` compiles the five choices an entry that names no
policy resolves to from a header of your own instead of the shipped ones — the choices are
compile-time values, so an unnamed call costs what it costs either way. CONTRIBUTING.md states what
such a header carries and what the option does not do.

As a submodule:

    git submodule add https://github.com/myamlak/boys external/boys

```cmake
add_subdirectory(external/boys boys)
target_link_libraries(app PRIVATE boys::boys)
```

The tree sets no global standard or flags. Tests and benchmarks are OFF unless this tree is the
top-level project. See the API documentation and [CONTRIBUTING.md](CONTRIBUTING.md) for the details.

For what each lane guarantees, where it stops and why, see
[docs/lane-contract.md](docs/lane-contract.md).

## Which option is fastest on your machine

The fastest entry depends on the host and the build flags rather than on the library: whether the
vector tier is present, whether a bare `a * b + c` is one rounding or two in your compiler's hands,
and how your arguments arrive all move the ranking. The library carries the measurement instead of a
recommendation — a probe you build and run where you deploy:

    cmake -S . -B build
    cmake --build build --target boys-option-probe
    ./build/Release/boys-option-probe      # the config directory is your generator's

It reports, per option, the cost per argument, the spread of that cost over the paired rounds it was
measured in, the machine load those rounds were taken under, and the accuracy that option delivered
against the certified fp64 lane — so you can see whether a faster option was faster at the same
accuracy or merely at a lower one. The comparison is paired: every option is called once in every
round, and two options are compared by the ratio of their times *within one round*, so a clock that
drifts through a run cancels in that ratio instead of being read as a difference between the two
options. Each pass carries runs of a fixed-work canary beside its rounds. It is a diagnostic that
gates nothing: a fixed work read by wall clock measures the clock as much as the load, so a decaying
clock widens the canary on a machine that is doing nothing else, and a rule that discarded a pass on
that would discard the measurement rather than the machine. What the ordering is made in is the
spread of the paired ratios, which the report measures. The reported figure is the option's
within-round ratio to the reference lane at the middle of the run's rounds, scaled by that lane's own
cost, with its spread printed beside it: the middle rather than a lower quartile, because the
reference's ratio to itself is one in every round, so scoring every other option at a lower quartile
of its ratio would give the reference the middle of its own rounds and its rivals less than the
middle of theirs — a ranking that turns on which row the run anchored on.

Options are ranked in classes, and a class is one precision and one question shape: what the option
hands back, either one argument's ladder up to that argument's own order or one common ladder over
an array of arguments at one top order. Every row of a class
answers the same question, so nothing inside one is
an answer to something else; the routes, schemes, partitions, packing axes and division forms a
caller does not choose are columns inside the class and compete in one ranking. The default is taken
from the certified double lane's precision, answering
the shape this probe's workload asks: a faster row of another precision, or of
the other shape is a different class and never a default candidate. Within that class the default is
the row the run's own figures put first, so the name it prints and the table it prints it beside
never disagree about which option is cheapest. **A class that held no row for a member of an axis
prints a `NOT COMPARED:` line naming that member**, because a default read as a race it never ran is
worse than no default: the partition axis is the one this library's shipped default states as a
comparison, and a run whose rows are all one partition has not compared partitions at all.

When a class cannot be ordered — a pair whose within-round ratio band straddles one, or too few
rounds for a band to exist — the run still ends with one combination, and it says how it reached it.
The options the class left tied are re-run alone at a longer protocol (more passes over more rounds,
set by `--refine-runs` and `--refine-factor`) and voted on. That vote is read against the run it
refines: where it names the same row, the report says the re-run confirmed it; where it names
another, the report prints both figures and says the class's top entries cannot be separated by the
run, and the default is the row the report's own table puts first — never a row its own figures show
behind another. A class that holds one option names that option: one entry is not a ranking, and
there is no alternative to it. A run in which no option produced a figure at all reports `CANNOT
DETERMINE` and the number of paired rounds a band needs, rather than a name it never measured — and
it offers no fallback read from a
table, because a name chosen that way would be a name this run cannot stand behind. What the refusal
does give you is the evidence: every option the class could not place behind its leader is printed
with the band that pair fell in and in how many rounds each was the slower of the two.

It also checks the clock rather than assuming it: options can draw the clock differently, since a
wider vector register runs at a lower frequency, so the confidence line says how far the
widest-moving pair's ratio travelled between the run's first and second half beside the resolution
that figure is read against — warning when it went further — and whether every option the comparison
put against another ran one arithmetic route.
`boys::RunOptionProbe` is the entry and `boys::ProbeOptions` moves the workload to your basis; the
text it prints says the result is about the machine it ran on.

## Which CUDA entry is cheapest on your card

**Before you time anything, warm the path.** The CUDA entries upload their coefficient tables on
first use, lazily and idempotently, so the *first* call on a given
device pays a real one-time cost that every later call does not. A loop timed from a cold start
measures the upload rather than the evaluation. Make one throwaway call of the entry you intend
to use — or call `boys::BoysCuda::InitializeTables` — and time everything after it.

A CUDA ranking is a statement about a card, not about the library: a part whose documented ratio of
single- to double-precision throughput is 2 orders the fp64 and fp32 lanes differently from one whose
ratio is 32, and a card whose compute capability predates the bf16 tensor instructions has no tensor
path at all. So the CUDA lane carries the same kind of measurement. `boys::RunDeviceOptionProbe` takes
a device ordinal, establishes that device's context before it allocates anything, and returns a
`boys::DeviceProbeReport` — the card's name and compute capability in the returned data, one figure
per entry, one class per precision and question
shape with the resolution that class was ordered at, and the entries it could not separate. The
entry's cost and its documented bound are reported together.
`boys-device-probe` is a thin driver over it for the terminal:

    cmake -S . -B build-cuda -DBUILD_CUDA=ON
    cmake --build build-cuda --target boys-device-probe
    ./build-cuda/Release/boys-device-probe     # the config directory is your generator's

The option space the probe ranks is the library's own, reported by `boys::BoysDeviceOptions()` in
`boys_cuda_options.hpp` beside the entries it describes. A row carries what a chooser needs to place
an option: the entry, the precision it computes in, the question shape it answers, the axis it varies
where it has one — the single fp32 entry's region-B exponential, whose two members carry different
bounds — the degree tables it reads, and the bound its own documentation states. The probe's rows are
a projection of that report, so a precision, a shape or an axis member added to the surface appears in
both without an edit to the probe, and an option this build does not serve is carried with the reason
rather than left out. The device accuracy gate reads the same report for the same bounds, so what a
chooser is told and what the gate certifies cannot come apart.

The figures are device time only. The arguments and the output are uploaded and allocated once,
before the first clock, and every figure is a CUDA event pair around many back-to-back launches into
those resident buffers, divided by the number of launches and by the number of arguments — so neither
the transfer nor the host's submission of a launch is what is being timed. The entries a caller
reaches through `boys_cuda.hpp` are timed as the library's own kernel; the device-callable entries of
`boys_cuda_device.hpp`, which are meant to run inside the caller's kernel, are timed by subtraction —
the caller's kernel with the call in it, less the same kernel with the call removed and the traffic
kept — and the report names which method produced each row.

Cost that is paid once per launch and not once per call would sit in a figure read at a single
argument count as a constant divided by that count, so every row is read at two counts in the same
round — `--count`, and `--count` times `--pair-factor`, four times apart by default — and its figure
is the count-independent cost the two readings extrapolate to, `(r·f₂ − f₁) / (r − 1)` for readings
`f₁` and `f₂` a factor `r` apart, which assumes the launch term falls as 1/count. Both raw readings
are printed beside the figure, so the correction can be checked instead of taken, and a launched row
whose two readings left no positive launch term to remove has no figure at all rather than one this
run cannot stand behind: raising both counts is what answers that. Separately, a kernel launched the
same way that does no Boys arithmetic at all gives the floor per launch, which the report states as a
fraction of the fastest launched figure's own cost per call — the diagnostic that says when a
workload is too small to carry its own launch, so that the ranking is of the launcher rather than of
the arithmetic. It is a diagnostic and not a correction: an entry's kernel starts more work than an
empty one and its own launch costs more than the floor does, so subtracting the floor would take out
part of the term it is meant to remove.

Such a figure is checked against the protocol that produced it as well. The rows a shape's answer leans
on are re-timed at a much smaller and a much larger number of launches inside the region —
`controlRepetitionsLow` and `controlRepetitionsHigh`, four and sixty-four by default — and each is
reduced at both counts to the count-independent figure the table carries, which is the quantity the check
is on rather than a reading beside it. Those two figures are compared against the resolution this run
measured: a figure that moves between them by more than that is a cost paid per region rather than per
call, and the row is set aside instead of named. The report prints the two figures, their signed
difference and the band the difference fell in, so a row's removal can be checked as well as counted.

The ordering is paired, the way the host probe's is. Every entry is launched once in every round,
and two entries are compared by the ratio of their times *within one round*, so a card whose clock
moves through a run — a boost that decays as the part heats — cancels in that ratio instead of being
read as a difference between the two entries. The figure a row carries is its within-round ratio to
the reference entry at the middle of the run's rounds, scaled by that entry's own cost, with its
spread printed beside it: the middle rather than a lower quartile, for the same reason as on the host
side — the reference's ratio to itself is one in every round, so a lower quartile would credit the
reference and debit its rivals by an amount that depends on where the anchor was put. The minimum is
kept in a column of its own, labelled as the entry's fastest single round, so a peak-clock reading
and the figure a caller meets can be read apart. Every pass launches a fixed-work kernel that does no Boys arithmetic beside its
rounds and reports that kernel's own spread; it is a diagnostic that gates nothing, because fixed
work read by the same clock the entries ran on measures the clock as much as the card, and a rule
that discarded a pass on it would discard the measurement rather than the machine. What a shape's
ordering is made in is the spread of the paired ratios. A shape whose own rounds cannot separate two
entries still ends with exactly one of them — always the one its own figures put first, the row its
own table lists first, so a caller never meets a default the figures beside it contradict. The
entries the shape could not place behind the leader are re-run alone at a longer protocol, set by
`--refine-runs` and `--refine-factor`, and what those runs decide is *how* that entry was reached:
a unanimous re-run that named it, a majority that named it, or a vote that named another entry, which
is a tie the runs could not break. Those are three answers of different strength, and the report says
which one it is making, with the vote printed run by run. A shape whose rows the run's own checks set
aside — a subtraction that resolved nothing, a figure the repetition control could not hold — is not
left without an answer: the rows it did produce a figure for go to that same stage, and the shape ends
with one of them, named with the way it was reached rather than as an ordering this shape's own
rounds established. A shape holding one entry names that entry, since there is no alternative to
name and one entry is not a ranking; a shape that produced no figure at all is the one case that
reports `CANNOT DETERMINE` and names no entry, and the reason says which count or which row was
missing. Where a shape was not ordered outright, every entry it could not place behind the leader is
printed with the band that pair fell in and in how many rounds each was the slower of the two.

The device probe checks the clock rather than assuming it. Two entries need not carry this card's
clock alike — a unit the part runs at a lower rate, a kernel long enough to heat it — so the
confidence line says how far the widest-moving pair's ratio travelled between the run's first and
second half beside the resolution that figure is read against, and warns when a pair moved further
than the run can order, because the conclusion is then a property of that run's clock as well as of
the entries.

`boys::DeviceProbeStatus` is how a bad device is reported — an ordinal that does not exist is a
status and not a crash, and the call does not throw for it.

The report's last block counts the option space rather than describing it: every member — a row of
`boys::BoysDeviceOptions()` — is placed in one state of the run
(measured, refused by this build with the library's reason and owed, not runnable on this card,
offered and producing no figure, or not asked for by the run's request), the states are summed
against the space's own total, and the classes the space admits are
held against the classes the report carries. `boys::DeviceOptionSpaceClosure` answers those counts
to a program, and `boys-device-probe` exits non-zero when they do not close: a member of the space
in no state is a failed run and not a remark, so the device half is auditable the way the host
gate's combination block makes the host half auditable.

Measuring a device other than the one the caller has been using does not disturb the caller's. The
probe sets its device before it allocates or uploads anything, and puts the calling thread's device
back before it returns. Every table is uploaded to whichever device is current when it is uploaded
and is guarded on that device, so a copy another device already holds is never written over and a
device the caller had already set up is not re-uploaded: the two devices' tables stay resident side
by side and each keeps its own. The one trace the probe leaves is the lane's record of which device
the tables last went to, which now names the device it measured — and that record is read by the
guard that compares it against the current device, so the caller's next call on their own device
uploads the same tables once more. That is redundant work rather than changed state, and it is
idempotent by construction: a record left naming another device makes an upload happen, never makes
one be skipped.

## Supported platforms

Every row below is a CI leg that runs on every push to `main` and every pull request. The
architecture column is what that leg's dispatch assertion holds it to: the AVX2 tier is present on
x86_64 and absent everywhere else, where the entry points dispatch to the scalar lanes.

The table is generated from `.github/workflows/ci.yml`, `.github/required-checks.txt` and
`.github/option-matrix.json` by `tools/gen_platform_table.py`. The CI legs re-check it, so it cannot
drift from what actually runs.

The `option-matrix` legs are not platform legs: each one builds and runs the accuracy gate in one
certified configuration of the library — a host crossed with a multiply-add route — and the list of
those configurations is generated from the library by `tools/gen_option_matrix.py`, so a member added
to an option axis reaches CI by regenerating that list rather than by editing the workflow.

What each leg's *build* is — the instruction sets it targets, its lane widths, whether a bare
`a * b + c` in it is a single rounding, and whether the compiled library calls the runtime's `fma` —
is recorded per leg in [docs/build-facts.md](docs/build-facts.md), from what the leg's own build
reports about itself.

"Required" means branch protection on `main` will not merge a pull request until that check is green.
Every platform leg that runs is required. The option-matrix legs are not required yet: a name is added
to `.github/required-checks.txt` only once the leg has reported green at least once, so that a
mistyped name cannot block every pull request.

<!-- platform-table:begin (generated by tools/gen_platform_table.py; do not edit) -->
| CI leg (the check name) | Runner | AVX2 tier | Required |
|---|---|---|---|
| `option-matrix (the configuration list)` | `ubuntu-latest` | n/a | no |
| `option-matrix linux-gcc fused` | `ubuntu-latest` | present | no |
| `option-matrix linux-gcc separate` | `ubuntu-latest` | present | no |
| `option-matrix linux-clang fused` | `ubuntu-latest` | present | no |
| `option-matrix linux-clang separate` | `ubuntu-latest` | present | no |
| `option-matrix windows-msvc fused` | `windows-latest` | present | no |
| `option-matrix windows-msvc separate` | `windows-latest` | present | no |
| `windows-msvc Release` | `windows-latest` | present | yes |
| `windows-msvc Debug` | `windows-latest` | present | yes |
| `linux-x86 gcc Release` | `ubuntu-latest` | n/a | yes |
| `linux-x86 gcc Debug` | `ubuntu-latest` | n/a | yes |
| `linux-x86 clang Release` | `ubuntu-latest` | n/a | yes |
| `linux-x86 clang Debug` | `ubuntu-latest` | n/a | yes |
| `linux-sanitizers asan+ubsan` | `ubuntu-latest` | n/a | yes |
| `linux-arm64 gcc Release` | `ubuntu-24.04-arm` | n/a | yes |
| `windows-arm64 msvc Release` | `windows-11-arm` | n/a | yes |
| `macos arm64` | `macos-latest` | absent | yes |
| `macos x64` | `macos-15-intel` | present | yes |
| `linux-x86 clang-tidy` | `ubuntu-latest` | n/a | yes |
<!-- platform-table:end -->

## Reproducibility

- `tests/data/boys_reference.csv` — the committed 45-digit reference grid.
- `tools/gen_boys_coefficients.py` — regenerates the Chebyshev tables and the grid from the cited
  formulas, using a pinned public `mpmath`. Its `--check` flag is the byte-identity proof, and one CI
  leg runs it: the linux-x86 gcc Release leg, in the workflow's Generator consistency step, which
  re-derives both artefacts into scratch copies and compares them byte for byte without writing
  either committed file. The other legs build and test without it. The emitted header is
  clang-format output, so
  `requirements-boys.txt` pins the formatter as well. The script reads `$CLANG_FORMAT` before
  searching `PATH`. Without a formatter it fails, rather than writing bytes the gate would report as
  drift.

## Citation

The published works the numerical claims rest on are listed in [CITATION.bib](CITATION.bib).

## License

BSD-3-Clause — see [LICENSE](LICENSE), `Copyright (c) 2026 Marcin Makowski`. Dependency details in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
