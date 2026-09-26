# The Boys function kernel

[![CI](https://github.com/myamlak/boys/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/myamlak/boys/actions/workflows/ci.yml)
[![docs](https://github.com/myamlak/boys/actions/workflows/docs.yml/badge.svg)](https://github.com/myamlak/boys/actions/workflows/docs.yml)
[![license](https://img.shields.io/badge/license-BSD--3--Clause-blue)](LICENSE)
[![version](https://img.shields.io/github/v/tag/myamlak/boys?label=latest%20release%20tag)](https://github.com/myamlak/boys/tags)

Self-contained C++23 evaluation of the Boys function family
F_n(x) = ∫₀¹ t^(2n) exp(−x t²) dt for n = 0..32 — the integral an electronic-structure program
evaluates at every order a shell quartet can ask for.

It ships the function in double and single precision, scalar and vectorised, with 16-bit storage
wrappers, a packed-half variant, an entry that evaluates a batch of arguments as a matrix product for
tensor cores, and optional CUDA kernels. It depends on nothing outside the C++ standard library, the
optional CUDA toolkit, and tables generated and committed with the source. Every entry is checked
against a reference grid computed to 45 digits.

## Quick start

    git clone https://github.com/myamlak/boys.git
    cmake -S . -B build
    cmake --build build
    ctest --test-dir build --output-on-failure

The tests are quiet when they pass: `ctest` prints a green line per test and nothing about what was
checked. To see the accuracy figures themselves, see [Check the figures
yourself](#check-the-figures-yourself) below.

For a one-file program against an already-built tree, compile and run it like this:

    g++ -std=c++23 -I include try.cpp -L build -lboys -o try
    ./try

The public headers need C++20 — that is the library's declared requirement, `cxx_std_20` on the
`boys` target; the tree's own tests are built as C++23. So `-std=c++20` compiles this same program.
On Windows with the Visual Studio generator the library lands in `build/Release/`, and the CMake
route below is the one that finds it without flags.

```cpp
#include <boys/boys.hpp>

int main()
{
    double f0 = boys::BoysSingle(0, 0.5);  // F_0(0.5)

    std::array<double, 8> out;
    boys::BoysAllOrders(7, 1.25, out.data());  // F_0..F_7(1.25)

    // Many arguments, all orders: the batch shape an integral engine needs.
    // Arguments may arrive in any order; the entry classifies, groups and
    // dispatches internally. out[k * count + i] = F_k(x[i]).
    const double x[3] = {0.5, 12.5, 30.0};
    std::array<double, 3 * 33> planes;
    boys::BoysAllN(32, x, planes.data(), 3);

    // Already non-decreasing? Say so and skip the sort.
    boys::BoysAllN(32, x, planes.data(), 3, boys::BoysSortedArgs{});

    // Each argument its own highest order — the shape a shell-quartet batch
    // has. The planes are the same, each column stopping at its own n[i];
    // the cells above n[i] are left as the caller left them.
    const int n[3] = {4, 12, 7};
    boys::BoysAllNAtOrders(n, x, planes.data(), 3);
}
```

**Choosing a lane.** [docs/consumer-perspective.md](docs/consumer-perspective.md) — how much accuracy
an integral calculation actually needs, and which of the library's evaluation lanes that leaves to
choose between.

**The accuracy contract.** [What each lane guarantees](#accuracy-contract) — the bound, and the
command that measures it on your machine.

The words this library uses — *lane*, *region*, *route*, *rung*, *tier*, *scheme*, *axis*, *gate* —
are defined on the documentation's landing page, together with the full API reference:
<https://myamlak.github.io/boys/>.

## Accuracy contract

The bound is |F̂_n(x) − F_n(x)| ≤ m·B, for every supported n, x and lane. Here m is the compile-time
accuracy multiplier of the call, and defaults to 1.

The library evaluates the function differently at different argument sizes, and one lane is tighter
over part of the range than over the rest. Every figure below holds for **all** x ≥ 0:

| Lane | Error bound |
|---|---|
| double single | ≤ m·5.5e-14 everywhere; ≤ m·3e-14 below x = 11.899848152108484; ≤ m·1e-15 below about x = 1.0855 |
| double batch, whether the top order is the batch's or each argument's | ≤ m·5.5e-14 |
| float single / batch | ≤ m·1.5e-7 |
| fp16 / bf16 | ≤ m·1e-7 + ½ ULP |
| native half, x ≥ 28.984375 | ≤ 8 ULP of the returned value |
| CUDA fp64 | same m·budgets as the CPU double lanes |
| CUDA fp32, `RegionBExp::kAccurate` (the default) | same m·budgets as the CPU float lanes |
| CUDA fp32, `RegionBExp::kFast` | ≤ m·1.5e-7 + 8e-8 — the lane's budget plus the corrected seed's own contribution |

"ULP" is the last representable digit of the result in the format concerned. On the C++ surface the
multiplier is any value at or above 1, with no upper end, and raising it loosens the bound and
reduces the work on the default route, which pays for the looser bound with fewer coefficients to
sum. The rational route's rung is derived by the same criterion over its own stored numerator and
denominator pair, and at the six multipliers this library names it certifies the stored pair
unchanged — so naming a rung there loosens the bound and changes nothing about the work. What the
criterion certifies, and the figure that makes the outcome checkable, are in
[docs/lane-contract.md](docs/lane-contract.md). The C surface takes a sampled set instead, listed in
its own header, and so do the CUDA lane's device-callable entries, because the rung a call names
has to be the one `BoysCuda::DeviceTables` was instantiated with: the relaxed degree tables are
resident for one rung at a time, and a call naming any other rung returns
`BoysDeviceStatus::kMultiplierNotResident` and writes nothing. m = 1 needs no such table and is
always served.

The rows above are bounds, and a bound is not the figure a lane delivers. Two lanes are delivered at
a different figure depending on one property of the build — whether the compiler fuses a bare
product-plus-add into a single rounding. **The architecture does not decide it**: of the six
configurations measured, gcc and AppleClang on arm64 contract one and MSVC on arm64 does not, so the
MSVC arm64 build delivers the x86-64 figures rather than its own architecture's. At the default
multiplier, against the committed reference grid:

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
bound. `RegionBExp::kAccurate` is the default and the arithmetic the batch entries already run.
`RegionBExp::kFast` is the hardware approximation with its argument-scaling residual removed.

The correction is what the recurrence asks for, and the recurrence is what makes the choice
non-trivial: its condition number is 7.6e4 at the region-B boundary at order 32, so an error in the
e^{-x} it consumes is amplified by that much there, while an approximation whose relative error
grows with the argument carries fifteen ulp at the boundary and thirty-five at the far end of the
region. The bare approximation therefore fails the lane's budget over a band of region B at the
highest order — predicted at 2.30e-7 against the 1.5e-7 budget, measured at 2.57e-7 — and returns
the wrong sign there. It is not offered at any multiplier, because the failing band is interior to
region B and there is no certified sub-range to restrict it to. Corrected, the seed's relative error
is flat at a few ulp, its contribution to the value is capped at 8e-8 by that same amplification
(measured 5.0e-8), and the option returns the function's sign at every cell the device gate audits.

No speed is claimed for either option. The corrected form's kernel is the same size statically as the
accurate one — 944 instructions against 944 at m = 1 — and the exponential is evaluated once per
element outside the order loop; what separates the two options is the bound, not a measured time.

The GPU and CPU float lanes agree to within 3.5e-7. That is a cross-lane statement about
|GPU − CPU| and not a bound on either lane's distance from F_n(x): the default CUDA fp32 entry is
held to 1.5e-7 above, and the fast option's looser bound is not covered by the 3.5e-7 figure.

### Check the figures yourself

Every figure above is measured, and the measurement ships in this tree. Build the gate and run it:

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
does not hold, so these figures cannot drift from the code without a red run. A figure is one host's:
your own run is the answer for your platform, your compiler and your arithmetic, and the rows are
compared cell by cell rather than as one number.

Run at revision f5c08f6, it printed this (an excerpt; the full run carries one row per lane and
region):

| Lane | Region | Worst delivered | Bound claimed |
|---|---|---|---|
| double single | A | 2.22e-16 | 1e-15 |
| double single | band | 3.22e-15 | 3e-14 |
| double single | B | 9.94e-15 | 3e-14 |
| double single | C | 5e-14 | 5.5e-14 |
| float single | all | 1.06e-07 | 1.5e-07 |
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
| C (`boys_c.h`) | Every entry validates and returns a status instead: `BOYS_ERROR_INVALID_ARGUMENT` (1) for an order outside [0, 32], a negative or NaN x, or a null pointer, and `BOYS_ERROR_UNSUPPORTED_MULTIPLIER` (2) for a multiplier outside the sampled set its header lists. No abort and no undefined behaviour. |
| CUDA | The single, batch and all-N entries return `boys::BoysStatus`, whose `kInvalidArgument` is the same validation as the C surface's; the device-callable entries take their orders as template arguments, so an order outside the range does not compile. |
| Native half (`BoysAllOrdersHalf2`, `BoysAllNF16Native`) | Same `assert`, and one precondition of its own: the entry covers region C only, so an x below the region-C boundary is a violation rather than a fallback to the table regions. |
| Output spans | The caller sizes them. A span shorter than the call writes is an out-of-bounds write with no check on any surface, and no entry reallocates or truncates. |

### Fit routes

The double lane's stored fits come in two routes, and a caller picks one per call with
`BoysAllOrdersWithRoute`. `BoysFitRoutes()` reports them: for each route and region, the interval
its fit covers, the argument its selector takes over at, how many coefficients it stores, the worst
error it was measured to deliver, and the bar it is certified against. Both routes hold the same bar
over the same interval; they differ in what they store to do it. The routes are alternatives, not
rungs: naming one changes only the fits that serve the intervals its rows report, and everywhere else
the entry runs the default route and returns its values bit for bit, so the lane's tighter per-order
1e-15 is untouched. A route name the build does not serve evaluates the default route rather than
returning something the caller did not ask for. **No speed is claimed for either route.**

The route composes with the evaluation scheme and with the accuracy multiplier into one `EvalPolicy`,
which every templated double-precision entry takes, and `BoysAllOrdersAtTier(tier, route, scheme,
...)` — with `BoysSingleAtTier` on the single-order shape — names the selectors at run time.

[docs/lane-contract.md](docs/lane-contract.md#the-two-fit-routes) carries this in full: the table of
routes with each one's stored count, measured error and bar; each route's rung and the criterion that
derives it; which entries carry a route and what that carriage delivers; and why the rational route
buys the interval's values and not a recursion seed.

#### The float lane's routes

The float lane's region-A table and its region-B seed are its own, and they come in the same two
routes. They are selected at run time with `BoysSingleF32WithRoute`, which is `BoysSingleF32` with
one thing changed — which fit supplies the lane's region-A seed and its region-B seed — and
`BoysFitRoutesF32()` reports them the way `BoysFitRoutes()` reports the double lane's. Region C holds
no coefficient under either route, so naming one there changes nothing.

**The same choice is open at compile time, and there it carries the scheme as well.** The lane's
entries take the `EvalPolicy` above, so a caller who templates on it names the route and the scheme as
the second template argument instead of calling `BoysSingleF32WithRoute`. At the reference multiplier
both fields are read: the route selects which family supplies the lane's fits, and the scheme selects
which of the Chebyshev family's two stored forms is summed. Past the reference multiplier the lane
serves the shipped pair alone — a rung cuts a fit by a table of effective degrees, and the degrees the
float lane's fits are cut by are derived from the lane's Chebyshev table, so a policy naming another
pair at a relaxed rung does not build; a build writes the measurement that says so. `BoysAllNF32`
forwards its policy to the batch entry's body once per argument.

The float lane's rational route covers the whole of region A from zero: the lane reads every order
from its own fit over the region, so there is no band boundary for a selector to take over at and
the row states no `servesFrom` above zero.

| Route | Region | Interval | Stored | Measured | Bar |
|---|---|---|---|---|---|
| chebyshev (default) | A | [0, 11.899848152108484) | 1067 | 1.06e-07 | 1.5e-07 |
| rational minimax | A | [0, 11.899848152108484) | 525 | 1.11e-07 | 1.5e-07 |
| chebyshev (default) | B | [11.899848152108484, 28.98933773882074) | 11 | 2.77e-08 | 1.5e-07 |
| rational minimax | B | [11.899848152108484, 28.98933773882074) | 6 | 7.50e-08 | 1.5e-07 |

The measured column is the accuracy gate's own worst over the committed reference grid, on the
entry the row names, every order 0..32 and every sample of the row's interval; the gate prints the
same sweep beside it. Both routes hold the lane's own bound, and the two region-A rows are the
comparison: the same interval, the same reference, the same arithmetic, 525 stored coefficients
against 1067. The rational route's pieces are its own cover of each order's interval rather than the
Chebyshev table's breaks, and both were accepted against the same criterion — half the lane's
tolerance, weighted by the downward recursion's gain, **on the coefficients as they are stored**,
because at this target the binary32 rounding is part of the fit and not a last-digit detail.

**The Chebyshev route's fits are stored in both of the forms the two schemes read**, one monomial
coefficient per Chebyshev coefficient: 1067 stored either way in region A and 11 in region B, the same
pieces, intervals and degrees, so naming a scheme chooses a table and not a shape. Over region A the
gate reads 1.06e-07 at the worst cell of the split Clenshaw reading and 7.68e-08 of the Horner
reading, over region B 2.77e-08 and 2.22e-08, all inside the lane's 1.5e-07 bar. The second form costs
this lane nothing, and the reason is the length of its pieces: the monomial coefficients of any one
piece sum in absolute value to at most 1.6, so rounding the monomial table to binary32 moves a value
by at most 0.64 of the bar even if every coefficient rounds against it, and the measured reading is
half the bar.

**The stored counts are upper bounds and not minima.** The degree search is a scan that stops at the
first count holding the target, not an exhaustive minimax search over the family, so a cheaper cover
of the same intervals may exist; the counts above are what the search found, and a reader comparing
them is comparing what the two tables cost as generated.

**Region B is the other direction.** There the rational seed stores 6 coefficients against the
Chebyshev seed's 11 and delivers 7.50e-08 against 2.77e-08. Half the coefficients at more error is a
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
and the same ≤ m·1e-15 bar the scalar region-A path holds, and naming it changes the region-A values
a caller receives — the shipped entry reaches most orders by a recursion from a seed where this lane
evaluates each order's own fit — with both inside the bound.

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

Every other combination the axis names is built and measured. A relaxed multiplier is answered by the
same effective-degree cut this library's other rungs are truncated by, applied at the degree the lane
reads; a route other than the shipped one is answered by that route's own region-A fits, whose pieces
cover the same per-order intervals as the shipped table. The gate's packing book carries a measured
row for every rung the tier enumeration declares, on both routes, at both schemes, through both
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
per call. Two partitions are offered and no spectrum between them: `FitGranularity::kShipped`, which
is the committed table and the default, and `FitGranularity::kNarrow`, a partition of region A and of
region B derived from the proved truncation bound below rather than placed by sampling.
`tools/gen_boys_coefficients.py --derive-partition` prints the design law's answer.

| partition | stored coefficients | stored rows | read per evaluation |
| --- | --- | --- | --- |
| `FitGranularity::kShipped` | 1339 | 67 | 19 to 21 |
| `FitGranularity::kNarrow` | 3476 | 316 | 11 |

**Narrowing is a trade and not a saving.** The coefficients an evaluation reads fall from 19–21 to 11
and the table a consumer carries grows from 1339 to 3476, with a piece lookup on every call. A
consumer whose cost is per evaluation gains; one whose cost is the table gains nothing and pays the
lookup.

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

**The member is certified.** Measured against the committed high-precision reference over the
interval its own pieces cover, `FitGranularity::kNarrow` delivers a worst absolute error of 2.22e-16
over region A at region A's published 1e-15 bar — the shipped table's own figure — and 7.21645e-16
over region B against the shipped seed's 9.9365e-15, a factor of 13.8. The gate's granularity block
carries all 32 of its rows, one per partition, scheme and call shape, each judged against the bar the
published table holds for the cell it ran in, with the worst cell named; and it reports the trade
above and the 481700 of 578888 axis cells (83.2%) that can discriminate, the rest carrying a bound at
least as large as the value itself. Those rows are counted apart from every other book the gate
reports, so nothing the library already published moves.

**Where a combination has no narrow table it is refused where it is named**, with the reason, rather
than answered from the shipped table: the rational minimax route (one numerator/denominator pair over
the whole interval), the relaxed rungs `m > 1` (which truncate the shipped fits to certified
effective degrees), and the single-precision lanes (which hold one coefficient set). The two
partitions are different fits of the same function over the same interval, so a substitution would
return the shipped values under the narrow partition's name.

The proved bound is what the partition is derived from, and it needs no sampling: for this function
`|F_n(z)| ≤ F_n(Re z)` holds exactly, so the max modulus on a Bernstein ellipse is at most its value
at the ellipse's leftmost point, and Trefethen's interpolant bound gives the truncation. Its
computation reproduces the figures this tree's generator carries from an earlier instrument
(6.4676e-16 and 3.14e-18, to five significant figures), and `docs/lane-contract.md` states the
derivation, the full trade curve and a quoted "19 → 7" measurement **that does not survive the
check**: at width 17.09/7 the target needs degree 10, not degree 6.

## Version

This tree is **version 2.0.0**. That number is written down once, in `CMakeLists.txt`'s
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
made clean for.

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
and how your arguments arrive all move the ranking. The library ships the measurement instead of a
recommendation — a probe you build and run where you deploy:

    cmake -S . -B build
    cmake --build build --target boys-option-probe
    ./build/Release/boys-option-probe      # the config directory is your generator's

It reports, per option, the cost per argument, the spread over the passes it was measured in, the
machine load each pass was taken under, and the accuracy that option delivered against the certified
fp64 lane — so you can see whether a faster option was faster at the same accuracy or merely at a
lower one. Every pass carries repeated runs of a fixed-work canary, and a pass is admitted only if
those runs agree closely enough with each other: a steady load slows every option alike and leaves
their order alone, an unsteady one is what corrupts a comparison, and the canary's own spread
measures that unsteadiness directly. A pass it cannot vouch for is discarded rather than averaged in,
and the figure is the minimum of the admitted passes.

When two options are closer than the resolution the run measured — the larger of the canary's widest
admitted spread and the leading option's own spread across its admitted passes, both measured rather
than assumed — it prints `CANNOT DETERMINE` and names what it could not separate rather than ordering
noise. `boys::RunOptionProbe` is the entry and `boys::ProbeOptions` moves the workload to your basis;
the text it prints says the result is about the machine it ran on.

## Which CUDA entry is cheapest on your card

A CUDA ranking is a statement about a card, not about the library: a part whose documented ratio of
single- to double-precision throughput is 2 orders the fp64 and fp32 lanes differently from one whose
ratio is 32, and a card whose compute capability predates the bf16 tensor instructions has no tensor
path at all. So the CUDA lane ships the same kind of measurement. `boys::RunDeviceOptionProbe` takes
a device ordinal, establishes that device's context before it allocates anything, and returns a
`boys::DeviceProbeReport` — the card's name and compute capability in the returned data, one figure
per entry with the spread it was taken under, one conclusion per question class with the resolution
that class was ordered at, and the entries it could not separate. `boys-device-probe` is a thin
driver over it for the terminal:

    cmake -S . -B build-cuda -DBUILD_CUDA=ON
    cmake --build build-cuda --target boys-device-probe
    ./build-cuda/Release/boys-device-probe     # the config directory is your generator's

The figures are device time only. The arguments and the output are uploaded and allocated once,
before the first clock, and every figure is a CUDA event pair around many back-to-back launches into
those resident buffers, divided by the number of launches and by the number of arguments — so neither
the transfer nor the host's submission of a launch is what is being timed. The entries a caller
reaches through `boys_cuda.hpp` are timed as the library's own kernel; the device-callable entries of
`boys_cuda_device.hpp`, which are meant to run inside the caller's kernel, are timed by subtraction —
the caller's kernel with the call in it, less the same kernel with the call removed and the traffic
kept — and the report names which method produced each row.

Two checks say what the timer is and is not measuring, and both are reported rather than assumed. One
entry is timed at two very different numbers of launches per region. A cost that repeats with the
launch rather than with the call is divided by a different number at each count, so it would move the
figure; a row whose figure moves further than that run can place the row is set aside and the class
falls to the next entry rather than shipping it. And a kernel launched the same way that does no Boys
arithmetic at all gives the floor per launch, which the report states as a fraction of the fastest
launched figure's own cost per call: on a platform whose host submission is expensive, a workload too
small to carry its own launch is a workload whose ranking is of the launcher. Raising
`boys::DeviceProbeOptions::count` is what answers that.

`boys::DeviceProbeStatus` is how a bad device is reported — an ordinal that does not exist is a
status and not a crash, and the call does not throw for it.

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
  formulas, using a pinned public `mpmath`. Its `--check` flag is the byte-identity proof, and every
  CI leg runs the full regeneration protocol. The emitted header is clang-format output, so
  `requirements-boys.txt` pins the formatter as well. The script reads `$CLANG_FORMAT` before
  searching `PATH`. Without a formatter it fails, rather than writing bytes the gate would report as
  drift.

## Citation

The published works the numerical claims rest on are listed in [CITATION.bib](CITATION.bib).

## License

BSD-3-Clause — see [LICENSE](LICENSE), `Copyright (c) 2026 Marcin Makowski`. Dependency details in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
