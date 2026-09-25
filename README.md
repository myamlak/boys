# The Boys function kernel

[![CI](https://github.com/myamlak/boys/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/myamlak/boys/actions/workflows/ci.yml)
[![docs](https://github.com/myamlak/boys/actions/workflows/docs.yml/badge.svg)](https://github.com/myamlak/boys/actions/workflows/docs.yml)
[![license](https://img.shields.io/badge/license-BSD--3--Clause-blue)](LICENSE)
[![version](https://img.shields.io/github/v/tag/myamlak/boys)](https://github.com/myamlak/boys/tags)

Self-contained C++23 evaluation of the Boys function family
F_n(x) = ∫₀¹ t^(2n) exp(−x t²) dt for n = 0..32.

It ships scalar fp64 and fp32 lanes, an AVX2 vector tier reached through the same entries, fp16 and
bf16 I/O wrappers, a native packed-half lane, a matrix-product entry for region A that can run on
tensor cores, and optional CUDA kernels. It depends on nothing outside the C++ standard library, the
optional CUDA toolkit, and the committed generated tables. Every lane is validated against a
committed 45-digit reference grid.

**Full API documentation: <https://myamlak.github.io/boys/>**

## Quick start

    git clone https://github.com/myamlak/boys.git
    cmake -S . -B build
    cmake --build build
    ctest --test-dir build --output-on-failure

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
}
```

## Accuracy contract

The bound is |F̂_n(x) − F_n(x)| ≤ m·B, for every supported n, x and lane. Here m is the compile-time
accuracy multiplier of the call, and defaults to 1.

The library evaluates the function differently at different argument sizes, and one lane is tighter
over part of the range than over the rest. Every figure below holds for **all** x ≥ 0:

| Lane | Error bound |
|---|---|
| double single | ≤ m·5.5e-14 everywhere; ≤ m·3e-14 below x = 11.899848152108484; ≤ m·1e-15 below about x = 1.0855 |
| double batch | ≤ m·5.5e-14 |
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
its own header.

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

The **CUDA fp32 lane's single entry** takes a certified choice of region-B exponential
(`boys::RegionBExp`), and each choice carries its own bound. `RegionBExp::kAccurate` is the default
and the arithmetic the batch entries already run. `RegionBExp::kFast` is the hardware approximation
with its argument-scaling residual removed.

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

### Fit routes

The double lane's stored fits come in two routes, and a caller picks one per call with
`BoysAllOrdersWithRoute`. `BoysFitRoutes()` reports them: for each route and region, the interval
its fit covers, the argument its selector takes over at, how many coefficients it stores, the worst
error it was measured to deliver, and the bar it is certified against. Both routes hold the same bar
over the same interval; they differ in what they store to do it. `FitRoute` names the choice, and
the default route is the fits this library has always shipped, value for value.

The routes are alternatives, not rungs: naming one changes only the fits that serve the intervals
its rows report. Everywhere else — outside them, and below the argument the region-A rational row
names — the entry runs the default route and returns its values bit for bit, so the lane's tighter
per-order 1e-15 is untouched. A route name the build does not serve evaluates the default route
rather than returning something the caller did not ask for.

| Route | Region | Interval | Stored | Measured | Bar |
|---|---|---|---|---|---|
| chebyshev (default) | A | [0, 11.899848152108484) | 1320 | 1.74e-16 | 3e-14 |
| rational minimax | A | [0, 11.899848152108484), from x = 1.0855252345349333 | 891 | 2.46e-14 | 3e-14 |
| chebyshev (default) | B | [11.899848152108484, 28.98933773882074) | 19 | 9.92e-15 | 5e-14 |
| rational minimax | B | [11.899848152108484, 28.98933773882074) | 12 | 4.46e-14 | 5e-14 |

Region A's two rows are the whole per-order table, one piece per order per band, and they are what
the two routes are compared on: the same intervals, the same reference, the same arithmetic. The
rational row's fit covers the same interval from zero, but its selector takes over per order, from
that order's own argument, because that is where the lane stops reading the order from its own fit
and reaches it from the band seed instead, at the band's figure rather than the order's. Below that
argument the order keeps the default route's value exactly. The argument in the row is the lowest of
those boundaries, and they are the same per-order arguments the lane's own dispatch uses. Region B's
two rows are the one seed each route evaluates there.

The default table's pieces do a second job that the rational route's do not: the batch entry's
relaxed path seeds its downward recursion from the top order's piece, and that recursion carries the
seed's error down to F_0 with a gain that grows with the argument — up to 1.04e5 near the right end
of the region, at order 12. The default pieces are fitted to hold their error after that gain, which
is why the table stores more than the plain figure its row reports would need, and why its stored
count is not the count the same bar alone would need. The rational route's pieces are read one order
at a time and take no recursion; **they are held to the bar its row states and not to the tighter
criterion that fitting for the recursion's gain would impose**, and a piece fitted for values alone
does not survive being used as that seed. Naming the rational route buys the interval's values, not a
recursion seed.

**No speed is claimed for either route.** A rational costs one division per order where the
Chebyshev form is division-free, so which is cheaper depends on the machine's divide-to-multiply
throughput, and the measurements that would settle it have not been taken. The routes are offered
because they are different shapes, not because one is known to be faster.

The route and the evaluation scheme compose: they are the two fields of one `EvalPolicy`, which every
templated double-precision entry takes as its second parameter, and `BoysAllOrdersWithRoute` names
both at run time. The route names the fits; the scheme names the summation the Chebyshev
coefficients are read in, and it reaches the parts of a call the named route's own fits do not serve.
A route whose own fit has one stored form — the rational minimax family — evaluates that fit the same
way under either scheme, so naming a scheme changes the values only where the shipped family answers.
Calls that name no policy at all compile the default pair, which is the Chebyshev route by the split
Clenshaw recurrence.

**The multiplier is a third selector, and it acts on the named route.** A rung cuts the route's own
stored fit to the degrees a criterion certifies for that multiplier, and the two routes' criteria
read different tables: the Chebyshev family's is a cut of its stored coefficient series, and the
rational family's a cut of its stored numerator and denominator pair. So the pair of selectors is a
combination rather than a redundancy, and it is offered on every entry, named at run time by
`BoysAllOrdersAtTier(tier, route, scheme, ...)`. Each rung of each route carries a measured
bound on the committed reference: [docs/lane-contract.md](docs/lane-contract.md) states what the two
criteria derive and publishes the figures the gate measures, including the one that says the
rational route's own rungs do not truncate at these six multipliers.

**The route is carried on every entry.** `BoysSingle`, `BoysAllOrders`, `BoysAllN` and `BoysFixedN`
all take the route the policy names, and each answers with the route's own fits rather than with the
shipped ones under its name: the two shapes that reach their values by a path of their own — the
plane entry's region-grouped path and the fixed-order entry's shaped path — are the shipped route's,
and a call naming the rational route is served by the per-argument body instead, which reads its fit
from the policy. The gate measures the carriage as a difference in the values rather than as a
sentence about the surface: naming the route changes what four entries return over the interval the
route's rows cover, and naming the default changes nothing.

It also measures the delivery: `BoysAllN` and `BoysFixedN` are swept over the whole committed grid,
every order, and judged against the named route's own bar for the region the argument falls in —
113,388 cells, worst delivered 5e-14 at n = 32, x = 28.98933773882074, no cell over. That bar is the
route's, not the entry's, so the row is the stronger of the two statements: the route's region-A row
promises 3e-14 where the entry promises 5.5e-14, and the entry holds the tighter figure over the
region, including the arguments below the route's own selector where the shipped lane answers.

### The packing axis

A packed lane keeps four doubles in a register and the call has to supply four of something. That
something is a choice, and it is the fourth field of `EvalPolicy`: `PackAxis::kArguments`, the
shipped axis, puts four arguments at one order in a register, and `PackAxis::kOrders` puts four
orders at one argument there — which is the axis `BoysAllOrders(nmax, x, out)` actually has, since
that entry computes every order at a single argument. `BoysPackAxes()` reports both, with the
interval each one's packed lane evaluates.

Every entry whose call shape has four orders to offer carries the orders axis, and there are two of
them: `BoysAllOrders`, and `BoysAllN`, whose planes are `out[k * count + i] = F_k(x[i])` — so an
argument's whole order vector is already what that entry writes. A plane call naming the axis takes
the entry's per-argument path and returns the all-orders entry's values under it **bit for bit**,
asserted with no tolerance, because the two call shapes reach one body. What the axis trades on the
plane entry is the region grouping: the shipped path groups the arguments by dispatch interval to
feed a lane that packs four *arguments*, and an orders-axis call has one argument to pack and so
nothing to group.

Two calls cannot form the axis, and both are refused where the call is named rather than answered.
`BoysFixedN` computes exactly one order at every argument of an array: a packed lane keeps four
orders, and this call has one, so there are not four to fill a lane with — that is the entry's
signature and not a body nobody built. And a relaxed multiplier on the axis is refused at every entry
that carries it, because the packed orders lane evaluates every stored fit at its full degree and
reads no effective-degree table, so it carries no rung.

The orders axis covers region A, `0 <= x < 11.899848152108484`, at the same per-order fits and the
same `m·1e-15` bar the scalar region-A path holds. At the split Clenshaw scheme its values are the
across-arguments lane's values bit for bit, order for order; past that interval the entry runs the
certified scalar lane one order at a time, so it answers for every argument the library accepts.
Naming it changes the region-A values a caller receives — the shipped entry reaches most orders by a
recursion from a seed where this lane evaluates each order's own fit — and both are inside the bound.

Measured on one machine (Intel Core i7-9850H, Linux `perf`, `uops_retired.retire_slots:u`), over
2,048,000 calls of `BoysAllOrders(32, x, out)` at 1,024 arguments in region A: the orders axis
retires **4,719,806,211** slots against the shipped entry's **6,687,926,524**, a ratio of **1.42**,
and against the across-arguments lane forced onto that shape it retires **4,719,806,211** against
**27,806,856,380**, a ratio of **5.89**. **No time is claimed, here or anywhere: a µop count is a
property of the CPU it was taken on, and a time taken on a loaded machine is not a measurement.**
`docs/lane-contract.md` carries the full table, both counters and the reproducing command — including
the finding that decides the design, that the same lane's gathered coefficient fetch is the fastest
of the three by instruction count and the slowest by retired slots.

Public function signatures and supported domains are stable within a major version. Bitwise outputs
are not. Internal region thresholds, seed selection, recursion order and dispatch logic may change
between minor releases, as long as the bounds above hold. Byte-for-byte reproducibility requires
pinning the release tag, the compiler and the build flags.

**Reporting a suspected violation.** Open an issue with the exact `(n, x)`, the lane and the region.
Reproduce it first with `ctest --test-dir build --output-on-failure`. If you suspect the generated
tables, add `python3 tools/gen_boys_coefficients.py --check`.

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
