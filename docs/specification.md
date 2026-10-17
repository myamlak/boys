# The Boys function kernel specification

This is the specification the [README](../README.md) points at: what each evaluation path guarantees,
the settings that select one, and the measurements behind the figures. None of it is needed to make
your first call. The words it uses — *lane*, *region*, *route*, *scheme*, *axis*, *gate* — are
defined on the documentation's landing page, together with the full API reference:
<https://myamlak.github.io/boys/>.

## Accuracy contract {#accuracy-contract}

The bound at a cell is **`1e-13 · F_0(x)`**, for every supported n, x and lane: the ladder's first
rung, `F_0(x)`, scaled by `1e-13`. Since F is decreasing in the order, the `1e-13 · F_n(x)` arm of
the pair is never the wider one, so the `max` of the two is the `F_0(x)` arm at every cell. Until
4.0.0 the bound instead carried a constant floor, `5.5e-14`, which the ladder-scaled figure
replaces: the same kind of rescue, scaled by the ladder's own first value rather than by a fixed
number.

**B is a fraction of the ladder's first rung, so read against the value it is not uniform.** The form
this contract had before was absolute — one figure for the whole range, whatever the value — and it
judged every cell against that one figure. At `n = 32, x = 29`, F is 1.93e-14 and the lane returned
6.87e-14 under that form — an error of 4.94e-14, inside the 5.5e-14 it carried, and 0.897 of it, so
that form passed the cell. Read relative to F, the same cell is **2.55 times the true value**. The
ladder-scaled bound holds there (the uniform grid, `FitGranularity::kUniform`, serves that argument:
its hi-bound is `kFlatHi`, 31.0). That is what the ladder-scaled form binds and the absolute form did
not: a budget spent where the value is tiny is spent where nothing needs it. What the same form does
not give is a figure the returned value holds to in its own right: on the committed grid the double
entries' delivered error is 3.496e-14 of `F_n(x)` at the worst cell below the region-C boundary,
`x = 28.98933773882074`, and at `n = 32` it is under `1e-13` of the value again above `x ≈ 93`;
between those, the bound is a larger fraction of the value, up to 140 % of it at the worst cell,
`n = 32, x = 31`.

B binds above the lane format's smallest normal — 2^-1022 for the double lanes, 2^-126 for the
float ones. Below it the returned value is subnormal or zero, the format's own floor is what
answers there, and no accuracy is claimed. To reproduce a figure: read the grid's `value` column
(`tests/data/boys_accuracy_gate_reference.csv`, 59,301 cells), call the entry at the same `(n, x)`,
and take the largest `|delivered − value| / |value|` over the cells whose `|value|` is above the
lane's floor. The gate does exactly this and prints the ratio beside every row.

**The contract is one figure per lane.** The library evaluates the function differently at different
argument sizes, and one lane is tighter over part of the range than over the rest — but a lane's
tighter regions sit inside its published figure rather than beside it. Every figure below holds for
**all** x ≥ 0:

| Lane | Error bound |
|---|---|
| double single | ≤ the larger of `1e-13 · F_n(x)` and `1e-13 · F_0(x)` — the lane's delivered error is far below the figure over the whole fitted range, and [lane-contract.md](lane-contract.md) states the per-region budgets its fits were cut against |
| double batch, whether the top order is the batch's or each argument's | ≤ the larger of `1e-13 · F_n(x)` and `1e-13 · F_0(x)` |
| float single / batch | ≤ 2^-22 · F_n(x) + 1e-7 in the plain-reciprocal form, or ≤ 1.5e-7 + 1e-7 in that form, whichever is the larger — the lane's arithmetic is binary32 and has an absolute floor, so its figure keeps the absolute figure it published before the relative one |
| fp16 / bf16 | ≤ 1.5e-7 + ½ ULP, or ≤ 2.5e-7 + ½ ULP in the plain-reciprocal form — this lane's figure is the fp32 engine's arithmetic stated absolutely, and it is left as it was: it is already a term of the value's own digit beside a floor, and it is the looser of the two figures wherever the float lane's relative figure falls below it |
| native half, x ≥ 28.984375 | ≤ 8 ULP of the returned value — a bound on the returned value's own last digit, which is what a relative bound is; no table, no constant, and no move |
| CUDA fp64 | ≤ the larger of `1e-13 · F_n(x)` and `1e-13 · F_0(x)` — the same contract as the CPU double lanes |
| CUDA fp32, `RegionBExp::kAccurate` (the default) | the same contract as the CPU float lanes, at the lane's own default division form |
| CUDA fp32, `RegionBExp::kFast` | ≤ 2^-22 · F_n(x) + 8e-8, or ≤ 1.5e-7 + 8e-8 where that is the larger — the lane's figure plus the corrected seed's own contribution, which is a term of the argument and not of the value |
| CUDA fp16 | ≤ 1e-7 + ½ ULP, or ≤ 2e-7 + ½ ULP in the plain-reciprocal form — the CUDA surface's own half entries, and not the host half lane's figure |

**Every figure in that table is the shipped build's.** Each row is a claim about the choices
[boys_build_defaults.hpp](../include/boys/boys_build_defaults.hpp) carries, and they are the figures
the accuracy gate below judges a default-configure build against. A configure that points
`BOYS_BUILD_DEFAULTS` at another header builds a library whose unnamed calls resolve to that
header's seven names instead — the header the option probe's `--emit-defaults` writes, or any
header satisfying the contract that file states — and no row above is a statement about such a
build: this page certifies nothing about a replaced build's delivered error. A replaced build's
figures are the ones its own measurement produced, and the option probe is the measurement that
tree carries: it measures the build it is compiled in, and its report prints each option's own
numbers beside the reference it compared them to. The accuracy gate marks the rows a replaced build
cannot be judged against — the double lane's published figure and the far-representation arms'
figures — and prints their readings rather than a verdict, and `boys-tests` takes the same reading
of its far-representation bars.

**Two combinations of the double lane hold the bound's ladder-scaled arm and not a figure read
against `F_n(x)`.** The bound above is the one the lanes' own entries hold, with margin: the gate's
recorded run puts the double lane's worst delivered error at `0.288` of it, at `n = 32, x = 31`. The
same cell is the worst the tree's own reference grid holds read against the value — a delivered
error of 1.396 · F_32(31) — and the grid's worst below the region-C boundary,
`x = 28.98933773882074`, is 3.496e-14 of `F_n(x)`. Two families of cells are held by the
ladder-scaled arm and not by a figure read against `F_n(x)`:

| Family | What it holds | The cell it is worst at |
|---|---|---|
| a lean partition's region-A fits and the ladder that carries the orders above them: `FitGranularity::kCoarsest`, `kNarrow`, and `EvalPolicy<>` as it resolves | the ladder-scaled bound, `0.12` of it at worst | the ladder's cell, n = 32, x = 10.7846; the documented cell n = 32, x = 10.7836 delivers 3.93e-09 · F_n(x), and a fit's cell, n = 8, x = 2.1613, delivers 6.68e-14 · F_n(x) |
| the rational-minimax route's high-order intervals | the ladder-scaled bound, `0.48` of it at worst | n = 2, x = 10.7936; the documented cell n = 32, x = 11.6879 delivers 4.69e-09 · F_n(x) |

So the double lane's bound as it stands is **the larger of 1e-13 · F_n(x) and 1e-13 · F_0(x)**, and
it is the wider arm, not the relative one, that the cells above need - measured at worst `0.12` and
`0.48` of the bound over the grid that carries them; the float lane's row states
the same union with its own two figures. Two changes, one per family, would raise each to the
relative figure. The first: an order whose value is far below its region's seed is carried up the
recurrence by the gain [boys_impl.hpp](../include/boys/boys_impl.hpp) states (`2 T(N, x) t / F_N` is
1.598e6 at kX0), so serve it from its own fit, as the shipped entries' per-order grid does, or carry
the recurrence in wider arithmetic. The second: the rational route's intervals between x = 11.5 and
12.0 store a fit that does not reach the bar their neighbours reach.

**One axis's members are not one figure.** `DivisionForm` is how every recurrence step divides, and
its three members are three arithmetics, not three spellings of one. Exact division rounds once per
step, the plain reciprocal rounds twice, and the refined reciprocal recovers the exact form's
rounding from the plain product.

On the **double** lane the three deliver one figure. Over the accuracy gate's reference grid,
59301 cells per form, the plain reciprocal leaves region A, region B and region
C where exact division has them and moves the extended band's worst absolute error from 3.22e-15 to
6.73e-15 — a factor of 2.1 (6.73e-15 / 3.22e-15), at the one region where the axis costs the double
lane anything, and
nothing anywhere else. The refined reciprocal is bit-identical to exact division in every cell. So
the double rows are figures under all three forms, with no term beside them.

On the **single-precision** lanes the plain reciprocal costs accuracy, so those lanes publish a
term for it beside their base: **the larger of 2^-22 · F_n(x) and 1.5e-7, plus 1e-7**. The term's
measured worst is 1.7514e-07 absolute, at n = 0, x = 9.74055, where the lane's exact and refined
forms deliver 1.08354e-07 at worst — the same absolute pair the lane carried when its figure was
stated absolutely, which is why the extra rounding is stated beside the figure rather than folded
into it: it is a term of the form and not of the value, so a relative figure does not absorb it.

The lane's 1.5e-7 arm is there for the same reason the term is. The relative figure alone does not
bind where the value is small, because this lane's error does not fall with the value: binary32
arithmetic leaves an absolute error of order 1e-7 behind on the way to a result of any size, and at
the cell above the lane's own exact form delivers 1.08354e-07 against a relative figure of
2^-22 · 0.283955 = 6.77e-08 — half again over it. So the figure the lane states is the larger of
the two, and the relative one binds above the value where it overtakes the absolute one, at
1.5e-7 / 2^-22 = 0.63. What raises the arm to the relative figure alone: the region-A seed and the
ladder evaluated in binary64 and rounded once. That is the double lane's arithmetic, and the floor
is what this lane's binary32 arithmetic costs.

The fp16
and bf16 lanes run that arithmetic and round at the boundary, so the same term stands beside their
base — which is the fp32 engine's arithmetic stated absolutely, 1.5e-7, rather than the relative
base the float lane now states — before the half digit above is added.

`BoysAccuracyGuaranteed` takes the form as an argument and answers the figure for the form you name.
A device lane's row is read at the form the caller names, as a host lane's is: the CUDA surface's
entries carry the axis and every entry of the device option space runs every form. The float and
double device lanes' rows state one figure for all three forms, which those forms deliver inside; the
half lane's row states the plain reciprocal's own term beside its base, because at a subnormal result
that form's rounding leaves the base — 1e-7 + 1e-7 = 2e-7.

"ULP" is the last representable digit of the result in the format concerned.

**Choosing nothing.** A caller that has picked a precision and no axis writes one name: each precision
has a named default, and so does the device lane. [lane-contract.md](lane-contract.md#the-default-policy-per-precision-and-per-device)
states what each selects, the bound it carries, and the command that prints the name and the in-force
default as numbers. Two of the choices — the evaluation scheme and the interval partition — were
set from the option probe's own runs, and **neither was settled by them**. The partition was not
chosen against its axis at all: every partition-bearing row of the runs that named it is one
partition, `narrow`, so the comparison that named it measured one member of a three-member axis
rather than ranking the whole axis. That default has therefore not been derived against the whole
axis. On the scheme axis the runs did compare both rows and did not separate them.
The route and the packing axis carry the settings the library has always shipped and have not been
ranked against a timing. The option probe below is what ranks any of them, on the machine it is run on.
**Which runs, and where this tree keeps them**: the committed default rows are in
`include/boys/boys_build_defaults.hpp`, one row per precision, shape and device class, each carrying
the run its combination came from, and a run's own record is the file it wrote, kept under
`tests/data/`.
**Each default is a choice between two ways of computing one answer and not between two accuracies**, so
a version that moves one costs no accuracy at any call site that names nothing.

The rows above are bounds, and a bound is not the figure a lane delivers. Two rows are delivered at a
different figure depending on which multiply-add the lane evaluates — one rounding or two — and the
build decides which that is, because a build that contracts a bare product-plus-add compiles the
separate selection onto the fused step: `RouteInForce` (`include/boys/backend.hpp`) reports the fused
route there whatever the selection says. Against the committed reference grid, built both ways — the
default fused route and the `BOYS_MULADD_SEPARATE=ON` separate route — on MSVC x64 Release at this
revision:

| lane | region | fused route | separate route | figure |
|---|---|---|---|---|
| double, single | 1.0855 ≤ x < 11.8998 | 3.22e-15 | 4.63e-15 | the larger of 1e-13 · F_n(x) and 1e-13 · F_0(x) |
| float, single | all arguments | 6.36e-08 | 7.49e-08 | the larger of 2^-22 · F_n(x) and 1.5e-7 |
| float, batch | all arguments | 1.08e-07 | 1.08e-07 | the larger of 2^-22 · F_n(x) and 1.5e-7 |

The delivered columns are absolute worst errors — each the worst cell of the sweep the gate's
recorded run holds — and the figure beside them is the lane's own: the relative one, or the larger
of it and the lane's absolute figure, held over the whole argument range the row names. The sweep is
a default-configure build's, as every figure on this page is, and the scope
[the contract states](#accuracy-contract) is theirs: a build whose default rows are a replacement's
moved the arithmetic under every cell here, and what it delivers is what its own option probe
measured. The delivered columns and the figure beside them are not a ratio of one another: what the
columns of this table show is what moves with the route, and the row's delivered cells are the whole
of what does.

The band row and the float single row are the whole of what moves: the double lane's other three
regions and the float batch lane deliver the same worst cell, to the digit, on both routes. A figure
is the compiler's as well as the route's: a build that contracts delivers the fused route whatever
the selection says, and g++ there reads 2.74e-15 on the band, 6.36e-08 on float single and 9.11e-08
on float batch, against MSVC's 3.22e-15, 6.36e-08 and 1.08e-07.

Every bound holds either way, and the configure step **measures** which arithmetic a build runs by
compiling and running a bare product-plus-add rather than inferring it from the architecture name;
the report prints the answer. The `BoysFixedN` entry's agreement with `BoysSingle` is exact where the
build does not contract that form and inside the single lane's bound everywhere. A contracting build
may fuse at one call site and not at another, so the entry's report prints how many of its
comparisons were bit-for-bit equal, and [lane-contract.md](lane-contract.md) carries the
counts.

The fp16 and bf16 rows are fp16 *I/O* around the fp32 engine, so their error is the engine's. Their
figures stay as they were, and the move to a relative contract is what leaves them alone: a bound
carried as a figure of the value's own last digit beside a term of the arithmetic is already the
shape the move is for, and both of their terms are looser than the float lane's base.

The native half lane (`BoysAllOrdersHalf2`, `BoysAllNF16Native`) is a different thing. It evaluates
region C's ladder in packed binary16, one correctly rounded operation per step. That buys two
arguments to a register. It costs the accuracy its own bound names: ≤ 8 ULP of the returned value,
measured at a worst of 4.3 ULP. That is a bound on the returned value's own last digit and not a
figure of its own, so the move to a relative contract leaves this lane exactly where it was: it
carries no table and no constant, and there is no figure to re-derive.

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
operand precision reaches 64-bit-grade accuracy on a 32-bit accumulator. These are absolute figures
beside a relative contract and they stay absolute: the transform's domain is region A, where the
smallest value any band of any order holds is 5.37e-3, so there is no range of small values for an
absolute figure to hide a relative gap in. **Those two bounds are
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
e^{-x} it consumes is amplified by that much there. The bare approximation's own relative error
grows with the argument: fifteen ulp at the boundary and thirty-five at the far end of the region.
It therefore fails the lane's budget over a band of region B at the highest order — predicted at
2.30e-7 against the 1.5e-7 budget, measured at 2.57e-7 — and returns
the wrong sign there. It is not offered, because the failing band is interior to
region B and there is no certified sub-range to restrict it to. Corrected, the seed's relative error
is flat at a few ulp, its contribution to the value is capped at 8e-8 by that same amplification
(measured 5.0e-8), and the option returns the function's sign at every cell the device gate audits.

No speed is claimed for either option. The corrected form's kernel is the same size statically as the
accurate one — 944 instructions against 944 — and the exponential is evaluated once per
element outside the order loop; what separates the two options is the bound, not a measured time.

The GPU and CPU float lanes agree to within 3.5e-7. That is a cross-lane statement about
|GPU − CPU| and not a bound on either lane's distance from F_n(x): the default CUDA fp32 entry is
held to the larger of 2^-22 · F_n(x) and 1.5e-7 above, and the fast option's looser bound is not
covered by the 3.5e-7 figure.

### Choosing a combination

The lanes above are one axis of five. A call is a lane, a fit route, an evaluation scheme, an interval
partition and a packing axis, and the library offers the product of all five:
2 routes × 2 schemes × 3 partitions × 2 axes, in 8 lanes — **192 combinations**, and every one of
them is a cell the gate below counts: certified and published, refused with the library's own reason
and owed, or a device cell this host cannot run. None delivers outside the bound its lane publishes
(the gate command below prints the three counts on the build it runs in).

The eight lanes are the host's double, single, half and bfloat16 lanes and the device's own
`kFp64Device`, `kFp32Device`, `kFp16Device` and `kBf16Device`, which are the four precisions the CUDA
surface's entries are built at. `BoysLaneContracts()` is the library's own list of them, one row per
`Precision` member in enumerator order, and the count above is that list's size. A device lane's
entries are not the host lane's beside it, which is why the default-policy table
keys a device class by the device lane: that table is keyed `(device, precision, shape)`, and the
device classes it holds are the twelve of those four lanes by the three questions the device option
probe ranks.
`BoysAccuracyGuaranteed(precision, n, x, ...)` returns the bound a combination carries **at the
cell it is asked about** — the lane's published form evaluated at `(n, x)`, plus the lane's own
additive term where it documents one — and `BoysAccuracyDelivered(...)` returns the figure it was
measured to deliver, which is the one to rank two combinations by. They are different questions, and
the `reading` field of the returned `AccuracyFigure` says which answer a figure is. The bound is the
contract and not the measured absolute figure: at x = 0 the double lane's figure is `1e-13`, and the
`5.5e-14` the lane's runs measured is documentation (`BoysLaneContracts()` publishes it and the
lane-contract page states where it crosses the contract). A combination the library does not carry,
or a cell outside the documented domain, has no figure: both accessors say so and give the library's
own reason rather than returning a number.

**Choosing a combination is a name.** The four structural axes —
route, scheme, partition, packing axis — are the fields of an `EvalPolicy`, so a combination is a type
the call site writes once and the compiler resolves where it is written: 8 lanes × 24 axis
combinations, **192 combinations, each reachable as a name**. Naming one costs nothing at the call —
there is no table to look a combination up in, no string to match and no search at run time, which is
why those axes are not call arguments. All of them are named and all of them are evaluated by
`tests/consumer_option_space.cpp`, which prints the counts per precision.
A caller holding a route and a scheme as values names the route where the call is compiled and the
scheme at the call instead:
`BoysAllOrdersWithRoute<kRoute>(scheme, nmax, x, out)` on the ladder shape. The single-order
float shape takes a route and no scheme — `BoysSingleF32WithRoute<kRoute>(n, x)` — so a scheme
there is a template argument rather than a call argument.

**A caller that has a target rather than a comparison asks it directly.** `QueryCombination(...)`
takes the same axes, the cell `(n, x)`, and the absolute error the caller needs, and answers with a
verdict beside
the numbers it was made on: `kGuaranteedInside` where the bound is at or below the tolerance,
`kDeliveredInside` where the bound is above it and the figure the combination's fits were measured
to deliver is at or below it, `kOutside` where neither is, and `kNotCarried` where the library does not have
the combination — no verdict, no figure, and the library's own reason. The bound decides first, so
`kGuaranteedInside` is the state a calculation's safety can rest on and `kDeliveredInside` is
explicitly not that state. The request and both figures come back with the verdict, so an answer can
be checked against what decided it.

**The bound is the lane's, and no other axis moves it** — what the other axes change is what a call
delivers, and the gate prints every combination's measured figure beside its bound. The per-lane
counts, the reasons, the two figures side by side and the tolerance table are in
[lane-contract.md](lane-contract.md#every-combination-its-bound-its-delivered-figure-and-its-tolerance).

### Check the figures yourself

Every figure above is measured, and the measurement is in this tree. Build the gate and run it:

    cmake -S . -B build -DBOYS_BUILD_TESTS=ON
    cmake --build build --config Release --target boys-accuracy-gate
    ./build/Release/boys-accuracy-gate --strict      # the config directory is your generator's

The first line names the option because the gate is a target of the test suite, and CMake caches what
a directory was configured with: the README's first recipe sets `BOYS_BUILD_TESTS=OFF`, and a
re-configure that does not name it again leaves it `OFF`. The target is then absent and the build
stops with `MSB1009`, naming `boys-accuracy-gate.vcxproj` as a project file that does not exist;
naming the option `ON` adds the suite's targets to the directory you already have. The gate is a
C++23 target, and the googletest it needs is vendored in this tree.

It sweeps the documented entries over the committed reference grid — 33 orders × 1797 arguments,
59,301 points — and prints, lane by lane and region by region, the worst error the lane delivered
beside the bound it claims. `--per-order` adds the same comparison order by order, naming the cells
the bound actually binds on; `--probe n x` prints one cell from every lane for one argument.

**The first five lines it prints are a record and not a failure.** They begin `EXCEEDED (record, not
judged)`, at the smallest arguments, and they are a slot the gate keeps from a withdrawn reading of
the header's table so the reason it was withdrawn stays visible: the lane does not meet that reading
in the band, and no verdict counts the lines. [lane-contract.md](lane-contract.md) says what the
reading was.

The run ends with a verdict line and an exit status: `PASS: every documented claim met at this
revision`, or the numbers of the claims that did not hold, with a non-zero status. `ctest` runs this
binary as one of its tests, but a passing `ctest` prints only how long the test took — the table is
in this binary's own output. CI runs this gate on every platform below and fails the build if a claim
does not hold. The contract table above is held against the gate: `tools/check_bound_transcripts.py`,
which CI runs as well, reads its rows against the gate's own constants, so a bound that moves on one
side and not the other is a red run. The two half lanes are that table's one exception, and the
checker names it: their figure is the half-quantum the gate composes at run time from the format of
the value it returned, so no constant states it.
The `Worst delivered` column is a run's own output, and it is held to one. `tests/data/` carries a run
of the gate kept as a file; `tools/check_bound_transcripts.py` reads the table below against that
run's rows, and `tools/check_recorded_run.py` reads the version and the content the run's own stamp
carries and recomputes the content from this tree. Both run in CI. So a change that moves a lane inside its bound leaves the gate green - the
bound still holds - and leaves the recorded run describing an older tree, which the second check
reports rather than passing over. The fix is a new run and a new transcription of it, never an edit to
the figure. A figure is one host's: your own run is the answer for your platform, your compiler and
your arithmetic, and the rows are compared cell by cell rather than as one number.

Run on the tree the run's file names, it printed this (an excerpt; the full run carries one row per
lane and region). **Both columns are that run's own output, the `Bound claimed` one included**: the
figures in it are the bounds the gate was judged against at the revision the run names, which is
older than the contract above — read the column as that run's record and not as this page's
contract, and read the delivered column beside it the same way. **The `C` row's two figures are in
units of the cell's own `F_n(x)`**: that row's worst cell is `n = 32, x = 31`, so it reads delivered
`1.4 × F_32(31)` against a claimed `4.85 × F_32(31)` — the 140 % error above, beside the bound that
carried it:

| Lane | Region | Worst delivered | Bound claimed |
|---|---|---|---|
| double single | A | 3.11e-15 | 1e-13 |
| double single | band | 1.68e-15 | 1e-13 |
| double single | B | 2.21e-16 | 1e-13 |
| double single | C | 1.4 | 4.85 |
| float single | all | 8.3e-08 | 2.38e-07 |
| fp16 store-half | single | 0.000122 | 0.000122 |
| bf16 store-half | single | 0.000976 | 0.000977 |

and ended `PASS: every documented claim met at this revision`.

The gate's region labels are the ones above: `A` is the part of the double lane's range below
x = 1.0855252345349333, `band` is the rest of the range below x = 11.899848152108484, and `B` and
`C` are the two regions beyond that. The `Bound claimed` column carries the figure the gate judged
each row against at that revision: the double lane's published figure, the larger of `1e-13 · F_n(x)`
and `1e-13 · F_0(x)`, the float lane's `2^-22 · F_n(x)` floored at `1.5e-7`, and the half lanes'
`1.5e-7` plus one half-ULP. The per-region absolute budgets the lane's fits were placed against
(`≤ 1e-15` on `A`, `≤ 3e-14` on `band` and `B`, `≤ 5.5e-14` on `C`) are measurements kept in
[lane-contract.md](lane-contract.md), and they are cut against the absolute figures the lanes
published before 4.0.0; the contract above states the one figure the lane holds to over all
four, in the ladder-scaled form.

### Arguments outside the documented domain

Every entry takes n in [0, 32], x ≥ 0, and output spans of the documented size. What a caller gets
for stepping outside that is not the same on every surface:

| Surface | What a violation does |
|---|---|
| C++ (`boys::`) | The check is `assert` from `<cassert>` — the call aborts in a build with assertions enabled, and a build with them cleared has no check at all. Nothing in this tree defines or clears `NDEBUG`, so a Release build gets it from the toolchain's own release flags, and a caller who wants the check keeps it by compiling this library with `NDEBUG` undefined. There is no fallback value and no clamping: an order outside [0, 32] indexes the order-dependent tables out of bounds, which is why every bound above reads "for every supported n, x". |
| C (`boys_c.h`) | Every entry validates and returns a status instead: `BOYS_ERROR_INVALID_ARGUMENT` (1) for an order outside [0, 32], a negative or NaN x, or a null pointer. No abort and no undefined behaviour. |
| CUDA | The single, batch and all-N entries return `boys::BoysStatus`, whose `kInvalidArgument` is the same validation as the C surface's; the device-callable entries take their orders as template arguments, so an order outside the range does not compile. |
| Native half (`BoysAllOrdersHalf2`, `BoysAllNF16Native`) | Same `assert`, and one precondition of its own: the entry covers region C only, so an x below the region-C boundary is a violation rather than a fallback to the table regions. |
| Output spans | The caller sizes them, and no entry reallocates or truncates. The four `std::span` overloads in `boys/boys_span.hpp` compare the extent they were handed against what the call writes and refuse a short one - a line naming the call and both extents, then an abort - and `BoysAllNAtOrders` refuses a top-order span that does not run with the arguments the same way. The pointer surfaces cannot make either check: a `double*` carries no size, so a span shorter than the call writes is an out-of-bounds write there. |

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
double-precision entry takes. The route entry itself names the route where the call is compiled and
takes the scheme as a value:
`BoysAllOrdersWithRoute<kRoute>(scheme, nmax, x, out)`. The float
lane's single-order shape takes a route and no scheme, `BoysSingleF32WithRoute<kRoute>(n, x)` — a
scheme there is the `EvalPolicy` below.

[lane-contract.md](lane-contract.md#the-two-fit-routes) carries this in full: the table of
routes with each one's stored count, measured error and bar; the criterion each is accepted against;
which entries carry a route and what that carriage delivers; and why the rational route
buys the interval's values and not a recursion seed.

#### The float lane's routes

The float lane's region-A table and its region-B seed are its own, and they come in the same two
routes. They are selected with `BoysSingleF32WithRoute`, which is `BoysSingleF32` with
one thing changed — which fit supplies the lane's region-A seed and its region-B seed — and
`BoysFitRoutesF32()` reports them the way `BoysFitRoutes()` reports the double lane's. Region C holds
no coefficient under either route, so naming one there changes nothing.

**The same choice is open through a policy as well, and there it carries the scheme too.** The lane's
entries take the `EvalPolicy` above, so a caller who templates on it names the route and the scheme as
the first two template arguments instead of calling `BoysSingleF32WithRoute`. Both fields are read:
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

[lane-contract.md](lane-contract.md#the-packing-axis-which-of-a-calls-values-share-a-vector)
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
value near it. Measured over region A and every order at four builds — MSVC x64 and g++ x86-64, each
at both route selections — the packed lane and the per-order lane differ in **0 of 879351 values**:
199815 over the region-A sweeps, 679371 over every nmax 0..32, and 165 past the ledger. The gate
carries a row for this axis beside the per-order one, judged against the committed reference grid at
the float lane's own 1.5e-7, and its worst cell reads **1.03e-07**, a ratio of 0.685 to the bar.

**On this lane the axis is not the cheaper way to get the values.** It retires 3.06 times fewer
instructions and 2.61 times fewer retired slots than the per-order loop it replaces, which is what
filling the register buys — and 3.13 times *more* instructions and 3.71 times more retired slots than
the float lane's default entry, which serves all 33 orders from one seed fit and a downward
recurrence. A caller naming this axis is
choosing the per-order lane's shape, and the axis is served because it was named; the figures and the
command that reproduces them are in [lane-contract.md](lane-contract.md#the-same-axis-on-the-single-precision-engines-eight-orders-to-a-register).

Every other combination the axis names is built and measured. A route other than the shipped one is
answered by that route's own region-A fits, whose pieces
cover the same per-order intervals as the shipped table; and the narrow partition is answered by the
very fits the scalar entry reads there, fetched one order at a time instead of by a stride. The gate's
packing book carries a measured
row for each of them, on both routes, at both schemes, through both
entries that carry the axis, and names the worst cell of each.

**Reporting a suspected violation.** Open an issue at <https://github.com/myamlak/boys/issues> with
the exact `(n, x)`, the lane and the region, and run `./build/Release/boys-accuracy-gate --strict`
first — it prints the delivered error beside
the bound for every lane and region and ends in a verdict, so an issue can quote the line that
failed rather than describe it. If you suspect the generated tables,
`python3 tools/gen_boys_coefficients.py --check` reports any drift: it re-derives them and the
reference grid and compares them byte for byte when the record committed beside them
(`tests/data/boys_coefficients_provenance.txt`) does not name the generator, the pins and the
environment in front of it, and otherwise verifies that record and the header's own read-back rules.
`ctest --test-dir build --output-on-failure` runs the whole suite.

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
496 of them on the double lane, each 1/16 wide — with each interval fitted at its own degree under a
read cap of 5, so the coefficients a call reads are that interval's degree and not a column of a
partition-wide count. The float lane carries a grid of its own rather than a re-cut of that one: 77
intervals of width 1/4 under a read cap of 4. `tools/gen_boys_coefficients.py --derive-partition`
prints each ladder and the degrees it chose.

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

**The gain the walk holds is the envelope; the most a call reaches is 2.1711.** The
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
the batch lane's pre-move figure, the absolute 5.5e-14 those rows are still judged at — the gate's
own measured rows for those fits, on the run's multiply-add route. The figures the generated header publishes for the two fitted rows are
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
own constant table carries (6.4676e-16 and 3.14e-18), and `lane-contract.md` states the
derivation and the full trade curve, including why a degree-6 piece of the width a sevenfold split
gives misses a 1e-14 target: at that width the target needs degree 10, not degree 6.

## Version

This tree is **version 4.0.0**. That number is written down once, in `CMakeLists.txt`'s
`project(boys VERSION ...)`, and the test suite fails if anything else disagrees with it. A caller
reads it at run time through `boys::VersionString()`, or as the constants `boys::kVersionMajor`,
`boys::kVersionMinor` and `boys::kVersionPatch`, in `boys/version.hpp` (included by
`boys/boys.hpp`).

**The accuracy promise is why 4.0.0 is a major one.** A lane's bound is stated as a multiple
of a value the call returns - the larger of `1e-13 · F_n(x)` and `1e-13 · F_0(x)`, in
`lane-contract.md` - where the lane carried a constant floor of its own, the absolute 5.5e-14,
before it. The change is what one promise means in another currency. The second arm is the one the
floor becomes: both rescue a cell whose own order's value has fallen below the figure, and the
ladder's first rung scales that rescue to the cell rather than fixing it, so the two agree where
`F_0(x)` is 0.55, at x about 2.46, and the ladder-scaled form is the stricter of the two beyond it.
Read against the relative arm rather than the floor, the ladder-scaled form is the looser of the
two wherever the ladder's first rung exceeds the cell's own value, and near the top of the range,
where the value is close to 1, it permits about twice the absolute error the constant did. A
published guarantee that a consumer pinned on changed, which is a break, and the version carries it.

The version badge at the top of this file shows the latest *release tag*. It names the same version
when a release is cut, and can be older than the tree you are reading.

Public function signatures and supported domains are stable within a major version. Bitwise outputs
are not. Internal region thresholds, seed selection, recursion order and dispatch logic may change
between minor releases, as long as the bounds above hold.

Byte-for-byte reproducibility requires pinning the release tag, the compiler and the build flags —
and *whose* compiler and flags is not one answer for the whole library, because it depends on the
call you write. An entry that names no policy at the fp64 lanes, and any policy whose packing axis
is the orders one, is instantiated in the library and takes the library's flags. A policy naming the
arguments packing axis, and the default fp32 ladder and batch, are instantiated in your translation
unit and take yours: the `With*` helpers all derive from `EvalPolicy<>`, whose packing axis is the
arguments one, so a call written through them is compiled where it is called. What moves a value is
FMA being available together with contraction being permitted; where it moves, it moves by less than
the absolute bound the lane publishes, and largest in relative terms at the smallest values the lane
returns. Bits do not travel across platforms either: fifteen of the twenty host rows read
`RegionBExp::kAccurate`, which is the platform's `exp`. What a pinned tag does give is one library
binary answering the same bits on any x86-64 tier, with or without AVX2.

## Building and consuming

Requires CMake (>= 3.25) and git. **Consuming the library needs C++20**; the tests and the
benchmark drivers are the targets that want C++23, and each is behind its own flag. The
leg-by-leg record of what is built and tested is the "Supported platforms" table below.

Optional builds: `-DBUILD_BENCHMARKS=ON` for the CPU benchmark drivers, which are local-only by
design, and `-DBUILD_CUDA=ON`, which needs the CUDA toolkit.

`-DBOYS_MULADD_SEPARATE=ON` moves the scalar multiply-add off the platform's fused call. On a target
with the fused instruction the fused step is one instruction and there is no call to remove, but the
option is not inert: it asks for two roundings where the fused call gives one. On MSVC x64
Release it delivers them — 4.63e-15 on the double single band where the fused route reads 3.22e-15,
and 7.49e-08 on the float single lane where the fused route reads 6.36e-08, which is the
separate-route column of the multiply-add table above. A compiler that contracts a bare
product-plus-add compiles the separate spelling onto the fused step, and the delivered figures are
then the same under either selection. On a target without the fused instruction the fused step is a
library call per recurrence step, and this option removes it. Where the compiler delivers it, it is a
**different arithmetic** — two roundings rather than one — and it has its own measured bound, which
`lane-contract.md` states lane by lane. `BoysBackends()` reports which route is in force.
`-DBOYS_WERROR=OFF` drops `-WX` for a consumer whose compiler warning noise this tree has not been
made clean for. `-DBOYS_BUILD_DEFAULTS=<header>` compiles the choices an entry that names no
policy resolves to — the host's five and the device lane's own division form and region-B
exponential — from a header of your own instead of the committed ones; the choices are compile-time
values, so an unnamed call costs what it costs either way. CONTRIBUTING.md states what such a
header carries and what the option does not do.

As a submodule:

    git submodule add https://github.com/myamlak/boys external/boys

```cmake
add_subdirectory(external/boys boys)
target_link_libraries(app PRIVATE boys::boys)
```

The tree sets no global standard or flags. Tests and benchmarks are OFF unless this tree is the
top-level project. See the API documentation and [CONTRIBUTING.md](../CONTRIBUTING.md) for the details.

For what each lane guarantees, where it stops and why, see
[lane-contract.md](lane-contract.md).

## Which option is fastest on your machine

The fastest entry depends on the host and the build flags rather than on the library: whether the
vector tier is present, whether a bare `a * b + c` is one rounding or two in your compiler's hands,
and how your arguments arrive all move the ranking. The library carries the measurement instead of a
recommendation — a probe you build and run where you deploy:

    cmake -S . -B build -DBOYS_BUILD_TESTS=ON
    cmake --build build --config Release --target boys-option-probe
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
the row the class's own rounds put first where they order it, and the row the refinement's vote named
where they cannot separate its members; where the vote named another tied row, that row is the
default and the figures of both are printed beside it. **A class that held no row for a member of an
axis prints a `NOT COMPARED:` line naming that member**, because a default read as a race it never
ran is worse than no default: the partition axis is the one this library's shipped default states as
a comparison, and a run whose rows are all one partition has not compared partitions at all.

When a class cannot be ordered — a pair whose within-round ratio band straddles one, or too few
rounds for a band to exist — the run still ends with one combination, and it says how it reached it.
The options the class left tied are re-run alone at a longer protocol (more passes over more rounds,
set by `--refine-runs` and `--refine-factor`) and voted on. That vote decides: options the class
cannot separate are settled by which was fastest in most runs, so the row it names is the default,
and the report says whether every one of those runs led with it or only most of them. The row the
report's own table puts first is kept beside it as the record of what the shorter protocol said, and
where the vote named another, the report prints both figures and says the class's top entries cannot
be separated by the run — never naming a row its own figures show behind another. A class that holds
one option names that option: one entry is not a ranking, and
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
per row, one class per precision and question
shape with the resolution that class was ordered at, and the entries it could not separate. A row's
cost and its documented bound are reported together.
`boys-device-probe` is a thin driver over it for the terminal:

    cmake -S . -B build-cuda -DBUILD_CUDA=ON
    cmake --build build-cuda --target boys-device-probe
    ./build-cuda/Release/boys-device-probe     # the config directory is your generator's

The option space the probe ranks is the library's own, reported by `boys::BoysDeviceOptions()` in
`boys_cuda_options.hpp` beside the entries it describes, **crossed with the division forms
`boys::BoysDivisionForms()` reports**: one figure per row per form, because every launched entry
takes the form as a trailing parameter and every device-callable one as a template argument, so the
same entry is a different row under each of the three. A row of the library's table carries what a
chooser needs to place an option: the entry, the precision it computes in, the question shape it
answers, the axis it varies where it has one — the single fp32 entry's region-B exponential, whose
two members carry different bounds — the degree tables it reads, and the bound its own documentation
states. The probe's rows are a projection of those two reports, so a precision, a shape, an axis
member or a division form added to the surface appears in both without an edit to the probe, and an
option this build does not serve is carried with the reason rather than left out. A row's name is its
entry's name, with the form's own segment where it runs a form other than the build's default — the
option probe's own grammar for its cells, because a name is what the rankings conclude on. The device
accuracy gate reads the same report for the same bounds, so what a chooser is told and what the gate
certifies cannot come apart.

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
entries still ends with exactly one of them: the entries it could not place behind the leader are
re-run alone at a longer protocol, set by `--refine-runs` and `--refine-factor`, and the entry that
led most of those runs is the shape's default — never a row its own figures show behind another.
The row the shape's own table lists first is kept beside it as the record of the shorter protocol
said, so a caller reads both. Whether the vote was unanimous, a majority, or split across several
entries is *how* that entry was reached: three answers of different strength, and the report says
which one it is making, with the vote printed run by run. A shape whose rows the run's own checks
set aside — a subtraction that resolved nothing, a figure the repetition control could not hold —
is not left without an answer: the rows it did produce a figure for go to that same stage, and the
shape ends with one of them, named with the way it was reached rather than as an ordering this
shape's own rounds established. A shape holding one entry names that entry, since there is no
alternative to name and one entry is not a ranking; a shape that produced no figure at all is the
one case that reports `CANNOT DETERMINE` and names no entry, and the reason says which count or
which row was
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

**The report's last block counts the option space rather than describing it.** Every member — a row
of `boys::BoysDeviceOptions()` crossed with a form of `boys::BoysDivisionForms()`, so the total is
the product of those two tables' own sizes — is placed in one state of the run: measured, refused by
this build with the library's reason and owed, not runnable on this card, offered and producing no
figure, or not asked for by the run's request. The states are summed against the space's own total,
and the classes the space admits are held against the classes the report carries.
`boys::DeviceOptionSpaceClosure` answers those counts to a program, and `boys-device-probe` exits
non-zero when they do not close: a member of the space
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

The `option-matrix` legs are not platform legs: each one builds and runs the accuracy gate and the
test suite in one certified configuration of the library — a host crossed with a multiply-add route —
and the list of those configurations is generated from the library by `tools/gen_option_matrix.py`, so
a member added to an option axis reaches CI by regenerating that list rather than by editing the
workflow.

What each leg's *build* is — the instruction sets it targets, its lane widths, whether a bare
`a * b + c` in it is a single rounding, and whether the compiled library calls the runtime's `fma` —
is recorded per leg in [build-facts.md](build-facts.md), from what the leg's own build
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
| `option-matrix linux-arm64 fused` | `ubuntu-24.04-arm` | absent | no |
| `option-matrix linux-arm64 separate` | `ubuntu-24.04-arm` | absent | no |
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
| `linux-cuda compile-only` | `ubuntu-latest` | n/a | no |
<!-- platform-table:end -->

## Reproducibility

- `tests/data/boys_reference.csv` — the committed 45-digit reference grid.
- `tools/gen_boys_coefficients.py` — regenerates the Chebyshev tables and the grid from the cited
  formulas, using a pinned public `mpmath`, and writes beside them a record of what produced them,
  `tests/data/boys_coefficients_provenance.txt`: a digest of its own source, one of
  `requirements-boys.txt`, the versions those pins name, and a digest of each artefact. One CI leg
  runs its `--check`: the linux-x86 gcc Release leg, in the workflow's Generator consistency step.
  Whenever that record's generator, pins or environment does not match the tree, the check
  re-derives both artefacts into scratch copies and compares them byte for byte without writing
  either committed file; otherwise it verifies the record's digests and the committed header's
  read-back rules. The other legs build and test without it. The emitted header is
  clang-format output, so
  `requirements-boys.txt` pins the formatter as well. The script reads `$CLANG_FORMAT` before
  searching `PATH`. Without a formatter it fails, rather than writing bytes the gate would report as
  drift.

## Citation

The library itself, and the published works its numerical claims rest on, are listed in
[CITATION.bib](../CITATION.bib).

## License

BSD-3-Clause — see [LICENSE](../LICENSE), `Copyright (c) 2026 Marcin Makowski`. Dependency details in
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
