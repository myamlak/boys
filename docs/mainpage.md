# API reference

The boys kernel evaluates the Boys function family
F_n(x) = ∫₀¹ t^(2n) exp(−x t²) dt for n = 0..32.
It does so in scalar fp64 and fp32 (double and single precision), an AVX2 vector tier behind the same
entries, fp16/bf16 I/O (half and bfloat16), a native packed-half lane, and optional CUDA lanes.

Source: the [GitHub repository](https://github.com/myamlak/boys).

Five pages go with this reference. Read them before any signature, and **if you have not called this
library before, start with the first**: it builds the library, runs four calls, and prints each one's
output beside the command that reproduces it. **If you are calling this library on a GPU, the fifth
is the one written for you.**

- \subpage md_docs_2getting-started "getting started: build it, call it, read the output"
- \subpage md_docs_2consumer-perspective "Choosing a lane: how much accuracy the calculation needs"
- \subpage md_docs_2lane-contract "The per-lane contract: where each bound holds, and where it stops"
- \subpage md_docs_2build-facts "Build facts: what a build of this library is, read out of the build"
- \subpage md_docs_2gpu-guide "Calling this library on a GPU: the two routes, what you allocate, and what you got"

## The words this library uses

Nine words carry the design. Each means something narrower here than it means elsewhere:

- **lane** — an entry plus the arithmetic behind it. The fp64 entries are "the double lane", the
  packed binary16 ones "the native half lane". A bound always describes one lane.
- **region** — an interval of the argument x. There are three. **A** is below
  x = 11.899848152108484, **B** runs from there to x = 28.98933773882074, **C** is at or above it.
  Each is evaluated differently, so a bound is stated per lane *and* region.
- **band** (the extended band) — the part of region A the per-order fits do not cover: from the
  argument at which an order stops being read from its own fit up to x = 11.899848152108484 where
  region B begins. That argument rises with the order — 1.0855252345349333 for orders 0 to 4,
  2.015297705335114 for 5 to 8, 4.897870299825657 for 9 to 16 and 10.783587858916762 for 17 to 32 —
  so the band's left edge is the lowest of them. One fit of its own serves the whole band, at degree
  24, and that fit's figure is 3e-14 where the per-order fits are held to 1e-15. A region-A figure is
  therefore stated for a range of arguments and not for the region as a whole.
- **route** — a table of stored fits serving a region. The double lane carries two, the default
  Chebyshev fits and a rational minimax alternative. Naming one with
  \ref boys::BoysAllOrdersWithRoute changes only the fits that serve the intervals its rows report.
- **scheme** — the summation a stored Chebyshev fit is read in: split Clenshaw or Horner. It is the
  second field of \ref boys::EvalPolicy, and changes values only where the default fit answers.
- **axis** (the packing axis) — which of a call's values share a vector register: four arguments at
  one order, the committed axis, or four orders at one argument (\ref boys::PackAxis).
- **partition** — how the fitted intervals are cut into pieces, and one of the axes a call site
  names. Three are carried: the committed cut (`FitGranularity::kCoarsest`), the narrower cut the
  proved truncation bound places, which is the default (`FitGranularity::kNarrow`), and a fixed grid
  of one uniform width over the whole fitted domain, every order fitted independently at one degree
  and no order built from another (`FitGranularity::kUniform`). All three cut one fit, so naming one
  changes the work and the storage, not the accuracy (\ref boys::FitGranularity).
- **cell** — one order at one argument: the unit every sweep on these pages counts in. The
  `--probe n x` flag of the accuracy gate prints one cell, from every lane, for one argument.
- **gate** — a program in this tree that measures the documented claims against the committed
  reference and fails when one does not hold. `boys-accuracy-gate` is the accuracy one; the platform,
  option-matrix and device legs have gates of their own.

## Entry points

All CPU entries are `noexcept`. Their preconditions are n in [0, 32], x >= 0, and output spans of
the documented size; **outside them the behaviour is undefined** - a release build faults on x < 0
and on a NaN in a batch, and answers a wrong value for n > 32. The CUDA lane reports through
\ref boys::BoysStatus instead.

Every CPU entry is templated on one policy, the five structural axes a call site names once and the
compiler resolves where it is written (\ref boys::EvalPolicy). The CUDA lane's entries carry no
`EvalPolicy`: `BoysCuda::SingleF32` takes the region-B exponential as a run-time argument, and the
device-callable entries of `boys/boys_cuda_device.hpp` take the division form their recurrence
divides in and the exponential their tables read as template parameters, each defaulting to the
lane's own member (\ref boys::kDefaultDeviceDivisionForm, \ref boys::kDefaultDeviceRegionBExp).
Every entry evaluates at the one accuracy this library carries, the full static accuracy of the
certified lane — the lane the accuracy gate holds to its published bound. The native half lane is the
exception: it has no stored fit, and no region but region C. See the accuracy contract below.

The first column is the entry's name in this reference. The second names the lane each group belongs
to, the call shape it answers — one order at one argument, the full ladder at one argument, one order
over an array of arguments, or every order over an array — and, where the group adds something to
that shape, what it adds.

| Entry point | Lane |
|---|---|
| \ref boys::BoysSingle, \ref boys::BoysAllOrders | scalar fp64, single argument / order batch F_0..F_nmax |
| \ref boys::BoysAllOrdersWithRoute, \ref boys::BoysFitRoutes | the double batch at a named fit route (\ref boys::FitRoute), and the report of which routes exist, what each promises and over what interval. A caller holding a route and a scheme as values names both there instead of in the entry's policy |
| \ref boys::BoysFixedN | fp64, one order over an array of arguments, strided. Takes a fit route through its policy |
| \ref boys::BoysAllN, \ref boys::BoysAllNAtOrders | fp64, all orders over an array of arguments, order-major planes. `BoysAllN` takes one top order for the batch and classifies, groups and dispatches internally (\ref boys::BoysSortedArgs skips the sort for a non-decreasing array). `BoysAllNAtOrders` takes each argument's own top order, with no padding of the arguments to a common order, and evaluates the per-argument body at each of them; that is the shape a shell-quartet batch has. Both take a fit route through their policy, and `BoysAllN` also takes a packing axis |
| \ref boys::BoysSingleF32, \ref boys::BoysSingleF32WithRoute, \ref boys::BoysFitRoutesF32, \ref boys::BoysAllOrdersF32, \ref boys::BoysFixedNF32, \ref boys::BoysAllNF32, \ref boys::BoysAllNAtOrdersF32 | scalar fp32; the batch forms seed in double. \ref boys::BoysSingleF32WithRoute is \ref boys::BoysSingleF32 with one thing changed: which fit supplies the lane's region-A and region-B seeds. It carries the lane's own two fit routes (\ref boys::FitRoute), and \ref boys::BoysFitRoutesF32 reports each one's interval, its stored count and the error it delivers. The entries also take the \ref boys::EvalPolicy the double lane's take, so both routes and both schemes are reachable as template arguments. `BoysAllNF32` is the float lane's \ref boys::BoysAllN — one top order for the batch, order-major planes — and `BoysFixedNF32` and `BoysAllNAtOrdersF32` are its \ref boys::BoysFixedN and \ref boys::BoysAllNAtOrders, which write one order over an array of arguments, strided, and each argument's own top order into those same planes |
| \ref boys::BoysSingleF16, \ref boys::BoysAllOrdersF16, \ref boys::BoysFixedNF16, \ref boys::BoysAllNF16, \ref boys::BoysAllNAtOrdersF16, \ref boys::BoysSingleBf16, \ref boys::BoysAllOrdersBf16 | fp16/bf16 scalar I/O around the fp32 engine. The fp16 entries are the float lane's shapes at the half budget: `BoysFixedNF16`, `BoysAllNF16` and `BoysAllNAtOrdersF16` are its fixed-N, all-N and all-N-at-orders shapes, each value computed by the fp32 engine and stored to half once |
| \ref boys::BoysAllOrdersHalf2, \ref boys::BoysAllNF16Native | native half: region C's ladder in packed binary16 (\ref boys::Half2), one correctly rounded half operation per step, two arguments to a register, results scaled by 2^15 (\ref boys::kHalfNativeScaleExponent) |
| \ref boys::BoysCuda::InitializeTables, \ref boys::BoysCuda::SingleF32, \ref boys::BoysCuda::AllOrdersF32, \ref boys::BoysCuda::AllNF32, \ref boys::BoysCuda::SingleF64, \ref boys::BoysCuda::AllOrdersF64, \ref boys::BoysCuda::AllNF64, \ref boys::BoysCuda::SingleF16, \ref boys::BoysCuda::AllOrdersF16, \ref boys::BoysCuda::AllNF16 | CUDA lane (optional build); device arrays with an opaque stream handle. The tables upload on first use, and `InitializeTables` is an optional warm-up. `AllOrders*` is the all-orders batch at a per-element order. `AllN*` is the device \ref boys::BoysAllN (one top order for the batch, order-major planes), and takes non-decreasing arguments, since it never sorts |
| \ref boys::BoysDeviceSingleF64, \ref boys::BoysDeviceAllOrdersF64, \ref boys::BoysDeviceAllNF64, \ref boys::BoysDeviceEachOrderF64 (and the f32 and fp16 siblings), \ref boys::BoysCuda::DeviceTables, \ref boys::BoysDeviceTables | CUDA lane, device-callable (`boys/boys_cuda_device.hpp`). These are the same arithmetic as `__device__` functions a caller's own kernel calls, at one argument the calling thread holds, so a fused integral kernel needs no round trip through global memory. `BoysCuda::DeviceTables` fills the \ref boys::BoysDeviceTables handle the entries take. The header is the whole of what the caller's build pays: no relocatable device code, no device link step, no library on the device side, and the arithmetic is inlined into the calling kernel |

## Measuring the options on this machine

Which entry is fastest depends on the host and the build flags, not the library. Three things move
the ranking: whether the vector tier is present, whether a bare `a * b + c` is one rounding or two in
the compiler's hands, and how the arguments arrive. \ref boys::RunOptionProbe measures the options
this build offers on the machine it is called on and returns an \ref boys::OptionProbeReport, which
\ref boys::FormatOptionProbe renders as the text a consumer reads.

It enumerates the options from the library rather than from a list: each option's arithmetic is
resolved against \ref boys::backend::BoysBackends. It measures each over the library's own call shape: one argument
set, every argument's own highest order, one seed and then upward recursion to that order. Each
option comes back with its cost per argument, the spread of that cost over the paired rounds it was
measured in, the machine load those rounds were taken under, and the accuracy it delivered against
the certified fp64 lane.

The comparison is paired: every option is called once in every round of every pass, and the
comparison between two options is the ratio of their times within one round. A clock drift common to
the round therefore cancels in it, instead of reading as a difference between the options. Each pass
carries runs of a fixed-work integer canary beside its rounds. It is a diagnostic and not a gate:
nothing fails on it. A fixed work read by wall clock measures the clock as much as the load, so a
decaying clock widens it on an idle machine. The ordering is made in the spread of the paired ratios,
which the report measures.

The reported figure is the option's within-round ratio to the reference lane at the middle of the
run's rounds, scaled by that lane's own cost, with its spread printed beside it. It is the middle
rather than a lower quartile, because the reference's ratio to itself is one in every round. Scoring
every other option at a lower quartile of its ratio would give the reference the middle of its own
rounds and its rivals less than the middle of theirs — a ranking that turns on which row the run
anchored on.

The report ranks inside classes, and a class is one precision and one question shape.
The question shape is what the option hands back: one argument's ladder up to its own order, or one
common ladder over an array of arguments. Every row of a class answers the same question.

The axes a caller does not choose — the fit route, the evaluation scheme, the partition of the
fitted regions and the packing axis — are columns inside the class, and compete in one ranking. The
default is taken from the certified double lane's precision, answering the shape the probe's workload
asks. A faster row of another precision or of the other shape is therefore never a candidate for it.
Within that class the default is the row the
class's own rounds put first where they order it. Where they cannot separate its members, it is the
row the refinement's vote named; where the vote named another tied row, both figures are printed
beside the name.

When a class cannot be ordered, the run still ends with one combination. A class cannot be ordered
when a pair's within-round ratio band straddles one, or when too few rounds exist for a band. The
tied options are then re-run alone at a longer protocol and voted on. The report says whether that
vote confirmed the row the figures put first or named another. When it named another, both figures are
printed and the class's top entries are reported as entries the run could not separate.

A class of one names its option, because one entry is not a ranking and there is no alternative to
it. A run that measured no figure at all reports `CANNOT DETERMINE` with the rounds a band needs,
rather than a name read from a table. Every option the run could not place behind the leader is
printed with the band that pair fell in, so the answer and the evidence missing for it are read
together.

The clock is checked rather than assumed, because options need not draw it alike: a wider vector
register runs at a lower frequency. The confidence line reports how far the widest-moving pair's
ratio travelled between the run's first and second half, beside the resolution that figure is read
against. It also reports whether every option the comparison put against another ran one arithmetic
route.

The result is about the machine it was measured on, and the report says so in its own output.

## Architecture

The AVX2 vector tier is x86_64-only by construction. Its kernels are AVX2/FMA intrinsics, and the
translation unit is compiled with `-mavx2 -mfma -mf16c` (GCC/Clang) or `/arch:AVX2` (MSVC).

On every other target the library still builds. There, the tier's region kernels are defined against
the certified scalar lanes instead. They carry the same contracts and the same numerics as the
scalar tails the x86 kernels already ran for their last `count % 4` elements.

\ref boys::BoysAvx2Available reports `false` on those targets. That predicate separates "the vector
tier is absent on this target" from "the vector tier is silently dead on this target". The CI matrix
therefore asserts it per architecture rather than trusting a green build.

Its answer is scoped to the caller as well as to the target. Which tier a translation unit is on is
the answer its own flags give: the tier's bodies are compiled where they are called, so a unit built
without the flag set compiles the certified scalar lanes, whose values are the vector lane's bit for
bit. The predicate's own question is the machine's — whether the processor carries the instruction
set — and the two answers may differ in either direction.

The region kernels themselves are internal. A caller that reached them directly would have to
partition its arguments by region first, the work \ref boys::BoysAllN exists to do. The entries
above are the surface.

## Types

- \ref boys::F16 and \ref boys::Bf16 — the half-precision I/O types (std::float16_t aliases where the
  toolchain ships them; self-contained wrappers on MSVC).
- \ref boys::BoysStatus — the CUDA lane's status enum.

## Accuracy contract

The bound is **relative**: |F̂_n(x) − F_n(x)| ≤ B · F_n(x), for every supported n, x and lane.

**B is a fraction of the value, so a small value is held to a small error.** B binds above the lane
format's smallest normal — 2^-1022 for the double lanes, 2^-126 for the float ones — below which
the return is subnormal or zero, the format's own floor is what answers there, and no accuracy is
claimed. Every figure below holds for **all** x ≥ 0:

*Single* is the entry that answers for one order at one argument; *batch* is the entries that answer
a whole call at once, and the two rows carrying those labels are both the double lane. The rows
state one figure per lane; a lane that is tighter over part of its range is inside its figure there
rather than stated beside it, and [the per-lane contract](lane-contract.md) carries the per-region
budgets each lane's fits were cut against.

| Lane | Error bound |
|---|---|
| double single | ≤ 1e-13 · F_n(x) — the lane's delivered error is far below the figure over the whole fitted range, and [lane-contract.md](lane-contract.md) states the per-region budgets its fits were cut against |
| double batch, whether the top order is the batch's or each argument's | ≤ 1e-13 · F_n(x) |
| float single / batch | ≤ 2^-22 · F_n(x) + 1e-7 in the plain-reciprocal form, or ≤ 1.5e-7 + 1e-7 in that form, whichever is the larger — the lane's arithmetic is binary32, and the absolute figure it published before the relative one is the floor under it |
| fp16 / bf16 | ≤ 1.5e-7 + ½ ULP, or ≤ 2.5e-7 + ½ ULP in the plain-reciprocal form — this lane's figure is the fp32 engine's arithmetic stated absolutely, and it is left as it was: it is already a term of the value's own digit beside a floor, and it is looser than the floor the float lane's figure keeps |
| native half, x ≥ 28.984375 | ≤ 8 ULP of the returned value — a bound on the returned value's own last digit, which is what a relative bound is; no table, no constant, and no move |
| CUDA fp64 | ≤ 1e-13 · F_n(x) — the same contract as the CPU double lanes |
| CUDA fp16 | ≤ 1e-7 + ½ ULP, or ≤ 2e-7 + ½ ULP in the plain-reciprocal form — the CUDA surface's own half entries, and not the host half lane's figure |
| CUDA fp32, `RegionBExp::kAccurate` (the default) | the same contract as the CPU float lanes, at the lane's own default division form |
| CUDA fp32, `RegionBExp::kFast` | ≤ 2^-22 · F_n(x) + 8e-8, or ≤ 1.5e-7 + 8e-8 where that is the larger — the lane's figure plus the corrected seed's own contribution, which is a term of the argument and not of the value |

**Every figure in that table is the shipped build's.** Each row is a claim about the choices
[boys_build_defaults.hpp](../include/boys/boys_build_defaults.hpp) carries. A configure that points
`BOYS_BUILD_DEFAULTS` at another header builds a library whose unnamed calls resolve to that
header's names instead, and no row above is a statement about such a build: its figures are the ones
its own option probe measured, and [the accuracy contract](specification.md#accuracy-contract)
states the scope in full.

The CUDA fp32 lane's single entry is the one device entry that takes a second, certified axis: which
region-B exponential it evaluates (see \ref boys::RegionBExp). Both options carry a bound of
their own, derived from the condition number of the region-B recurrence and confirmed by the device
gate's sweep. The bare hardware approximation, whose relative error grows with the argument, is not
offered. The batch single entry takes the option as an argument; the
device-callable one takes it as a template argument, since there it replaces an arithmetic inside
the caller's own kernel rather than branching within one.

"ULP" is the last representable digit of the result in the format concerned.

The rows above are bounds, and a bound is not the figure a lane delivers. Two rows deliver a
different figure depending on which multiply-add the lane evaluates — one rounding or two. The build
decides which that is, because a build that contracts a bare product-plus-add compiles the separate
selection onto the fused step: the `RouteInForce` the backend report is built from asks that question
and returns the fused route there whatever the selection says. Against the committed
reference grid, built both ways — the default fused route and the `BOYS_MULADD_SEPARATE=ON` separate
route — measured on MSVC x64 Release:

| lane | region | fused route | separate route | figure |
|---|---|---|---|---|
| double, single | 1.0855 ≤ x < 11.8998 | 3.22e-15 | 4.63e-15 | 1e-13 · F_n(x) |
| float, single | all arguments | 6.36e-08 | 7.49e-08 | the larger of 2^-22 · F_n(x) and 1.5e-7 |
| float, batch | all arguments | 1.08e-07 | 1.08e-07 | the larger of 2^-22 · F_n(x) and 1.5e-7 |

The delivered columns are absolute worst errors — each the worst cell of the sweep the gate's
recorded run holds — and the figure beside them is the lane's own, the larger of its relative arm
and its absolute floor, held over the whole argument range the row names. The two are not a ratio of
one another: what the columns of this table
show is what moves with the route, and the row's delivered cells are the whole of what does.

The band row and the float single row are the whole of what moves: the double lane's other three
regions and the float batch lane deliver the same worst cell, to the digit, on both routes. A figure
is the compiler's as well as the route's: a build that contracts delivers the fused route whatever
the selection says, and g++ there reads 2.74e-15 on the band, 6.36e-08 on float single and 9.11e-08
on float batch, against MSVC's 3.22e-15, 6.36e-08 and 1.08e-07.

Every bound holds either way. A build must not infer which arithmetic it runs from the name of its
architecture: the configure step measures it by compiling and running a bare product-plus-add, and
\ref boys::backend::BoysBackends reports the answer. The `BoysFixedN` entry's agreement with
`BoysSingle` is exact where the build does not contract that form, and inside the single lane's
bound everywhere. A contracting build may fuse at one call site and not at another, so the entry's
report prints how many of its comparisons were bit-for-bit equal.
[the per-lane contract](lane-contract.md) carries the counts.

### Checking these figures

The measurement is in this tree. Build it and run it:

    cmake --build <build> --target boys-accuracy-gate
    <build>/Release/boys-accuracy-gate --strict      # the config directory is your generator's

It sweeps the documented entries over the committed reference grid (33 orders × 1796 arguments,
59,268 points) and prints, lane by lane and region by region, the worst error the lane delivered
beside the bound it claims. The run ends in `PASS: every documented claim met at this revision`, or
in the numbers of the claims that did not hold and a non-zero status. `--per-order` extends the
comparison to every order, and `--probe n x` prints one cell from every lane for one argument.
`ctest` runs the same binary as one of its tests, but a passing `ctest` prints only how long the test
took.

The double lane's stored fits come in two routes: the Chebyshev fits that are the default, and a
rational minimax alternative. \ref boys::BoysFitRoutes reports each one's interval, the argument its
selector takes over at, its stored coefficient count, the error it was measured to deliver and the
bar it is certified against. The two hold the same bar over the same interval, and differ in what
they store to reach it. Naming one changes only the fits that serve the intervals its rows report;
everywhere else the entry runs the default route and returns its values bit for bit.

The domain each route states is its own. Region A's rational route takes over per order, from the
argument at which the lane stops reading that order from its own fit and reaches it from the band
seed, and the row states the lowest of those arguments. Below it an order keeps the default route's
value bit for bit, so the tighter per-order figure the lane documents there is untouched.

The native half lane is the one lane whose figure is not a fraction of the value: it is stated in
ULP of the returned value, ≤ 8 ULP, measured at a worst of 4.3. A bound on the returned value's own
last digit is already what a relative bound is, and this lane has no table and no stored fit to
move: it covers region C only, and its results are the scaled values 2^15 F_k(x).

Its domain ends at F_k(x) = 2^-29, which is x about 359, 128 and 30 at orders 3, 4 and 8. Past that
argument the return becomes a subnormal half and then a zero, by design. The lane claims nothing
there, and a caller that has to be right there wants the double or float lane.

Public signatures and supported domains are stable within a major version. Bitwise outputs are not;
pin the release tag, the compiler and the flags for exact reproducibility — and read the
reproducibility paragraph in the specification, because whose compiler and flags decide a value
depends on the call. A call that names no policy at the fp64 lanes is instantiated in the library;
one written through a `With*` helper is instantiated in your own translation unit.

The full contract statement and the region definitions are in the header comments (see
\ref boys::BoysSingle). The certified boundaries are pinned by the committed reference grid
(`tests/data/boys_reference.csv`).

Where each lane's bound holds, where it stops, and what sets its ceiling are in
[the per-lane contract and limitations](lane-contract.md).

## Indexes

- [Functions](globals_func.html) — the complete function index
- [Classes](annotated.html) — F16, Bf16, and the CUDA namespace
- [Files](files.html) — boys.hpp, f16.hpp, boys_cuda.hpp
