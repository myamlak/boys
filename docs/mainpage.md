# API reference

The boys kernel evaluates the Boys function family
F_n(x) = ∫₀¹ t^(2n) exp(−x t²) dt for n = 0..32.
It does so in scalar fp64 and fp32, an AVX2 vector tier behind the same entries, fp16/bf16 I/O, a
native packed-half lane, and optional CUDA lanes.

Source and quick start: the [GitHub repository](https://github.com/myamlak/boys).

Two pages go with this reference. Read both before any signature:

- \subpage md_docs_2consumer-perspective "Choosing a lane: how much accuracy the calculation needs"
- \subpage md_docs_2lane-contract "The per-lane contract: where each bound holds, and where it stops"

## The words this library uses

Six words carry the design. Each means something narrower here than it means elsewhere:

- **lane** — an entry plus the arithmetic behind it. The fp64 entries are "the double lane", the
  packed binary16 ones "the native half lane". A bound always describes one lane.
- **region** — an interval of the argument x. There are three. **A** is below
  x = 11.899848152108484, **B** runs from there to x = 28.98933773882074, **C** is at or above it.
  Each is evaluated differently, so a bound is stated per lane *and* region.
- **route** — a table of stored fits serving a region. The double lane carries two, the default
  Chebyshev fits and a rational minimax alternative. Naming one with
  \ref boys::BoysAllOrdersWithRoute changes only the fits that serve the intervals its rows report.
- **scheme** — the summation a stored Chebyshev fit is read in: split Clenshaw or Horner. It is the
  second field of \ref boys::EvalPolicy, and changes values only where the default fit answers.
- **axis** (the packing axis) — which of a call's values share a vector register: four arguments at
  one order, the committed axis, or four orders at one argument (\ref boys::PackAxis).
- **gate** — a program in this tree that measures the documented claims against the committed
  reference and fails when one does not hold. `boys-accuracy-gate` is the accuracy one; the platform,
  option-matrix and device legs have gates of their own.

## Entry points

All CPU entries are `noexcept` and total. Their preconditions are n in [0, 32], x >= 0, and output
spans of the documented size. The CUDA lane reports through \ref boys::BoysStatus instead.

Every entry is templated on one policy, the five structural axes a call site names once and the
compiler resolves where it is written (\ref boys::EvalPolicy). Every entry evaluates at the one
accuracy this library carries, the full static accuracy of the certified lane. The native half lane
is the exception: it has no stored fit, and no region but region C. See the accuracy contract below.

| Entry point | Lane |
|---|---|
| \ref boys::BoysSingle, \ref boys::BoysAllOrders | scalar fp64, single argument / order batch F_0..F_nmax |
| \ref boys::BoysAllOrdersWithRoute, \ref boys::BoysFitRoutes | the double batch at a named fit route (\ref boys::FitRoute), and the report of which routes exist, what each promises and over what interval. A caller holding a route and a scheme as values names both there instead of in the entry's policy |
| \ref boys::BoysFixedN | fp64, one order over an array of arguments, strided. Takes a fit route through its policy |
| \ref boys::BoysAllN, \ref boys::BoysAllNAtOrders | fp64, all orders over an array of arguments, order-major planes. `BoysAllN` takes one top order for the batch and classifies, groups and dispatches internally (\ref boys::BoysSortedArgs skips the sort for a non-decreasing array). `BoysAllNAtOrders` takes each argument's own top order, with no padding of the arguments to a common order, and evaluates the per-argument body at each of them; that is the shape a shell-quartet batch has. Both take a fit route through their policy, and `BoysAllN` also takes a packing axis |
| \ref boys::BoysSingleF32, \ref boys::BoysSingleF32WithRoute, \ref boys::BoysFitRoutesF32, \ref boys::BoysAllOrdersF32, \ref boys::BoysAllNF32 | scalar fp32; the batch forms seed in double. \ref boys::BoysSingleF32WithRoute is \ref boys::BoysSingleF32 with one thing changed: which fit supplies the lane's region-A and region-B seeds. It carries the lane's own two fit routes (\ref boys::FitRoute), and \ref boys::BoysFitRoutesF32 reports each one's interval, its stored count and the error it delivers. The entries also take the \ref boys::EvalPolicy the double lane's take, so both routes and both schemes are reachable as template arguments. `BoysAllNF32` is the float lane's \ref boys::BoysAllN: one top order for the batch, order-major planes |
| \ref boys::BoysSingleF16, \ref boys::BoysAllOrdersF16, \ref boys::BoysSingleBf16, \ref boys::BoysAllOrdersBf16 | fp16/bf16 scalar I/O around the fp32 engine |
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
carries runs of a fixed-work integer canary beside its rounds. It is a diagnostic that gates nothing:
a fixed work read by wall clock measures the clock as much as the load, so a decaying clock widens it
on an idle machine. What the ordering is made in is the spread of the paired ratios, which the report
measures.

The reported figure is the option's within-round ratio to the reference lane at the middle of the
run's rounds, scaled by that lane's own cost, with its spread printed beside it. It is the middle
rather than a lower quartile, because the reference's ratio to itself is one in every round. Scoring
every other option at a lower quartile of its ratio would give the reference the middle of its own
rounds and its rivals less than the middle of theirs — a ranking that turns on which row the run
anchored on.

The classes the report ranks inside are one precision and one question shape.
The question shape is what the option hands back: one argument's ladder up to its own order, or one
common ladder over an array of arguments. Every row of a class answers the same question.

The axes a caller does not choose — the fit route, the evaluation scheme, the partition of the
fitted regions and the packing axis — are columns inside the class, and compete in one ranking. The
default is taken from the certified double lane's precision, answering the shape the probe's workload
asks. A faster row of another precision or of the other shape is therefore never a candidate for it.
Within that class the default
is the row the run's own figures put first, so the name printed and the table printed beside it
never disagree about which option is cheapest.

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

The region kernels themselves are internal. A caller that reached them directly would have to
partition its arguments by region first, the work \ref boys::BoysAllN exists to do. The entries
above are the surface.

## Types

- \ref boys::F16 and \ref boys::Bf16 — the half-precision I/O types (std::float16_t aliases where the
  toolchain ships them; self-contained wrappers on MSVC).
- \ref boys::BoysStatus — the CUDA lane's status enum.

## Accuracy contract

The bound is |F̂_n(x) − F_n(x)| ≤ B for every supported n, x and lane. Every figure holds for
**all** x ≥ 0:

| Lane | Error bound |
|---|---|
| double single | ≤ 5.5e-14 everywhere; ≤ 3e-14 below x = 11.899848152108484; ≤ 1e-15 below about x = 1.0855 |
| double batch, whether the top order is the batch's or each argument's | ≤ 5.5e-14 |
| float single / batch | ≤ 1.5e-7, or ≤ 2.5e-7 in the plain-reciprocal form |
| fp16 / bf16 | ≤ 1.5e-7 + ½ ULP, or ≤ 2.5e-7 + ½ ULP in the plain-reciprocal form |
| native half, x ≥ 28.984375 | ≤ 8 ULP of the returned value |
| CUDA fp64 | the same budgets as the CPU double lanes |
| CUDA fp32, `RegionBExp::kAccurate` (the default) | the same budgets as the CPU float lanes |
| CUDA fp32, `RegionBExp::kFast` | ≤ 1.5e-7 + 8e-8, the lane's budget plus the corrected seed's contribution |

The CUDA fp32 lane's single entry is the one device entry that takes a second, certified axis: which
region-B exponential it evaluates (see \ref boys::RegionBExp). Both options carry a bound of
their own, derived from the condition number of the region-B recurrence and confirmed by the device
gate's sweep. The bare hardware approximation, whose relative error grows with the argument, is not
offered. The batch single entry takes the option as an argument; the
device-callable one takes it as a template argument, since there it replaces an arithmetic inside
the caller's own kernel rather than branching within one.

"ULP" is the last representable digit of the result in the format concerned.

The rows above are bounds, and a bound is not the figure a lane delivers. Two lanes deliver a
different figure depending on one property of the build: whether the compiler fuses a bare
product-plus-add into a single rounding. **The architecture does not decide it.** Of the six
configurations measured, gcc and AppleClang on arm64 contract one, and MSVC on arm64 does not. The
MSVC arm64 build therefore delivers the x86-64 figures rather than its own architecture's. At the
default, against the committed reference grid:

| lane | region | contracted | not contracted | bound |
|---|---|---|---|---|
| double, single | 1.0855 ≤ x < 11.8998 | 3.29e-15 | 3.22e-15 | 3e-14 |
| float, single | all arguments | 1.29e-07 | 1.06e-07 | 1.5e-07 |
| float, batch | all arguments | 1.29e-07 | 1.08e-07 | 1.5e-07 |

Those three rows are the whole of what moves. The double lane's other three regions deliver the same
worst cell on both arithmetics. The not-contracted column is measured on MSVC on arm64, MSVC on
x86-64, clang on x86-64 and AppleClang on x86-64. The contracted column is measured on AppleClang on
arm64 and reproduced on x86-64 by building with `-mfma`.

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

It sweeps the documented entries over the committed reference grid (33 orders × 1718 arguments,
56,694 points) and prints, lane by lane and region by region, the worst error the lane delivered
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

The native half lane is the one exception to the region-budget form above. It covers region C only.
Its results are the scaled values 2^15 F_k(x), and its bound is stated in ULP of the returned value:
≤ 8 ULP, measured at a worst of 4.3.

Its domain ends at F_k(x) = 2^-29, which is x about 359, 128 and 30 at orders 3, 4 and 8. Past that
argument the return becomes a subnormal half and then a zero, by design. The lane claims nothing
there, and a caller that has to be right there wants the double or float lane.

Public signatures and supported domains are stable within a major version. Bitwise outputs are not;
pin the release tag, the compiler and the flags for exact reproducibility.

The full contract statement and the region definitions are in the header comments (see
\ref boys::BoysSingle). The certified boundaries are pinned by the committed reference grid
(`tests/data/boys_reference.csv`).

Where each lane's bound holds, where it stops, and what sets its ceiling are in
[the per-lane contract and limitations](lane-contract.md).

## Indexes

- [Functions](globals_func.html) — the complete function index
- [Classes](annotated.html) — F16, Bf16, and the CUDA namespace
- [Files](files.html) — boys.hpp, f16.hpp, boys_cuda.hpp
