# API reference

The boys kernel evaluates the Boys function family
F_n(x) = ∫₀¹ t^(2n) exp(−x t²) dt, n = 0..32, in scalar fp64 and fp32, an AVX2 vector tier behind
the same entries, fp16/bf16 I/O, a native packed-half lane, and optional CUDA lanes.

Source and quick start: the [GitHub repository](https://github.com/myamlak/boys).

## Entry points

All CPU entries are `noexcept` and total. Their preconditions are n in [0, 32], x >= 0, and output
spans of the documented size. The CUDA lane reports through \ref boys::BoysStatus instead.

Every entry is templated on one compile-time accuracy multiplier, `kAccuracyMultiplier`, which
defaults to 1. The native half lane is the exception: it has no relaxable resource, and no region but
region C. See the accuracy contract below.

| Entry point | Lane |
|---|---|
| \ref boys::BoysSingle, \ref boys::BoysAllOrders | scalar fp64, single argument / order batch F_0..F_nmax |
| \ref boys::BoysAllOrdersWithRoute, \ref boys::BoysFitRoutes | the double batch at a named fit route (\ref boys::FitRoute), and the report of which routes exist, what each promises and over what interval |
| \ref boys::BoysFixedN | fp64, one order over an array of arguments, strided |
| \ref boys::BoysAllN, \ref boys::BoysAllNAtOrders | fp64, all orders over an array of arguments, order-major planes. `BoysAllN` takes one top order for the batch and classifies, groups and dispatches internally (\ref boys::BoysSortedArgs skips the sort for a non-decreasing array); `BoysAllNAtOrders` takes each argument's own top order, which is the shape a shell-quartet batch has — no padding of the arguments to a common order — and evaluates the per-argument body at each of them |
| \ref boys::BoysSingleF32, \ref boys::BoysAllOrdersF32, \ref boys::BoysAllNF32 | scalar fp32 (the batch forms seed in double); `BoysAllNF32` is the float lane's \ref boys::BoysAllN, one top order for the batch and order-major planes |
| \ref boys::BoysSingleF16, \ref boys::BoysAllOrdersF16, \ref boys::BoysSingleBf16, \ref boys::BoysAllOrdersBf16 | fp16/bf16 scalar I/O around the fp32 engine |
| \ref boys::BoysAllOrdersHalf2, \ref boys::BoysAllNF16Native | native half: region C's ladder in packed binary16 (\ref boys::Half2), one correctly rounded half operation per step, two arguments to a register, results scaled by 2^15 (\ref boys::kHalfNativeScaleExponent) |
| \ref boys::BoysCuda::InitializeTables, \ref boys::BoysCuda::SingleF32, \ref boys::BoysCuda::AllOrdersF32, \ref boys::BoysCuda::AllNF32, \ref boys::BoysCuda::SingleF64, \ref boys::BoysCuda::AllOrdersF64, \ref boys::BoysCuda::AllNF64, \ref boys::BoysCuda::SingleF16, \ref boys::BoysCuda::AllOrdersF16, \ref boys::BoysCuda::AllNF16 | CUDA lane (optional build); device arrays with an opaque stream handle; the tables upload on first use, and `InitializeTables` is an optional warm-up; `AllOrders*` is the all-orders batch at a per-element order, while `AllN*` is the device \ref boys::BoysAllN (one top order for the batch, order-major planes) and takes non-decreasing arguments, since it never sorts |

## Measuring the options on this machine

Which entry is fastest is a property of the host and the build flags rather than of the library:
whether the vector tier is present, whether a bare `a * b + c` is one rounding or two in the
compiler's hands, and how the arguments arrive all move the ranking. \ref boys::RunOptionProbe
measures the options this build offers on the machine it is called on and returns an \ref
boys::OptionProbeReport, which \ref boys::FormatOptionProbe renders as the text a consumer reads.

It enumerates the options from the library rather than from a list — each option's arithmetic is
resolved against \ref boys::backend::BoysBackends, and the relaxed rungs are the ones \ref
boys::QueryTier reports as served — and measures each over the library's own call shape: one
argument set, every argument's own highest order, one seed and then upward recursion to that order.
Each option comes back with its cost per argument, the spread of that cost over the passes it was
measured in, the machine load each pass was taken under, and the accuracy it delivered against the
certified fp64 lane. Every pass carries repeated runs of a fixed-work integer canary, and the pass is
admitted only if those runs agree within \ref boys::ProbeOptions::canarySpreadThreshold — a steady load
slows every option alike and leaves their order alone, an unsteady one is what corrupts a comparison,
and the canary's own spread measures that unsteadiness directly. A pass it cannot vouch for is
discarded rather than averaged in, and the reported figure is the minimum of the admitted passes.

When two options are closer than the resolution the run measured — the larger of the canary's widest
admitted spread and the leading option's own spread across its admitted passes, both measured rather
than assumed — the report says `CANNOT DETERMINE` and names what it could not separate instead of
ordering noise. The result is about the machine it was measured on, and the report says so in its own
output.

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
partition its arguments by region first, which is the work \ref boys::BoysAllN exists to do. The
entries above are the surface.

## Types

- \ref boys::F16 and \ref boys::Bf16 — the half-precision I/O types (std::float16_t aliases where the
  toolchain ships them; self-contained wrappers on MSVC).
- \ref boys::BoysStatus — the CUDA lane's status enum.

## Accuracy contract

The bound is |F̂_n(x) − F_n(x)| ≤ m·B for every supported n, x and lane, with m the accuracy
multiplier of the call. It defaults to 1 and runs to 65536. Every figure holds for **all** x ≥ 0:

| Lane | Error bound |
|---|---|
| double single | ≤ m·5.5e-14 everywhere; ≤ m·3e-14 below x = 11.899848152108484; ≤ m·1e-15 below about x = 1.0855 |
| double batch, whether the top order is the batch's or each argument's | ≤ m·5.5e-14 |
| float single / batch | ≤ m·1.5e-7 |
| fp16 / bf16 | ≤ m·1e-7 + ½ ULP |
| native half, x ≥ 28.984375 | ≤ 8 ULP of the returned value |
| CUDA fp64 | same m·budgets as the CPU double lanes |
| CUDA fp32, `RegionBExp::kAccurate` (the default) | same m·budgets as the CPU float lanes |
| CUDA fp32, `RegionBExp::kFast` | ≤ m·1.5e-7 + 8e-8, the lane's budget plus the corrected seed's contribution |

The CUDA fp32 lane's single entry is the one device entry that takes a second, certified axis: which
region-B exponential it evaluates (see \ref boys::RegionBExp). Both options ship with a bound of
their own, derived from the condition number of the region-B recurrence and confirmed by the device
gate's sweep; the bare hardware approximation, whose relative error grows with the argument, is not
offered at any multiplier.

"ULP" is the last representable digit of the result in the format concerned.

The double lane's stored fits come in two routes — the Chebyshev fits that are the default, and a
rational minimax alternative — and \ref boys::BoysFitRoutes reports each one's interval, the
argument its selector takes over at, its stored coefficient count, the error it was measured to
deliver and the bar it is certified against. The two hold the same bar over the same interval and
differ in what they store to reach it. Naming one changes only the fits that serve the intervals its
rows report; everywhere else the entry runs the default route and returns its values bit for bit.
The domain each route states is its own: region A's rational route takes over per order, from the
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
