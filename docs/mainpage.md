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
| \ref boys::BoysFixedN | fp64, one order over an array of arguments, strided |
| \ref boys::BoysAllN | fp64, all nmax + 1 orders over an array of arguments, order-major planes; classifies, groups and dispatches internally (\ref boys::BoysSortedArgs skips the sort for a non-decreasing array) |
| \ref boys::BoysSingleF32, \ref boys::BoysAllOrdersF32 | scalar fp32 (the batch form seeds in double) |
| \ref boys::BoysSingleF16, \ref boys::BoysAllOrdersF16, \ref boys::BoysSingleBf16, \ref boys::BoysAllOrdersBf16 | fp16/bf16 scalar I/O around the fp32 engine |
| \ref boys::BoysAllOrdersHalf2, \ref boys::BoysAllNF16Native | native half: region C's ladder in packed binary16 (\ref boys::Half2), one correctly rounded half operation per step, two arguments to a register, results scaled by 2^15 (\ref boys::kHalfNativeScaleExponent) |
| \ref boys::BoysCuda::InitializeTables, \ref boys::BoysCuda::SingleF32, \ref boys::BoysCuda::AllOrdersF32, \ref boys::BoysCuda::AllNF32, \ref boys::BoysCuda::SingleF64, \ref boys::BoysCuda::AllOrdersF64, \ref boys::BoysCuda::AllNF64, \ref boys::BoysCuda::SingleF16, \ref boys::BoysCuda::AllOrdersF16, \ref boys::BoysCuda::AllNF16 | CUDA lane (optional build); device arrays with an opaque stream handle; the tables upload on first use, and `InitializeTables` is an optional warm-up; `AllOrders*` is the all-orders batch at a per-element order, while `AllN*` is the device \ref boys::BoysAllN (one top order for the batch, order-major planes) and takes non-decreasing arguments, since it never sorts |

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

The bound is |F̂_n(x) − F_n(x)| ≤ m·B_region for every supported n, x, lane and region, with m the
accuracy multiplier of the call:

| Lane | region A | extended band | region B | region C |
|---|---|---|---|---|
| double single | ≤ m·1e-15 | ≤ m·3e-14 | ≤ m·3e-14 | ≤ m·5.5e-14 |
| double batch | ≤ m·5.5e-14 | ≤ m·5.5e-14 | ≤ m·5.5e-14 | ≤ m·5.5e-14 |
| float single / batch | ≤ m·1.5e-7 | ≤ m·1.5e-7 | ≤ m·1.5e-7 | ≤ m·1.5e-7 |
| fp16 / bf16 | ≤ m·1e-7 + ½ ULP | ≤ m·1e-7 + ½ ULP | ≤ m·1e-7 + ½ ULP | ≤ m·1e-7 + ½ ULP |
| native half | — | — | — | ≤ 8 ULP of the returned value |
| CUDA fp64 | same m·budgets as the CPU double lanes, verified directly | | | |
| CUDA fp32 | same m·budgets as the CPU float lanes; GPU-vs-CPU 3.5e-7 | | | |

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
