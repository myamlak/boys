# API reference

The boys kernel evaluates the Boys function family
F_n(x) = ∫₀¹ t^(2n) exp(−x t²) dt, n = 0..32, in scalar fp64/fp32,
AVX2 region-sorted SIMD, fp16/bf16 I/O, and optional CUDA lanes.
Source and quick start: the [GitHub repository](https://github.com/myamlak/boys).

## Entry points

All CPU entries are `noexcept` and total: preconditions are n in
[0, 32], x >= 0, and output spans of the documented size. The CUDA
lane reports through \ref boys::BoysStatus instead. Every
entry is templated on one compile-time accuracy multiplier
`kAccuracyMultiplier` (default 1; see the accuracy contract below).

| Entry point | Lane |
|---|---|
| \ref boys::BoysSingle, \ref boys::BoysBatch | scalar fp64, single argument / order batch F_0..F_nmax |
| \ref boys::BoysFixedN | fp64, one order over an array of arguments, strided |
| \ref boys::BoysSingleF32, \ref boys::BoysBatchF32 | scalar fp32 (the batch form seeds in double) |
| \ref boys::BoysRegionASimd, \ref boys::BoysRegionBSimd, \ref boys::BoysRegionCSimd | AVX2 region-sorted fp64; arrays pre-partitioned by region, gate on \ref boys::BoysAvx2Available |
| \ref boys::BoysSingleF16, \ref boys::BoysBatchF16, \ref boys::BoysSingleBf16, \ref boys::BoysBatchBf16 | fp16/bf16 scalar I/O around the fp32 engine |
| \ref boys::BoysRegionASimdF16, \ref boys::BoysRegionBSimdF16, \ref boys::BoysRegionCSimdF16 | fp16 AVX2 region lanes (F16C) |
| \ref boys::BoysRegionASimdBf16, \ref boys::BoysRegionBSimdBf16, \ref boys::BoysRegionCSimdBf16 | bf16 AVX2 region lanes |
| \ref boys::BoysCuda::InitializeTables, \ref boys::BoysCuda::SingleF32, \ref boys::BoysCuda::BatchF32, \ref boys::BoysCuda::SingleF64, \ref boys::BoysCuda::BatchF64, \ref boys::BoysCuda::SingleF16, \ref boys::BoysCuda::BatchF16 | CUDA lane (optional build); device arrays with an opaque stream handle; the tables upload on first use, `InitializeTables` is an optional warm-up |

## Types

- \ref boys::F16 and \ref boys::Bf16 — the half-precision I/O types
  (std::float16_t aliases where the toolchain ships them;
  self-contained wrappers on MSVC).
- \ref boys::BoysStatus — the CUDA lane's status enum.

## Accuracy contract

The bound is |F̂_n(x) − F_n(x)| ≤ m·B_region for every supported n, x,
lane, and region, m the accuracy multiplier of the call:

| Lane | region A | extended band | region B | region C |
|---|---|---|---|---|
| double single | ≤ m·1e-15 | ≤ m·3e-14 | ≤ m·3e-14 | ≤ m·5.5e-14 |
| double batch | ≤ m·5.5e-14 | ≤ m·5.5e-14 | ≤ m·5.5e-14 | ≤ m·5.5e-14 |
| float single / batch | ≤ m·1.5e-7 | ≤ m·1.5e-7 | ≤ m·1.5e-7 | ≤ m·1.5e-7 |
| fp16 / bf16 | ≤ m·1e-7 + ½ ULP | ≤ m·1e-7 + ½ ULP | ≤ m·1e-7 + ½ ULP | ≤ m·1e-7 + ½ ULP |
| CUDA fp64 | same m·budgets as the CPU double lanes, verified directly | |
| CUDA fp32 | same m·budgets as the CPU float lanes; GPU-vs-CPU 3.5e-7 | |

Public signatures and supported domains are stable within a major
version; bitwise outputs are not (pin the release tag, compiler, and
flags for exact reproducibility). The full contract statement, the
region definitions, and the certified boundaries are in the
accompanying paper (see CITATION.bib in the repository).

## Indexes

- [Functions](globals_func.html) — the complete function index
- [Classes](annotated.html) — F16, Bf16, and the CUDA namespace
- [Files](files.html) — boys.hpp, f16.hpp, boys_cuda.hpp
