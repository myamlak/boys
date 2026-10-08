# Source map

This page is for a reader who knows C++ and numerical analysis and nothing about this tree: it
says which file implements what, and which test holds each thing to its contract. It is a map, not
a specification — what a lane guarantees, and why the figures are what they are, is in
[the per-lane contract](lane-contract.md) and [the accuracy contract](specification.md#accuracy-contract).

Every claim below is read off the files it names. Where this page could not tell what a file is
for, it says so instead of guessing.

## The build, in one paragraph

`CMakeLists.txt` at the root defines the `boys` library target and every test, example and
benchmark target. The public headers are under `include/boys/`, the translation units under
`src/`, the tests and the committed reference data under `tests/` and `tests/data/`, the checkers
and generators under `tools/`, the CI workflow under `.github/workflows/ci.yml`, and the two
vendored dependencies under `third_party/`. `docs/` holds the published pages.

## The three regions

The argument line is cut into regions, and each is evaluated differently. The boundaries are
`kX0 = 11.899848152108484` and `kX1 = 28.98933773882074`, both committed in
`include/boys/boys_coefficients.hpp`.

| Region | Interval | What it is | The file that implements it | The test that holds it |
|---|---|---|---|---|
| A | `[0, kX0)` | Per-order piecewise Chebyshev fits, summed by a split Clenshaw recurrence or by Horner. Its top part — the **extended band** — is one fit of its own read by upward recursion, dispatched per order at the tier thresholds | Piece lookup and summation: `include/boys/boys_impl.hpp` (`RegionAValue`, `ChebyshevFit`, `ClenshawSplit`, `FitSum`, `RegionBExtendedSeed`). The stored fits: `include/boys/boys_coefficients.hpp` (generated). The partitions: `ShippedRegionAPartition`, `NarrowRegionAPartition` and the `UniformFit` specializations, all in `boys_impl.hpp` | `tests/boys_accuracy_test.cpp` (per-region budgets on the grid), `tests/boys_across_orders_test.cpp` (the packed-orders lane), `tests/boys_transform_test.cpp` (the transform lane), `tests/boys_accuracy_gate.cpp` (every documented bound) |
| A, vector tier | as above | The same fits evaluated four arguments or four orders to a register, and the AVX2 arithmetic behind them | `src/boys_simd.cpp` (across arguments), `src/boys_orders_simd.cpp` with `src/boys_orders_simd.hpp` (across orders; its cells are declared `extern template` in `boys_impl.hpp` and defined here), `src/boys_backend_simd.hpp` (the 4-wide double arithmetic) | `tests/boys_across_orders_test.cpp`, `tests/boys_orders_f32_test.cpp`, `tests/boys_muladd_route_simd_test.cpp` |
| B | `[kX0, kX1)` | One `F_0` fit plus upward recursion; the exponential that seeds it is an axis | `include/boys/boys_impl.hpp` (`ShippedRegionBSeed`, `NarrowRegionBSeed`, `RegionBHalfExp`, `UpwardStep`); the seed's coefficients in `boys_coefficients.hpp` | `tests/boys_accuracy_test.cpp`, `tests/boys_boundary_test.cpp` |
| C | `[kX1, inf)` | The asymptotic form `1/2 sqrt(pi/x)` plus upward recursion; no stored fit | `include/boys/boys_impl.hpp` (`kBoysHalfSqrtPi`, `BoysAllNBodyRegionC`), the vector tier in `src/boys_simd.cpp` (`BoysRegionCSimd`) | `tests/boys_accuracy_test.cpp`, `tests/boys_half_native_test.cpp` |
| Which region one argument takes | — | The dispatch itself: zero, tabulated, extended, middle, asymptotic | `include/boys/boys_impl.hpp`, `BoysPath` and `BoysClassifyPath` | the region-boundary cases in `tests/boys_accuracy_test.cpp` and `tests/boys_boundary_test.cpp` |

Two more things are stated per region rather than per lane and live in the same two places:
`include/boys/boys_effective_degrees.hpp` carries the lane roles the region budgets and the
truncation amplification are read through, and `tools/gen_boys_coefficients.py` is what places the
fits and emits the boundaries and degrees.

## The lanes, and the entry points that serve them

A *lane* is an entry plus the arithmetic behind it, and a bound always describes one lane. Five
precision families carry host entries - double, float, fp16 I/O, bf16 I/O and the native packed
half - and the device lane adds four `Precision` members of its own. The entries are declared in
`include/boys/boys.hpp`, defined in `include/boys/boys_impl.hpp`, and instantiated out of line in
`src/boys.cpp`, `src/boys_simd.cpp` and `src/boys_orders_simd.cpp`.

| Lane | `Precision` member | Entry points | What it is | The test that holds it |
|---|---|---|---|---|
| double | `kFp64` | `BoysSingle`, `BoysAllOrders`, `BoysFixedN`, `BoysAllN`, `BoysAllNAtOrders`, `BoysAllOrdersWithRoute`, and the `*Checked` spellings | scalar fp64, the certified lane every other is measured against | `tests/boys_accuracy_test.cpp`, `tests/boys_all_n_test.cpp`, `tests/boys_fixed_n_test.cpp`, `tests/boys_test.cpp` |
| float | `kFp32` | `BoysSingleF32`, `BoysSingleF32WithRoute`, `BoysAllOrdersF32`, `BoysFixedNF32`, `BoysAllNF32`, `BoysAllNAtOrdersF32` | scalar fp32; the batch forms seed in double | `tests/boys_accuracy_test.cpp`, `tests/boys_orders_f32_test.cpp` |
| fp16 I/O | `kFp16` | `BoysSingleF16`, `BoysAllOrdersF16`, `BoysFixedNF16`, `BoysAllNF16`, `BoysAllNAtOrdersF16` | 16-bit storage around the float engine; types in `include/boys/f16.hpp` | `tests/f16_test.cpp`, `tests/boys_accuracy_test.cpp` |
| bf16 I/O | `kBf16` | `BoysSingleBf16`, `BoysAllOrdersBf16` | the same engine and budget, stored in this format | `tests/f16_test.cpp`, `tests/boys_accuracy_test.cpp` |
| native packed half | none — a lane with entries and no `Precision` member | `BoysAllOrdersHalf2`, `BoysAllNF16Native` | region C's ladder in packed binary16, two arguments to a register, results scaled by `2^15`; the arithmetic is defined by `include/boys/half2.hpp` and the bodies are in `src/boys_half_native.cpp` | `tests/boys_half_native_test.cpp` |
| CUDA, host-launched | `kFp64Device`, `kFp32Device`, `kFp16Device`, `kBf16Device` | `boys::BoysCuda::SingleF64`, `AllOrdersF64`, `AllNF64` and the f32 and f16 siblings, `InitializeTables` | device arrays with an opaque stream handle; declared in `include/boys/boys_cuda.hpp` | `tests/boys_cuda_test.cpp`, `tests/boys_cuda_route_test.cpp`, `tests/boys_cuda_accuracy_gate.cpp` |
| CUDA, device-callable | as above | `boys::BoysDeviceSingleF64`, `BoysDeviceAllOrdersF64`, `BoysDeviceAllNF64`, `BoysDeviceEachOrderF64` and the f32 and fp16 siblings | the same arithmetic inside the caller's own kernel; declared in `include/boys/boys_cuda_device.hpp` | `tests/boys_accuracy_gate_fast_device.cu`, `tests/boys_cuda_device_demo.cu`, `tests/consumer_cuda_defaults.cu` |

The C surface is `include/boys/boys_c.h` with its bodies in `src/boys_c.cpp`, one entry per lane
group above; `tests/boys_c_test.cpp` holds it to the C++ entries. `include/boys/boys_span.hpp` adds
`std::span` forwarders and `include/boys/status.hpp` the shared status enum the CUDA and C surfaces
report through.

`tests/boys_test.cpp` and `tests/boys_cuda_test.cpp` carry no file-level comment. They are the
bodies of the `boys-tests` and `boys-cuda-tests` cases; what their cases cover is read from their
case names, and this page does not guess past that.

## The option axes

An axis is a choice a call site can name. Five are the host call site's — the fit route, the
evaluation scheme, the packing axis, the fit granularity and the division form — and they are the
five axes a host `EvalPolicy` holds. The region-B exponential is the device lane's own choice and
is named at the device entry rather than in the host policy. The multiply-add route is the build's,
and no call site names it.

| Axis | The enumeration | The file that defines it | Where it is read | The test that holds it |
|---|---|---|---|---|
| fit route | `FitRoute` | `include/boys/accuracy.hpp` | `RouteFit` and the route tables in `include/boys/boys_impl.hpp`; the stored fits in `boys_coefficients.hpp` | `tests/boys_accuracy_test.cpp`, `tests/consumer_option_space.cpp` |
| evaluation scheme | `EvalScheme` | `include/boys/accuracy.hpp` | `ChebyshevValue` and `FitSum` in `boys_impl.hpp`; the across-orders lane has its own summation enumeration, `OrdersScheme`, in `src/boys_orders_simd.hpp` | `tests/boys_across_orders_test.cpp`, `tests/consumer_option_space.cpp`, `tools/check_enum_arms_covered.py` (over `OrdersScheme`, whose switches must name every member) |
| region-B exponential | `RegionBExp` | `include/boys/accuracy.hpp` | `RegionBHalfExp` and `RegionBHalfExpF32` in `boys_impl.hpp`; the device's own bodies in `include/boys/boys_cuda_arithmetic.hpp` | `tests/boys_accuracy_test.cpp`, `tests/boys_cuda_accuracy_gate.cpp` (the device sweep over both members), `tests/consumer_option_space.cpp` |
| fit granularity | `FitGranularity` | `include/boys/accuracy.hpp` | the partitions and `FitAnswersPartition` in `boys_impl.hpp`; the per-piece rows in `boys_coefficients.hpp` | `tests/boys_fixture_pins_test.cpp`, the `build_defaults_uniform` tree built by CI, `tools/check_class_combinations.py` |
| division form | `DivisionForm` | `include/boys/accuracy.hpp` | `DivideExact`, `DividePlain`, `DivideByReciprocal`, `DivideStep` in `boys_impl.hpp` | `tests/build_defaults_division_form.hpp` (the fixture that moves it on the host), `tests/boys_cuda_test.cpp` (all three on the device), `tools/check_combination_bounds.py` |
| packing axis | `PackAxis` | `include/boys/backend.hpp` | the vector lanes in `src/boys_simd.cpp` and `src/boys_orders_simd.cpp`. The device's reading of the same question is `DevicePacking` in `include/boys/boys_cuda_options.hpp`, which adds the `kNotApplicable` member a one-order entry states | `tests/boys_across_orders_test.cpp`, `tools/check_host_entry_bijection.py` |
| multiply-add route (the build's, not a call-site one) | `MulAddRoute` | `include/boys/backend.hpp` | the named arithmetic backends in `src/boys_backend_simd.hpp` and `src/boys_backend_registry.hpp`; the device's is `include/boys/boys_cuda_muladd.hpp` | `tests/boys_backend_test.cpp`, `tests/boys_muladd_route_test.cpp`, `tests/boys_muladd_route_simd_test.cpp` |

The choices a call naming no policy resolves to are the **build-defaults seam**,
`include/boys/boys_build_defaults.hpp`: seven names, five for a host class and two for a device
one, each checked against the enumerators of the axis it initialises. A build replaces the file
with `-DBOYS_BUILD_DEFAULTS=<header>`; the fixtures that move one axis at a time are under
`tests/build_defaults_*.hpp`, and `tests/boys_build_defaults_test.cpp` and
`tests/boys_defaults.cpp` read what the seam resolved. The row table itself — `DefaultPolicyRow`,
`BOYS_DEFAULT_POLICY_BUILD_ROWS` and its device half — is in `include/boys/boys.hpp`, and the rows
it is filled from are in the seam file.

## Host and device

| Piece | File | Compiled by | The test that holds it |
|---|---|---|---|
| The host surface, its entries and its policy table | `include/boys/boys.hpp` | the host compiler | `tests/consumer_umbrella.cpp` (a translation unit whose only library include is this header, exercising the documented surface), `tests/boys_test.cpp` |
| The host implementation: regions, recurrences, entry bodies | `include/boys/boys_impl.hpp` | the host compiler | `tests/boys_accuracy_test.cpp`, `tests/boys_across_orders_test.cpp` |
| The scalar instantiations and the library's own reports (`BoysFitRoutes`, `BoysEvalSchemes`, `BoysPackAxes`, `BoysFitGranularities`, `BoysDivisionForms`, `BoysRegionBExps`, `BoysLaneContracts`, `BoysAccuracyGuaranteed`, `BoysAccuracyDelivered`, `QueryCombination`) | `src/boys.cpp` | the host compiler | `tests/boys_accuracy_gate.cpp` (the bound it publishes for each lane), `tests/consumer_defaults_match_docs.cpp` (the default policy table a document states) |
| The CUDA host-side class, CUDA-header-free | `include/boys/boys_cuda.hpp` | the host compiler | `tests/boys_cuda_test.cpp`, `tests/consumer_cuda_defaults_host.cpp` |
| The CUDA status layer, the option rows and `BoysDeviceOptions()` | `src/boys_cuda.cpp` | the host compiler | `tests/boys_cuda_probe_test.cpp` (the space and its closure), `tools/check_device_entry_bijection.py` |
| The CUDA kernels and the table upload | `src/boys_cuda.cu` | nvcc | `tests/boys_cuda_accuracy_gate.cpp`, `tests/boys_cuda_test.cpp` |
| The device-callable entries and the arithmetic they run | `include/boys/boys_cuda_device.hpp`, `include/boys/boys_cuda_arithmetic.hpp` | nvcc | `tests/boys_accuracy_gate_fast_device.cu`, `tests/boys_cuda_device_demo.cu` |
| The handle a caller passes, and the lanes its tables are cut into | `include/boys/boys_device_tables.hpp` | both | `tests/boys_cuda_route_test.cu`, `tests/boys_build_defaults_test.cpp` |
| Which multiply-add the device arithmetic is built with | `include/boys/boys_cuda_muladd.hpp` | both | `tests/boys_cuda_route_test.cpp`, `tests/boys_cuda_route_test.cu` |

**The device option space is a header of its own for a compile-time reason.**
`include/boys/boys_cuda_options.hpp` carries the rows — one `DeviceOptionInfo` per `DeviceEntry`,
reached through `BoysDeviceOptions()` — and it has two readers: the host surface that declares the
entries, and the device functions that implement them. The second is compiled by nvcc in a
translation unit that has to stay free of the library's C++23 headers, and it cannot take
`boys_cuda.hpp`, which pulls in the whole library through `boys.hpp`. So the rows live in the
options header, and a probe's rows and an accuracy gate's claims are projections of one report
rather than lists that drift from it. The same constraint is why the axis enumerations are in
`accuracy.hpp` and `backend.hpp` rather than beside the kernel: a header naming them must not have
to carry the kernel.

**The policy layer beside it is a second list of the same surface, and it is not a table.**
`include/boys/boys_cuda_policy.hpp` defines the policy-templated members — `<Entry>WithPolicy`, one
per option the rows book — so that a caller can reach a device combination by naming a policy
rather than by knowing an entry name. It is included at the end of `boys_cuda.hpp` because every
definition there names a declaration of the class above, and it includes that header back at its
top so a translation unit that includes it alone still compiles: the same continuation
`boys_impl.hpp` is to `boys.hpp`. Two checkers hold the two lists to each other:
`tools/check_device_entry_bijection.py` reads the rows against the members the class declares, and
`tools/check_device_booked_surface.py` reads those members against the bodies the class's own
translation unit defines.

For a reader who comes to `boys_cuda_policy.hpp` expecting a table: it carries none. It has no
array of rows, no accessor returning rows and no count constant — the option space's rows are in
`boys_cuda_options.hpp`, and what this header adds to them is one policy-templated member per
option the rows book.

The device probe is `include/boys/boys_cuda_probe.hpp` with its host side in
`src/boys_cuda_probe.cpp`, its device side in `src/boys_cuda_probe_kernels.cu` and the shared
option list in `src/boys_cuda_probe_entries.hpp`; the host probe is `include/boys/boys_probe.hpp`
with its bodies in `src/boys_probe.cpp`.

## Generated, recorded, and hand-written

These are easy to confuse, and the difference decides what a change to one costs. A **generated**
file is re-derived by a tool and its committed bytes are a claim about that tool. A **recorded** file
is a run kept as a file: nothing re-derives it, so it is re-made by running the program again. A
**hand-written** file is edited, and the checker beside it is what holds it to the code.

| Artifact | Written by | Refresh with | The check that holds it |
|---|---|---|---|
| `include/boys/boys_coefficients.hpp`, `tests/data/boys_reference.csv` | `tools/gen_boys_coefficients.py` | run it with no flag; it rewrites both | its own `--check`, re-deriving both and comparing byte for byte |
| `tests/data/boys_accuracy_gate_reference.csv` | `tools/gen_boys_accuracy_gate_reference.py` | run it with no flag | its own `--check` |
| The platform table inside `docs/specification.md` | `tools/gen_platform_table.py`, from the workflow and `.github/required-checks.txt` | run it with no flag | its own `--check` |
| The row region inside `docs/build-facts.md` | `tools/gen_build_facts.py` | `--record <file>` with a captured probe row; it merges the row in and re-renders the document | its own `--check`, which also refuses a table holding no CI leg's row |
| `.github/option-matrix.json` | `tools/gen_option_matrix.py`, from the library's own declarations | run it with no flag | its own `--check`, in both directions |
| The `DeviceEntryArithmeticOf` block in `include/boys/boys_cuda_options.hpp` | `tools/gen_entry_aliases.py`, from the entries' own bodies | run it with no flag | its own `--check` |
| `tests/data/boys_accuracy_gate_run.txt` | a run of `boys-accuracy-gate`, kept as a file | build the gate, run it, write its output there | `tools/check_recorded_run.py` (is it this tree's run) and `tools/check_bound_transcripts.py` (do the documents quote it) |
| `tests/data/boys_option_probe_report.txt` | a run of `boys-option-probe` | build the driver, run it, write its output there | `tools/check_class_combinations.py --report`, `tools/combination_matrix.py` |
| `tests/data/boys_device_probe_report.txt` | a run of `boys-device-probe` on a machine with a card | as above | `tools/status.py` and the device checkers that read it |
| `tools/host-shapes.json` | a person | edit it when a host entry moves between shapes | `tools/check_host_entry_bijection.py`, which reads it against the headers in both directions |
| The seam, `include/boys/boys_build_defaults.hpp` | a person — the committed five host names and two device ones | or: `tools/splice_default_rows.py --emitted <file> --seam <seam>` copies the rows a probe run emitted with `--emit-defaults` into it, row for row | `tools/check_seam_macros.py` requires every name it offers to be expanded somewhere in `include/` or `src/`; the CI legs that build against `tests/build_defaults_*.hpp` are what compiles a moved axis |

A *recorded* file is a run kept as a file: it is not re-derived, so a check that reads one is
asking about the revision the run names, and the answer changes only when someone re-makes the
run. `tests/data/boys_reference.csv` is the opposite — it is generated, and the committed bytes are
a claim about what its generator produces.

## The probes

| Program | What it emits | Who consumes it |
|---|---|---|
| `benchmarks/boys_option_probe.cpp` (`boys-option-probe`) | the host option probe's report: the options this build carries on this machine, their cost per argument, the spread of the paired rounds, and the verdict | a person; the committed report under `tests/data/`; `tools/combination_matrix.py` |
| `benchmarks/boys_device_probe.cpp` (`boys-device-probe`) | the device option probe's report: the same question for the card, with the card's own name | a person; the committed device report; `tools/status.py` |
| `tests/boys_build_facts.cpp` (`boys-build-facts`) | one line per build fact — the target ISA, the lane widths, whether a bare `a*b+c` rounds once in this build, whether the compiled library reaches the runtime's `fma` | `tools/gen_build_facts.py --record`; the CI step that compares a row against `docs/build-facts.md` |
| `tests/boys_accuracy_gate.cpp` (`boys-accuracy-gate`) | the gate's report: every documented lane and region against the committed grid, worst delivered beside the bound | a person; the committed recorded run; the specification page's printed-run table through `tools/check_bound_transcripts.py` |
| `tests/boys_avx2_probe.cpp` (`boys-avx2-probe`) | one line: what `BoysAvx2Available()` reports on this leg, beside what the leg expects | ctest, per leg; the expectation comes from `BOYS_EXPECT_AVX2` |
| `tests/boys_cuda_axis_sensitivity.cpp` (`boys-cuda-axis-sensitivity`) | a per-cell report: whether each row of the device option table really reads the axes its coordinates state | a person. It is built by CMake and registered as no test, because the cells it reports are pairs the library documents as one arithmetic under two names |
| `tests/boys_boundary_standalone.cpp` (`boys-boundary-standalone`) | the smallest argument at which the erf-seeded upward recursion matches the reference series to 5e-14, per order limit | ctest, and the CI steps that compile and run it over `tests/data/boys_reference.csv` |

The two option probes read a clock and none of the others does. No probe's figure is a fact about
the library alone: a ranking is a property of the machine, the build and the card, which is why
the probes report what they measured rather than naming a winner from a table.

## The test tiers

`ctest` is one runner over several tiers. `BOYS_BUILD_TESTS` is on by default; `-DBUILD_CUDA=ON`
adds the device tier and `-DBUILD_BENCHMARKS=ON` the benchmark drivers, which are built and never
registered.

| Tier | What it holds | The ctest names |
|---|---|---|
| The kernel contract suite | the per-region accuracy budgets on the committed grid, the C surface, the region boundaries, the fp16/bf16 conversion contract, the across-orders and transform lanes, the backend and multiply-add route contracts, the build-defaults seam | `boys-tests` |
| The gate | every documented per-lane, per-region bound, re-measured against the committed 45-digit grid | `boys-accuracy-gate` (run with `--strict`) |
| The standalone boundary check | the recursion boundary, compiled outside the library's own targets and run over the committed grid | `boys-boundary-standalone` |
| The consumer checks | that a translation unit including only `<boys/boys.hpp>` reaches the whole documented surface, that one naming what the library does not pre-instantiate links nothing, that the defaults a call naming nothing compiles are the documented ones, that every combination the space serves has a name a caller can write | `boys-consumer-umbrella`, `boys-consumer-header-only`, `boys-consumer-defaults`, `boys-consumer-defaults-match-docs`, `boys-consumer-option-space` |
| The defaults report | what this build's unnamed calls resolve to, class by class, read from the table the build carries | `boys-defaults` |
| The dispatch assertion | that `BoysAvx2Available()` reports what the architecture requires, so a wrong CPUID bit cannot leave the vector tier silently dead behind a green run | `boys-avx2-probe` (built only when `BOYS_EXPECT_AVX2` is supplied) |
| The build facts | this build's own row, against the row `docs/build-facts.md` records for the leg | `boys-build-facts` |
| The examples | every program under `examples/`, run rather than only built | `boys-example-<name>` |
| The documents against the programs | that the runs `README.md` and `docs/getting-started.md` quote are what the examples print | `boys-example-transcripts` (registered only on an MSVC build with the committed defaults and the fused route, with a Python3 interpreter found) |
| The device tier | the CUDA host entries, the three division forms, the device probe's report shape and its device-free cases, the device accuracy gate, the device refusal reasons | `boys-cuda-tests`, `boys-cuda-tests-divform-control`, `boys-cuda-probe-tests`, `boys-cuda-accuracy-gate`, `boys-cuda-route-tests`, `boys-consumer-cuda-defaults`, `boys-cuda-refusal-reason-tests` |

The checkers under `tools/` are the other half of the same idea — they read the tree rather than
run it, and most of them are armed as steps of `.github/workflows/ci.yml` rather than as ctest
cases. `python tools/check_all.py` runs every one the workflow arms on its full leg, and
`--fast` runs the ones that do not pay for a derivation or a compile. `CONTRIBUTING.md` says which
check belongs to which kind of edit.
