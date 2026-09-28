# Build facts

What a build of this library actually is, read out of the build: the instruction sets it targets,
whether a bare `a * b + c` written in it is a single rounding, whether the compiled library reaches
the runtime's `fma`, which arithmetic backends it carries and what each of them answers, and the
machine each statement was observed on.

Every line here is deterministic — a preprocessor fact, a compile-time constant, a symbol read out of
the library's own object code, or the operating system's report of the processor. **Nothing on this
page is a measurement.** A duration taken on a shared build machine describes the machine and its
neighbours as much as the code, and none of the questions below needs one: each is a yes-or-no fact
about a build, and the same build answers it identically on every run.

This page is a record of builds, not of machines. A machine's contribution is named — the processor
the row was seen on, the count of logical processors, the compiler version, the date the row was
recorded — and it is reported rather than gated, because the runners these rows come from are
ephemeral and the same build leg draws different hardware from run to run. **Read a row as a
statement about an architecture and a flag set, never as a statement about the machine that produced
it.**

The rows are printed by `tests/boys_build_facts.cpp`, built as the `boys-build-facts` target. Every
CI leg that builds anything builds it, runs it under that leg's own check name, and compares what it
finds against the row recorded here for that leg: a fact that moves turns the leg red, and a leg whose
row has not been recorded yet prints the row it would record and says so instead of passing quietly.
`tools/gen_build_facts.py` is what folds a printed row into this page.

## What these facts mean for a consumer

Two of the facts decide the shape of a multiply-add in a caller's kernel, and both belong to the
build rather than to the source. The same expression is two different arithmetics in two builds, and
the same call is two different costs:

- **A build that does not contract has a different fastest option.** `contract.this-tu.fp64` and
  `contract.this-tu.fp32` say whether a plain `a * b + c` compiled the way this build compiles is a
  single rounding. Where the answer is 1, the compiler emits one fused instruction from that bare
  expression, and an explicit fused call is a second way of writing something the build already does.
  Where the answer is 0, the same bare expression is two roundings, and a kernel that wants the fused
  value has to name it — `std::fma`, or this library's own `Fused`. **A route tuned on one build is
  not the route that build's neighbour measures fastest on, because it is not the same arithmetic.**
- **`fma.route` says what naming it costs.** `out-of-line-call` means the library's objects carry an
  undefined reference to the runtime's `fma`: every fused operation is a call into the C library.
  `no-call` means they carry no such reference — on a target with an FMA instruction the compiler
  emits that instruction instead. This is a fact of the compiled artifact, not of the source, and it
  is why the two builds of one source on one machine are not interchangeable in a cost model.
- **`backend.*.contracts` is the same question asked inside the library**, per arithmetic backend,
  because the library compiles its packed arithmetic under different flags than its scalar
  arithmetic. Its answer can differ from `contract.this-tu.*` in the same row, and that is expected:
  each answers for the translation units whose flags it was compiled with.
- **`simd.target`, `simd.bits` and `simd.lanes.*`** say how much packed arithmetic the build targets
  and how many values one packed operation covers — the shape a caller's batch wants.
- **`runtime.avx2`** is the library's own `BoysAvx2Available()` on the machine the row ran on: 1 where
  the AVX2 tier runs, 0 where the entry points dispatch to the scalar lanes.

This page claims no speed for anything. It records the facts a cost decision depends on; which route
wins on a given build and machine is a question for a measurement taken on that machine.

## The format

One line per fact, `key=value`, with no spaces around the `=`. The first line of a row is the format
tag `boys.build-facts/1`; a fenced block that does not open with that tag is prose and is never read
as a row. A value is a single line — whitespace runs inside one are collapsed. A fact the platform
cannot establish is printed `unestablished`, with a `fma.route.reason` line where there is a reason;
nothing is inferred from a neighbouring fact, and no fact is omitted.

| Key | What it says |
|---|---|
| `leg` | the CI leg the row belongs to (that leg's check name), or a derived name for a developer build |
| `recorded` | the date the row was folded into this page. Stamped by the generator: the probe itself reads no clock |
| `config` | the build configuration (`Release`, `Debug`, ...) |
| `cpu.model`, `cpu.model.source` | the processor the row was observed on, and where that report came from (`cpuid`, `/proc/cpuinfo`, `sysctl`, `PROCESSOR_IDENTIFIER`) |
| `cpu.logical` | logical processors the platform reports |
| `os`, `arch`, `pointer.bits`, `endian` | what the build targets |
| `compiler.id`, `compiler.version`, `compiler.standard` | the compiler, its version, and the standard level it was told to use |
| `sanitizers` | the sanitizers this build carries, or `none` |
| `macro.<NAME>` | 1 when that preprocessor macro is defined in this build, 0 when it is not. The list is fixed: SSE2, AVX, AVX2, FMA, F16C, AVX512F, NEON and the arm64 arithmetic-feature macros |
| `simd.target`, `simd.bits` | the packed arithmetic the build targets, and its width in bits |
| `simd.lanes.fp64`, `.fp32`, `.fp16` | values one packed operation covers, per format |
| `backend.<name>.contracts` | whether a bare `a * b + c` in that backend's arithmetic is a single rounding, as the library's `BoysBackends()` reports it |
| `contract.this-tu.fp64`, `.fp32` | the same question asked by the translation unit this row was printed from — the flags a consumer of this build gets |
| `runtime.avx2` | whether the library's AVX2 tier runs on the machine the row ran on |
| `fma.route` | `out-of-line-call` when the library's objects reference the runtime's `fma`, `no-call` when they do not, `unestablished` when it could not be read |
| `fma.route.symbols` | which of `fma` and `fmaf` were found, or `none` |
| `fma.route.reason` | why the route is unestablished, when it is |
| `fma.route.artifact`, `fma.route.tool` | the artifact that was read and the tool that read it |

The keys `leg`, `recorded`, `cpu.model`, `cpu.model.source`, `cpu.logical`, `compiler.version`,
`compiler.standard`, `fma.route.tool` and `fma.route.artifact` are **tags**: they say where and by
whom a row was observed rather than what the build does. A leg reports a tag that has moved and does
not fail on it; every other key is gated.

Recording a row, from a leg's captured output or a developer build's:

    python tools/gen_build_facts.py --record build-facts-row.txt

And the page is re-checked against its own rows, and against the legs in the workflow, with:

    python tools/gen_build_facts.py --check

## Recorded rows

<!-- build-facts:begin (generated by tools/gen_build_facts.py; do not edit) -->
0 of the 19 CI legs have a row here, and 7 row(s) were recorded from builds that are not CI legs.

Each row was observed by the build it describes, and the lines a row carries are the facts of that one build.

#### `local-linux-x86-64-avx2 gcc-15.2.0 Release`

```text
boys.build-facts/1
leg=local-linux-x86-64-avx2 gcc-15.2.0 Release
config=Release
cpu.model=Intel(R) Core(TM) i7-9850H CPU @ 2.60GHz
cpu.model.source=cpuid
cpu.logical=12
os=linux
arch=x86-64
pointer.bits=64
endian=little
compiler.id=gcc
compiler.version=15.2.0
compiler.standard=202302
sanitizers=none
macro.__SSE2__=1
macro.__AVX__=1
macro.__AVX2__=1
macro.__FMA__=1
macro.__F16C__=0
macro.__AVX512F__=0
macro.__ARM_NEON=0
macro.__ARM_FEATURE_FMA=0
macro.__ARM_FEATURE_FP16_SCALAR_ARITHMETIC=0
macro.__ARM_FEATURE_FP16_VECTOR_ARITHMETIC=0
simd.target=avx2
simd.bits=256
simd.lanes.fp64=4
simd.lanes.fp32=8
simd.lanes.fp16=16
backend.scalar-fp64.contracts=1
backend.scalar-fp32.contracts=1
backend.avx2-fp64.contracts=1
backend.avx2-fp32.contracts=1
contract.this-tu.fp64=1
contract.this-tu.fp32=1
runtime.avx2=1
fma.route=no-call
fma.route.symbols=none
fma.route.artifact=libboys.a
fma.route.tool=nm
recorded=2026-09-24
```

#### `local-linux-x86-64-sse2 clang-21.1.8 (6ubuntu1) Release`

```text
boys.build-facts/1
leg=local-linux-x86-64-sse2 clang-21.1.8 (6ubuntu1) Release
config=Release
cpu.model=Intel(R) Core(TM) i7-9850H CPU @ 2.60GHz
cpu.model.source=cpuid
cpu.logical=12
os=linux
arch=x86-64
pointer.bits=64
endian=little
compiler.id=clang
compiler.version=21.1.8 (6ubuntu1)
compiler.standard=202302
sanitizers=none
macro.__SSE2__=1
macro.__AVX__=0
macro.__AVX2__=0
macro.__FMA__=0
macro.__F16C__=0
macro.__AVX512F__=0
macro.__ARM_NEON=0
macro.__ARM_FEATURE_FMA=0
macro.__ARM_FEATURE_FP16_SCALAR_ARITHMETIC=0
macro.__ARM_FEATURE_FP16_VECTOR_ARITHMETIC=0
simd.target=sse2
simd.bits=128
simd.lanes.fp64=2
simd.lanes.fp32=4
simd.lanes.fp16=8
backend.scalar-fp64.contracts=0
backend.scalar-fp32.contracts=0
backend.avx2-fp64.contracts=1
backend.avx2-fp32.contracts=1
contract.this-tu.fp64=0
contract.this-tu.fp32=0
runtime.avx2=1
fma.route=out-of-line-call
fma.route.symbols=fma,fmaf
fma.route.artifact=libboys.a
fma.route.tool=nm
recorded=2026-09-24
```

#### `local-linux-x86-64-sse2 gcc-15.2.0 Debug address,undefined`

```text
boys.build-facts/1
leg=local-linux-x86-64-sse2 gcc-15.2.0 Debug address,undefined
config=Debug
cpu.model=Intel(R) Core(TM) i7-9850H CPU @ 2.60GHz
cpu.model.source=cpuid
cpu.logical=12
os=linux
arch=x86-64
pointer.bits=64
endian=little
compiler.id=gcc
compiler.version=15.2.0
compiler.standard=202302
sanitizers=address,undefined
macro.__SSE2__=1
macro.__AVX__=0
macro.__AVX2__=0
macro.__FMA__=0
macro.__F16C__=0
macro.__AVX512F__=0
macro.__ARM_NEON=0
macro.__ARM_FEATURE_FMA=0
macro.__ARM_FEATURE_FP16_SCALAR_ARITHMETIC=0
macro.__ARM_FEATURE_FP16_VECTOR_ARITHMETIC=0
simd.target=sse2
simd.bits=128
simd.lanes.fp64=2
simd.lanes.fp32=4
simd.lanes.fp16=8
backend.scalar-fp64.contracts=0
backend.scalar-fp32.contracts=0
backend.avx2-fp64.contracts=0
backend.avx2-fp32.contracts=0
contract.this-tu.fp64=0
contract.this-tu.fp32=0
runtime.avx2=1
fma.route=unestablished
fma.route.reason=the symbol tool did not run
fma.route.artifact=libboys.a
fma.route.tool=nm
recorded=2026-09-24
```

#### `local-linux-x86-64-sse2 gcc-15.2.0 Release`

```text
boys.build-facts/1
leg=local-linux-x86-64-sse2 gcc-15.2.0 Release
config=Release
cpu.model=Intel(R) Core(TM) i7-9850H CPU @ 2.60GHz
cpu.model.source=cpuid
cpu.logical=12
os=linux
arch=x86-64
pointer.bits=64
endian=little
compiler.id=gcc
compiler.version=15.2.0
compiler.standard=202302
sanitizers=none
macro.__SSE2__=1
macro.__AVX__=0
macro.__AVX2__=0
macro.__FMA__=0
macro.__F16C__=0
macro.__AVX512F__=0
macro.__ARM_NEON=0
macro.__ARM_FEATURE_FMA=0
macro.__ARM_FEATURE_FP16_SCALAR_ARITHMETIC=0
macro.__ARM_FEATURE_FP16_VECTOR_ARITHMETIC=0
simd.target=sse2
simd.bits=128
simd.lanes.fp64=2
simd.lanes.fp32=4
simd.lanes.fp16=8
backend.scalar-fp64.contracts=0
backend.scalar-fp32.contracts=0
backend.avx2-fp64.contracts=1
backend.avx2-fp32.contracts=1
contract.this-tu.fp64=0
contract.this-tu.fp32=0
runtime.avx2=1
fma.route=out-of-line-call
fma.route.symbols=fma,fmaf
fma.route.artifact=libboys.a
fma.route.tool=nm
recorded=2026-09-24
```

#### `local-windows-x86-64-avx2 msvc-19.51.36256 Release`

```text
boys.build-facts/1
leg=local-windows-x86-64-avx2 msvc-19.51.36256 Release
config=Release
cpu.model=Intel(R) Core(TM) i7-9850H CPU @ 2.60GHz
cpu.model.source=cpuid
cpu.logical=12
os=windows
arch=x86-64
pointer.bits=64
endian=little
compiler.id=msvc
compiler.version=19.51.36256
compiler.standard=202400
sanitizers=none
macro.__SSE2__=0
macro.__AVX__=1
macro.__AVX2__=1
macro.__FMA__=0
macro.__F16C__=0
macro.__AVX512F__=0
macro.__ARM_NEON=0
macro.__ARM_FEATURE_FMA=0
macro.__ARM_FEATURE_FP16_SCALAR_ARITHMETIC=0
macro.__ARM_FEATURE_FP16_VECTOR_ARITHMETIC=0
simd.target=avx2
simd.bits=256
simd.lanes.fp64=4
simd.lanes.fp32=8
simd.lanes.fp16=16
backend.scalar-fp64.contracts=0
backend.scalar-fp32.contracts=0
backend.avx2-fp64.contracts=0
backend.avx2-fp32.contracts=0
contract.this-tu.fp64=0
contract.this-tu.fp32=0
runtime.avx2=1
fma.route=no-call
fma.route.symbols=none
fma.route.artifact=boys.lib
fma.route.tool=link.exe
recorded=2026-09-24
```

#### `local-windows-x86-64-sse2 msvc-19.51.36256 Debug`

```text
boys.build-facts/1
leg=local-windows-x86-64-sse2 msvc-19.51.36256 Debug
config=Debug
cpu.model=Intel(R) Core(TM) i7-9850H CPU @ 2.60GHz
cpu.model.source=cpuid
cpu.logical=12
os=windows
arch=x86-64
pointer.bits=64
endian=little
compiler.id=msvc
compiler.version=19.51.36256
compiler.standard=202400
sanitizers=none
macro.__SSE2__=0
macro.__AVX__=0
macro.__AVX2__=0
macro.__FMA__=0
macro.__F16C__=0
macro.__AVX512F__=0
macro.__ARM_NEON=0
macro.__ARM_FEATURE_FMA=0
macro.__ARM_FEATURE_FP16_SCALAR_ARITHMETIC=0
macro.__ARM_FEATURE_FP16_VECTOR_ARITHMETIC=0
simd.target=sse2
simd.bits=128
simd.lanes.fp64=2
simd.lanes.fp32=4
simd.lanes.fp16=8
backend.scalar-fp64.contracts=0
backend.scalar-fp32.contracts=0
backend.avx2-fp64.contracts=0
backend.avx2-fp32.contracts=0
contract.this-tu.fp64=0
contract.this-tu.fp32=0
runtime.avx2=1
fma.route=out-of-line-call
fma.route.symbols=fma,fmaf
fma.route.artifact=boys.lib
fma.route.tool=link.exe
recorded=2026-09-24
```

#### `local-windows-x86-64-sse2 msvc-19.51.36256 Release`

```text
boys.build-facts/1
leg=local-windows-x86-64-sse2 msvc-19.51.36256 Release
config=Release
cpu.model=Intel(R) Core(TM) i7-9850H CPU @ 2.60GHz
cpu.model.source=cpuid
cpu.logical=12
os=windows
arch=x86-64
pointer.bits=64
endian=little
compiler.id=msvc
compiler.version=19.51.36256
compiler.standard=202400
sanitizers=none
macro.__SSE2__=0
macro.__AVX__=0
macro.__AVX2__=0
macro.__FMA__=0
macro.__F16C__=0
macro.__AVX512F__=0
macro.__ARM_NEON=0
macro.__ARM_FEATURE_FMA=0
macro.__ARM_FEATURE_FP16_SCALAR_ARITHMETIC=0
macro.__ARM_FEATURE_FP16_VECTOR_ARITHMETIC=0
simd.target=sse2
simd.bits=128
simd.lanes.fp64=2
simd.lanes.fp32=4
simd.lanes.fp16=8
backend.scalar-fp64.contracts=0
backend.scalar-fp32.contracts=0
backend.avx2-fp64.contracts=0
backend.avx2-fp32.contracts=0
contract.this-tu.fp64=0
contract.this-tu.fp32=0
runtime.avx2=1
fma.route=out-of-line-call
fma.route.symbols=fma,fmaf
fma.route.artifact=boys.lib
fma.route.tool=link.exe
recorded=2026-09-24
```

### CI legs with no recorded row yet

Each of these legs prints its own row on its next run, under the step that runs the probe. Record it with

    python tools/gen_build_facts.py --record <the captured block>

| CI leg | Runner label |
|---|---|
| `option-matrix (the configuration list)` | `ubuntu-latest` |
| `option-matrix linux-gcc fused` | `ubuntu-latest` |
| `option-matrix linux-gcc separate` | `ubuntu-latest` |
| `option-matrix linux-clang fused` | `ubuntu-latest` |
| `option-matrix linux-clang separate` | `ubuntu-latest` |
| `option-matrix windows-msvc fused` | `windows-latest` |
| `option-matrix windows-msvc separate` | `windows-latest` |
| `windows-msvc Release` | `windows-latest` |
| `windows-msvc Debug` | `windows-latest` |
| `linux-x86 gcc Release` | `ubuntu-latest` |
| `linux-x86 gcc Debug` | `ubuntu-latest` |
| `linux-x86 clang Release` | `ubuntu-latest` |
| `linux-x86 clang Debug` | `ubuntu-latest` |
| `linux-sanitizers asan+ubsan` | `ubuntu-latest` |
| `linux-arm64 gcc Release` | `ubuntu-24.04-arm` |
| `windows-arm64 msvc Release` | `windows-11-arm` |
| `macos arm64` | `macos-latest` |
| `macos x64` | `macos-15-intel` |
| `linux-x86 clang-tidy` | `ubuntu-latest` |
<!-- build-facts:end -->
