# Calling this library on a GPU

This page is the device half of `docs/getting-started.md`. That page takes a reader who has never
called this library and walks them through the CPU entries, one question at a time. This one does
the same for the CUDA lane: which of the two routes answers your problem, what a call looks like,
what you must allocate and pass, what the library refuses and why that is a feature, and how to
read what you got.

It is written for a reader who knows CUDA and does not know this library. Nothing here assumes you
have read another page.

## Before anything: is there a CUDA lane in your build?

The lane is off unless you ask for it. From the top-level `CMakeLists.txt`:

```cmake
option(BUILD_CUDA
    "Build the CUDA lane (needs the CUDA toolkit; CI compiles it and runs no kernel, so the device figures stay a local gate)"
    OFF)
```

So `cmake -S . -B build-cuda -DBUILD_CUDA=ON` and a CUDA toolkit on the machine. The last clause of
that description is worth reading before you plan any measurement: continuous integration compiles
and links the lane and **runs no kernel**, so every device figure is a figure somebody took on a
machine with a card in it. That is a statement about where the numbers come from, and it is taken up
again under *What you got: the bound*, below.

## The two routes, and which one is yours

The library offers the same arithmetic two ways, and they are different questions rather than two
spellings of one.

| | a launched kernel | a `__device__` function |
|---|---|---|
| header | `boys/boys_cuda.hpp` | `boys/boys_cuda_device.hpp` |
| what runs | the library's own kernel, launched for you | the library's arithmetic, inlined into **your** kernel |
| what goes in | device arrays: orders, arguments, output | one `(order, x)` pair the calling thread already holds |
| what comes back | a `BoysStatus` for the launch | a `BoysDeviceStatus` per call, written into your own output |
| the question it answers | "I have a batch of arguments in global memory" | "I formed `x` in this thread and need the ladder here and now" |
| what it costs | a kernel launch and the memory traffic to get the batch there | instructions in your kernel, plus the tables' compile |

The choice is not a preference. It is the shape of your program. If your arguments already exist as
a device array — a batch of shell quartets built by another kernel — the launched route is what you
want, and it is one call. If your kernel forms `x` per thread and needs `F_0..F_n` for that thread's
own `x`, the launched route would make you write every `x` to global memory, launch a second kernel,
synchronise and read the answers back. The device header states the cost of that round trip in its
own words:

> The batch entries of this lane evaluate an array of arguments instead, so a caller has to
> materialise every x to global memory, launch a second kernel, synchronise and read the results
> back — two extra memory passes per integral batch. The entries here take one (order, x) per call
> and return the values to the calling thread, so the fused kernel needs neither the pass out nor
> the pass back.

Both routes read the same tables and answer at the same bounds. A caller that writes its own kernel
"gets the entry's own arithmetic, not an approximation of it" (`boys_cuda.hpp`). So this is a
question about your program's shape, never about accuracy.

## What you own, and what the library owns

Start here, because it is the sentence that surprises people. From the file comment of
`boys_cuda.hpp`:

> All entry points take raw device pointers — the caller owns the device memory
> (cudaMalloc/cudaMemcpy/cudaFree) and the stream.

The library allocates nothing on your behalf and frees nothing. Every device pointer you hand an
entry is yours: you `cudaMalloc` it, you fill it, you `cudaFree` it. The stream is yours too, and it
is an opaque `void*` in this header — the header is deliberately kept free of CUDA runtime headers
so that a plain C++ translation unit can include it, and it is `cudaStream_t` inside the `.cu` file.
Pass `nullptr` for the default stream.

What the library does own is its coefficient tables. They upload on first use, once per device,
lazily and idempotently, and they live as long as the process does. You never allocate them and you
never free them.

## Route 1: the library launches the kernel

Three shapes per precision, and the shape is the question you are asking. Each is declared on
`boys::BoysCuda`. Here is the double family, quoted from `boys/boys_cuda.hpp`:

```cpp
static BoysStatus SingleF64(
    const int* n,
    const double* x,
    double* out,
    std::size_t count,
    void* stream,
    DivisionForm form = kDefaultDeviceDivisionForm);
```

```cpp
static BoysStatus AllOrdersF64(
    const int* n,
    const double* x,
    double* out,
    std::size_t count,
    void* stream,
    DivisionForm form = kDefaultDeviceDivisionForm);
```

```cpp
static BoysStatus AllNF64(
    int nmax,
    const double* x,
    double* out,
    std::size_t count,
    void* stream,
    DivisionForm form = kDefaultDeviceDivisionForm);
```

The float family is the same three with `F32` in place of `F64` and `float` in place of `double` for
`out`. The half family stores through `F16`, which is the library's own 16-bit type and the element
type of both the argument array and the output:

```cpp
static BoysStatus SingleF16(
    const int* n,
    const F16* x,
    F16* out,
    std::size_t count,
    void* stream,
    DivisionForm form = kDefaultDeviceDivisionForm);
```

**Read `Single*` against the other two in the parameter list, not in the name.** `Single*` takes a
device array of orders, one per element. `AllOrders*` takes a device array of orders *and* answers
every order up to each element's own top order. `AllN*` takes **one** top order as a plain `int`
— the whole batch shares it — and takes no order array at all.

**The output layout differs between the two batch shapes, and getting it wrong is silent.** For
`AllOrders*`, from the header:

> Output layout: out[order * count + i] = F_order(x[i]), order 0..nmax[i]: plane l holds F_l(x[i])
> for the arguments whose order reaches l.

So the planes are order-major, and `out` must hold `count * (kMaxBoysOrder + 1)` values — the
header's own `\param out` says exactly that. `AllN*` uses the same order-major layout, and the
header says why that matters:

> Output layout: out[k * count + i] = F_k(x[i]), k = 0..nmax — the order-major planes the CPU lane's
> BoysAllN returns, so a batch moved between the CPU and the device lanes is not transposed.

**`AllN*` is the one entry with a precondition on the order of your arguments, and the header
states it together with what to do instead.** From `AllNF32`'s own documentation:

> \pre x[i - 1] <= x[i] for every i in [1, count): the arguments are non-decreasing. An unsorted
> batch still returns correct values — the kernel classifies each argument itself — but the ordering
> lands that classification on one path per warp. This entry does not sort its arguments (an
> internal sort is a launch, a permutation and device scratch, and the caller that builds x is the
> one that knows its order): a batch in arbitrary order is AllOrdersF32 with the order array set to
> nmax.

That last clause is the answer if your arguments arrive unsorted: use `AllOrders*` and fill the order
array with your one top order. Note the shape of the precondition — it is about speed, not about
correctness. An unsorted batch to `AllN*` does not return a wrong answer.

A call is accepted when the launch is accepted, not when the kernel has finished. The tree's own
consumer — `tests/boys_cuda_accuracy_gate.cpp` — synchronises after every call before it reads the
output, and that is what any caller that wants its values does too.

### The return value: `BoysStatus`

```cpp
enum class BoysStatus {
    kSuccess = 0, ///< the call succeeded
    kInvalidArgument, ///< a parameter was invalid (see the entry's contract)
    kDeviceError, ///< a CUDA operation failed
};
```

The CUDA lane is the one fallible surface of this library, and the header is explicit that it
reports here rather than through exceptions: "table uploads, parameter validation, and launches
report through this status (never exceptions)". `kSuccess` is 0, and when a `kDeviceError` comes
back "the caller can use the CUDA runtime's own error reporting (cudaGetLastError, stream capture)
for the detail" — the enum carries no payload. `kInvalidArgument` is the entry's own validation of
what you handed it.

## Route 2: the entry inside your own kernel

This route has one extra step, and it is the whole of the interface you have to learn.

### The handle

```cpp
struct BoysDeviceTables {
    const int* pieceStart = nullptr;
    const double* pieceA = nullptr;
    /* ... */
};
```

That is `boys::BoysDeviceTables`, from `boys/boys_device_tables.hpp`, abbreviated — it is a large
plain-data struct of device pointers and degrees, and the entries' own to read. The header's own
description of it is what you need:

> Plain data — device pointers and degrees — one handle for every precision. Fill it with
> `BoysCuda::DeviceTables`, then pass it by value into a kernel and hand it to an entry; a kernel
> parameter lives in the constant bank, so the handle itself costs no global memory traffic. The
> tables it points at are the library's: they live as long as the process does and are read-only.
> The handle owns nothing and the caller copies it.

You fill it once, on the host:

```cpp
static BoysStatus DeviceTables(BoysDeviceTables* out);
```

> \returns kSuccess after filling \c out with the current device's table addresses; kDeviceError
> when the upload or an address query fails.

Two properties of the handle are easy to miss and both are stated by the header.

**It is per device.** "A handle names the device that was current when it was filled, as every other
entry of the CUDA lane names the current device: a caller that runs on more than one device fills one
handle per device."

**It is passed by value, and you should say so in the signature.** The device header gives the
reason and the spelling:

> Pass it by value into the kernel — it is plain data, and kernel parameters live in the constant
> bank — and declare that parameter \c __grid_constant__ so that taking its address inside the
> kernel does not copy it to local memory per thread

```cpp
__global__ void Fused(__grid_constant__ const boys::BoysDeviceTables tables,
                      const double* rho, const double* d2, ...)
```

"A caller on a toolchain without `__grid_constant__` passes the same value without the qualifier and
reads the handle through a generic pointer, which is correct and slower; the arithmetic is identical
either way."

### The entries, and what they cost your build

Four shapes per precision. Here is the double ladder and the single, quoted from
`boys/boys_cuda_device.hpp`:

```cpp
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllOrdersF64(const BoysDeviceTables& tables,
                                                   int order,
                                                   double x,
                                                   double* out,
                                                   int capacity);
```

```cpp
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceSingleF64(
    const BoysDeviceTables& tables,
    int order,
    double x,
    double* out);
```

The all-N shape takes its top order as a **template argument** rather than as a parameter, which is
what lets the compiler bound and unroll the recursion loops:

```cpp
template <DivisionForm kForm = kDefaultDeviceDivisionForm, int kTopOrder, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllNF64(const BoysDeviceTables& tables,
                                              double x,
                                              double* out);
```

And the fourth shape is for callers who do not want a ladder alive at all — it hands one value at a
time to a sink you supply:

```cpp
template <DivisionForm kForm = kDefaultDeviceDivisionForm, typename Sink, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceEachOrderF64(const BoysDeviceTables& tables,
                                                   int order,
                                                   double x,
                                                   Sink sink);
```

The header states why that fourth shape exists, and it is a register-pressure answer rather than a
style one:

> A caller keeping the whole ladder live holds order + 1 doubles — 33 of them at the top order, which
> is the real cost of this interface and the reason the entries write into the caller's array rather
> than into an internal one: the caller's own liveness decides what stays in registers, and a caller
> that consumes each order as it arrives can hold none.

The `capacity` parameter on the ladder shape is the same idea from the other side: it is the number
of values your array actually holds, so an order your array cannot receive is reported rather than
written past the end.

**What including this header costs your build** is the header's own promise, and it is unusually
cheap:

> The entries are header-defined device functions with no device-side symbol to link: including this
> header is the whole of what the caller's build pays. No relocatable device code (\c -rdc=true), no
> device link step, no library on the link line, no additional compilation flag — a kernel that calls
> an entry compiles with the same nvcc command line as one that does not — and the arithmetic is
> inlined into the calling kernel, so there is no call overhead per order.

### The return value: `BoysDeviceStatus`

```cpp
enum class BoysDeviceStatus : int {
    kSuccess = 0, ///< the values were written
    kTablesNotReady,
    kOrderOutOfRange,
    kCapacityTooSmall,
};
```

A different enum from the launched route's, because the failure modes are different: there is no
launch to fail, so what can go wrong is that the handle has no tables, that the order is outside
`0..kMaxBoysOrder`, or that your array is too small. The header states the ordering and the
guarantee, and both matter to a caller writing a branch:

> The checks are ordered tables, then order, then capacity: a request that is malformed is reported
> as malformed, so a caller branching on these statuses always has one thing to fix.

> Every failure is a BoysDeviceStatus the caller branches on, and a call that fails writes nothing:
> no truncated ladder, no partial fill.

## Naming a combination instead of an entry

The entries above are a catalogue, and for a fused kernel you do not want a catalogue — you want to
say *which* evaluation and get it. That is what the policy layer is for, and it is the same
mechanism the host entries use.

Every class of the device surface carries a policy-templated member beside the entries it has, named
after the class with `WithPolicy` after it. The spelling, from `boys/boys_cuda_policy.hpp`:

```cpp
BoysCuda::AllOrdersF64WithPolicy<>             (n, x, out, count, stream)   - the seam's row
BoysCuda::AllOrdersF64WithPolicy<Policy>(...)  - the kernel the policy names
BoysCuda::AllOrdersF64(...)                    - the shipped entry, unchanged
```

And here is that member as it is declared, for the double ladder:

```cpp
template <EvalPolicyLike Policy =
              DefaultPolicy<Precision::kFp64Device, Shape::kAllOrders, Device::kDevice>>
static BoysStatus AllOrdersF64WithPolicy(
    const int* n, const double* x, double* out, std::size_t count, void* stream);
```

The suffix is load-bearing and not decoration, and the header says why: a second function of an
entry's own name would make that name an overload set, and the address of an overload set cannot be
taken where the pointer type is deduced — a use the shipped tests make. With the suffix every entry
stays the one function of its name.

### The twelve classes

A class is a precision crossed with the shape of your question. The four precisions of this
surface are `Precision::kFp64Device`, `Precision::kFp32Device`, `Precision::kFp16Device` and
`Precision::kBf16Device`, and the
three shapes are `Shape::kSingle`, `Shape::kAllOrders` and `Shape::kAllN`. Every cell of that cross
has a member of the policy layer, and a policy for one is written
`DefaultPolicy<Precision, Shape, Device::kDevice>` — exactly as the declaration above spells it.

The half lane's two stores are two of those precisions and not one: the class is keyed by the format
your return is stored in, so `kBf16Device` is the class a bfloat16 call resolves and `kFp16Device`
the one an fp16 call does. They are one arithmetic at one budget and two classes, and the two carry
different half digits — 2^-8 against 2^-11 — which is why the library keys them apart rather than
giving one of them the other's figure.

**Every one of the twelve has a row in the committed default seam**, in
`include/boys/boys_build_defaults.hpp`, so `BoysCuda::AllOrdersF64WithPolicy<>` with a policy
argument left out reaches the combination this build wrote for that class. Read that file's own
words about where those twelve rows came from before you treat one as tuned:

> a choice, not a measurement: no device run stands, so each of these twelve rows states the fallback
> names above at its own lane's budget

**That is worth pausing on.** The host's rows in that same file carry a measured figure beside them
— `measured: ... ns per argument on this host`. The twelve device rows do not, and the file states the
reason rather than leaving you to guess it: no device run stands. The reason is one this library
applies everywhere: a device figure is a card's, and a gate built with CUDA on that card is where it
is taken. So the device default rows are the library's fallback names at the class's own budget, and
the file says so rather than dressing them up. The two axes a device policy can move have their own
names in that file too: `BOYS_BUILD_DEFAULT_DEVICE_DIVISION_FORM` and
`BOYS_BUILD_DEFAULT_DEVICE_REGION_B_EXP`, read once as `boys::kDefaultDeviceDivisionForm` and
`boys::kDefaultDeviceRegionBExp`.

## What the library refuses, and why that is a feature

There are two kinds of refusal, and they happen at two different times.

### At compile time: a combination with no kernel

This is the one to understand before you write a call, because it is the library's central promise
about this surface rather than an inconvenience. From `boys/boys_cuda_policy.hpp`:

> A combination the surface **has no kernel for is a compile error**, and the message names the
> combination and states that no entry of the class answers it. Nothing here falls back to a nearby
> kernel: a policy naming a reading the lane does not carry is never answered out of another
> reading's tables, which is the substitution the whole seam machinery of this library exists to
> prevent.

Read that as a guarantee about your results. In a numerical library the dangerous failure is not a
call that does not compile — it is a call that compiles and quietly runs something other than what
you asked for. You would get a number back, it would look plausible, and nothing would tell you it
came from different tables. Refusing at compile time makes that outcome impossible: you cannot ship a
binary in which a policy naming one reading was answered out of another's.

**In practice it means a missing combination is loud.** The build stops, and the diagnostic names the
combination you asked for and says no entry of the class answers it. That is a statement about what
this revision carries, not about what is possible.

### At run time: a malformed request

The launched entries validate their parameters and report `kInvalidArgument`. The device-callable
entries report the three statuses above, and a refused call writes nothing.

## What you got: the bound

The library's accuracy contract is a **bound**, not an observation: for every supported `n`, `x`,
lane and region the returned value satisfies `|F̂_n(x) − F_n(x)| ≤ B` at a published `B`.

**The bound is per lane, and a lane is not a precision.** On this surface the four device classes
`Precision::kFp64Device`, `kFp32Device`, `kFp16Device` and `kBf16Device` are four lanes with four rows
and not one row in four spellings — the device's half lane is not the host's, its two stores are two
lanes of their own with a figure each, and its double lane's figure
is its own statement rather than the host's row read across. `boys::BoysLaneContracts()` publishes
one row per lane, and the entry that answers "what does this combination guarantee" is
`boys::BoysAccuracyGuaranteed`, which takes the whole combination and returns an `AccuracyFigure`:

```cpp
AccuracyFigure BoysAccuracyGuaranteed(Precision precision,
                                      FitRoute route,
                                      EvalScheme scheme,
                                      PackAxis axis,
                                      FitGranularity granularity,
                                      DivisionForm form = kDefaultDivisionForm,
                                      RegionBExp exp = kDefaultHostRegionBExp) noexcept;
```

Its `reading` field says whether you are looking at the guaranteed figure or the delivered one, so
neither can be read as the other.

**Where the figures are written down.** [The per-lane contract](lane-contract.md) carries the
per-lane bound tables, the four device rows among them, and
[the API reference's contract table](mainpage.md) states the CUDA rows. Both are documents. The
authority behind them is a program, and this is the part to take away:

**The device half carries no certified per-class accuracy figure of its own at this revision.**
There is no valid device run standing behind one, which is the same fact the seam file states about
its twelve rows and the header states about the lane's coverage. So this page quotes you no device
accuracy number: it would be a number no program printed, which is exactly the defect this library's
documentation exists to avoid. What it tells you instead is where the number lives, and how to get
the one that is true for your card.

### Getting a figure that is true for your card

That is what the device probe is for, and the reasoning is the same one the CPU half of this library
gives for the host — see the option-probe section of `docs/getting-started.md`. Which entry is
cheapest is a property of the card and not of the library.

    cmake -S . -B build-cuda -DBUILD_CUDA=ON
    cmake --build build-cuda --target boys-device-probe
    ./build-cuda/Release/boys-device-probe     # the config directory is your generator's

`boys::RunDeviceOptionProbe` takes a `boys::DeviceProbeOptions` — whose `device` field is the
ordinal "as the CUDA runtime numbers it", and of which the probe "establishes this device's context
before it allocates anything" — and returns a `boys::DeviceProbeReport` carrying the card's name and
compute capability, one figure per row, and the entries it could not separate. **A row's cost and its
documented bound are reported together** — so the probe is where you read both the ranking and the
bound, from a program, on your hardware.

The option space the probe ranks is the library's own and is not a list written beside it:
`boys::BoysDeviceOptions()` in `boys/boys_cuda_options.hpp` reports one row per option of the
surface, and the probe takes its rows from there. **That is also the answer to "which entries exist
in my build"** — read the report rather than a document, because a document cannot know.

Two cautions the library states in its own voice, and both are easy to get wrong:

**Warm the path before you time anything.** The tables upload on first use, so a loop timed from a
cold start measures the upload rather than the evaluation. Make one throwaway call of the entry you
intend to use — or call `BoysCuda::InitializeTables` — and time everything after it.

**A device figure is a card's.** The probe builds that caveat into its own output from what the
runtime reports: `This ranking is about this card: %s, compute capability %d.%d.` — and where your
part documents a ratio of single- to double-precision throughput, it says in the same breath that
how far the fp64 lane sits behind the fp32 lane here is that ratio "and not the library's". A
ranking taken on another part is not evidence about yours.

### Checking the lane's own claims

`boys-cuda-accuracy-gate` is where the device lane's figures come from, and its own file comment
gives the command:

    cmake --build <build> --config Release --target boys-cuda-accuracy-gate
    <build>/Release/boys-cuda-accuracy-gate [--reference <grid.csv>]
                                            [--digit-reference <grid.csv>]

It is the device counterpart of `boys-accuracy-gate`, the program behind the host figures in
[the API reference's contract table](mainpage.md). Three properties of it are worth knowing because
each one is a thing that could otherwise go quietly wrong:

- **It says so when it cannot measure.** With no CUDA device present it prints
  `RESULT: evidence absent - no CUDA device on this machine` and exits with status 2, rather than
  reporting a clean run over nothing.
- **It takes the option space from the library.** Its coverage and its claims are walked from
  `boys::BoysDeviceOptions()`, so a row added to the surface is a row the gate owes an answer for.
- **It fails loudly.** Non-zero when any measured cell is over its bound, and its rows are compared
  against the reference grid it is pointed at rather than against a number written beside them.

## The route the multiply-add takes, and how to read it

Whether a bare `a * b + c` in this build is one rounding or two is a property of your build, and the
device lane honours the same selection the host does: `BOYS_MULADD_SEPARATE`. A caller reads which
route the device arithmetic runs off a single accessor:

```cpp
static constexpr backend::MulAddRoute MulAddRouteInForce() noexcept {
    return detail::kDeviceMulAddRoute;
}
```

It is the device's counterpart of the host's `RouteInForce<T>()` and of the `route` field of
`BoysBackends()` — one computation serves both precisions of the lane, so one reader answers for
both. It is a report and not a restatement of the selection: the device spells both routes out
rather than leaving the choice to the device compiler's contraction setting, so the fused step is the
fused intrinsic and the separate step is "a product rounded once and then summed, a form no setting
has anything left to fuse". That the delivered arithmetic is the named one is measured, not asserted
— `boys-cuda-route-tests` holds each route to its own reference, bit for bit, over a fixed value set:

    cmake --build <build> --target boys-cuda-route-tests   # needs -DBUILD_CUDA=ON
    <build>/Release/boys-cuda-route-tests

## What this page does not state, and why

Two things are deliberately absent, and naming them is part of using this page correctly.

**It does not give you a complete list of entry names.** The device surface is large: it carries one
entry per combination of a route, a scheme, a partition and a packing axis for the shapes those are
coordinates of, so names like `BoysCuda::AllOrdersF64NarrowOrdersRat` exist beside the catalogue
above. Those names are a mechanical spelling of the axes and not a second design, and the axes
themselves — `FitRoute`, `EvalScheme`, `FitGranularity`, `PackAxis` — are defined on
[the API reference's landing page](mainpage.md).

The policy header states plainly why a page should not be the list, and it is the direction this
surface is moving in: the combinations used to be reachable "only by knowing its name, and a class
has no default at all", which is what the policy layer was added to change. **So the stable way to
reach a combination is the policy layer**, and the authoritative list of what your build carries is
`BoysDeviceOptions()` at run time — a document cannot be that list, because a document goes stale
the moment an entry is added. This page names only the entries it has read out of the headers.

**It quotes no device accuracy figure.** See *What you got: the bound* above: no valid device run
stands behind one at this revision, so a figure printed here would be one no program printed. Where
the figures live is stated, and the commands that produce them on your card are given.

## Where to go next

- `docs/getting-started.md` in this tree — the CPU half of this page, task by task.
- [The per-lane contract](lane-contract.md) — the bound each lane guarantees, over which interval,
  and where it stops; the default policy per precision and per device.
- [The API reference](mainpage.md) — every entry, every type, and the vocabulary this library uses.
- [Build facts](build-facts.md) — what a build of this library is, read out of the build.
