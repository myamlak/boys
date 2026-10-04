# Getting started

This library evaluates the Boys function,

    F_n(x) = integral from 0 to 1 of t^(2n) exp(-x t^2) dt,   n = 0..32,  x >= 0

the integral an electronic-structure program needs at every order a shell quartet can ask for. This
page is the task-shaped route in. Each section is a question you would actually ask, the code that
answers it, and what that code prints. Every transcript below is what one of those programs printed,
and the command that reproduces it is beside it. Nothing on this page is a figure you have to take on
trust.

The precise guarantees, the evaluation settings, and the measurements behind them are in
[docs/lane-contract.md](lane-contract.md) and the [API reference](mainpage.md). This page links to
them where a guarantee is involved. It does not restate them.

## Build it, and run the programs below

    cmake -S . -B build
    cmake --build build
    ctest --test-dir build --output-on-failure     # runs every example

Each example is also a program you can compile on its own against an already-built tree:

    c++ -std=c++20 -I include examples/00_first_call.cpp -L build -lboys -o first && ./first

The headers need C++20, and the compile line above is all they ask for. The build in the
block before it is the contributor's: it compiles the test suite as well, which wants C++23. On
Windows with the Visual Studio generator the library lands in `build/Release/`. The CMake route above
is the one that finds it without flags.

---

## What does it look like to call this?

`examples/00_first_call.cpp` — four calls cover most of what a program does with this function.

```cpp
double ladder[boys::kMaxBoysOrder + 1] = {};
boys::BoysAllOrders(6, 3.5, ladder);          // F_0..F_6 at x = 3.5

const double x[3] = {0.25, 4.0, 30.0};
double out[3] = {};
boys::BoysFixedN(2, x, out, 3);               // out[i] = F_2(x[i])

const double batch[4] = {0.0, 0.25, 4.0, 30.0};
double grid[4 * 7] = {};
boys::BoysAllN(6, batch, grid, 4);            // grid[k * 4 + i] = F_k(x[i])

boys::BoysSingle(3, 1.25);                    // F_3(1.25)
```

    F_3(1.25)  = 0.055476132923077535
    F_0(3.5)   = 0.46984703520162352
    F_3(3.5)   = 0.011831383583716678
    F_6(3.5)   = 0.0040954447623731674
    F_2(0.25)  = 0.1675331909073505
    F_2(4)     = 0.017525782161993068
    F_2(30)    = 0.00013483513281636802
    F_0(30)    = 0.1618021593796416
    F_6(30)    = 3.6049670926598394e-08

All four are `noexcept` and total. The requirements on the values are `n` in `[0, 32]` and `x >= 0`,
and both hold for anything a basis set produces. The three calls that write into an array ask one
thing more: the buffer has to hold the values the call writes. Nothing checks it.

**Next:** if you want one of these three shapes specifically, read on. If you want the whole
catalogue at once, jump to [Which entry do I call?](#which-entry-do-i-call).

---

## I want one value: F_n(x) at a single argument

`examples/01_one_order.cpp` — one call, and the error the library guarantees for it.

```cpp
const double f = boys::BoysSingle(n, x);
```

    F_3(1.25) = 0.055476132923077535
    guaranteed error <= 5.5e-14  (from throughout, every region)
    BoysAllOrders agrees to 0

The second line is the whole accuracy story in one figure: **the answer is within 5.5e-14 of the true
F_3(1.25)**. It is not an estimate of the error somebody observed. It is the bound the library
guarantees for this call, read out of the library at run time rather than typed into a document. The
phrase in parentheses after it says how far that figure reaches. The argument line is cut into
regions, and this is the lane's bound over every one of them. If your calculation needs an error below
that, see [I need to know what the library guarantees before I rely on
it](#i-need-to-know-what-the-library-guarantees-before-i-rely-on-it).

**Next:** `BoysSingle` computes each order on its own, so reach for the ladder below if you need more
than one. It is the same call an integral code makes.

---

## I want every order at one argument

`examples/02_all_orders.cpp` — a shell quartet needs F_0 through F_nmax at the same argument. One call
hands over the whole ladder, instead of one call per order.

```cpp
double ladder[boys::kMaxBoysOrder + 1] = {};
boys::BoysAllOrders(nmax, x, ladder);         // ladder[k] = F_k(x)
```

    k   F_6(3.5)
    0   0.46984703520162352
    1   0.062807093111329287
    2   0.02260341370166705
    3   0.011831383583716678
    4   0.0075174716662426075
    5   0.0053514087962664245
    6   0.0040954447623731674
    worst gap against BoysSingle: 0 (guaranteed error <= 5.5e-14)

The entry writes `nmax + 1` doubles. The example declares the full `kMaxBoysOrder + 1`, so the same
buffer serves any later call without a second array.

**Next:** if your arguments are an array and the order is one number, the call below turns the loop
inside out. That is the shape that gets help from the vector units.

---

## I want one order over many arguments

`examples/03_arguments_array.cpp` — the same order at every argument, written where you want it.

```cpp
boys::BoysFixedN(n, x, column, count, stride);   // column[i * stride] = F_n(x[i])
```

    n = 2, stride 2
    x = 0.25    out[ 0] = 0.1675331909073505
    x = 1       out[ 2] = 0.10026879814501725
    x = 4       out[ 4] = 0.017525782161993068
    x = 12.5    out[ 6] = 0.0012030139279729953
    x = 30      out[ 8] = 0.00013483513281636802
    worst gap against BoysSingle: 0 (guaranteed error <= 5.5e-14)

The stride is why the entry exists: the answer can go straight into a column of a larger structure,
with no intermediate copy. Leave it out and it defaults to 1.

**Next:** every order over many arguments at once — the batch shape — which is the next section.

---

## I want every order over many arguments

`examples/08_shell_quartet_batch.cpp` — the batch shape an integral engine calls: a whole shell
quartet's arguments at once, every order at each of them.

```cpp
boys::BoysAllN(nmax, x, out, count);            // out[k * count + i] = F_k(x[i])
boys::BoysAllNAtOrders(n, x, out, count);       // each argument at its own top order
```

    five arguments, each at its own top order
      F_3(0.5) = 0.097222024416930064
      F_6(2) = 0.014008835839082863
      F_2(7.5) = 0.0042700136194305542
      F_8(0.125) = 0.052602850434881922
      F_4(20) = 8.1278555493588482e-06
    values the per-argument call produces: 28
    values the padded call produces:        45
    produced for nothing:                   17
    cells compared against the padded call:  28, worst difference 0
    guaranteed error per value:             5.5e-14  (throughout, every region)
    every value positive and at most one

This is the entry to reach for when the arguments are the batch and the orders are the ladder: it
groups the arguments once for the whole call rather than a ladder at a time, which the header records
as considerably cheaper than `nmax + 1` single evaluations. The planes come out order-major, so
nothing has to be transposed afterwards, and the arguments may arrive in any order.

When the arguments do not share a top order — a quartet whose four shells differ — the padded call
pays for cells nobody asked for, 17 of the 45 in the run above. `BoysAllNAtOrders` writes only the
values you asked for, and it is the column the padded call is checked against.

**Next:** if you want a specific evaluation, the next section is how you name it.

---

## I want to name a specific evaluation

**First, a warning, because this one is a trap.** One of the settings below is called `kNarrow`. That
name reads as "narrower, therefore more careful". It is not. `kNarrow` and `kCoarsest` cut the fitted
interval into pieces of different widths and carry a different number of pieces; **both meet the same
published error bound.** Choosing the wrong one changes the work, not the accuracy. It costs you
nothing visible: both compile, both run, both are correct, so you would never find out you had picked
backwards. `kNarrow` is what a call site that names no partition reads. `kCoarsest` is what the program
below names. If you have not measured a preference, name no policy at all and take the default. The
bound the library publishes is the lane's, and no partition moves it.

`examples/05_policy.cpp` — a policy is named in the template argument list, never constructed. The
values below are the program's own ladder, at the one argument and the one top order it sets for
itself at the top of the file.

```cpp
using Shipped = boys::EvalPolicy<boys::FitRoute::kChebyshev, boys::EvalScheme::kSplitClenshaw,
                                 boys::BoysBudget::kFloat, boys::PackAxis::kArguments,
                                 boys::FitGranularity::kCoarsest,
                                 boys::DivisionForm::kExactDivision>;

boys::BoysAllOrders<Shipped>(nmax, x, named_ladder);
```

    k   default policy            named policy
    0   0.54629197178514799      0.54629197178514799
    1   0.092841394632249843     0.092841394632249843
    2   0.039287837054570153     0.039287837054570153
    3   0.022870837329790391     0.022870837329790391
    4   0.015602172536926787     0.015602172536926787
    worst disagreement: 0 (guaranteed error <= 5.5e-14)

The two agree here to the last bit, and that is expected rather than a coincidence: the settings
select how the same approximation is summed, not what is approximated. The places to change it are
the ones where they do **not** agree. Finding those on your machine is what
[the option probe](#i-want-to-know-which-option-is-fastest-on-this-machine) is for.

The six settings the call above writes, in the order they are written: which stored fit serves the
interval, how its coefficients are summed, an internal precision budget, whether vector lanes hold
four arguments or four orders, how finely the fitted interval is cut, and how the recursion's
divisions are performed. The policy carries a seventh — which exponential seeds a region-B ladder —
which the call above leaves at its default. Each is documented beside its type in
[the API reference](mainpage.md), and all seven are listed under [What you can and cannot choose at
run time](#what-you-can-and-cannot-choose-at-run-time).

**Next:** you have named a specific evaluation. Before you rely on it, ask what it guarantees.

---

## I need to know what the library guarantees before I rely on it

`examples/06_what_it_guarantees.cpp` — ask with your tolerance and get a verdict, rather than
comparing a table by eye.

```cpp
const boys::CombinationCoverage answer = boys::QueryCombination(
    boys::Precision::kFp64, boys::FitRoute::kChebyshev, boys::EvalScheme::kHorner,
    boys::PackAxis::kArguments, boys::FitGranularity::kNarrow,
    tolerance);
```

    the double entry, every order, as it stands:
      requested    verdict                  guaranteed   measured
      1e-12        YES - guaranteed       5.5e-14      4.1e-14  [throughout, every region]
      1e-10        YES - guaranteed       5.5e-14      4.1e-14  [throughout, every region]
      1e-08        YES - guaranteed       5.5e-14      4.1e-14  [throughout, every region]
    1e-20 is answered NO

The distinction the verdict makes matters for a calculation. **`YES - guaranteed` means the bound the
library documents for that combination is at or below what you asked for.** The middle state,
`only measured, not guaranteed`, means a figure exists but it is an observation rather than a promise.
It is deliberately not treated as a pass. A request nothing can meet comes back `NO`, not as a
quietly rounded yes. The last line above is that case.

"Is this accurate enough?" and "which of these two is more accurate?" are different questions with
different answers, and the function above answers only the first. For the second, `BoysAccuracyDelivered`
returns the figure a combination was *measured* to deliver. The [API reference](mainpage.md) documents
both accessors and says which reading a calculation's safety may rest on.

**Next:** `YES - guaranteed` tells you the answer is good enough. The option probe tells you which of
the good-enough options is fastest where you are.

---

## I want to know which option is fastest on this machine

`examples/07_option_probe.cpp` — this is a property of the machine and the build flags, so no
document can answer it for you. It has to be measured where it will run.

```cpp
const boys::OptionProbeReport report = boys::RunOptionProbe(options);
```

    measured 2018 options on 12 logical processors

    fp64 all-orders - 72 options ranked
      fastest: uniform-pack-orders-horner-exact-division-fp64 at 44.54 ns/argument
      the run could not separate 2 of them from the leader
        uniform-pack-orders-horner-plain-reciprocal-fp64
        uniform-pack-orders-horner-fp64

    recommended here: uniform-pack-orders-horner-exact-division-fp64

**The line counts are that build's, not this one's.** The transcript above was taken on the build this
document was written against, and the option space has changed size since, so **your counts will
differ from the two headline numbers** — that is the space growing and shrinking, not a fault in your
build. The counts to read are the ones your own run prints.

**The important line is the third one, not the second.** Of the 72 options in that group, the run
could not place 2 behind the leader. The "fastest" row is therefore one of three that are effectively
tied, and that is the real answer. The probe says so rather than inventing a winner from a timing
difference too small to measure. Read the result as *a group at the top and a tail behind it*. Being
on the leader is worth little. Being well behind the group is worth fixing.

**Every figure in that transcript is a property of this host and this run.** The nanosecond figures
are wall-clock measurements. Which options tie follows from them, and the headline counts follow the
build. What carries to your machine is the shape of the answer — a leader, a tied group, and a tail —
not the numbers in it.

That distinction is also why these figures should not be copied. On the machine that wrote this page
the same program was run several times while other work was going on. The leader changed each time,
and so did the length of the tied group. A timing taken on a busy machine measures the machine. Run it
on a machine you are not otherwise using, and re-run it if you change the build flags.

The example runs a deliberately short protocol so it can run inside a build. The command-line tool
`boys-option-probe` runs the full one and prints every group, every figure beside its spread, and the
machine load the rounds were taken under.

Once you have a preference, name it at your call site — [the section
above](#i-want-to-name-a-specific-evaluation) — or point `BOYS_BUILD_DEFAULTS` at a header carrying
your own choices for a whole build: the five a host class falls back to and the device lane's own
two. Every call site that names nothing then compiles the choices
your probe measured. `CONTRIBUTING.md` describes the second route.

---

## Which entry do I call?

By what you want, not by what the library calls things:

| I want | Entry | One clause on when |
|---|---|---|
| F_n(x) for one order, one argument | `BoysSingle` | the simplest call; use it when you have one value |
| F_0..F_nmax at one argument | `BoysAllOrders` | a shell quartet's ladder; this is the common case |
| F_n at every argument of an array | `BoysFixedN` | when the order is fixed and the arguments vary; takes an output stride |
| F_0..F_nmax at every argument of an array | `BoysAllN` | the batch shape of an integral engine; the arguments may arrive in any order |
| ...with each argument's own top order | `BoysAllNAtOrders` | a shell-quartet batch, where no argument is padded to a common order |
| ...when you know the arguments are already sorted | `BoysAllN` with `BoysSortedArgs{}` | skips the internal sort when your loop already produces a non-decreasing array |
| any of the above in single precision | `BoysSingleF32`, `BoysAllOrdersF32`, `BoysFixedNF32`, `BoysAllNF32`, `BoysAllNAtOrdersF32` | when the rest of your kernel is `float` |
| half-precision storage | `BoysSingleF16`, `BoysAllOrdersF16`, `BoysFixedNF16`, `BoysAllNF16`, `BoysAllNAtOrdersF16`, `BoysSingleBf16`, `BoysAllOrdersBf16` | 16-bit I/O around the single-precision engine |
| an answer on a GPU | `boys/boys_cuda.hpp` | device arrays; see [below](#on-a-gpu-the-first-call-is-the-slow-one) |
| the same arithmetic inside your own CUDA kernel | `boys/boys_cuda_device.hpp` | when a round trip through global memory would cost more than the evaluation |

The complete list, with every overload and the arithmetic behind it, is the entry-point table in the
[API reference](mainpage.md).

## What you can and cannot choose at run time

The settings that select an evaluation fall into two kinds. Two can be named by a value the program
computes. The rest can only be named where the call is compiled:

| Setting | Can you name it at run time? | How |
|---|---|---|
| which stored fit serves the interval | **yes** | `BoysAllOrdersWithRoute`, `BoysSingleF32WithRoute` |
| how the coefficients are summed | **yes** | the `(route, scheme)` overload of `BoysAllOrdersWithRoute` — the ladder shape; the float lane's single-order entry takes a route and no scheme |
| the internal precision budget | no | template argument only — `EvalPolicy`'s third parameter |
| the packing axis (whether a vector register holds four arguments or four orders) | no | template argument only — `EvalPolicy`'s fourth parameter |
| how finely the fitted interval is cut | no | template argument only — `EvalPolicy`'s fifth parameter |
| how the recursion divides | no | template argument only — `EvalPolicy`'s sixth parameter |
| which exponential seeds a region-B ladder | no | template argument only — `EvalPolicy`'s seventh parameter |

The ones that cannot be named late are fixed when the translation unit is compiled. **This revision
offers no run-time entry for them.** A choice that has to be made per input record needs an `if` over
two instantiations at the call site, not a value passed into one.

## On a GPU, the first call is the slow one

The GPU entries upload coefficient tables on first use, lazily and idempotently. **The first call on
a given device therefore pays a real one-time cost that every later
call does not.** A caller timing a loop from a cold start measures the upload, not the evaluation.

If you are benchmarking or budgeting latency, warm the path with one throwaway call of the entry
you intend to use, and time everything after it. A warm-up is also what `BoysCuda::InitializeTables`
is for, if you would rather pay the cost where you can see it than inside your first real call.

## What you do not pay for

A call that names no policy is resolved entirely at compile time. It compiles to
that one call — no run-time dispatch, no branch on an axis, nothing to predict. **The library has
already compiled that specialization into its own archive**, so your translation unit links against
it instead of instantiating the kernel a second time. An unnamed call therefore costs you neither
run-time selection nor template-instantiation time. Naming another policy is
what instantiates the version you asked for, in your translation unit.

## Where the specification lives

- [The per-lane contract](lane-contract.md) — the bound each lane (one entry together with the
  arithmetic behind it) guarantees, over which interval, and where it stops.
- [Choosing a lane](consumer-perspective.md) — how much accuracy an integral calculation actually
  needs, and which lanes that leaves.
- [The API reference](mainpage.md) — every entry, every type, and the vocabulary this library uses,
  defined.
